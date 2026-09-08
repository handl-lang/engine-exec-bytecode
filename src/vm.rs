//! Bytecode VM internals: scope stack, numeric helpers, and the execution
//! frame.
//!
//! [`VmFrame`] executes a compiled [`Vec<Instr>`](crate::Instr) program
//! against an operand stack and a scope stack. [`Instr::EvalChild`] and
//! [`Instr::Call`] recurse into [`VmFrame::exec_node`], which compiles the
//! node's scheme (via a per-node cache) and executes.
//!
//! ## Value shapes
//!
//! The v2 VM operates on the same tagged JSON value shapes as the v2
//! tree-walk handler:
//!
//! - **Pair** (`first`): `{"__pair__": [a, b]}`.
//! - **Either** (`Choice` / `tag`): `{"__either__": "left"|"right", "value": v}`.
//! - **Arrow-value** (`App`): `{"__arrow__": true, "params": [...], "body": <idx>, "env": {...}}`.
//!
//! ## Loop execution model
//!
//! [`Instr::LoopCheck`] implements causal/strict feedback. The compiler
//! emits `Push(Null), Pair` to seed the initial pair `(incoming, Null)`, then
//! the body code, then `LoopCheck(loop_back)`. Each iteration: the body
//! consumes the pair from the stack and leaves its result. `LoopCheck` pops
//! the result and inspects its shape:
//! - **Either left** -> push the carried value, fall through (done).
//! - **Either right** -> the carried value is the new state; reconstruct
//!   `pair(original_input, new_state)`, push it, jump to `loop_back`.
//! - **Pair** (not an Either) -> single-step mode: push `fst`, fall through.
//!
//! A max-iteration counter (100_000) prevents infinite loops. The
//! `original_input` is saved by a look-ahead in the `Push` handler: when
//! `Push(Null)` is followed by `Pair` and no loop is active, the stack top
//! (the incoming value) is saved as `loop_input` before `Pair` consumes it.
//!
//! ## Native ops and `incoming`
//!
//! In the tree-walk, `incoming` is passed explicitly to each `eval_inner`
//! call. In the VM, the incoming value flows through the operand stack. For
//! 0-arg native ops that use `incoming` (`id`, `tag`, `is_pos`, `dec`,
//! `inc`, `half_floor`, `inc_fst`, `countdown_step`, `infinite_step`), the
//! incoming value IS the top of the stack -- the op pops it and pushes its
//! result. For ops with args (`add`, `sub`, etc.), the args are on the stack
//! and `incoming` is not used. For `closure` (0-arg, no incoming), nothing is
//! popped.
//!
//! ## Derived step ops
//!
//! All four derived step ops from ADR-0052 are supported:
//! - `loop_step` (the ergonomic `loop` surface lowering): the `NativeOp`
//!   handler pops the loop pair from the operand stack (below the
//!   inline-compiled args) and passes it as `incoming` to
//!   [`VmFrame::apply_native_v2`], which applies the body closure via
//!   [`VmFrame::apply_app`].
//! - `while_step`/`for_step`/`fix_step`: the `DerivedStepOp` instruction
//!   re-evaluates the arg sub-programs per iteration with the current value
//!   (extracted from the loop pair) as incoming, then dispatches to
//!   [`VmFrame::apply_native_v2`]. See the [`instr`](crate::instr) module
//!   docs for details.

use std::rc::Rc;

use handl_language::{Ast, AstNode, AstRef, LangError, Language, Value};
use handl_plugin_engine_exec::pattern_match::{self, PatternMatchCtx};
use handl_value::{TaggedValue, tagged_to_value};
use rustc_hash::FxHashMap;

use crate::compile::compile;
use crate::instr::Instr;

// ---------------------------------------------------------------------------
// Scope stack — identical semantics to the tree-walk interpreter
// ---------------------------------------------------------------------------

/// A scope stack: a flat `Vec<(Rc<str>, TaggedValue)>` of bindings with
/// scope markers, searched innermost-first.
///
/// Mirrors the tree-walk's `ScopeStack` exactly: `push_scope` records a
/// mark; lookup searches innermost-first (linear scan from the end);
/// closure calls swap in the captured environment, bind params, evaluate
/// the body, then restore the original scope stack. For the small scopes
/// typical in handler programs (3-10 variables), a linear scan is faster
/// than `HashMap` because there is no hashing overhead.
pub(crate) struct ScopeStack {
    /// Flat list of (name, value) bindings, innermost last.
    pub(crate) bindings: Vec<(Rc<str>, TaggedValue)>,
    /// Stack of mark positions — each `push_scope` records the current
    /// length, each `pop_scope` truncates back to that length.
    pub(crate) marks: Vec<usize>,
    /// The captured environment root for the current closure invocation.
    /// Searched after `bindings` so user-defined scopes shadow captured
    /// variables. Stored as `Rc` so closure calls avoid deep-cloning the
    /// entire captured `HashMap`.
    pub(crate) captured_root: Option<Rc<FxHashMap<Rc<str>, TaggedValue>>>,
}

impl ScopeStack {
    pub(crate) fn new() -> Self {
        Self {
            bindings: Vec::new(),
            marks: vec![0],
            captured_root: None,
        }
    }

    /// Look up a variable, searching innermost bindings first (linear scan
    /// from the end), then falling back to the captured root (if any).
    pub(crate) fn lookup(&self, name: &str) -> Option<&TaggedValue> {
        // Search innermost bindings first (reverse linear scan).
        for (n, v) in self.bindings.iter().rev() {
            if n.as_ref() == name {
                return Some(v);
            }
        }
        // Fall back to captured root.
        self.captured_root.as_ref().and_then(|m| m.get(name))
    }

    /// Define a variable in the current (innermost) scope.
    pub(crate) fn define(&mut self, name: &str, value: TaggedValue) {
        self.bindings.push((Rc::from(name), value));
    }

    /// Push a new empty scope by recording the current bindings length.
    pub(crate) fn push_scope(&mut self) {
        self.marks.push(self.bindings.len());
    }

    /// Pop the innermost scope by truncating bindings back to the last
    /// mark. Never pops the root scope (the `marks` stack always keeps at
    /// least one entry).
    pub(crate) fn pop_scope(&mut self) {
        if self.marks.len() > 1 {
            if let Some(mark) = self.marks.pop() {
                self.bindings.truncate(mark);
            }
        }
    }

    /// Capture all visible variables into a single flat map.
    ///
    /// Used by the `closure` primitive to snapshot the current environment.
    /// Inner scopes shadow outer ones (later entries overwrite earlier).
    pub(crate) fn capture(&self) -> FxHashMap<Rc<str>, TaggedValue> {
        self.bindings.iter().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Numeric helpers — identical semantics to the tree-walk interpreter
// ---------------------------------------------------------------------------

/// A number that can be either integer or float.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) enum Num {
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
}

impl Num {
    /// Convert to `f64`.
    pub(crate) fn to_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
}

/// Extract a number from a [`TaggedValue`].
pub(crate) fn as_num(v: &TaggedValue) -> Result<Num, LangError> {
    match v {
        TaggedValue::Int(i) => Ok(Num::Int(*i)),
        TaggedValue::Float(f) => Ok(Num::Float(*f)),
        _ => Err(LangError::engine(format!("expected number, got {:?}", v))),
    }
}

/// Convert a [`Num`] back to a [`TaggedValue`].
pub(crate) fn num_value(n: Num) -> TaggedValue {
    match n {
        Num::Int(i) => TaggedValue::Int(i),
        Num::Float(f) => TaggedValue::Float(f),
    }
}

/// Whether a [`TaggedValue`] is truthy.
///
/// `false`, `null`, `0`, and `""` are falsy; everything else is truthy.
pub(crate) fn is_truthy(v: &TaggedValue) -> bool {
    match v {
        TaggedValue::Bool(b) => *b,
        TaggedValue::Null => false,
        TaggedValue::Int(i) => *i != 0,
        TaggedValue::Float(f) => *f != 0.0,
        TaggedValue::Str(s) => !s.is_empty(),
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Pair / Either value-shape helpers (mirror the v2 tree-walk)
// ---------------------------------------------------------------------------

/// Construct a pair value `(a, b)`.
fn pair(a: TaggedValue, b: TaggedValue) -> TaggedValue {
    handl_value::pair(a, b)
}

/// Extract the first component of a pair value, or `None` if not a pair.
fn pair_fst(v: &TaggedValue) -> Option<&TaggedValue> {
    handl_value::pair_fst(v)
}

/// Extract the second component of a pair value, or `None` if not a pair.
fn pair_snd(v: &TaggedValue) -> Option<&TaggedValue> {
    handl_value::pair_snd(v)
}

/// Return the type tag string for a [`TaggedValue`] (ADR-0060).
fn type_tag(v: &TaggedValue) -> &'static str {
    match v {
        TaggedValue::Null => "null",
        TaggedValue::Bool(_) => "bool",
        TaggedValue::Int(_) | TaggedValue::Float(_) => "number",
        TaggedValue::Str(_) => "string",
        TaggedValue::Arr(_) => "array",
        TaggedValue::Obj(_) => "object",
        TaggedValue::Pair(..) => "object",
        TaggedValue::EitherLeft(_) | TaggedValue::EitherRight(_) => "object",
        TaggedValue::Arrow { .. } => "object",
        // ADR-0083 §2.1: `Bytes` is a scalar type in `value_types.yaml`.
        TaggedValue::Bytes(_) => "bytes",
        // ADR-0083 §2.2: opaque foreign resource handle.
        TaggedValue::Resource(_) => "resource",
    }
}

/// Construct a `left` Either value.
fn either_left(v: TaggedValue) -> TaggedValue {
    handl_value::either_left(v)
}

/// Construct a `right` Either value.
fn either_right(v: TaggedValue) -> TaggedValue {
    handl_value::either_right(v)
}

/// Whether an Either value is a `left`.
fn is_either_left(v: &TaggedValue) -> bool {
    handl_value::is_either_left(v)
}

/// Whether an Either value is a `right`.
fn is_either_right(v: &TaggedValue) -> bool {
    handl_value::is_either_right(v)
}

/// Extract the carried value from an Either.
fn either_value(v: &TaggedValue) -> Option<&TaggedValue> {
    handl_value::either_value(v)
}

// ---------------------------------------------------------------------------
// VmFrame -- the v2 execution context
// ---------------------------------------------------------------------------

/// The execution frame for the v2 bytecode VM.
///
/// Owns the scope stack and a per-node v2 compilation cache. The `ast` and
/// `lang` references are immutable; `modules` is the optional module registry
/// for `Foreign` dispatch and `import_resolver` is the optional resolver for
/// `import "path"` statements.
pub(crate) struct VmFrame<'a, 'h> {
    /// The AST being evaluated.
    ast: &'a Ast,
    /// The language definition (for scheme lookup during compilation).
    lang: &'a Language,
    /// The module registry (for `Foreign` dispatch).
    modules: Option<&'h Rc<dyn handl_platform::ModuleLookup>>,
    /// The import resolver for `import "path"` statements (Phase 9.5).
    import_resolver: Option<&'h Rc<dyn handl_plugin_engine_exec::ImportResolver>>,
    /// Per-node v2 compilation cache: `AstRef index -> compiled instructions`.
    cache: FxHashMap<usize, Rc<[Instr]>>,
    /// The scope stack.
    scopes: ScopeStack,
    /// Reusable operand stack for `exec_instrs_inner`. Taken via
    /// `mem::take` at the start of each call and returned at the end, so
    /// the allocation is reused across calls. Reentrant calls (when
    /// `apply_app`/`exec_node_inner` re-enters `exec_instrs_inner`) find
    /// `scratch_stack` empty and allocate fresh — this is rare and
    /// correct (the inner call's stack is discarded when the outer call
    /// puts its own stack back).
    scratch_stack: Vec<TaggedValue>,
}

impl<'a, 'h> VmFrame<'a, 'h> {
    /// Create a new v2 frame with a fresh cache and scope stack.
    pub(crate) fn new(
        ast: &'a Ast,
        lang: &'a Language,
        modules: Option<&'h Rc<dyn handl_platform::ModuleLookup>>,
        import_resolver: Option<&'h Rc<dyn handl_plugin_engine_exec::ImportResolver>>,
    ) -> Self {
        Self {
            ast,
            lang,
            modules,
            import_resolver,
            cache: FxHashMap::default(),
            scopes: ScopeStack::new(),
            scratch_stack: Vec::new(),
        }
    }

    /// Define a global variable in the root scope.
    pub(crate) fn define_global(&mut self, name: &str, value: TaggedValue) {
        self.scopes.define(name, value);
    }
}

/// `PatternMatchCtx` impl for the bytecode VM (ADR-0061).
///
/// The VM implements the trait directly: `define_binding` delegates to
/// the scope stack, and `eval_pattern_node` delegates to `exec_node`
/// (which uses `self.ast` internally; the `ast` parameter from the trait
/// is the same AST).
impl<'a, 'h> PatternMatchCtx for VmFrame<'a, 'h> {
    fn define_binding(&mut self, name: &str, value: Value) {
        self.scopes.define(name, TaggedValue::from(value));
    }

    fn eval_pattern_node(&mut self, _ast: &Ast, node_ref: AstRef) -> Result<Value, LangError> {
        let val = self.exec_node_inner(node_ref)?;
        Ok(tagged_to_value(&val))
    }
}

impl<'a, 'h> VmFrame<'a, 'h> {
    /// Compile (if not cached) and execute a single node using v2 bytecode.
    ///
    /// Looks up `lang.eval[node.kind]` to find the node's `Scheme`, compiles
    /// it to v2 instructions (or uses the no-scheme fallback
    /// `[LoadPayload, Return]`), caches the result, and executes. On
    /// subsequent calls for the same [`AstRef`], the cached instructions are
    /// reused.
    ///
    /// This is the public(crate) boundary: returns a
    /// [`serde_json::Value`], converting from the internal
    /// [`TaggedValue`] representation via [`tagged_to_value`].
    pub(crate) fn exec_node(&mut self, node_ref: AstRef) -> Result<Value, LangError> {
        let val = self.exec_node_inner(node_ref)?;
        Ok(tagged_to_value(&val))
    }

    /// Internal node evaluator — the [`TaggedValue`] counterpart of
    /// [`exec_node`](Self::exec_node). Returns a [`TaggedValue`].
    pub(crate) fn exec_node_inner(&mut self, node_ref: AstRef) -> Result<TaggedValue, LangError> {
        let node = self.ast.at(node_ref)?;

        // Compile-on-demand with cache.
        if !self.cache.contains_key(&node_ref.index()) {
            let scheme_def = self.lang.eval.get_by_identity(&node.kind);
            let instrs = match scheme_def {
                Some(def) => compile(&def.scheme, self.lang, self.ast, node)?,
                None => vec![Instr::LoadPayload, Instr::Return],
            };
            self.cache.insert(node_ref.index(), Rc::from(instrs));
        }

        // Rc-clone the instruction slice (cheap refcount bump) to avoid
        // borrow conflict with recursive exec_node calls that may insert
        // into the cache.
        let instrs = self
            .cache
            .get(&node_ref.index())
            // INFALLIBLE: we just inserted this key (or it was cached from a
            // prior call) on the line above, so the lookup cannot miss.
            .expect("just inserted or cached")
            .clone();
        self.exec_instrs_inner(&instrs, node, None)
    }

    /// Execute a program of v2 instructions against the given current node.
    ///
    /// `incoming` is the value fed by an enclosing combinator (top-level
    /// calls pass `None`). The operand stack is fresh; the scope stack
    /// (`self.scopes`) is shared across recursive `exec_node` calls.
    ///
    /// This is the public(crate) boundary: accepts and returns
    /// [`serde_json::Value`], converting to/from the internal
    /// [`TaggedValue`] representation.
    pub(crate) fn exec_instrs(
        &mut self,
        instrs: &[Instr],
        node: &AstNode,
        incoming: Option<Value>,
    ) -> Result<Value, LangError> {
        let val = self.exec_instrs_inner(instrs, node, incoming.map(TaggedValue::from))?;
        Ok(tagged_to_value(&val))
    }

    /// Internal instruction executor — the [`TaggedValue`] counterpart of
    /// [`exec_instrs`](Self::exec_instrs). Operates on a `Vec<TaggedValue>`
    /// operand stack and returns a [`TaggedValue`].
    pub(crate) fn exec_instrs_inner(
        &mut self,
        instrs: &[Instr],
        node: &AstNode,
        incoming: Option<TaggedValue>,
    ) -> Result<TaggedValue, LangError> {
        // Reuse the scratch stack allocation across calls. `mem::take`
        // leaves `self.scratch_stack` empty, so reentrant calls (via
        // `apply_app`/`exec_node_inner`) allocate fresh — this is rare and
        // correct. We put our stack back at the end, overwriting whatever
        // the inner call left.
        let mut stack = std::mem::take(&mut self.scratch_stack);
        stack.clear();
        // Push the incoming value onto the stack so that 0-arg ops using
        // `op_uses_incoming` (like `id`) can pop it. At the top level
        // (incoming = None), the stack starts empty — matching the
        // tree-walk's `incoming: None` behavior.
        if let Some(v) = &incoming {
            stack.push(v.clone());
        }
        let mut current_incoming = incoming;
        let mut loop_counter: usize = 0;
        const MAX_ITERS: usize = 100_000;
        let mut loop_input: Option<TaggedValue> = None;

        let mut ip = 0;
        while ip < instrs.len() {
            match &instrs[ip] {
                Instr::Push(v) => {
                    // Look-ahead: if this is Push(Null) followed by Pair and
                    // no loop is active, save the stack top as the original
                    // loop input. The compiler emits Push(Null), Pair as the
                    // loop prologue; the stack top is the incoming value that
                    // Pair will consume into the initial pair.
                    if loop_input.is_none()
                        && loop_counter == 0
                        && v.is_null()
                        && ip + 1 < instrs.len()
                        && matches!(instrs[ip + 1], Instr::Pair)
                    {
                        loop_input = stack.last().cloned();
                    }
                    stack.push(TaggedValue::from(v.clone()));
                }
                Instr::Load(name) => {
                    let val = self
                        .scopes
                        .lookup(name)
                        .cloned()
                        .unwrap_or_else(|| TaggedValue::Str(Rc::clone(name)));
                    stack.push(val);
                }
                Instr::LoadPayloadVar => {
                    let var_name = node
                        .payload
                        .as_str()
                        .ok_or_else(|| {
                            LangError::engine(
                                "expected string payload for $payload var".to_string(),
                            )
                        })?
                        .to_string();
                    let val = self
                        .scopes
                        .lookup(&var_name)
                        .cloned()
                        .unwrap_or_else(|| TaggedValue::Str(Rc::from(var_name.as_str())));
                    stack.push(val);
                }
                Instr::LoadPayload => {
                    stack.push(TaggedValue::from(node.payload.clone()));
                }
                Instr::EvalChild(n) => {
                    let children = node
                        .payload
                        .get("children")
                        .and_then(|c| c.as_array())
                        .ok_or_else(|| {
                            LangError::engine_node(
                                "expected children array in payload".to_string(),
                                node,
                            )
                        })?;
                    let arena_idx = children.get(*n).and_then(|v| v.as_u64()).ok_or_else(|| {
                        LangError::engine_node(format!("child {} not found", n), node)
                    })?;
                    let child_ref = AstRef::new(arena_idx as usize);
                    let result = self.exec_node_inner(child_ref)?;
                    stack.push(result);
                }
                Instr::Store(name) => {
                    let val = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Store".to_string(), node)
                    })?;
                    self.scopes.push_scope();
                    self.scopes.define(name, val);
                }
                Instr::StorePayloadName => {
                    let val = stack.pop().ok_or_else(|| {
                        LangError::engine_node(
                            "stack underflow on StorePayloadName".to_string(),
                            node,
                        )
                    })?;
                    let bind_name = node
                        .payload
                        .get("name")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            LangError::engine_node(
                                "expected name in payload for $payload extend_env",
                                node,
                            )
                        })?
                        .to_string();
                    self.scopes.push_scope();
                    self.scopes.define(&bind_name, val);
                }
                Instr::PopScope => {
                    self.scopes.pop_scope();
                }
                Instr::Call(n) => {
                    let mut args = Vec::with_capacity(*n);
                    for _ in 0..*n {
                        args.push(stack.pop().ok_or_else(|| {
                            LangError::engine_node("stack underflow on Call".to_string(), node)
                        })?);
                    }
                    args.reverse();
                    let callee = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call".to_string(), node)
                    })?;
                    // ADR-0060: unpack single array arg for variable-arity
                    // call. When Call receives a single argument that is an
                    // array, unpack it into individual arguments.
                    let actual_args: Vec<TaggedValue> = if args.len() == 1 && args[0].is_array() {
                        args[0]
                            .as_array()
                            .ok_or_else(|| {
                                LangError::engine_node(
                                    "Call: expected array argument".to_string(),
                                    node,
                                )
                            })?
                            .iter()
                            .cloned()
                            .collect()
                    } else {
                        args
                    };
                    let result = self.apply_app(&callee, &actual_args)?;
                    stack.push(result);
                }
                // Fixed-arity Call variants: avoid the per-call `Vec`
                // allocation for the common 1/2/3-arg cases. Args are pushed
                // in source order, so they're popped in reverse order. The
                // ADR-0060 single-array-arg unpacking is replicated for
                // `Call1` (the only arity where it can apply).
                Instr::Call1 => {
                    let arg0 = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call1".to_string(), node)
                    })?;
                    let callee = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call1".to_string(), node)
                    })?;
                    // ADR-0060: unpack single array arg for variable-arity
                    // call.
                    let result = if arg0.is_array() {
                        let actual: Vec<TaggedValue> = arg0
                            .as_array()
                            .ok_or_else(|| {
                                LangError::engine_node(
                                    "Call1: expected array argument".to_string(),
                                    node,
                                )
                            })?
                            .iter()
                            .cloned()
                            .collect();
                        self.apply_app(&callee, &actual)?
                    } else {
                        self.apply_app(&callee, std::slice::from_ref(&arg0))?
                    };
                    stack.push(result);
                }
                Instr::Call2 => {
                    let arg1 = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call2".to_string(), node)
                    })?;
                    let arg0 = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call2".to_string(), node)
                    })?;
                    let callee = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call2".to_string(), node)
                    })?;
                    let result = self.apply_app(&callee, &[arg0, arg1])?;
                    stack.push(result);
                }
                Instr::Call3 => {
                    let arg2 = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call3".to_string(), node)
                    })?;
                    let arg1 = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call3".to_string(), node)
                    })?;
                    let arg0 = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call3".to_string(), node)
                    })?;
                    let callee = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Call3".to_string(), node)
                    })?;
                    let result = self.apply_app(&callee, &[arg0, arg1, arg2])?;
                    stack.push(result);
                }
                Instr::ChoiceJump(left_target, right_target) => {
                    let either = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on ChoiceJump".to_string(), node)
                    })?;
                    let val = either_value(&either).cloned().ok_or_else(|| {
                        LangError::engine_node("Choice: Either missing value".to_string(), node)
                    })?;
                    if is_either_left(&either) {
                        stack.push(val);
                        ip = *left_target;
                        continue;
                    } else if is_either_right(&either) {
                        stack.push(val);
                        ip = *right_target;
                        continue;
                    } else {
                        return Err(LangError::engine(
                            "Choice: incoming is not an Either".to_string(),
                        ));
                    }
                }
                Instr::Jump(target) => {
                    ip = *target;
                    continue;
                }
                Instr::RouteOnType(tag) => {
                    let val = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on RouteOnType".to_string(), node)
                    })?;
                    if type_tag(&val) == tag.as_str() {
                        stack.push(either_left(val));
                    } else {
                        stack.push(either_right(val));
                    }
                }
                Instr::NativeOp(op, n) => {
                    // For 0-arg ops that use `incoming`: the incoming value
                    // is on top of the stack. Pop it and pass as `incoming`.
                    // For other ops: pop n args, use current_incoming.
                    //
                    // Derived step ops (`loop_step`) are a hybrid: they carry
                    // sub-scheme args (the body closure) compiled inline, but
                    // they also consume the loop pair `(in, state)` that the
                    // enclosing `Loop` left on the stack below those args.
                    // Pop the args first, then pop the pair as `incoming`.
                    let is_derived_step = op_is_derived_step(op);
                    let inc = if is_derived_step {
                        // Filled below after popping args (the pair sits
                        // beneath the args on the stack).
                        None
                    } else if *n == 0 && op_uses_incoming(op) {
                        stack.pop()
                    } else {
                        current_incoming.take()
                    };
                    let mut args = Vec::with_capacity(*n);
                    for _ in 0..*n {
                        args.push(stack.pop().ok_or_else(|| {
                            LangError::engine_node("stack underflow on NativeOp".to_string(), node)
                        })?);
                    }
                    args.reverse();
                    let inc = if is_derived_step {
                        Some(stack.pop().ok_or_else(|| {
                            LangError::engine(
                                "derived step op: missing loop pair on stack".to_string(),
                            )
                        })?)
                    } else {
                        inc
                    };
                    let result = self.apply_native_v2(op, &args, node, inc)?;
                    stack.push(result);
                }
                Instr::ForeignOp(name, n) => {
                    let mut args = Vec::with_capacity(*n);
                    for _ in 0..*n {
                        args.push(stack.pop().ok_or_else(|| {
                            LangError::engine_node("stack underflow on ForeignOp".to_string(), node)
                        })?);
                    }
                    args.reverse();
                    let modules = self.modules.ok_or_else(|| {
                        LangError::engine_node(format!("unregistered foreign: {}", name), node)
                    })?;
                    let func = modules.lookup(name).ok_or_else(|| {
                        LangError::engine_node(format!("unregistered foreign: {}", name), node)
                    })?;
                    let val_args: Vec<Value> = args.iter().map(tagged_to_value).collect();
                    let mut pctx = handl_platform::EvalContext::new();
                    let result = func
                        .call(&val_args, &mut pctx)
                        .map_err(|e| LangError::engine_node(e.to_string(), node))?;
                    stack.push(TaggedValue::from(result));
                }
                Instr::FirstOp(start, end) => {
                    // Pop the pair from the stack, extract fst/snd, run the
                    // inner sub-program at [start, end) with fst as incoming
                    // on a fresh stack, then re-pair the result with snd.
                    let p = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on FirstOp".to_string(), node)
                    })?;
                    let a = pair_fst(&p)
                        .ok_or_else(|| {
                            LangError::engine_node("FirstOp: not a pair".to_string(), node)
                        })?
                        .clone();
                    let b = pair_snd(&p)
                        .ok_or_else(|| {
                            LangError::engine_node("FirstOp: not a pair".to_string(), node)
                        })?
                        .clone();
                    let sub = &instrs[*start..*end];
                    let result = self.exec_instrs_inner(sub, node, Some(a))?;
                    stack.push(pair(result, b));
                    ip = *end;
                    continue;
                }
                Instr::Pair => {
                    let a = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Pair".to_string(), node)
                    })?;
                    let b = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Pair".to_string(), node)
                    })?;
                    stack.push(pair(b, a));
                }
                Instr::LoopCheck(loop_back) => {
                    loop_counter += 1;
                    if loop_counter > MAX_ITERS {
                        return Err(LangError::engine(
                            "Loop: exceeded max iterations".to_string(),
                        ));
                    }
                    let result = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on LoopCheck".to_string(), node)
                    })?;
                    if is_either_left(&result) {
                        let out = either_value(&result).cloned().ok_or_else(|| {
                            LangError::engine_node(
                                "Loop: left Either missing value".to_string(),
                                node,
                            )
                        })?;
                        stack.push(out);
                        loop_counter = 0;
                        loop_input = None;
                    } else if is_either_right(&result) {
                        let new_state = either_value(&result).cloned().ok_or_else(|| {
                            LangError::engine_node(
                                "Loop: right Either missing value".to_string(),
                                node,
                            )
                        })?;
                        let orig = loop_input.clone().unwrap_or(TaggedValue::Null);
                        stack.push(pair(orig, new_state));
                        ip = *loop_back;
                        continue;
                    } else {
                        let out = pair_fst(&result).ok_or_else(|| {
                            LangError::engine(
                                "Loop: body did not return a pair or Either".to_string(),
                            )
                        })?;
                        stack.push(out.clone());
                        loop_counter = 0;
                        loop_input = None;
                    }
                }
                Instr::Return => {
                    let result = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on Return".to_string(), node)
                    })?;
                    // Return the scratch stack allocation for reuse by the
                    // next `exec_instrs_inner` call.
                    self.scratch_stack = stack;
                    return Ok(result);
                }
                // --- Typed arithmetic/comparison bytecodes ---
                //
                // Pop two values (second arg on top, first arg below) and
                // push the result directly — no `Vec` allocation, no string
                // dispatch. Logic mirrors `apply_native_v2`'s `add`/`sub`/
                // `mul`/`div`/`mod`/`eq`/`neq`/`lt`/`lte`/`gt`/`gte` arms
                // exactly, so results are identical to the `NativeOp` path.
                Instr::Add | Instr::Sub | Instr::Mul | Instr::Div | Instr::Mod => {
                    let b = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on arithmetic op".to_string(), node)
                    })?;
                    let a = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on arithmetic op".to_string(), node)
                    })?;
                    let result = eval_bin_arith(&instrs[ip], a, b)?;
                    stack.push(result);
                }
                Instr::Eq | Instr::Neq | Instr::Lt | Instr::Lte | Instr::Gt | Instr::Gte => {
                    let b = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on comparison op".to_string(), node)
                    })?;
                    let a = stack.pop().ok_or_else(|| {
                        LangError::engine_node("stack underflow on comparison op".to_string(), node)
                    })?;
                    let result = eval_bin_cmp(&instrs[ip], a, b)?;
                    stack.push(result);
                }
                // --- Object construction bytecodes ---
                Instr::EvalNode(idx) => {
                    // Evaluate the AST node at the given absolute arena index
                    // and push the result. Used by `MakeObj` to evaluate
                    // object entry value nodes (which are nested inside
                    // `entry` nodes, not direct children of the `object`
                    // node).
                    let result = self.exec_node_inner(AstRef::new(*idx))?;
                    stack.push(result);
                }
                Instr::MakeObj { keys } => {
                    // Pop `keys.len()` values (in reverse order — values were
                    // pushed in source order) and construct an object with
                    // the pre-allocated keys. The keys are `Rc<str>` stored
                    // in the instruction (allocated once at compile time),
                    // so only a refcount bump is needed per use — no
                    // per-iteration `Rc::from(key)` allocation.
                    let mut obj =
                        FxHashMap::with_capacity_and_hasher(keys.len(), Default::default());
                    for key in keys.iter().rev() {
                        let val = stack.pop().ok_or_else(|| {
                            LangError::engine_node("stack underflow on MakeObj".to_string(), node)
                        })?;
                        obj.insert(Rc::clone(key), val);
                    }
                    stack.push(TaggedValue::Obj(Rc::new(obj)));
                }
                // --- Closure construction bytecode ---
                Instr::Closure { params, body } => {
                    // Capture the current environment and construct a
                    // `TaggedValue::Arrow`. Identical to the `closure`
                    // native op, but avoids the per-call `Vec` allocation
                    // for args, the string `match op` dispatch, and the
                    // node payload extraction — `params` and `body` are
                    // pre-computed at compile time.
                    let captured = self.scopes.capture();
                    stack.push(TaggedValue::Arrow {
                        params: Rc::clone(params),
                        body: *body,
                        env: Rc::new(captured),
                        home: None,
                    });
                }
                Instr::DerivedStepOp(op, ranges) => {
                    // Pop the loop pair (in, state) from the stack — left
                    // by the enclosing Loop. Extract the current value
                    // (in on first step, state thereafter).
                    let inc = stack.pop().ok_or_else(|| {
                        LangError::engine_node(
                            "DerivedStepOp: missing loop pair on stack".to_string(),
                            node,
                        )
                    })?;
                    let in_val = pair_fst(&inc).cloned().unwrap_or(TaggedValue::Null);
                    let state = pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null);
                    let current = if state.is_null() {
                        in_val.clone()
                    } else {
                        state.clone()
                    };
                    // Determine the incoming value for the arg sub-programs.
                    // - while_step / fix_step: pass `current` (the evolving
                    //   value) to cond/body.
                    // - for_step: pass `acc` (the second component of the
                    //   pair) to body — not the full pair. The pair is
                    //   (counter, acc); the body operates on `acc`.
                    let arg_incoming = if op == "for_step" {
                        if state.is_null() {
                            // First step: in_val = pair(count, initial).
                            // acc = snd(in_val).
                            pair_snd(&in_val).cloned().unwrap_or(TaggedValue::Null)
                        } else {
                            // Later steps: state = pair(counter, acc).
                            pair_snd(&state).cloned().unwrap_or(TaggedValue::Null)
                        }
                    } else {
                        current
                    };
                    // Re-evaluate each arg sub-program with `arg_incoming`
                    // as incoming. The sub-programs are at [start, end) in
                    // the current instruction stream.
                    let mut eval_args: Vec<TaggedValue> = Vec::with_capacity(ranges.len());
                    for &(start, end) in ranges {
                        let sub = &instrs[start..end];
                        let result =
                            self.exec_instrs_inner(sub, node, Some(arg_incoming.clone()))?;
                        eval_args.push(result);
                    }
                    // Dispatch to apply_native_v2 with the fresh results
                    // and the loop pair as incoming.
                    let result = self.apply_native_v2(op, &eval_args, node, Some(inc))?;
                    stack.push(result);
                }
            }
            ip += 1;
        }
        stack.pop().ok_or_else(|| {
            LangError::engine_node("empty operand stack at end of program".to_string(), node)
        })
    }

    /// Apply a native v2 op. Mirrors the v2 tree-walk's `apply_native` exactly.
    ///
    /// `incoming` is the value flowing through the arrow (used by `id`,
    /// `tag`, `is_pos`, `dec`, `inc`, `half_floor`, `inc_fst`,
    /// `countdown_step`, `infinite_step`).
    fn apply_native_v2(
        &mut self,
        op: &str,
        args: &[TaggedValue],
        node: &AstNode,
        incoming: Option<TaggedValue>,
    ) -> Result<TaggedValue, LangError> {
        match op {
            "id" => Ok(incoming
                .or_else(|| args.first().cloned())
                .unwrap_or(TaggedValue::Null)),
            "tag" => {
                let v = incoming
                    .or_else(|| args.first().cloned())
                    .unwrap_or(TaggedValue::Null);
                if is_truthy(&v) {
                    Ok(either_left(v))
                } else {
                    Ok(either_right(v))
                }
            }
            "closure" => {
                let payload = &node.payload;
                let (params, body) = if let (Some(p), Some(b)) = (
                    payload.get("params").and_then(|v| v.as_array()),
                    payload.get("body").and_then(|v| v.as_u64()),
                ) {
                    let param_names: Vec<String> = p
                        .iter()
                        .map(|v| {
                            v.as_str()
                                .ok_or_else(|| {
                                    LangError::engine_node(
                                        "closure: param must be a string".to_string(),
                                        node,
                                    )
                                })
                                .map(String::from)
                        })
                        .collect::<Result<_, _>>()?;
                    (param_names, b as usize)
                } else if let Some(children) = payload.get("children").and_then(|c| c.as_array()) {
                    let param_idx = children.first().and_then(|v| v.as_u64()).ok_or_else(|| {
                        LangError::engine_node("closure: missing param child".to_string(), node)
                    })? as usize;
                    let body_idx = children.get(1).and_then(|v| v.as_u64()).ok_or_else(|| {
                        LangError::engine_node("closure: missing body child".to_string(), node)
                    })? as usize;
                    let param_node = self.ast.at(AstRef::new(param_idx))?;
                    // Collection-vs-single params is a shape question
                    // (ADR-0081): a shape declaring a variadic role (e.g.
                    // `params`) is a param list; the legacy `paren`/`array`
                    // kinds remain only as the role-less fallback.
                    let param_names: Vec<String> = if self.lang.has_variadic_role(&param_node.kind)
                    {
                        let param_children = param_node
                            .payload
                            .get("children")
                            .and_then(|c| c.as_array())
                            .ok_or_else(|| {
                                LangError::engine_node(
                                    "closure: params missing children".to_string(),
                                    node,
                                )
                            })?;
                        let mut names = Vec::with_capacity(param_children.len());
                        for pc in param_children {
                            let pidx = pc.as_u64().ok_or_else(|| {
                                LangError::engine(
                                    "closure: param index must be a number".to_string(),
                                )
                            })? as usize;
                            let pnode = self.ast.at(AstRef::new(pidx))?;
                            let pname = self
                                .lang
                                .param_name(self.ast, pnode)
                                .map_err(|e| LangError::engine_node(e.to_string(), node))?;
                            names.push(pname);
                        }
                        names
                    } else {
                        let pname = self
                            .lang
                            .param_name(self.ast, param_node)
                            .map_err(|e| LangError::engine_node(e.to_string(), node))?;
                        vec![pname]
                    };
                    (param_names, body_idx)
                } else {
                    return Err(LangError::engine(
                        "closure: missing params/body in payload".to_string(),
                    ));
                };
                let captured = self.scopes.capture();
                Ok(TaggedValue::Arrow {
                    params: Rc::from(params),
                    body,
                    env: Rc::new(captured),
                    home: None,
                })
            }
            "array" => {
                // Two payload shapes for `array`:
                // - Rich: `{"elements": [v1, v2, ...]}` — literal values.
                // - Pratt-parsed: `{"children": [elem_idx, ...]}` — each
                //   element is an arena index to evaluate.
                if let Some(elements) = node.payload.get("elements") {
                    Ok(TaggedValue::from(elements.clone()))
                } else if let Some(children) =
                    node.payload.get("children").and_then(|c| c.as_array())
                {
                    let mut result = Vec::with_capacity(children.len());
                    for entry in children {
                        let idx = entry.as_u64().ok_or_else(|| {
                            LangError::engine_node(
                                "array: element index must be a number".to_string(),
                                node,
                            )
                        })? as usize;
                        result.push(self.exec_node_inner(AstRef::new(idx))?);
                    }
                    Ok(TaggedValue::Arr(Rc::from(result)))
                } else {
                    Err(LangError::engine(
                        "array: missing elements/children in payload".to_string(),
                    ))
                }
            }
            "object" => {
                // Two payload shapes for `object`:
                // - Rich: `{"entries": [{"key": "k", "value": v}, ...]}`.
                // - Pratt-parsed: `{"children": [entry_idx, ...]}` where each
                //   entry node has `{"children": [key_idx, value_idx]}`.
                if let Some(entries) = node.payload.get("entries").and_then(|v| v.as_array()) {
                    let mut obj = FxHashMap::default();
                    for entry in entries {
                        let key = entry.get("key").and_then(|v| v.as_str()).ok_or_else(|| {
                            LangError::engine_node("object: entry missing key".to_string(), node)
                        })?;
                        let value = entry.get("value").ok_or_else(|| {
                            LangError::engine_node("object: entry missing value".to_string(), node)
                        })?;
                        if handl_language::is_reserved_value_key(key) {
                            return Err(LangError::engine(format!(
                                "object: key '{}' is reserved",
                                key
                            )));
                        }
                        obj.insert(Rc::from(key), TaggedValue::from(value.clone()));
                    }
                    Ok(TaggedValue::Obj(Rc::new(obj)))
                } else if let Some(children) =
                    node.payload.get("children").and_then(|c| c.as_array())
                {
                    let start = self
                        .lang
                        .child_role_index(&node.kind, "entries")?
                        .unwrap_or(0);
                    let mut obj = FxHashMap::default();
                    for entry in children.iter().skip(start) {
                        let entry_idx = entry.as_u64().ok_or_else(|| {
                            LangError::engine_node(
                                "object: entry index must be a number".to_string(),
                                node,
                            )
                        })? as usize;
                        let entry_node = self.ast.at(AstRef::new(entry_idx))?;
                        let entry_children = entry_node
                            .payload
                            .get("children")
                            .and_then(|c| c.as_array())
                            .ok_or_else(|| {
                                LangError::engine_node(
                                    "object: entry node missing children".to_string(),
                                    node,
                                )
                            })?;
                        let key_role = self
                            .lang
                            .child_role_index(&entry_node.kind, "key")?
                            .unwrap_or(0);
                        let val_role = self
                            .lang
                            .child_role_index(&entry_node.kind, "value")?
                            .unwrap_or(1);
                        let key_idx = entry_children
                            .get(key_role)
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| {
                                LangError::engine_node(
                                    "object: entry missing key child".to_string(),
                                    node,
                                )
                            })? as usize;
                        let val_idx = entry_children
                            .get(val_role)
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| {
                                LangError::engine("object: entry missing value child".to_string())
                            })? as usize;
                        // Object key: a literal key child (the entry shape
                        // declares `literal_children: { key: [...] }`,
                        // ADR-0081) uses its payload as the key string
                        // directly — don't evaluate it. Otherwise (computed
                        // key), evaluate the key expression and use its
                        // string representation. Bracket computed keys
                        // `[expr]:` are desugared to `paren { expr }`
                        // before eval (ADR-0058), so they fall into the
                        // `else` branch. Mirrors the v2 tree-walk's object
                        // prim. Role-less shapes fall back to the legacy
                        // `var`/`string` kind heuristic.
                        let key_node = self.ast.at(AstRef::new(key_idx))?;
                        let is_literal_key = self
                            .lang
                            .literal_child(&entry_node.kind, "key", &key_node.kind)
                            .unwrap_or_else(|| {
                                matches!(key_node.kind.as_str(), "var" | "string")
                            });
                        let key = if is_literal_key {
                            key_node
                                .payload
                                .as_str()
                                .ok_or_else(|| {
                                    LangError::engine(
                                        "object: literal key missing string payload".to_string(),
                                    )
                                })?
                                .to_string()
                        } else {
                            let key_val = self.exec_node_inner(AstRef::new(key_idx))?;
                            if let Some(s) = key_val.as_str() {
                                s.to_string()
                            } else {
                                tagged_to_value(&key_val).to_string()
                            }
                        };
                        let value = self.exec_node_inner(AstRef::new(val_idx))?;
                        if handl_language::is_reserved_value_key(&key) {
                            return Err(LangError::engine(format!(
                                "object: key '{}' is reserved",
                                key
                            )));
                        }
                        obj.insert(Rc::from(key.as_str()), value);
                    }
                    Ok(TaggedValue::Obj(Rc::new(obj)))
                } else {
                    Err(LangError::engine(
                        "object: missing entries/children in payload".to_string(),
                    ))
                }
            }
            "index" => {
                // `index { children: [target, index] }` — evaluate the two
                // child nodes, matching the tree-walk/cranelift `index`
                // dispatch. Falls back to legacy `payload.target`/
                // `payload.index` for hand-built ASTs.
                let (target, index) = if let Some(children) =
                    node.payload.get("children").and_then(|c| c.as_array())
                {
                    if children.len() < 2 {
                        return Err(LangError::engine_node(
                            "index: expected at least 2 children".to_string(),
                            node,
                        ));
                    }
                    let target_idx = children[0].as_u64().ok_or_else(|| {
                        LangError::engine_node(
                            "index: target index must be a number".to_string(),
                            node,
                        )
                    })? as usize;
                    let index_idx = children[1].as_u64().ok_or_else(|| {
                        LangError::engine_node(
                            "index: index index must be a number".to_string(),
                            node,
                        )
                    })? as usize;
                    let target = self.exec_node_inner(AstRef::new(target_idx))?;
                    let index = self.exec_node_inner(AstRef::new(index_idx))?;
                    (tagged_to_value(&target), tagged_to_value(&index))
                } else {
                    let target = node.payload.get("target").ok_or_else(|| {
                        LangError::engine_node("index: missing target in payload".to_string(), node)
                    })?;
                    let index = node.payload.get("index").ok_or_else(|| {
                        LangError::engine_node("index: missing index in payload".to_string(), node)
                    })?;
                    (target.clone(), index.clone())
                };
                match (target, index) {
                    (Value::Array(arr), Value::Number(n)) => {
                        let i = n.as_u64().ok_or_else(|| {
                            LangError::engine(
                                "index: index must be a non-negative integer".to_string(),
                            )
                        })? as usize;
                        let v = arr.get(i).cloned().ok_or_else(|| {
                            LangError::engine_node(
                                format!("index: index {} out of bounds", i),
                                node,
                            )
                        })?;
                        Ok(TaggedValue::from(v))
                    }
                    (Value::Object(obj), Value::String(key)) => {
                        let v = obj.get(&key).cloned().ok_or_else(|| {
                            LangError::engine_node(format!("index: key '{}' not found", key), node)
                        })?;
                        Ok(TaggedValue::from(v))
                    }
                    _ => Err(LangError::engine(
                        "index: target must be array/object and index must be number/string"
                            .to_string(),
                    )),
                }
            }
            "assign" => {
                // ADR-0053: `assign { children: [var, value] }` — evaluate
                // child 1 (value) and return it. Fall back to legacy
                // `payload.value` for hand-built ASTs.
                if let Some(children) = node.payload.get("children").and_then(|c| c.as_array()) {
                    if children.len() >= 2 {
                        let value_idx = children[1].as_u64().ok_or_else(|| {
                            LangError::engine_node(
                                "assign: value index must be a number".to_string(),
                                node,
                            )
                        })? as usize;
                        return self.exec_node_inner(AstRef::new(value_idx));
                    }
                }
                let value = node.payload.get("value").ok_or_else(|| {
                    LangError::engine_node("assign: missing value in payload".to_string(), node)
                })?;
                Ok(TaggedValue::from(value.clone()))
            }
            "add" => bin_num(args, |a, b| a + b),
            "sub" => bin_num(args, |a, b| a - b),
            "mul" => bin_num(args, |a, b| a * b),
            "div" => {
                let a = as_num(&args[0])?;
                let b = as_num(&args[1])?;
                if b.to_f64() == 0.0 {
                    return Err(LangError::engine_node("division by zero".to_string(), node));
                }
                match (a, b) {
                    (Num::Int(a), Num::Int(b)) => Ok(TaggedValue::Int(a / b)),
                    (a, b) => Ok(TaggedValue::Float(a.to_f64() / b.to_f64())),
                }
            }
            "mod" => {
                let a = as_num(&args[0])?;
                let b = as_num(&args[1])?;
                if b.to_f64() == 0.0 {
                    return Err(LangError::engine_node("modulo by zero".to_string(), node));
                }
                Ok(num_value(match (a, b) {
                    (Num::Int(a), Num::Int(b)) => Num::Int(a % b),
                    (a, b) => Num::Float(a.to_f64() % b.to_f64()),
                }))
            }
            "neg" => {
                let a = as_num(&args[0])?;
                Ok(num_value(match a {
                    Num::Int(i) => Num::Int(-i),
                    Num::Float(f) => Num::Float(-f),
                }))
            }
            "eq" => Ok(TaggedValue::Bool(handl_value::eq(&args[0], &args[1]))),
            "neq" => Ok(TaggedValue::Bool(!handl_value::eq(&args[0], &args[1]))),
            "lt" => cmp_num(args, |a, b| a < b),
            "gt" => cmp_num(args, |a, b| a > b),
            "lte" => cmp_num(args, |a, b| a <= b),
            "gte" => cmp_num(args, |a, b| a >= b),
            "pow" => {
                let a = as_num(&args[0])?;
                let b = as_num(&args[1])?;
                Ok(num_value(Num::Float(a.to_f64().powf(b.to_f64()))))
            }
            "compare" => {
                let a = as_num(&args[0])?;
                let b = as_num(&args[1])?;
                let cmp = a.to_f64().partial_cmp(&b.to_f64()).ok_or_else(|| {
                    LangError::engine_node("compare: incomparable values".to_string(), node)
                })?;
                Ok(num_value(Num::Int(match cmp {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                })))
            }
            // ADR-0060: call_host — takes a string callee (args[0]) and an
            // args array (args[1]), looks up the callee in the module
            // registry, and invokes the host function with the array
            // elements as individual arguments.
            "call_host" => {
                let callee = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                    LangError::engine_node("call_host: callee must be a string".to_string(), node)
                })?;
                let modules = self.modules.ok_or_else(|| {
                    LangError::engine_node("call_host: no module registry".to_string(), node)
                })?;
                let func = modules.lookup(callee).ok_or_else(|| {
                    LangError::engine_node(
                        format!("call_host: function '{}' not found", callee),
                        node,
                    )
                })?;
                let call_args: Vec<Value> = args
                    .get(1)
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().map(tagged_to_value).collect())
                    .unwrap_or_default();
                let mut pctx = handl_platform::EvalContext::new();
                let val = func
                    .call(&call_args, &mut pctx)
                    .map_err(|e| LangError::engine_node(e.to_string(), node))?;
                Ok(TaggedValue::from(val))
            }
            // ADR-0060: proj_rest — evaluate all children after index 0 and
            // return them as a JSON array.
            "proj_rest" => {
                let children = node
                    .payload
                    .get("children")
                    .and_then(|c| c.as_array())
                    .ok_or_else(|| {
                        LangError::engine(
                            "proj_rest: expected children array in payload".to_string(),
                        )
                    })?;
                let mut vals = Vec::new();
                for entry in children.iter().skip(1) {
                    let idx = entry.as_u64().ok_or_else(|| {
                        LangError::engine_node(
                            "proj_rest: child index must be a number".to_string(),
                            node,
                        )
                    })? as usize;
                    vals.push(self.exec_node_inner(AstRef::new(idx))?);
                }
                Ok(TaggedValue::Arr(Rc::from(vals)))
            }
            "typeof" => {
                let arg = args
                    .first()
                    .or(incoming.as_ref())
                    .unwrap_or(&TaggedValue::Null);
                Ok(TaggedValue::Str(Rc::from(type_tag(arg))))
            }
            "not" => Ok(TaggedValue::Bool(!is_truthy(&args[0]))),
            "and" => Ok(TaggedValue::Bool(
                is_truthy(&args[0]) && is_truthy(&args[1]),
            )),
            "or" => Ok(TaggedValue::Bool(
                is_truthy(&args[0]) || is_truthy(&args[1]),
            )),
            "is_pos" => {
                let v = incoming.unwrap_or(TaggedValue::Null);
                let n = as_num(&v)?;
                Ok(TaggedValue::Bool(n.to_f64() > 0.0))
            }
            "dec" => {
                let v = incoming.unwrap_or(TaggedValue::Null);
                let n = as_num(&v)?;
                Ok(num_value(match n {
                    Num::Int(i) => Num::Int(i - 1),
                    Num::Float(f) => Num::Float(f - 1.0),
                }))
            }
            "inc" => {
                let v = incoming.unwrap_or(TaggedValue::Null);
                let n = as_num(&v)?;
                Ok(num_value(match n {
                    Num::Int(i) => Num::Int(i + 1),
                    Num::Float(f) => Num::Float(f + 1.0),
                }))
            }
            "half_floor" => {
                let v = incoming.unwrap_or(TaggedValue::Null);
                let n = as_num(&v)?;
                Ok(num_value(match n {
                    Num::Int(i) => Num::Int(i / 2),
                    Num::Float(f) => Num::Float((f / 2.0).floor()),
                }))
            }
            "inc_fst" => {
                let inc = args
                    .first()
                    .cloned()
                    .or(incoming)
                    .unwrap_or(TaggedValue::Null);
                let a = pair_fst(&inc)
                    .ok_or_else(|| {
                        LangError::engine_node("inc_fst: input is not a pair".to_string(), node)
                    })?
                    .clone();
                let b = pair_snd(&inc)
                    .ok_or_else(|| {
                        LangError::engine_node("inc_fst: input is not a pair".to_string(), node)
                    })?
                    .clone();
                let a_n = as_num(&a)?;
                Ok(pair(
                    num_value(match a_n {
                        Num::Int(i) => Num::Int(i + 1),
                        Num::Float(f) => Num::Float(f + 1.0),
                    }),
                    b,
                ))
            }
            "countdown_step" => {
                let inc = args
                    .first()
                    .cloned()
                    .or(incoming)
                    .unwrap_or(TaggedValue::Null);
                let a = pair_fst(&inc).cloned().ok_or_else(|| {
                    LangError::engine_node("countdown_step: input is not a pair".to_string(), node)
                })?;
                let b = pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null);
                let result_val = pair_fst(&a).cloned().ok_or_else(|| {
                    LangError::engine(
                        "countdown_step: in must be pair(result, initial)".to_string(),
                    )
                })?;
                let initial = pair_snd(&a).cloned().unwrap_or(TaggedValue::Null);
                let state = if b.is_null() { initial } else { b };
                let n = as_num(&state)?;
                if n.to_f64() > 0.0 {
                    Ok(either_right(num_value(match n {
                        Num::Int(i) => Num::Int(i - 1),
                        Num::Float(f) => Num::Float(f - 1.0),
                    })))
                } else {
                    Ok(either_left(result_val))
                }
            }
            "infinite_step" => {
                let inc = args
                    .first()
                    .cloned()
                    .or(incoming)
                    .unwrap_or(TaggedValue::Null);
                let b = pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null);
                Ok(either_right(b))
            }
            // Ergonomic `loop` surface lowering (ADR-0052). args = [body_proc]
            // — the body closure, evaluated from the carried sub-scheme
            // arg. `incoming` is the loop pair `(in, state)` popped from the
            // operand stack by the `NativeOp` handler (see `op_is_derived_step`).
            // The body closure is applied to the current state (in on the first
            // step, state thereafter); it returns:
            //   { state: v }  → either_right(v) (continue)
            //   { out: v }    → either_left(v)  (terminate, backward compat)
            //   bare value    → either_left(v)  (terminate, v0.1 `return v`)
            // Param destructuring: if the body closure has >1 params and the
            // current state is an object, bind each param to the corresponding
            // field by name. Mirrors the v2 tree-walk's `eval_derived_step`
            // `loop_step` arm exactly.
            "loop_step" => {
                let inc = incoming.unwrap_or(TaggedValue::Null);
                let in_val = pair_fst(&inc).cloned().unwrap_or(TaggedValue::Null);
                let state = pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null);
                let body_proc = args.first().ok_or_else(|| {
                    LangError::engine_node("loop_step: missing body closure".to_string(), node)
                })?;
                let current = if state.is_null() { in_val } else { state };
                let call_args: Vec<TaggedValue> = if let Some(obj_state) = current.as_object() {
                    if let TaggedValue::Arrow { params, .. } = body_proc {
                        if params.len() > 1 {
                            params
                                .iter()
                                .map(|p| {
                                    obj_state
                                        .get(p.as_str())
                                        .cloned()
                                        .unwrap_or(TaggedValue::Null)
                                })
                                .collect()
                        } else {
                            vec![current.clone()]
                        }
                    } else {
                        vec![current.clone()]
                    }
                } else {
                    vec![current.clone()]
                };
                let result = self.apply_app(body_proc, &call_args)?;
                // v0.1: `return v` produces a bare value (not wrapped in { out: v }).
                // Recognize bare values as exit signals, and { state: v } as continue.
                // { out: v } is accepted for backward compatibility.
                if let Some(obj) = result.as_object() {
                    if let Some(new_state) = obj.get("state") {
                        // Continue with new state.
                        Ok(either_right(new_state.clone()))
                    } else if let Some(out) = obj.get("out") {
                        // Exit with out value (backward compat with { out: v } protocol).
                        Ok(either_left(out.clone()))
                    } else {
                        // Object without out/state keys — treat as exit value (return).
                        Ok(either_left(result.clone()))
                    }
                } else {
                    // Non-object (number, string, bool, null, array) — exit with the value.
                    Ok(either_left(result.clone()))
                }
            }
            // `while_step` derived step op (ADR-0052 §"Derived forms").
            // args = [cond_result, body_result] — re-evaluated per iteration
            // by the `DerivedStepOp` handler with the current value as
            // incoming. `incoming` is the loop pair (in, state).
            // Current = in (first step) or state (later). If cond is truthy,
            // run body and continue with its result; else terminate.
            "while_step" => {
                let inc = incoming.unwrap_or(TaggedValue::Null);
                let in_val = pair_fst(&inc).cloned().unwrap_or(TaggedValue::Null);
                let state = pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null);
                let current = if state.is_null() { in_val } else { state };
                let cond_val = args.first().cloned().unwrap_or(TaggedValue::Null);
                if is_truthy(&cond_val) {
                    let body_val = args.get(1).cloned().unwrap_or(TaggedValue::Null);
                    Ok(either_right(body_val))
                } else {
                    Ok(either_left(current))
                }
            }
            // `for_step` derived step op. args = [body_result]. State is
            // pair(counter, acc). First step: initialize from in = pair(count,
            // initial). While counter > 0, run body on acc and continue with
            // pair(counter-1, new_acc); else terminate with acc.
            "for_step" => {
                let inc = incoming.unwrap_or(TaggedValue::Null);
                let in_val = pair_fst(&inc).cloned().unwrap_or(TaggedValue::Null);
                let state = pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null);
                let (counter, acc) = if state.is_null() {
                    let count = pair_fst(&in_val).cloned().unwrap_or(TaggedValue::Null);
                    let initial = pair_snd(&in_val).cloned().unwrap_or(TaggedValue::Null);
                    (count, initial)
                } else {
                    let c = pair_fst(&state).cloned().unwrap_or(TaggedValue::Null);
                    let a = pair_snd(&state).cloned().unwrap_or(TaggedValue::Null);
                    (c, a)
                };
                let count_n = as_num(&counter)?;
                if count_n.to_f64() > 0.0 {
                    let body_val = args.first().cloned().unwrap_or(TaggedValue::Null);
                    let new_counter = match count_n {
                        Num::Int(i) => num_value(Num::Int(i - 1)),
                        Num::Float(f) => num_value(Num::Float(f - 1.0)),
                    };
                    Ok(either_right(pair(new_counter, body_val)))
                } else {
                    Ok(either_left(acc))
                }
            }
            // `fix_step` derived step op. args = [body_result]. Current = in
            // (first step) or state (later). If body result equals current,
            // terminate (fixpoint); else continue with the result.
            "fix_step" => {
                let inc = incoming.unwrap_or(TaggedValue::Null);
                let in_val = pair_fst(&inc).cloned().unwrap_or(TaggedValue::Null);
                let state = pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null);
                let current = if state.is_null() { in_val } else { state };
                let new_val = args.first().cloned().unwrap_or(TaggedValue::Null);
                if handl_value::eq(&new_val, &current) {
                    Ok(either_left(new_val))
                } else {
                    Ok(either_right(new_val))
                }
            }
            // `bind` op: arrow-calculus env extension (ADR-0052). Mirrors the
            // v2 tree-walk's `prim_bind`. The `bind` node's payload has
            // `name` and `children: [value_idx, body_idx]`; evaluate value,
            // extend env with name=value, evaluate body, pop scope.
            "bind" => self.prim_v2_bind(node),
            // `eval_let` — v0.1 `let` evaluation. Same payload format as
            // `bind` (`{"name": ..., "children": [value, body]}`), so
            // dispatch to the same handler.
            "eval_let" => self.prim_v2_bind(node),
            // ADR-0060: json node payload is the JSON template string. When
            // the json node has children (desugared from json{...}(args)),
            // child 0 is the template node and children 1..N are args.
            // Evaluate args and substitute $0, $1, … into the template.
            "json" => {
                if let Some(children) = node.payload.get("children").and_then(|c| c.as_array()) {
                    if children.len() >= 2 {
                        let template_idx = children[0].as_u64().ok_or_else(|| {
                            LangError::engine_node(
                                "json: child index must be a number".to_string(),
                                node,
                            )
                        })? as usize;
                        let template_node = self.ast.at(AstRef::new(template_idx))?;
                        let template = template_node.payload.as_str().ok_or_else(|| {
                            LangError::engine_node(
                                "json: template payload must be a string".to_string(),
                                node,
                            )
                        })?;
                        let mut s = template.to_string();
                        for (i, entry) in children[1..].iter().enumerate() {
                            let idx = entry.as_u64().ok_or_else(|| {
                                LangError::engine_node(
                                    "json: child index must be a number".to_string(),
                                    node,
                                )
                            })? as usize;
                            let val = self.exec_node_inner(AstRef::new(idx))?;
                            let replacement = serde_json::to_string(&tagged_to_value(&val))
                                .map_err(|e| {
                                    LangError::engine_node(format!("json: {}", e), node)
                                })?;
                            s = s.replace(&format!("${}", i), &replacement);
                        }
                        return serde_json::from_str::<Value>(&s)
                            .map_err(|e| {
                                LangError::engine_node(format!("json: parse error: {}", e), node)
                            })
                            .map(TaggedValue::from);
                    }
                }
                let expr = node.payload.as_str().ok_or_else(|| {
                    LangError::engine_node("json: payload must be a string".to_string(), node)
                })?;
                serde_json::from_str::<Value>(expr)
                    .map_err(|e| LangError::engine_node(format!("json: parse error: {}", e), node))
                    .map(TaggedValue::from)
            }
            // ADR-0053: the data-driven `if → match` desugar rewrites `if`
            // nodes into `match` nodes before eval, so the v2 bytecode path
            // must handle `match`/`handle` directly.
            "match" => self.prim_v2_match(node),
            "handle" => self.prim_v2_handle(node),
            // v0.1: `product` — categorical product `***` (ADR-0074).
            // Receives [f_result, g_result] and pairs them.
            "product" => {
                let left = args.first().cloned().unwrap_or(TaggedValue::Null);
                let right = args.get(1).cloned().unwrap_or(TaggedValue::Null);
                Ok(pair(left, right))
            }
            // v0.1: `nullish` — nullish coalescing `??` (ADR-0074 A.3).
            // If left is some(v), return v; if none or null, return right.
            "nullish" => {
                let left = args.first().cloned().unwrap_or(TaggedValue::Null);
                let right = args.get(1).cloned().unwrap_or(TaggedValue::Null);
                if left.is_null() {
                    Ok(right)
                } else if left.get("_tag").and_then(|t| t.as_str()) == Some("Some") {
                    Ok(left.get("value").cloned().unwrap_or(TaggedValue::Null))
                } else if left.get("_tag").and_then(|t| t.as_str()) == Some("None") {
                    Ok(right)
                } else {
                    Ok(right)
                }
            }
            // v0.1: `opt_chain` — optional chaining `?.` (ADR-0074 A.4).
            // args = [input, field_name]. If some(v) → some(v.field);
            // if none or null → none.
            "opt_chain" => {
                let input = args.first().cloned().unwrap_or(TaggedValue::Null);
                let field_name = args.get(1).and_then(|v| v.as_str()).ok_or_else(|| {
                    LangError::engine_node(
                        "opt_chain: field name must be a string".to_string(),
                        node,
                    )
                })?;
                if input.is_null() {
                    Ok(handl_value::tval!({"_tag": "None"}))
                } else if input.get("_tag").and_then(|t| t.as_str()) == Some("Some") {
                    let inner = input.get("value").cloned().unwrap_or(TaggedValue::Null);
                    let field_val = inner.get(field_name).cloned().unwrap_or(TaggedValue::Null);
                    Ok(handl_value::tval!({"_tag": "Some", "value": field_val}))
                } else if input.get("_tag").and_then(|t| t.as_str()) == Some("None") {
                    Ok(handl_value::tval!({"_tag": "None"}))
                } else {
                    let field_val = input.get(field_name).cloned().unwrap_or(TaggedValue::Null);
                    Ok(handl_value::tval!({"_tag": "Some", "value": field_val}))
                }
            }
            // Phase 9.5: `import "path"` loads a module via the configured
            // resolver and binds its top-level exports into the current scope.
            "import" => self.prim_v2_import(args, node),
            // ADR-0054 / ADR-0079: `do { stmt; expr }` evaluates children in
            // sequence in a new scope and returns the last child's value. It
            // also handles `assign` children as bindings.
            "eval_block" => self.prim_v2_eval_block(node),
            _ => Err(LangError::engine_node(
                format!("unknown v2 native op: {}", op),
                node,
            )),
        }
    }

    /// `bind` native op: arrow-calculus env extension (ADR-0052). Mirrors
    /// the v2 tree-walk's `prim_bind`. The `bind` node's payload has
    /// `name` and `children: [value_idx, body_idx]`. Evaluates the value
    /// child, pushes a scope with `name = value`, evaluates the body child
    /// in the extended env, then pops the scope.
    fn prim_v2_bind(&mut self, node: &AstNode) -> Result<TaggedValue, LangError> {
        let name = node
            .payload
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LangError::engine_node("bind: missing name".to_string(), node))?;
        let children = node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LangError::engine_node("bind: expected children array in payload".to_string(), node)
            })?;
        let value_idx =
            children.first().and_then(|v| v.as_u64()).ok_or_else(|| {
                LangError::engine_node("bind: missing value child".to_string(), node)
            })? as usize;
        let body_idx =
            children.get(1).and_then(|v| v.as_u64()).ok_or_else(|| {
                LangError::engine_node("bind: missing body child".to_string(), node)
            })? as usize;
        // Evaluate the value in the current env.
        let value = self.exec_node_inner(AstRef::new(value_idx))?;
        // Extend env with name = value, evaluate body, pop.
        self.scopes.push_scope();
        self.scopes.define(name, value);
        let result = self.exec_node_inner(AstRef::new(body_idx));
        self.scopes.pop_scope();
        result
    }

    /// `import` native op (Phase 9.5): load a module via the configured
    /// [`ImportResolver`] and bind its top-level exports into the current
    /// scope. Mirrors the v2 tree-walk's `prim_import`.
    fn prim_v2_import(
        &mut self,
        args: &[TaggedValue],
        node: &AstNode,
    ) -> Result<TaggedValue, LangError> {
        // Resolve the module path: prefer the evaluated child (`args[0]`),
        // fall back to the node's `path` payload field.
        let path: String = args
            .first()
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| {
                node.payload
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .ok_or_else(|| {
                LangError::engine_node(
                    "import: missing module path (expected string argument or 'path' payload)"
                        .to_string(),
                    node,
                )
            })?;
        let resolver = self.import_resolver.ok_or_else(|| {
            LangError::engine_node("import: no import resolver configured".to_string(), node)
        })?;
        let bindings = resolver.resolve_import(&path).map_err(|e| {
            LangError::engine_node(format!("import \"{}\": {}", path, e), node)
        })?;
        // If an alias is present (`from "path" import * as alias`), bind the
        // whole module object to that name. Otherwise bind each top-level
        // export into the current scope.
        if let Some(alias) = node.payload.get("alias").and_then(|v| v.as_str()) {
            self.scopes.define(alias, TaggedValue::from(bindings.clone()));
        } else if let Value::Object(map) = bindings {
            for (name, val) in map {
                self.scopes.define(&name, TaggedValue::from(val.clone()));
            }
        }
        // `import` is a statement; it produces no value.
        Ok(TaggedValue::Null)
    }

    /// Resolve a child arena index by role from a node's payload children.
    fn role_child_arena(&self, node: &AstNode, role: &str) -> Result<Option<usize>, LangError> {
        let Some(idx) = self.lang.child_role_index(&node.kind, role)? else {
            return Ok(None);
        };
        let arena = node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.get(idx))
            .and_then(|v| v.as_u64())
            .map(|i| i as usize);
        Ok(arena)
    }

    /// `eval_block` native op: evaluate the children of a `block` node in
    /// sequence in a new scope, returning the last child's value. `assign`
    /// children (shapes declaring `target`/`value` roles) are bound into the
    /// block scope. Mirrors the v2 tree-walk's `eval_block`.
    fn prim_v2_eval_block(&mut self, node: &AstNode) -> Result<TaggedValue, LangError> {
        let children = node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LangError::engine_node(
                    "eval_block: expected children array in payload".to_string(),
                    node,
                )
            })?;
        if children.is_empty() {
            return Ok(TaggedValue::Null);
        }

        self.scopes.push_scope();
        let mut result = TaggedValue::Null;
        for entry in children.iter() {
            let idx = entry.as_u64().ok_or_else(|| {
                LangError::engine_node(
                    "eval_block: child index must be a number".to_string(),
                    node,
                )
            })? as usize;
            let child_node = self.ast.at(AstRef::new(idx))?;

            let target_arena = self.role_child_arena(&child_node, "target")?;
            let value_arena = self.role_child_arena(&child_node, "value")?;
            if let (Some(target_idx), Some(value_idx)) = (target_arena, value_arena) {
                // Binding child: evaluate the value and bind the target
                // name. A `target` child whose shape declares a `subject`
                // role (e.g. a type annotation — the `subject`/`annot` role
                // convention, ADR-0081) delegates to that child for the
                // bound name.
                let mut target_node = self.ast.at(AstRef::new(target_idx))?;
                let var_name = loop {
                    if let Some(s) = target_node.payload.as_str() {
                        break s.to_string();
                    }
                    match self.role_child_arena(target_node, "subject")? {
                        Some(subject_idx) => {
                            target_node = self.ast.at(AstRef::new(subject_idx))?;
                        }
                        None => {
                            return Err(LangError::engine_node(
                                "eval_block: binding target must be an identifier".to_string(),
                                node,
                            ));
                        }
                    }
                };
                let value = self.exec_node_inner(AstRef::new(value_idx))?;
                self.scopes.define(&var_name, value.clone());
                result = value;
            } else {
                result = self.exec_node_inner(AstRef::new(idx))?;
            }
        }
        self.scopes.pop_scope();
        Ok(result)
    }

    /// `match` native op: pattern-match a scrutinee against a list of arms.
    ///
    /// Mirrors the v2 tree-walk's `prim_match` and cranelift's `prim_match`.
    /// The `match` node's payload has `children: [scrut_idx, block_idx]`.
    /// The block node (kind `block`) has `children: [arm_idx, ...]`. Each
    /// arm node (kind `arm`) has `children: [pattern_idx, body_idx]`. The
    /// first matching arm's body is evaluated in a scope holding the pattern
    /// bindings; if no arm matches, an error is returned.
    ///
    /// ADR-0053: the data-driven `if → match` desugar rewrites `if` nodes
    /// into `match` nodes before eval, so the v2 bytecode path must handle
    /// the `match` op directly.
    fn prim_v2_match(&mut self, node: &AstNode) -> Result<TaggedValue, LangError> {
        let children = node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LangError::engine_node("match: missing children in payload".to_string(), node)
            })?;
        let scrut_idx = children.get(0).and_then(|v| v.as_u64()).ok_or_else(|| {
            LangError::engine_node("match: missing scrutinee child".to_string(), node)
        })? as usize;
        let block_idx =
            children.get(1).and_then(|v| v.as_u64()).ok_or_else(|| {
                LangError::engine_node("match: missing block child".to_string(), node)
            })? as usize;

        // Evaluate the scrutinee.
        let scrut_val = self.exec_node_inner(AstRef::new(scrut_idx))?;
        // Convert to Value for pattern matching (which works on serde_json::Value).
        let scrut_value = tagged_to_value(&scrut_val);

        // Get the block node's children (the arm indices).
        let block_node = self.ast.at(AstRef::new(block_idx))?;
        let arm_indices = block_node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LangError::engine_node("match: block node missing children".to_string(), node)
            })?;

        // Try each arm.
        for arm_entry in arm_indices {
            let arm_idx = arm_entry.as_u64().ok_or_else(|| {
                LangError::engine_node("match: arm index must be a number".to_string(), node)
            })? as usize;
            let arm_node = self.ast.at(AstRef::new(arm_idx))?;
            let arm_children = arm_node
                .payload
                .get("children")
                .and_then(|c| c.as_array())
                .ok_or_else(|| {
                    LangError::engine_node("match: arm node missing children".to_string(), node)
                })?;
            let pattern_idx = arm_children
                .get(0)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    LangError::engine_node("match: arm missing pattern child".to_string(), node)
                })? as usize;
            let body_idx = arm_children
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    LangError::engine_node("match: arm missing body child".to_string(), node)
                })? as usize;

            // Interpret the pattern node structurally (ADR-0061: shared
            // pattern matching logic).
            self.scopes.push_scope();
            let matched =
                pattern_match::try_pattern(self.ast, AstRef::new(pattern_idx), &scrut_value, self);
            if let Ok(true) = matched {
                let result = self.exec_node_inner(AstRef::new(body_idx));
                self.scopes.pop_scope();
                return result;
            }
            self.scopes.pop_scope();
            matched?;
        }
        Err(LangError::engine_node(
            "match: no arm matched".to_string(),
            node,
        ))
    }

    /// `handle` native op: evaluate a body, then dispatch the result to the
    /// first matching handler clause.
    ///
    /// Mirrors the v2 tree-walk's `prim_handle` and cranelift's
    /// `prim_handle`. Structurally identical to `match` (children =
    /// [body_idx, block_idx], block contains `arm` nodes with
    /// [pattern_idx, handler_idx]). The body is evaluated first; its result
    /// is matched against clause patterns. If no clause matches, the body
    /// value is returned as-is (not an error).
    fn prim_v2_handle(&mut self, node: &AstNode) -> Result<TaggedValue, LangError> {
        let children = node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LangError::engine_node("handle: missing children in payload".to_string(), node)
            })?;
        let body_idx =
            children.get(0).and_then(|v| v.as_u64()).ok_or_else(|| {
                LangError::engine_node("handle: missing body child".to_string(), node)
            })? as usize;
        let block_idx = children.get(1).and_then(|v| v.as_u64()).ok_or_else(|| {
            LangError::engine_node("handle: missing block child".to_string(), node)
        })? as usize;

        // Evaluate the body first.
        let body_val = self.exec_node_inner(AstRef::new(body_idx))?;
        // Convert to Value for pattern matching.
        let body_value = tagged_to_value(&body_val);

        // Get the block node's children (the clause indices).
        let block_node = self.ast.at(AstRef::new(block_idx))?;
        let clause_indices = block_node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LangError::engine_node("handle: block node missing children".to_string(), node)
            })?;

        // Try each clause.
        for clause_entry in clause_indices {
            let clause_idx = clause_entry.as_u64().ok_or_else(|| {
                LangError::engine_node("handle: clause index must be a number".to_string(), node)
            })? as usize;
            let clause_node = self.ast.at(AstRef::new(clause_idx))?;
            let clause_children = clause_node
                .payload
                .get("children")
                .and_then(|c| c.as_array())
                .ok_or_else(|| {
                    LangError::engine_node("handle: clause node missing children".to_string(), node)
                })?;
            let pattern_idx = clause_children
                .get(0)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    LangError::engine_node("handle: clause missing pattern child".to_string(), node)
                })? as usize;
            let handler_idx = clause_children
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    LangError::engine_node("handle: clause missing handler child".to_string(), node)
                })? as usize;

            self.scopes.push_scope();
            let matched =
                pattern_match::try_pattern(self.ast, AstRef::new(pattern_idx), &body_value, self);
            if let Ok(true) = matched {
                let result = self.exec_node_inner(AstRef::new(handler_idx));
                self.scopes.pop_scope();
                return result;
            }
            self.scopes.pop_scope();
            matched?;
        }

        // No clause matched — return the body value as-is.
        Ok(body_val)
    }

    /// Apply an `App` callee: closure application only (ADR-0060).
    ///
    /// Host-function dispatch (string callee) has been moved to the
    /// `call_host` native op. This helper only handles arrow-value
    /// (closure) application: extract params/body/env, build a fresh
    /// scope from the captured env, bind params, evaluate the body.
    fn apply_app(
        &mut self,
        callee: &TaggedValue,
        args: &[TaggedValue],
    ) -> Result<TaggedValue, LangError> {
        // Closure application — match on TaggedValue::Arrow directly.
        let (params, body_idx, captured_env) = match callee {
            TaggedValue::Arrow {
                params, body, env, ..
            } => (params.clone(), *body, Rc::clone(env)),
            _ => {
                return Err(LangError::engine(
                    "callee is not an arrow-value".to_string(),
                ));
            }
        };
        let body_ref = AstRef::new(body_idx);
        if params.len() != args.len() {
            return Err(LangError::engine(format!(
                "arrow-value expects {} args, got {}",
                params.len(),
                args.len()
            )));
        }
        // Arrow-value invocation (ADR-0052): the arrow carries its
        // definition-time env as data. We install the captured env as the
        // `captured_root` (a cheap `Rc` clone — no deep HashMap clone),
        // push a fresh scope for params, and evaluate the body. The
        // caller's env is never mutated — we save the pre-call binding/marks
        // lengths and `captured_root`, then truncate back after the body
        // returns. This avoids allocating a fresh `ScopeStack` (two `Vec`
        // allocations) on every closure call — critical for `loop_step`
        // where `apply_app` runs once per iteration.
        let saved_captured = self.scopes.captured_root.take();
        let saved_marks_len = self.scopes.marks.len();
        let saved_bindings_len = self.scopes.bindings.len();
        self.scopes.captured_root = Some(captured_env);
        self.scopes.push_scope();
        for (param, arg) in params.iter().zip(args.iter()) {
            self.scopes.define(param, arg.clone());
        }
        let result = self.exec_node_inner(body_ref);
        // Restore: truncate bindings and marks to pre-call state, discarding
        // any scopes the body may have pushed, and restore the captured root.
        self.scopes.bindings.truncate(saved_bindings_len);
        self.scopes.marks.truncate(saved_marks_len);
        self.scopes.captured_root = saved_captured;
        result
    }
}

/// Whether a native op uses the `incoming` value (as opposed to only args).
///
/// Ops that use `incoming` with 0 args pop the incoming value from the stack
/// (it's the value flowing through the arrow). Ops that don't use `incoming`
/// (like `closure`) don't pop anything for a 0-arg call.
fn op_uses_incoming(op: &str) -> bool {
    matches!(
        op,
        "id" | "tag"
            | "is_pos"
            | "dec"
            | "inc"
            | "half_floor"
            | "inc_fst"
            | "countdown_step"
            | "infinite_step"
    )
}

/// Whether a native op is a handler-interpreted derived step op
/// (ADR-0052 §"Derived forms"). These ops carry sub-scheme `Arr.args`
/// compiled inline *and* consume the loop pair `(in, state)` that the
/// enclosing `Loop` combinator left on the operand stack below those args.
/// The `NativeOp` handler pops the args first, then pops the pair as
/// `incoming`, before dispatching to [`VmFrame::apply_native_v2`].
///
/// Only `loop_step` (the ergonomic `loop` surface lowering) is handled here
/// — its body arg is a closure evaluated once, then applied per
/// iteration via [`VmFrame::apply_app`]. The remaining derived step ops
/// (`while_step`/`for_step`/`fix_step`) use the [`Instr::DerivedStepOp`]
/// instruction instead, which re-evaluates their cond/body sub-programs
/// per iteration with the current value as incoming.
fn op_is_derived_step(op: &str) -> bool {
    matches!(op, "loop_step")
}

// ---------------------------------------------------------------------------
// Numeric op helpers (mirror the v2 tree-walk)
// ---------------------------------------------------------------------------

/// Binary numeric op helper. Preserves int-int arithmetic.
fn bin_num<F>(args: &[TaggedValue], f: F) -> Result<TaggedValue, LangError>
where
    F: FnOnce(f64, f64) -> f64,
{
    let a = as_num(&args[0])?;
    let b = as_num(&args[1])?;
    match (a, b) {
        (Num::Int(a), Num::Int(b)) => Ok(TaggedValue::Int(f(a as f64, b as f64) as i64)),
        (a, b) => Ok(TaggedValue::Float(f(a.to_f64(), b.to_f64()))),
    }
}

/// Numeric comparison helper.
fn cmp_num<F>(args: &[TaggedValue], f: F) -> Result<TaggedValue, LangError>
where
    F: FnOnce(f64, f64) -> bool,
{
    let a = as_num(&args[0])?;
    let b = as_num(&args[1])?;
    Ok(TaggedValue::Bool(f(a.to_f64(), b.to_f64())))
}

/// Evaluate a typed arithmetic bytecode (`Add`/`Sub`/`Mul`/`Div`/`Mod`).
///
/// `a` is the first argument (popped second), `b` is the second argument
/// (popped first/top). Mirrors the corresponding `apply_native_v2` arms
/// exactly: int-int preserves integer arithmetic; if either is a float,
/// both are promoted to `f64`. `Div`/`Mod` error on a zero divisor.
fn eval_bin_arith(instr: &Instr, a: TaggedValue, b: TaggedValue) -> Result<TaggedValue, LangError> {
    let a = as_num(&a)?;
    let b = as_num(&b)?;
    match instr {
        Instr::Add => Ok(num_value(match (a, b) {
            (Num::Int(a), Num::Int(b)) => Num::Int((a as f64 + b as f64) as i64),
            (a, b) => Num::Float(a.to_f64() + b.to_f64()),
        })),
        Instr::Sub => Ok(num_value(match (a, b) {
            (Num::Int(a), Num::Int(b)) => Num::Int((a as f64 - b as f64) as i64),
            (a, b) => Num::Float(a.to_f64() - b.to_f64()),
        })),
        Instr::Mul => Ok(num_value(match (a, b) {
            (Num::Int(a), Num::Int(b)) => Num::Int((a as f64 * b as f64) as i64),
            (a, b) => Num::Float(a.to_f64() * b.to_f64()),
        })),
        Instr::Div => {
            if b.to_f64() == 0.0 {
                return Err(LangError::engine("division by zero".to_string()));
            }
            Ok(match (a, b) {
                (Num::Int(a), Num::Int(b)) => TaggedValue::Int(a / b),
                (a, b) => TaggedValue::Float(a.to_f64() / b.to_f64()),
            })
        }
        Instr::Mod => {
            if b.to_f64() == 0.0 {
                return Err(LangError::engine("modulo by zero".to_string()));
            }
            Ok(num_value(match (a, b) {
                (Num::Int(a), Num::Int(b)) => Num::Int(a % b),
                (a, b) => Num::Float(a.to_f64() % b.to_f64()),
            }))
        }
        _ => Err(LangError::engine(format!(
            "eval_bin_arith: not an arithmetic instruction: {:?}",
            instr
        ))),
    }
}

/// Evaluate a typed comparison bytecode (`Eq`/`Neq`/`Lt`/`Lte`/`Gt`/`Gte`).
///
/// `a` is the first argument (popped second), `b` is the second argument
/// (popped first/top). `Eq`/`Neq` use [`handl_value::eq`] (matching the
/// `NativeOp` path exactly, including cross-type equality semantics); the
/// ordered comparisons convert both operands to `f64` via [`as_num`].
fn eval_bin_cmp(instr: &Instr, a: TaggedValue, b: TaggedValue) -> Result<TaggedValue, LangError> {
    match instr {
        Instr::Eq => Ok(TaggedValue::Bool(handl_value::eq(&a, &b))),
        Instr::Neq => Ok(TaggedValue::Bool(!handl_value::eq(&a, &b))),
        Instr::Lt => {
            let a = as_num(&a)?;
            let b = as_num(&b)?;
            Ok(TaggedValue::Bool(a.to_f64() < b.to_f64()))
        }
        Instr::Lte => {
            let a = as_num(&a)?;
            let b = as_num(&b)?;
            Ok(TaggedValue::Bool(a.to_f64() <= b.to_f64()))
        }
        Instr::Gt => {
            let a = as_num(&a)?;
            let b = as_num(&b)?;
            Ok(TaggedValue::Bool(a.to_f64() > b.to_f64()))
        }
        Instr::Gte => {
            let a = as_num(&a)?;
            let b = as_num(&b)?;
            Ok(TaggedValue::Bool(a.to_f64() >= b.to_f64()))
        }
        _ => Err(LangError::engine(format!(
            "eval_bin_cmp: not a comparison instruction: {:?}",
            instr
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use handl_language::{Ast, AstNode, EvalSchemeDef, Language, Scheme, SchemeVersion};
    use handl_platform::{EvalContext, Function, Module, ModuleLookup};
    use handl_plugin_engine_exec::ImportResolver;
    use handl_plugin_module::ModuleRegistry;
    use serde_json::json;
    use std::rc::Rc;

    /// Build a simple AST for testing.
    fn make_ast(nodes: Vec<AstNode>) -> Ast {
        let mut ast = Ast::new();
        for n in nodes {
            ast.push(n);
        }
        ast
    }

    /// A dummy node with an empty-object payload.
    fn dummy_node() -> AstNode {
        AstNode::new("test").with_payload(json!({}))
    }

    /// Evaluate a standalone v2 scheme with a fresh frame and dummy node.
    fn eval_standalone(scheme: &Scheme) -> Result<Value, LangError> {
        let ast = make_ast(vec![dummy_node()]);
        let lang = Language::new();
        let node = ast.at(AstRef::new(0)).unwrap();
        let instrs = compile(scheme, &lang, &ast, node)?;
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        frame.exec_instrs(&instrs, node, None)
    }

    /// Construct a pair value `{"__pair__": [a, b]}` as a JSON Value
    /// (for test assertions and scheme literals at the public API boundary).
    fn pair(a: Value, b: Value) -> Value {
        serde_json::json!({"__pair__": [a, b]})
    }

    /// Construct a `left` Either value as a JSON Value.
    fn either_left(v: Value) -> Value {
        serde_json::json!({"__either__": "left", "value": v})
    }

    /// Construct a `right` Either value as a JSON Value.
    fn either_right(v: Value) -> Value {
        serde_json::json!({"__either__": "right", "value": v})
    }

    // 1. arr const

    #[test]
    fn arr_const_returns_literal() {
        let s = Scheme::arr_native_lit("const", vec![], json!(42));
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn arr_const_null_when_no_literal() {
        let s = Scheme::arr_native("const", vec![]);
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, Value::Null);
    }

    // 2. arr proj_env

    #[test]
    fn arr_proj_env_bound_lookup() {
        let s = Scheme::arr_native_lit("proj_env", vec![], "x");
        let ast = make_ast(vec![dummy_node()]);
        let lang = Language::new();
        let node = ast.at(AstRef::new(0)).unwrap();
        let instrs = compile(&s, &lang, &ast, node).unwrap();
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        frame.define_global("x", TaggedValue::from(json!(7)));
        let result = frame.exec_instrs(&instrs, node, None).unwrap();
        assert_eq!(result, json!(7));
    }

    #[test]
    fn arr_proj_env_unbound_returns_name() {
        let s = Scheme::arr_native_lit("proj_env", vec![], "unknown");
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, Value::from("unknown"));
    }

    // 3. Seq const >>> id

    #[test]
    fn seq_const_then_id() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], json!(5)),
            Scheme::arr_native("id", vec![]),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(5));
    }

    // 4. Bind shape

    #[test]
    fn bind_shape_extend_env_then_proj_env() {
        let s = Scheme::seq(
            Scheme::seq(
                Scheme::arr_native_lit("const", vec![], json!(5)),
                Scheme::arr_native_lit("extend_env", vec![], "x"),
            ),
            Scheme::arr_native_lit("proj_env", vec![], "x"),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(5));
    }

    // 5. First on a pair

    #[test]
    fn first_id_on_pair() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], pair(json!(3), json!(7))),
            Scheme::first(Scheme::arr_native("id", vec![])),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, pair(json!(3), json!(7)));
    }

    #[test]
    fn first_const_on_pair() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], pair(json!(3), json!(7))),
            Scheme::first(Scheme::arr_native_lit("const", vec![], json!(99))),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, pair(json!(99), json!(7)));
    }

    // 6. Choice on Either

    #[test]
    fn choice_left() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], either_left(json!(5))),
            Scheme::choice(
                Scheme::arr_native_lit("const", vec![], json!(1)),
                Scheme::arr_native_lit("const", vec![], json!(2)),
            ),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(1));
    }

    #[test]
    fn choice_right() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], either_right(json!(5))),
            Scheme::choice(
                Scheme::arr_native_lit("const", vec![], json!(1)),
                Scheme::arr_native_lit("const", vec![], json!(2)),
            ),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(2));
    }

    // 7. tag + Choice (Branch lowering)

    #[test]
    fn tag_choice_true_yields_then() {
        let s = Scheme::seq(
            Scheme::seq(
                Scheme::arr_native_lit("const", vec![], json!(true)),
                Scheme::arr_native("tag", vec![]),
            ),
            Scheme::choice(
                Scheme::arr_native_lit("const", vec![], "yes"),
                Scheme::arr_native_lit("const", vec![], "no"),
            ),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!("yes"));
    }

    #[test]
    fn tag_choice_false_yields_else() {
        let s = Scheme::seq(
            Scheme::seq(
                Scheme::arr_native_lit("const", vec![], json!(false)),
                Scheme::arr_native("tag", vec![]),
            ),
            Scheme::choice(
                Scheme::arr_native_lit("const", vec![], "yes"),
                Scheme::arr_native_lit("const", vec![], "no"),
            ),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!("no"));
    }

    // 8. Loop causal single-step

    #[test]
    fn loop_causal_single_step() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], json!(10)),
            Scheme::loop_feedback(Scheme::arr_native("inc_fst", vec![])),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(11));
    }

    // 8b. Loop multi-step (Either-based)

    #[test]
    fn loop_multi_step_either_countdown() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], pair(json!(42), json!(5))),
            Scheme::loop_feedback(Scheme::arr_native("countdown_step", vec![])),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn loop_multi_step_max_iterations_guard() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], json!(1)),
            Scheme::loop_feedback(Scheme::arr_native("infinite_step", vec![])),
        );
        let err = eval_standalone(&s).unwrap_err();
        assert!(
            matches!(err, LangError::Engine { ref message, .. } if message.contains("max iterations"))
        );
    }

    // 8c. `loop_step` derived op — the ergonomic `loop` surface lowering
    // (ADR-0052). These tests build the canonical `loop` lowering
    // `Seq(proj_child(0), Loop(loop_step(proj_child(1))))` by hand with a
    // real arrow body, exercising param destructuring, `if`, arithmetic,
    // and object construction — the same shape `hnl` source parses to.

    /// Wire the eval schemes needed for the `loop` surface: `var`, arithmetic,
    /// `if`, `object`, `function` (closure), and `loop`.
    fn loop_lang() -> Language {
        let mut lang = Language::new();
        // var -> proj_env $payload (looks up the var name in the env)
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.var",
                "var",
                Scheme::arr_native_lit("proj_env", vec![], Value::String("$payload".into())),
            ))
            .unwrap();
        // binary.eq / sub / mul -> native op over the two children
        for (id, kind, op) in [
            ("e.eq", "binary.eq", "eq"),
            ("e.sub", "binary.sub", "sub"),
            ("e.mul", "binary.mul", "mul"),
        ] {
            lang.eval
                .merge(EvalSchemeDef::new(
                    id,
                    kind,
                    Scheme::arr_native(
                        op,
                        vec![
                            Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                            Scheme::arr_native_lit("proj_child", vec![], json!(1u64)),
                        ],
                    ),
                ))
                .unwrap();
        }
        // if -> Seq(Seq(proj_child(0), tag), Choice(proj_child(1), proj_child(2)))
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.if",
                "if",
                Scheme::seq(
                    Scheme::seq(
                        Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                        Scheme::arr_native("tag", vec![]),
                    ),
                    Scheme::choice(
                        Scheme::arr_native_lit("proj_child", vec![], json!(1u64)),
                        Scheme::arr_native_lit("proj_child", vec![], json!(2u64)),
                    ),
                ),
            ))
            .unwrap();
        // object -> native `object` prim (reads the node's children payload)
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.object",
                "object",
                Scheme::arr_native("object", vec![]),
            ))
            .unwrap();
        // function -> closure prim
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.function",
                "function",
                Scheme::arr_native("closure", vec![]),
            ))
            .unwrap();
        // return -> proj_child(0) (v0.1: `return v` evaluates to `v`)
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.return",
                "return",
                Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
            ))
            .unwrap();
        // loop -> Seq(proj_child(0), Loop(loop_step(proj_child(1))))
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.loop",
                "loop",
                Scheme::seq(
                    Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                    Scheme::loop_feedback(Scheme::arr_native(
                        "loop_step",
                        vec![Scheme::arr_native_lit("proj_child", vec![], json!(1u64))],
                    )),
                ),
            ))
            .unwrap();
        lang
    }

    /// `loop 42 (x ~> return x)` — single-step terminate,
    /// one param (no destructuring), scalar state. Expected: 42.
    #[test]
    fn loop_step_single_step_terminate() {
        let lang = loop_lang();
        // Arena:
        //   0: var "x"
        //   1: return {children: [0]}      (return x)
        //   2: function {params: ["x"], body: 1}
        //   3: lit 42         4: loop {children: [3, 2]}
        let ast = make_ast(vec![
            AstNode::new("var").with_payload(json!("x")), // 0
            AstNode::new("return").with_payload(json!({"children": [0]})), // 1
            AstNode::new("function").with_payload(json!({"params": ["x"], "body": 1})), // 2
            AstNode::new("lit").with_payload(json!(42)),  // 3
            AstNode::new("loop").with_payload(json!({"children": [3, 2]})), // 4
        ]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(4)).unwrap();
        assert_eq!(result, json!(42));
    }

    /// `loop { n: 3, acc: 1 }([n, acc] ~> if n == 0 then return acc
    /// else { state: { n: n - 1, acc: acc * n } })` — multi-step with param
    /// destructuring, `if`, arithmetic, nested object construction.
    /// Computes 3! = 6.
    #[test]
    fn loop_step_factorial_destructured() {
        let lang = loop_lang();
        // Arena (see `loop_lang` for the schemes):
        //   0: lit "n"        1: lit "acc"      2: lit "state"
        //   3: lit 0          4: lit 1
        //   5: var "n"        6: var "acc"
        //   7: binary.eq   [5, 3]    (n == 0)
        //   8: binary.sub  [5, 4]    (n - 1)
        //   9: binary.mul  [6, 5]    (acc * n)
        //  10: entry [0, 8]          (n: n - 1)
        //  11: entry [1, 9]          (acc: acc * n)
        //  12: object {children: [10, 11]}   (inner state {n, acc})
        //  13: var "acc"
        //  14: return {children: [13]}       (then: return acc)
        //  15: entry [2, 12]         (state: inner)
        //  16: object {children: [15]}       (else {state: {...}})
        //  17: if {children: [7, 14, 16]}
        //  18: function {params: ["n","acc"], body: 17}
        //  19: lit {n: 3, acc: 1}
        //  20: loop {children: [19, 18]}
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!("n")),     // 0
            AstNode::new("lit").with_payload(json!("acc")),   // 1
            AstNode::new("lit").with_payload(json!("state")), // 2
            AstNode::new("lit").with_payload(json!(0)),       // 3
            AstNode::new("lit").with_payload(json!(1)),       // 4
            AstNode::new("var").with_payload(json!("n")),     // 5
            AstNode::new("var").with_payload(json!("acc")),   // 6
            AstNode::new("binary.eq").with_payload(json!({"children": [5, 3]})), // 7
            AstNode::new("binary.sub").with_payload(json!({"children": [5, 4]})), // 8
            AstNode::new("binary.mul").with_payload(json!({"children": [6, 5]})), // 9
            AstNode::new("entry").with_payload(json!({"children": [0, 8]})), // 10
            AstNode::new("entry").with_payload(json!({"children": [1, 9]})), // 11
            AstNode::new("object").with_payload(json!({"children": [10, 11]})), // 12
            AstNode::new("var").with_payload(json!("acc")),   // 13
            AstNode::new("return").with_payload(json!({"children": [13]})), // 14
            AstNode::new("entry").with_payload(json!({"children": [2, 12]})), // 15
            AstNode::new("object").with_payload(json!({"children": [15]})), // 16
            AstNode::new("if").with_payload(json!({"children": [7, 14, 16]})), // 17
            AstNode::new("function").with_payload(json!({"params": ["n", "acc"], "body": 17})), // 18
            AstNode::new("lit").with_payload(json!({"n": 3, "acc": 1})), // 19
            AstNode::new("loop").with_payload(json!({"children": [19, 18]})), // 20
        ]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(20)).unwrap();
        assert_eq!(result, json!(6));
    }

    /// Object construction with `var` keys exercises the `MakeObj`
    /// bytecode path (static keys pre-allocated at compile time). Uses the
    /// same `loop_lang` schemes. The object `{ a: 1, b: 2 }` is built with
    /// `var` key nodes, so the compiler emits `EvalNode` + `MakeObj`
    /// instead of `NativeOp("object", 0)`.
    #[test]
    fn makeobj_static_var_keys() {
        let lang = loop_lang();
        // Arena:
        //   0: var "a"       (key)
        //   1: var "b"       (key)
        //   2: lit 1         (value for a)
        //   3: lit 2         (value for b)
        //   4: entry {children: [0, 2]}   (a: 1)
        //   5: entry {children: [1, 3]}   (b: 2)
        //   6: object {children: [4, 5]}  ({ a: 1, b: 2 })
        let ast = make_ast(vec![
            AstNode::new("var").with_payload(json!("a")), // 0
            AstNode::new("var").with_payload(json!("b")), // 1
            AstNode::new("lit").with_payload(json!(1)),   // 2
            AstNode::new("lit").with_payload(json!(2)),   // 3
            AstNode::new("entry").with_payload(json!({"children": [0, 2]})), // 4
            AstNode::new("entry").with_payload(json!({"children": [1, 3]})), // 5
            AstNode::new("object").with_payload(json!({"children": [4, 5]})), // 6
        ]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(6)).unwrap();
        assert_eq!(result, json!({"a": 1, "b": 2}));
    }

    /// Object construction with `string` keys also exercises the `MakeObj`
    /// bytecode path. The keys are `string` kind nodes with string payloads.
    #[test]
    fn makeobj_static_string_keys() {
        let lang = loop_lang();
        // Arena:
        //   0: string "x"    (key)
        //   1: string "y"    (key)
        //   2: lit 10        (value for x)
        //   3: lit 20        (value for y)
        //   4: entry {children: [0, 2]}
        //   5: entry {children: [1, 3]}
        //   6: object {children: [4, 5]}
        let ast = make_ast(vec![
            AstNode::new("string").with_payload(json!("x")), // 0
            AstNode::new("string").with_payload(json!("y")), // 1
            AstNode::new("lit").with_payload(json!(10)),     // 2
            AstNode::new("lit").with_payload(json!(20)),     // 3
            AstNode::new("entry").with_payload(json!({"children": [0, 2]})), // 4
            AstNode::new("entry").with_payload(json!({"children": [1, 3]})), // 5
            AstNode::new("object").with_payload(json!({"children": [4, 5]})), // 6
        ]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(6)).unwrap();
        assert_eq!(result, json!({"x": 10, "y": 20}));
    }

    /// Object construction with a computed (dynamic) key falls back to
    /// `NativeOp("object", n)` — the `MakeObj` path is not used. Verifies
    /// the fallback produces correct results.
    #[test]
    fn makeobj_dynamic_key_fallback() {
        let lang = loop_lang();
        // Arena:
        //   0: lit "dyn"     (computed key — `lit` kind, not var/string)
        //   1: lit 99        (value)
        //   2: entry {children: [0, 1]}
        //   3: object {children: [2]}
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!("dyn")), // 0
            AstNode::new("lit").with_payload(json!(99)),    // 1
            AstNode::new("entry").with_payload(json!({"children": [0, 1]})), // 2
            AstNode::new("object").with_payload(json!({"children": [2]})), // 3
        ]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(3)).unwrap();
        assert_eq!(result, json!({"dyn": 99}));
    }

    // 8d. `while_step` / `for_step` / `fix_step` derived step ops.
    // These use the `DerivedStepOp` instruction which re-evaluates arg
    // sub-programs per iteration with the current value as incoming.

    /// `while` loop: countdown from 5 to 0 using `is_pos`/`dec` ops.
    /// `while_loop(is_pos, dec)` applied to 5.
    #[test]
    fn while_step_countdown() {
        // Scheme: Seq(const 5, Loop(while_step(is_pos, dec)))
        // is_pos: native op, 0 args, uses incoming (checks > 0)
        // dec: native op, 0 args, uses incoming (subtracts 1)
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], json!(5)),
            Scheme::loop_feedback(Scheme::arr_native(
                "while_step",
                vec![
                    Scheme::arr_native("is_pos", vec![]),
                    Scheme::arr_native("dec", vec![]),
                ],
            )),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(0));
    }

    /// `while` loop with zero iterations: cond is immediately false.
    #[test]
    fn while_step_zero_iterations() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], json!(0)),
            Scheme::loop_feedback(Scheme::arr_native(
                "while_step",
                vec![
                    Scheme::arr_native("is_pos", vec![]),
                    Scheme::arr_native("dec", vec![]),
                ],
            )),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(0));
    }

    /// `for` loop: accumulate 5 increments from 0. `for_loop(inc)` applied
    /// to pair(5, 0).
    #[test]
    fn for_step_accumulate() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], pair(json!(5), json!(0))),
            Scheme::loop_feedback(Scheme::arr_native(
                "for_step",
                vec![Scheme::arr_native("inc", vec![])],
            )),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(5));
    }

    /// `for` loop with zero iterations: counter is 0, return initial.
    #[test]
    fn for_step_zero_iterations() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], pair(json!(0), json!(99))),
            Scheme::loop_feedback(Scheme::arr_native(
                "for_step",
                vec![Scheme::arr_native("inc", vec![])],
            )),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(99));
    }

    /// `fix` loop: repeatedly halve 100 until fixpoint at 0.
    /// `fix_loop(half_floor)` applied to 100.
    #[test]
    fn fix_step_halve_to_zero() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], json!(100)),
            Scheme::loop_feedback(Scheme::arr_native(
                "fix_step",
                vec![Scheme::arr_native("half_floor", vec![])],
            )),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(0));
    }

    /// `fix` loop already at fixpoint: 0 → 0.
    #[test]
    fn fix_step_already_at_fixpoint() {
        let s = Scheme::seq(
            Scheme::arr_native_lit("const", vec![], json!(0)),
            Scheme::loop_feedback(Scheme::arr_native(
                "fix_step",
                vec![Scheme::arr_native("half_floor", vec![])],
            )),
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(0));
    }

    // 9. App with a closure value

    #[test]
    fn app_closure_applies_body() {
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.var",
                "var",
                Scheme::arr_native_lit("proj_env", vec![], "$payload"),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("var").with_payload(json!("x")),
            AstNode::new("test").with_payload(json!({})),
        ]);
        let closure = json!({
            "__arrow__": true,
            "params": ["x"],
            "body": 0,
            "env": {},
        });
        let s = Scheme::app(
            Scheme::arr_native_lit("const", vec![], closure),
            vec![Scheme::arr_native_lit("const", vec![], json!(41))],
        );
        let node = ast.at(AstRef::new(1)).unwrap();
        let instrs = compile(&s, &lang, &ast, node).unwrap();
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_instrs(&instrs, node, None).unwrap();
        assert_eq!(result, json!(41));
    }

    // 10. Arithmetic parity

    #[test]
    fn native_add() {
        let s = Scheme::arr_native(
            "add",
            vec![
                Scheme::arr_native_lit("const", vec![], json!(2)),
                Scheme::arr_native_lit("const", vec![], json!(3)),
            ],
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(5));
    }

    #[test]
    fn native_eq() {
        let s = Scheme::arr_native(
            "eq",
            vec![
                Scheme::arr_native_lit("const", vec![], json!(2)),
                Scheme::arr_native_lit("const", vec![], json!(2)),
            ],
        );
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(true));
    }

    // 11. Unknown op

    #[test]
    fn unknown_native_op_errors() {
        let s = Scheme::arr_native("bogus", vec![]);
        let err = eval_standalone(&s).unwrap_err();
        assert!(
            matches!(err, LangError::Engine { ref message, .. } if message.contains("unknown v2 native op"))
        );
    }

    // 12. Foreign dispatch

    /// A host function that doubles its numeric argument.
    struct DoubleFn;

    impl Function for DoubleFn {
        fn call(
            &self,
            args: &[Value],
            _ctx: &mut EvalContext,
        ) -> Result<Value, handl_platform::EvalError> {
            Ok(Value::from(args[0].as_i64().unwrap() * 2))
        }
    }

    struct DoubleModule;

    impl Module for DoubleModule {
        fn name(&self) -> &str {
            "test"
        }
        fn function(&self, name: &str) -> Option<&dyn Function> {
            if name == "double" {
                Some(&DoubleFn)
            } else {
                None
            }
        }
    }

    fn double_registry() -> Rc<dyn ModuleLookup> {
        Rc::new(ModuleRegistry::new().with(DoubleModule))
    }

    #[test]
    fn foreign_arr_dispatches_to_module() {
        let s = Scheme::arr_foreign(
            "double",
            vec![Scheme::arr_native_lit("const", vec![], json!(21))],
        );
        let ast = make_ast(vec![dummy_node()]);
        let lang = Language::new();
        let node = ast.at(AstRef::new(0)).unwrap();
        let instrs = compile(&s, &lang, &ast, node).unwrap();
        let reg = double_registry();
        let mut frame = VmFrame::new(&ast, &lang, Some(&reg), None);
        let result = frame.exec_instrs(&instrs, node, None).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn foreign_arr_no_modules_errors() {
        let s = Scheme::arr_foreign(
            "double",
            vec![Scheme::arr_native_lit("const", vec![], json!(21))],
        );
        let err = eval_standalone(&s).unwrap_err();
        assert!(
            matches!(err, LangError::Engine { ref message, .. } if message.contains("unregistered foreign"))
        );
    }

    // 13. exec_node (v2 scheme from lang.eval)

    #[test]
    fn exec_node_add() {
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.add",
                "binary.add",
                Scheme::arr_native(
                    "add",
                    vec![
                        Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                        Scheme::arr_native_lit("proj_child", vec![], json!(1u64)),
                    ],
                ),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(2)),
            AstNode::new("lit").with_payload(json!(3)),
            AstNode::new("binary.add").with_payload(json!({"children": [0, 1]})),
        ]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(2)).unwrap();
        assert_eq!(result, json!(5));
    }

    #[test]
    fn exec_node_branch_true() {
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.if",
                "if",
                Scheme::seq(
                    Scheme::seq(
                        Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                        Scheme::arr_native("tag", vec![]),
                    ),
                    Scheme::choice(
                        Scheme::arr_native_lit("proj_child", vec![], json!(1u64)),
                        Scheme::arr_native_lit("proj_child", vec![], json!(2u64)),
                    ),
                ),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(true)),
            AstNode::new("lit").with_payload(json!(1)),
            AstNode::new("lit").with_payload(json!(2)),
            AstNode::new("if").with_payload(json!({"children": [0, 1, 2]})),
        ]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(3)).unwrap();
        assert_eq!(result, json!(1));
    }

    #[test]
    fn exec_node_branch_false() {
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.if",
                "if",
                Scheme::seq(
                    Scheme::seq(
                        Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                        Scheme::arr_native("tag", vec![]),
                    ),
                    Scheme::choice(
                        Scheme::arr_native_lit("proj_child", vec![], json!(1u64)),
                        Scheme::arr_native_lit("proj_child", vec![], json!(2u64)),
                    ),
                ),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(false)),
            AstNode::new("lit").with_payload(json!(1)),
            AstNode::new("lit").with_payload(json!(2)),
            AstNode::new("if").with_payload(json!({"children": [0, 1, 2]})),
        ]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(3)).unwrap();
        assert_eq!(result, json!(2));
    }

    // 14. No-scheme fallback

    #[test]
    fn no_scheme_returns_payload() {
        let lang = Language::new();
        let ast = make_ast(vec![AstNode::new("custom").with_payload(json!("hello"))]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_node(AstRef::new(0)).unwrap();
        assert_eq!(result, json!("hello"));
    }

    // 15. Version check

    #[test]
    fn v2_engine_accepts_v2_lang() {
        assert!(SchemeVersion::v2().supports(&SchemeVersion::v2()));
    }

    // 16. import op

    #[test]
    fn import_op_binds_module_bindings_into_scope() {
        struct StubResolver {
            bindings: Value,
        }
        impl ImportResolver for StubResolver {
            fn resolve_import(&self, _path: &str) -> Result<Value, LangError> {
                Ok(self.bindings.clone())
            }
        }

        let resolver: Rc<dyn ImportResolver> =
            Rc::new(StubResolver { bindings: json!({"x": 42, "y": "hi"}) });

        let mut lang = Language::new();
        // var -> proj_env $payload
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.var",
                "var",
                Scheme::arr_native_lit("proj_env", vec![], Value::from("$payload")),
            ))
            .unwrap();
        // stmt.import -> import(proj_child(0))
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.stmt.import",
                "stmt.import",
                Scheme::arr_native(
                    "import",
                    vec![Scheme::arr_native_lit("proj_child", vec![], Value::from(0u64))],
                ),
            ))
            .unwrap();

        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!("m.hnl")),
            AstNode::new("stmt.import").with_payload(json!({"children": [0]})),
            AstNode::new("var").with_payload(json!("x")),
        ]);

        let mut frame = VmFrame::new(&ast, &lang, None, Some(&resolver));
        // Execute the import; it produces no value.
        let import_result = frame.exec_node(AstRef::new(1)).unwrap();
        assert!(import_result.is_null());

        // The imported binding `x` should now be in scope.
        let result = frame.exec_node(AstRef::new(2)).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn import_op_errors_without_resolver() {
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.stmt.import",
                "stmt.import",
                Scheme::arr_native(
                    "import",
                    vec![Scheme::arr_native_lit("const", vec![], Value::from("m.hnl"))],
                ),
            ))
            .unwrap();
        let ast = make_ast(vec![AstNode::new("stmt.import").with_payload(json!({}))]);
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let err = frame.exec_node(AstRef::new(0)).unwrap_err();
        assert!(format!("{}", err).contains("no import resolver"));
    }

    #[test]
    fn eval_block_imports_and_returns_binding() {
        struct StubResolver {
            bindings: Value,
        }
        impl ImportResolver for StubResolver {
            fn resolve_import(&self, _path: &str) -> Result<Value, LangError> {
                Ok(self.bindings.clone())
            }
        }

        let resolver: Rc<dyn ImportResolver> =
            Rc::new(StubResolver { bindings: json!({"x": 42 }) });

        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.var",
                "var",
                Scheme::arr_native_lit("proj_env", vec![], Value::from("$payload")),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.block",
                "block",
                Scheme::arr_native("eval_block", vec![]),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.stmt.import",
                "stmt.import",
                Scheme::arr_native(
                    "import",
                    vec![Scheme::arr_native_lit("proj_child", vec![], Value::from(0u64))],
                ),
            ))
            .unwrap();

        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!("m.hnl")),
            AstNode::new("stmt.import").with_payload(json!({"children": [0]})),
            AstNode::new("var").with_payload(json!("x")),
            AstNode::new("block").with_payload(json!({"children": [1, 2]})),
        ]);

        let mut frame = VmFrame::new(&ast, &lang, None, Some(&resolver));
        let result = frame.exec_node(AstRef::new(3)).unwrap();
        assert_eq!(result, json!(42));
    }
}

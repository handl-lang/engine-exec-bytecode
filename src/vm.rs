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
use handl_plugin_engine_exec::native_ops::{
    self, NativeOpCtx, Num, OpService, as_num, bin_add, build_loop_step_args, is_truthy,
    loop_step_outcome, num_value, step_decision, type_tag,
};
use handl_plugin_engine_exec::pattern_match::{self, PatternMatchCtx};
use handl_plugin_engine_exec_tree_walk::home_frame;
use handl_value::{TaggedValue, tagged_to_value};
use rustc_hash::FxHashMap;

use crate::compile::compile;
use crate::instr::{ArithOp, Instr};

// ---------------------------------------------------------------------------
// Ffi registry — the host/OS function table for `Instr::Ffi` (ADR-0103 §2.4)
// ---------------------------------------------------------------------------

/// A foreign-function entry for the [`Instr::Ffi`] hardware boundary.
///
/// Each entry is a closure that receives the popped argument values (as
/// `serde_json::Value`) and returns a result value or an error message. The
/// registry is keyed by `(module, name)`; when `Ffi` does not find an entry
/// it falls back to the existing [`Instr::ForeignOp`] dispatch via
/// [`ModuleLookup`](handl_platform::ModuleLookup).
pub(crate) type FfiFn = Rc<dyn Fn(&[Value]) -> Result<Value, String>>;

/// A registry of `(module, name)` -> host function mappings for `Ffi`.
///
/// Minimal in-process table; the VM owns one per frame (empty by default).
/// Tests register mock foreign functions via [`VmFrame::register_ffi`].
#[derive(Default)]
pub(crate) struct FfiRegistry {
    fns: FxHashMap<(String, String), FfiFn>,
}

impl FfiRegistry {
    /// Create an empty registry.
    pub(crate) fn new() -> Self {
        Self {
            fns: FxHashMap::default(),
        }
    }

    /// Register a host function under `(module, name)`.
    pub(crate) fn register(&mut self, module: &str, name: &str, f: FfiFn) {
        self.fns.insert((module.to_string(), name.to_string()), f);
    }

    /// Look up the host function for `(module, name)`.
    pub(crate) fn lookup(&self, module: &str, name: &str) -> Option<&FfiFn> {
        self.fns.get(&(module.to_string(), name.to_string()))
    }
}

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
    /// Host-provided globals (e.g. `self` and `__arg_N`), searched after
    /// local/captured bindings so user-defined scopes shadow them. Held in
    /// a cheaply-cloned `Rc` so it is excluded from `capture()` and shared
    /// across closure calls. Mirrors the tree-walk `ScopeStack::globals`.
    pub(crate) globals: Rc<FxHashMap<Rc<str>, TaggedValue>>,
    /// Set when `assign` updates a binding in an *enclosing* scope (below
    /// the innermost mark) with a different value. Read and cleared by
    /// `eval_rec` fixpoint detection (`prim_v2_eval_rec`) — such writes are
    /// the ADR-0074 feedback wires: the `while`/`do-while`/`for` desugars
    /// update `__continue`/`__xs` (and user bindings like `x = x + 1`)
    /// from inside `if`/`match` branch blocks, and a pass that produces
    /// any such write must not count as converged. Mirrors the tree-walk
    /// `ScopeStack::enclosing_updated`.
    pub(crate) enclosing_updated: bool,
}

impl ScopeStack {
    pub(crate) fn new() -> Self {
        Self {
            bindings: Vec::new(),
            marks: vec![0],
            captured_root: None,
            globals: Rc::new(FxHashMap::default()),
            enclosing_updated: false,
        }
    }

    /// Set the host-provided globals map (an `Rc` clone, cheap). Globals
    /// are accessible via `lookup()` but excluded from `capture()`.
    pub(crate) fn set_globals(&mut self, globals: Rc<FxHashMap<Rc<str>, TaggedValue>>) {
        self.globals = globals;
    }

    /// Look up a variable, searching innermost bindings first (linear scan
    /// from the end), then the captured root (if any), then host-provided
    /// globals. Mirrors the tree-walk `ScopeStack::lookup` ordering.
    pub(crate) fn lookup(&self, name: &str) -> Option<&TaggedValue> {
        // Search innermost bindings first (reverse linear scan).
        for (n, v) in self.bindings.iter().rev() {
            if n.as_ref() == name {
                return Some(v);
            }
        }
        // Fall back to captured root.
        if let Some(v) = self.captured_root.as_ref().and_then(|m| m.get(name)) {
            return Some(v);
        }
        // Fall back to host-provided globals.
        self.globals.get(name)
    }

    /// Define a variable in the current (innermost) scope.
    #[inline]
    pub(crate) fn define(&mut self, name: &str, value: TaggedValue) {
        self.bindings.push((Rc::from(name), value));
    }

    /// Assign `name = value`: update the nearest existing binding (the
    /// innermost *visible* binding — a reverse linear scan over `bindings`,
    /// which mirrors the tree-walk `ScopeStack::assign` scope walk), or
    /// define it in the innermost scope when the name is not already bound.
    /// See the tree-walk engine for the ADR-0074 feedback-wire rationale.
    ///
    /// When the update lands below the innermost mark (a cross-scope write)
    /// and the value actually changed, `enclosing_updated` is set so the
    /// `eval_rec` fixpoint check can observe the write — exactly like the
    /// tree-walk `ScopeStack::assign`.
    pub(crate) fn assign(&mut self, name: &str, value: TaggedValue) {
        let inner_start = self.marks.last().copied().unwrap_or(0);
        for (pos, (n, v)) in self.bindings.iter_mut().enumerate().rev() {
            if n.as_ref() == name {
                if !handl_value::eq(v, &value) {
                    *v = value;
                    // Only cross-scope writes are fixpoint-relevant: an
                    // update to a binding defined earlier in the *same*
                    // innermost scope is ordinary sequencing and must not
                    // keep a `rec` pass alive.
                    if pos < inner_start {
                        self.enclosing_updated = true;
                    }
                }
                return;
            }
        }
        self.define(name, value);
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
    /// Includes the parent closure's captured env (if any) so that nested
    /// closures retain access to bindings from all enclosing closure scopes.
    pub(crate) fn capture(&self) -> FxHashMap<Rc<str>, TaggedValue> {
        let mut captured = FxHashMap::default();
        // Include parent closure's captured env first (lower priority).
        if let Some(parent_captured) = &self.captured_root {
            for (k, v) in parent_captured.iter() {
                captured.insert(Rc::clone(k), v.clone());
            }
        }
        // Then include current scopes (higher priority, overwrites parent).
        for (n, v) in &self.bindings {
            captured.insert(Rc::clone(n), v.clone());
        }
        captured
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

/// Flatten a right-nested `Pair` tree into a flat argument list (mirrors
/// the v2 tree-walk's `flatten_pair`): `Pair(a, Pair(b, c))` → `[a, b, c]`.
fn flatten_pair(v: &TaggedValue) -> Vec<TaggedValue> {
    match v {
        TaggedValue::Pair(p) => {
            let mut left = flatten_pair(&p.0);
            left.extend(flatten_pair(&p.1));
            left
        }
        other => vec![other.clone()],
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

/// Convergence comparison for `eval_rec` bindings.
///
/// Closures (`TaggedValue::Arrow`) are compared by params and body index
/// only, NOT by their captured env. In a `rec` fixpoint, each pass
/// re-evaluates a closure binding and creates a new closure that captures
/// the current scope (which now includes other rec bindings). The captured
/// env naturally differs each pass even though the closure's code is
/// identical, so a full structural `eq` would never converge. Comparing
/// only params + body lets the fixpoint converge after the env stabilizes.
/// Non-closure values use the standard structural `eq`. Mirrors the
/// tree-walk `rec_value_eq`.
fn rec_value_eq(a: &TaggedValue, b: &TaggedValue) -> bool {
    match (a, b) {
        (
            TaggedValue::Arrow {
                params: p1,
                body: b1,
                ..
            },
            TaggedValue::Arrow {
                params: p2,
                body: b2,
                ..
            },
        ) => p1.len() == p2.len() && p1.iter().zip(p2.iter()).all(|(a, b)| a == b) && b1 == b2,
        _ => handl_value::eq(a, b),
    }
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
    /// The foreign-function registry for [`Instr::Ffi`] (ADR-0103 §2.4).
    /// Empty by default; tests register mock host functions via
    /// [`VmFrame::register_ffi`]. When `Ffi` finds no entry here it falls
    /// back to [`Instr::ForeignOp`] dispatch via `modules`.
    ffi_registry: FfiRegistry,
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
            ffi_registry: FfiRegistry::new(),
        }
    }

    /// Define a global variable in the root scope (for unit tests).
    pub(crate) fn define_global(&mut self, name: &str, value: TaggedValue) {
        self.scopes.define(name, value);
    }

    /// Register a mock foreign function for [`Instr::Ffi`] (for unit tests).
    ///
    /// The function receives the popped argument values (as `serde_json::
    /// Value`) and returns a result value or an error message.
    pub(crate) fn register_ffi(
        &mut self,
        module: &str,
        name: &str,
        f: impl Fn(&[Value]) -> Result<Value, String> + 'static,
    ) {
        self.ffi_registry.register(module, name, Rc::new(f));
    }

    /// Set the host-provided globals map (e.g. `self`, `__arg_N`).
    pub(crate) fn set_globals(&mut self, globals: Rc<FxHashMap<Rc<str>, TaggedValue>>) {
        self.scopes.set_globals(globals);
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

    fn global_value(&self, name: &str) -> Option<Value> {
        self.scopes.globals.get(name).map(tagged_to_value)
    }

    fn lang(&self) -> Option<&Language> {
        Some(self.lang)
    }
}

impl<'a, 'h> VmFrame<'a, 'h> {
    /// Compile (if not cached) and execute a single node using v2 bytecode.
    ///
    /// Looks up `lang.eval[node.kind]` to find the node's `Term`, compiles
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
                Instr::EvalChildOr(n) => {
                    // Like EvalChild, but tolerates an absent child index
                    // (e.g. an optional `else` child) — pushes null.
                    let arena_idx = node
                        .payload
                        .get("children")
                        .and_then(|c| c.as_array())
                        .and_then(|children| children.get(*n))
                        .and_then(|v| v.as_u64());
                    match arena_idx {
                        Some(i) => {
                            let result = self.exec_node_inner(AstRef::new(i as usize))?;
                            stack.push(result);
                        }
                        None => stack.push(TaggedValue::Null),
                    }
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
                    // Array-pattern arrows receive the array as a single
                    // argument, so they must not be unpacked.
                    let is_array_pattern = matches!(
                        &callee,
                        TaggedValue::Arrow {
                            array_pattern: true,
                            ..
                        }
                    );
                    let actual_args: Vec<TaggedValue> =
                        if !is_array_pattern && args.len() == 1 && args[0].is_array() {
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
                    // call. Array-pattern arrows receive the array as a
                    // single argument, so they must not be unpacked.
                    let is_array_pattern = matches!(
                        &callee,
                        TaggedValue::Arrow {
                            array_pattern: true,
                            ..
                        }
                    );
                    let result = if !is_array_pattern && arg0.is_array() {
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
                    // `call_host` (ADR-0060 `call` scheme) also consumes a
                    // stack-carried incoming below its args: `ChoiceJump`
                    // pushes the routed callee, then `proj_rest` pushes the
                    // arg array — so the callee sits beneath the args, like
                    // a derived step op's loop pair.
                    let is_derived_step = op_is_derived_step(op) || op == "call_host";
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
                Instr::Closure {
                    params,
                    rest,
                    body,
                    array_pattern,
                } => {
                    // Capture the current environment and construct a
                    // `TaggedValue::Arrow`. Identical to the `closure`
                    // native op, but avoids the per-call `Vec` allocation
                    // for args, the string `match op` dispatch, and the
                    // node payload extraction — `params` and `body` are
                    // pre-computed at compile time.
                    let captured = self.scopes.capture();
                    stack.push(TaggedValue::Arrow {
                        params: Rc::clone(params),
                        rest: rest.clone(),
                        body: *body,
                        env: Rc::new(captured),
                        home: None,
                        array_pattern: *array_pattern,
                        // ADR-0106 item 2 step 1: the bytecode engine's body is
                        // an instruction index, not an AST node, so no
                        // self-describing definition is minted yet (step 3).
                        body_def: None,
                    });
                }
                // --- ADR-0103 hardware-boundary instructions ---
                //
                // `Arith(op)` is the consolidated ALU instruction. It mirrors
                // the typed bytecodes (`Add`/`Sub`/…/`Eq`/`Lt`/…) and the
                // arithmetic `NativeOp` arms exactly, so results are
                // identical. Unary ops (`Neg`/`Not`) pop one value; all
                // others pop two (`b` on top, `a` below).
                Instr::Arith(op) => {
                    let result = match op {
                        ArithOp::Neg | ArithOp::Not => {
                            let a = stack.pop().ok_or_else(|| {
                                LangError::engine_node("stack underflow on Arith".to_string(), node)
                            })?;
                            eval_arith_unary(*op, a)?
                        }
                        _ => {
                            let b = stack.pop().ok_or_else(|| {
                                LangError::engine_node("stack underflow on Arith".to_string(), node)
                            })?;
                            let a = stack.pop().ok_or_else(|| {
                                LangError::engine_node("stack underflow on Arith".to_string(), node)
                            })?;
                            eval_arith_bin(*op, a, b)?
                        }
                    };
                    stack.push(result);
                }
                Instr::Ffi { module, name, n } => {
                    let mut args = Vec::with_capacity(*n);
                    for _ in 0..*n {
                        args.push(stack.pop().ok_or_else(|| {
                            LangError::engine_node("stack underflow on Ffi".to_string(), node)
                        })?);
                    }
                    args.reverse();
                    // First, consult the Ffi registry (ADR-0103 host/OS
                    // boundary). If no entry is registered, fall back to the
                    // existing `ForeignOp` dispatch via `ModuleLookup` so
                    // already-registered host functions keep working.
                    if let Some(f) = self.ffi_registry.lookup(module, name) {
                        let val_args: Vec<Value> = args.iter().map(tagged_to_value).collect();
                        let result = f(&val_args).map_err(|e| {
                            LangError::engine_node(format!("ffi {}::{}: {}", module, name, e), node)
                        })?;
                        stack.push(TaggedValue::from(result));
                    } else {
                        let modules = self.modules.ok_or_else(|| {
                            LangError::engine_node(
                                format!("unregistered ffi: {}::{}", module, name),
                                node,
                            )
                        })?;
                        let func = modules.lookup(name).ok_or_else(|| {
                            LangError::engine_node(
                                format!("unregistered ffi: {}::{}", module, name),
                                node,
                            )
                        })?;
                        let val_args: Vec<Value> = args.iter().map(tagged_to_value).collect();
                        let mut pctx = handl_platform::EvalContext::new();
                        let result = func
                            .call(&val_args, &mut pctx)
                            .map_err(|e| LangError::engine_node(e.to_string(), node))?;
                        stack.push(TaggedValue::from(result));
                    }
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

    /// Apply a native v2 op.
    ///
    /// `incoming` is the value flowing through the arrow (used by `id`,
    /// `tag`, `is_pos`, `dec`, `inc`, `half_floor`, `inc_fst`,
    /// `countdown_step`, `infinite_step`).
    ///
    /// The op semantics are defined once in
    /// `plugin-engine-exec::native_ops` (ADR-0044 §7): this engine supplies
    /// only the globals — the scope stack, AST-node execution, and arrow
    /// application — through [`VmCtx`]. The residual `match` below owns
    /// ops that still require this engine's machinery (imports, module
    /// calls, `eval_block`, closures, FFI) and are migrated to the shared
    /// table as their services are declared.
    fn apply_native_v2(
        &mut self,
        op: &str,
        args: &[TaggedValue],
        node: &AstNode,
        incoming: Option<TaggedValue>,
    ) -> Result<TaggedValue, LangError> {
        {
            let mut ctx = VmCtx { frame: self, node };
            match native_ops::apply_native_op(op, args, None, incoming.clone(), &mut ctx) {
                Ok(Some((v, _pushed))) => return Ok(v),
                Ok(None) => {}
                Err(e) => return Err(e.with_node(node)),
            }
        }
        match op {
            "closure" => {
                let payload = &node.payload;
                let (params, rest, body, array_pattern) = if let (Some(p), Some(b)) = (
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
                    let rest_name = payload
                        .get("rest")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    (param_names, rest_name, b as usize, false)
                } else if let Some(children) = payload.get("children").and_then(|c| c.as_array()) {
                    // `params`/`body` child positions resolve by named
                    // role (ADR-0081); positional 0/1 is the role-less
                    // fallback.
                    let params_role = self
                        .lang
                        .child_role_index(&node.kind, "params")?
                        .unwrap_or(0);
                    let body_role = self.lang.child_role_index(&node.kind, "body")?.unwrap_or(1);
                    let param_idx = children
                        .get(params_role)
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            LangError::engine_node("closure: missing param child".to_string(), node)
                        })? as usize;
                    let body_idx = children
                        .get(body_role)
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            LangError::engine_node("closure: missing body child".to_string(), node)
                        })? as usize;
                    let param_node = self.ast.at(AstRef::new(param_idx))?;
                    // `param_names` resolves fixed params plus an optional
                    // rest param (`...name`) data-driven via
                    // `pattern_role`/variadic metadata (ADR-0081).
                    let (param_names, rest_name) = self
                        .lang
                        .param_names(self.ast, param_node)
                        .map_err(|e| LangError::engine_node(e.to_string(), node))?;
                    let array_pattern = self.lang.is_array_pattern(param_node);
                    (param_names, rest_name, body_idx, array_pattern)
                } else {
                    return Err(LangError::engine(
                        "closure: missing params/body in payload".to_string(),
                    ));
                };
                let captured = self.scopes.capture();
                Ok(TaggedValue::Arrow {
                    params: Rc::from(params),
                    rest: rest.map(Rc::from),
                    body,
                    env: Rc::new(captured),
                    home: None,
                    array_pattern,
                    // ADR-0106 item 2 step 1: see `body_def` on `TaggedValue`.
                    body_def: None,
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
                        // Spread: if the child is a `rest`-roled node
                        // (`pattern_role: rest`; from `...expr` syntax),
                        // flatten its value into the array if it's an
                        // array. `kind == "rest"` is the role-less
                        // fallback (ADR-0081). Mirrors the tree-walk
                        // implementation.
                        let is_rest = self
                            .ast
                            .get(AstRef::new(idx))
                            .map(|n| {
                                self.lang
                                    .pattern_role_is(&n.kind, "rest")
                                    .unwrap_or_else(|| n.kind == "rest")
                            })
                            .unwrap_or(false);
                        if is_rest {
                            let val = self.exec_node_inner(AstRef::new(idx))?;
                            if let TaggedValue::Arr(arr) = val {
                                result.extend(arr.iter().cloned());
                            } else {
                                result.push(val);
                            }
                        } else {
                            result.push(self.exec_node_inner(AstRef::new(idx))?);
                        }
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
                    // ADR-0074 v0.1: `{ expr }` (bare-expression block) vs
                    // `{ key: val }` (object). When ALL children from the
                    // `entries` role are non-`entry` nodes (bare
                    // expressions), treat as a block: evaluate each and
                    // return the last value. When any child is an `entry`
                    // node, treat as an object. `kind == "entry"` is the
                    // role-less fallback (ADR-0081). Mirrors the v2
                    // tree-walk's object prim.
                    let all_bare = children.iter().skip(start).all(|c| {
                        c.as_u64()
                            .and_then(|idx| self.ast.at(AstRef::new(idx as usize)).ok())
                            .map(|n| {
                                !self
                                    .lang
                                    .pattern_role_is(&n.kind, "entry")
                                    .unwrap_or_else(|| n.kind == "entry")
                            })
                            .unwrap_or(true)
                    });
                    if all_bare && !children.is_empty() {
                        // Block semantics: evaluate all, return last.
                        let mut last = TaggedValue::Null;
                        for entry in children.iter().skip(start) {
                            let idx = entry.as_u64().ok_or_else(|| {
                                LangError::engine_node(
                                    "object: entry index must be a number".to_string(),
                                    node,
                                )
                            })? as usize;
                            last = self.exec_node_inner(AstRef::new(idx))?;
                        }
                        return Ok(last);
                    }
                    let mut obj = FxHashMap::default();
                    for entry in children.iter().skip(start) {
                        let entry_idx = entry.as_u64().ok_or_else(|| {
                            LangError::engine_node(
                                "object: entry index must be a number".to_string(),
                                node,
                            )
                        })? as usize;
                        let entry_node = self.ast.at(AstRef::new(entry_idx))?;
                        // Spread entry: if the child is not an `entry`-roled
                        // node, evaluate it and merge its fields into the
                        // object. This implements `{ ...s, field: val }`
                        // syntax. `kind == "entry"` is the role-less
                        // fallback (ADR-0081). Mirrors the v2 tree-walk's
                        // object prim.
                        let is_entry = self
                            .lang
                            .pattern_role_is(&entry_node.kind, "entry")
                            .unwrap_or_else(|| entry_node.kind == "entry");
                        if !is_entry {
                            let spread_val = self.exec_node_inner(AstRef::new(entry_idx))?;
                            if let Some(spread_obj) = spread_val.as_object() {
                                for (k, v) in spread_obj {
                                    obj.insert(Rc::clone(k), v.clone());
                                }
                            }
                            continue;
                        }
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
                        // prim. The `var`/`string` fallback below is for the
                        // non-lowering path and role-less shapes; it is
                        // bypassed when `BytecodeEngine::with_lowering()` has
                        // normalized `string` keys to `var` and the `entry`
                        // shape declares `literal_children` for `key`.
                        let key_node = self.ast.at(AstRef::new(key_idx))?;
                        let is_literal_key = self
                            .lang
                            .literal_child(&entry_node.kind, "key", &key_node.kind)
                            .unwrap_or_else(|| matches!(key_node.kind.as_str(), "var" | "string"));
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
            "assign" => {
                // ADR-0053: `assign { children: [target, value] }` —
                // evaluate the `value` child (its position resolves by
                // named role; index 1 is the role-less fallback,
                // ADR-0081) and return it. Fall back to legacy
                // `payload.value` for hand-built ASTs.
                if let Some(children) = node.payload.get("children").and_then(|c| c.as_array()) {
                    if children.len() >= 2 {
                        let value_role = self
                            .lang
                            .child_role_index(&node.kind, "value")?
                            .unwrap_or(1);
                        let value_idx = children
                            .get(value_role)
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| {
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
            "call_host" => {
                // Mirror the tree-walk: the callee (e.g. `str.length`
                // evaluating to the string "str.length") arrives as
                // `incoming`; `args[0]` is the `proj_rest` arg array.
                let callee = incoming
                    .as_ref()
                    .and_then(|v| v.as_str())
                    .or_else(|| args.first().and_then(|v| v.as_str()))
                    .ok_or_else(|| {
                        LangError::engine_node(
                            "call_host: callee must be a string".to_string(),
                            node,
                        )
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
                // Arg array position: when the callee arrives as `incoming`
                // (canonical `call` scheme), the `proj_rest` arg array is
                // `args[0]`; the legacy convention puts it at `args[1]`.
                let arg_src = if incoming.as_ref().and_then(|v| v.as_str()).is_some() {
                    args.first()
                } else {
                    args.get(1)
                };
                let call_args: Vec<Value> = arg_src
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
            // `loop_step` — the ergonomic `loop` surface (ADR-0052). The
            // body closure is applied to the current state via this
            // engine's `apply_app`; the shared `build_loop_step_args`/
            // `loop_step_outcome` define the argument mapping and the
            // `{ state: v }`/`{ out: v }`/bare-value outcome semantics.
            "loop_step" => {
                let inc = incoming.unwrap_or(TaggedValue::Null);
                let in_val = pair_fst(&inc).cloned().unwrap_or(TaggedValue::Null);
                let state = pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null);
                let body_proc = args.first().ok_or_else(|| {
                    LangError::engine_node("loop_step: missing body closure".to_string(), node)
                })?;
                let current = if state.is_null() { in_val } else { state };
                let call_args = build_loop_step_args(body_proc, &current);
                let result = self.apply_app(body_proc, &call_args)?;
                Ok(loop_step_outcome(result))
            }
            // `fold_step` — the ArrowLoop protocol used by the `loop` fold
            // node (ADR-0120 Phase 1). args = [step closure]; each
            // iteration applies the step to the pair (loop-input, state)
            // carried in `incoming`; the step result is itself the Either
            // consumed by `LoopCheck`.
            "fold_step" => {
                let inc = incoming.unwrap_or(TaggedValue::Null);
                let step_proc = args.first().ok_or_else(|| {
                    LangError::engine_node("fold_step: missing step closure".to_string(), node)
                })?;
                let call_args = vec![
                    pair_fst(&inc).cloned().unwrap_or(TaggedValue::Null),
                    pair_snd(&inc).cloned().unwrap_or(TaggedValue::Null),
                ];
                self.apply_app(step_proc, &call_args)
            }
            // Derived step ops (ADR-0052 §"Derived forms"). This engine's
            // interpretation is eager: `Instr::DerivedStepOp` re-evaluates
            // each arg sub-program per iteration and passes the results in
            // `args`; the step *decision* is the shared [`step_decision`]
            // semantics, with `eval_arg` reading the pre-evaluated args.
            "while_step" | "for_step" | "fix_step" => {
                let inc = incoming.unwrap_or(TaggedValue::Null);
                let mut eval_arg =
                    |i: usize, _current: &TaggedValue| -> Result<TaggedValue, LangError> {
                        args.get(i).cloned().ok_or_else(|| {
                            LangError::engine_node(format!("{op}: missing arg {i}"), node)
                        })
                    };
                step_decision(op, &inc, &mut eval_arg).map_err(|e| e.with_node(node))
            }
            // `bind` op: arrow-calculus env extension (ADR-0052). Mirrors the
            // v2 tree-walk's `prim_bind`. The `bind` node's payload has
            // `name` and `children: [value_idx, body_idx]`; evaluate value,
            // extend env with name=value, evaluate body, pop scope.
            "bind" => self.prim_v2_bind(node),
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
            "field_access" => {
                let target = args.first().cloned().unwrap_or(TaggedValue::Null);
                let field = args.get(1).cloned().unwrap_or(TaggedValue::Null);
                match (target, field) {
                    (TaggedValue::Obj(map), TaggedValue::Str(key)) => {
                        Ok(map.get(key.as_ref()).cloned().unwrap_or(TaggedValue::Null))
                    }
                    (TaggedValue::Arr(arr), field) => {
                        let i = field
                            .as_f64()
                            .filter(|n| *n >= 0.0 && n.fract() == 0.0)
                            .map(|n| n as usize)
                            .ok_or_else(|| {
                                LangError::engine_node(
                                    "field_access: array index must be a non-negative integer"
                                        .to_string(),
                                    node,
                                )
                            })?;
                        arr.get(i).cloned().ok_or_else(|| {
                            LangError::engine_node(
                                format!(
                                    "field_access: index {i} out of bounds (len {})",
                                    arr.len()
                                ),
                                node,
                            )
                        })
                    }
                    (TaggedValue::Str(s), TaggedValue::Str(key)) => {
                        let joined: Rc<str> = Rc::from(format!("{}.{}", s.as_ref(), key.as_ref()));
                        Ok(TaggedValue::Str(joined))
                    }
                    (target, field) => Err(LangError::engine_node(
                        format!(
                            "field_access: target must be object/array, field must be string/number; got {} and {}",
                            type_tag(&target),
                            type_tag(&field),
                        ),
                        node,
                    )),
                }
            }
            // Phase 9.5: `import "path"` loads a module via the configured
            // resolver and binds its top-level exports into the current scope.
            "import" => self.prim_v2_import(args, node),
            // ADR-0054 / ADR-0079: `do { stmt; expr }` evaluates children in
            // sequence in a new scope and returns the last child's value. It
            // also handles `assign` children as bindings.
            "eval_block" => self.prim_v2_eval_block(node),
            // ADR-0074: `rec` — recursive computation region, iterated to a
            // fixpoint (mirrors the v2 tree-walk's `eval_rec`).
            "eval_rec" => self.prim_v2_eval_rec(node),
            // ADR-0074 follow-up #3: `>>>` (`flow`) — function application
            // with a single input argument. Unlike `call` (which unpacks a
            // `proj_rest` array of call args), `flow` passes the left
            // operand through as one argument, only destructuring a
            // categorical `Pair` (`***`) into multiple positional args.
            // Mirrors the v2 tree-walk's `flow` arm exactly.
            "flow" => {
                if args.len() != 2 {
                    return Err(LangError::engine_node(
                        format!("flow: expected 2 args, got {}", args.len()),
                        node,
                    ));
                }
                let input = &args[0];
                let callee = &args[1];
                let actual_args: Vec<TaggedValue> = if matches!(input, TaggedValue::Pair(_)) {
                    flatten_pair(input)
                } else {
                    vec![input.clone()]
                };
                if let Some(name) = callee.as_str() {
                    let modules = self.modules.ok_or_else(|| {
                        LangError::engine_node("flow: no module registry".to_string(), node)
                    })?;
                    let func = modules.lookup(name).ok_or_else(|| {
                        LangError::engine_node(format!("flow: function '{}' not found", name), node)
                    })?;
                    let call_args: Vec<Value> = actual_args.iter().map(tagged_to_value).collect();
                    let mut pctx = handl_platform::EvalContext::new();
                    let val = func
                        .call(&call_args, &mut pctx)
                        .map_err(|e| LangError::engine_node(e.to_string(), node))?;
                    Ok(TaggedValue::from(val))
                } else {
                    self.apply_app(callee, &actual_args)
                }
            }
            // `export` and `record_proj` are owned by the shared table.
            "runtime_assert" => {
                let children = node
                    .payload
                    .get("children")
                    .and_then(|c| c.as_array())
                    .ok_or_else(|| {
                        LangError::engine_node(
                            "runtime_assert: expected children array in payload".to_string(),
                            node,
                        )
                    })?;
                let pred_idx = self
                    .lang
                    .child_role_index(&node.kind, "predicate")?
                    .unwrap_or(0);
                let expr_idx = self.lang.child_role_index(&node.kind, "expr")?.unwrap_or(1);
                let pred_arena =
                    children
                        .get(pred_idx)
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            LangError::engine_node(
                                format!("runtime_assert: predicate child {} not found", pred_idx),
                                node,
                            )
                        })? as usize;
                let expr_arena =
                    children
                        .get(expr_idx)
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            LangError::engine_node(
                                format!("runtime_assert: expr child {} not found", expr_idx),
                                node,
                            )
                        })? as usize;
                let pred_val = self.exec_node_inner(AstRef::new(pred_arena))?;
                let expr_val = self.exec_node_inner(AstRef::new(expr_arena))?;
                if !matches!(pred_val, TaggedValue::Arrow { .. }) {
                    return Ok(expr_val);
                }
                let check_result = self.apply_app(&pred_val, &[expr_val.clone()])?;
                if is_truthy(&check_result) {
                    Ok(expr_val)
                } else {
                    Err(LangError::engine_node(
                        "runtime_assert: assertion failed — predicate returned false".to_string(),
                        node,
                    ))
                }
            }
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
        let bindings = resolver
            .resolve_import(&path)
            .map_err(|e| LangError::engine_node(format!("import \"{}\": {}", path, e), node))?;
        // Binding semantics mirror the tree-walk `prim_import` (ADR-0096):
        // `names` (non-"*") projects a single export, `alias` alone binds
        // the whole module object, otherwise each export is spliced into
        // the current scope. A `names` array supports braced import lists
        // such as `from "p" import { a, b as x }`.
        let names_val = node.payload.get("names");
        let alias = node.payload.get("alias").and_then(|v| v.as_str());
        if let Some(arr) = names_val.and_then(|v| v.as_array()) {
            for item in arr {
                let (imported, local) = match item {
                    Value::String(s) => (s.as_str(), s.as_str()),
                    Value::Object(obj) => {
                        let n = obj.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
                            LangError::engine_node(
                                "import: missing 'name' in braced import entry".to_string(),
                                node,
                            )
                        })?;
                        let a = obj.get("as").and_then(|v| v.as_str()).unwrap_or(n);
                        (n, a)
                    }
                    _ => {
                        return Err(LangError::engine_node(
                            "import: invalid braced import entry".to_string(),
                            node,
                        ));
                    }
                };
                if imported == "*" {
                    return Err(LangError::engine_node(
                        "import: cannot use '*' inside a braced import list".to_string(),
                        node,
                    ));
                }
                let exported = match &bindings {
                    Value::Object(map) => map.get(imported).cloned(),
                    _ => None,
                };
                let value = exported.ok_or_else(|| {
                    LangError::engine_node(
                        format!("import \"{}\": module has no export `{}`", path, imported),
                        node,
                    )
                })?;
                self.scopes.define(local, TaggedValue::from(value));
            }
        } else if let Some(imported) = names_val.and_then(|v| v.as_str()).filter(|n| *n != "*") {
            let exported = match &bindings {
                Value::Object(map) => map.get(imported).cloned(),
                _ => None,
            };
            let value = exported.ok_or_else(|| {
                LangError::engine_node(
                    format!("import \"{}\": module has no export `{}`", path, imported),
                    node,
                )
            })?;
            self.scopes
                .define(alias.unwrap_or(imported), TaggedValue::from(value));
        } else if let Some(alias) = alias {
            self.scopes
                .define(alias, TaggedValue::from(bindings.clone()));
        } else if let Value::Object(map) = bindings {
            for (name, val) in map {
                self.scopes.define(&name, TaggedValue::from(val.clone()));
            }
        }
        // `import` is a statement; it produces no value.
        Ok(TaggedValue::Null)
    }

    /// Resolve a child arena index by role from a node's payload children.
    /// Arity-aware (`child_role_index_for`): roles after an `optional` slot
    /// shift when the slot is absent (ADR-0081 optional roles).
    fn role_child_arena(&self, node: &AstNode, role: &str) -> Result<Option<usize>, LangError> {
        let Some(idx) = self.lang.child_role_index_for(node, role)? else {
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
                LangError::engine_node("eval_block: child index must be a number".to_string(), node)
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
                // `x = y` updates the nearest enclosing binding when `x`
                // is already bound (ADR-0074 feedback wires); otherwise it
                // defines a block-local binding.
                self.scopes.assign(&var_name, value.clone());
                result = value;
            } else {
                result = self.exec_node_inner(AstRef::new(idx))?;
            }
        }
        self.scopes.pop_scope();
        Ok(result)
    }

    /// Whether `node`'s shape declares the given child role (ADR-0081).
    fn role_declared(&self, node: &AstNode, role: &str) -> bool {
        matches!(self.lang.child_role_index(&node.kind, role), Ok(Some(_)))
    }

    /// Whether `node` is a binding (assign-like) node: its shape declares a
    /// `target` role. Mirrors the tree-walk `is_assign_node`.
    fn is_assign_node(&self, node: &AstNode) -> bool {
        self.role_declared(node, "target")
    }

    /// Whether `node` is a return node: its shape declares a `value` role
    /// but no `target` role (a binding declares both). Mirrors the
    /// tree-walk `is_return_node`.
    fn is_return_node(&self, node: &AstNode) -> bool {
        self.role_declared(node, "value") && !self.role_declared(node, "target")
    }

    /// Resolve the bound name of a binding's `target` child: a bare
    /// `var`-like node carries the name in its payload; a wrapper whose
    /// shape declares a `subject` role (e.g. a type annotation — the
    /// `subject`/`annot` role convention, ADR-0081) delegates to that
    /// child. Mirrors the tree-walk `binding_target_name`.
    fn binding_target_name(
        &self,
        target_idx: usize,
        err_ctx: &str,
        err_node: &AstNode,
    ) -> Result<String, LangError> {
        let mut node = self.ast.at(AstRef::new(target_idx))?;
        // Unwrap `subject`-roled wrappers (annotation-like nodes).
        for _ in 0..8 {
            if let Some(name) = node.payload.as_str() {
                return Ok(name.to_string());
            }
            let subject_idx = match self.role_child_arena(node, "subject")? {
                Some(i) => i,
                None => break,
            };
            node = self.ast.at(AstRef::new(subject_idx))?;
        }
        node.payload.as_str().map(str::to_string).ok_or_else(|| {
            LangError::engine_node(
                format!("{err_ctx}: binding target must be an identifier"),
                err_node,
            )
        })
    }

    /// `eval_rec` native op: recursive computation region (ADR-0074).
    /// Mirrors the v2 tree-walk's `prim_eval_rec` exactly.
    ///
    /// The `rec` node's children are the flattened block bindings. Each
    /// child is an expression:
    /// - `assign` children (`a = expr`) define bindings that may reference
    ///   each other cyclically.
    /// - `return` children exit the region early with the return value.
    /// - Any other expression is evaluated for its side effects / result.
    ///
    /// The engine iterates to a fixpoint: on each pass, every binding's
    /// value expression is re-evaluated using the current values of other
    /// bindings. When no binding's value changes between passes, the
    /// fixpoint is reached. A fuel limit (10000 iterations) prevents
    /// infinite loops. `return` inside a `rec` region exits the region
    /// early with the return value. The last child's value is the result
    /// of the region (unless a `return` exits early).
    ///
    /// Bindings are NOT pre-initialized. On the first pass, a binding that
    /// references another not-yet-computed binding will either find it in
    /// an enclosing scope (if it exists there) or encounter an error (e.g.
    /// arithmetic on an unbound variable). Errors during binding evaluation
    /// are caught and the binding is skipped for that pass — it will be
    /// retried on the next pass once its dependencies have values.
    fn prim_v2_eval_rec(&mut self, node: &AstNode) -> Result<TaggedValue, LangError> {
        let children = node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LangError::engine_node(
                    "eval_rec: expected children array in payload".to_string(),
                    node,
                )
            })?;
        if children.is_empty() {
            return Ok(TaggedValue::Null);
        }

        // Push a new scope for the rec region. Bindings will be defined
        // here as they are computed.
        self.scopes.push_scope();

        // Track the previous value of each binding for change detection.
        // Initialize from the enclosing scope so we can detect the first
        // change correctly.
        let mut prev_values: FxHashMap<Rc<str>, TaggedValue> = FxHashMap::default();
        let mut binding_names: Vec<Rc<str>> = Vec::new();
        for entry in children {
            let child_idx = entry.as_u64().ok_or_else(|| {
                LangError::engine_node("eval_rec: child index must be a number".to_string(), node)
            })? as usize;
            let child_node = self.ast.at(AstRef::new(child_idx))?;
            if self.is_assign_node(child_node) {
                if let Some(ac) = child_node
                    .payload
                    .get("children")
                    .and_then(|c| c.as_array())
                {
                    if ac.len() >= 2 {
                        let var_idx = self
                            .role_child_arena(child_node, "target")?
                            .or_else(|| ac[0].as_u64().map(|i| i as usize))
                            .ok_or_else(|| {
                                LangError::engine_node(
                                    "eval_rec: assign var index must be a number".to_string(),
                                    node,
                                )
                            })?;
                        if let Ok(name) = self.binding_target_name(var_idx, "eval_rec", node) {
                            // Seed from the enclosing scope (before the
                            // push). Since we already pushed a scope,
                            // lookup will search innermost-first, but the
                            // inner scope is empty at this point, so it
                            // falls through.
                            let outer_val = self
                                .scopes
                                .lookup(&name)
                                .cloned()
                                .unwrap_or(TaggedValue::Null);
                            let name_rc = Rc::from(name.as_str());
                            prev_values.insert(Rc::clone(&name_rc), outer_val);
                            binding_names.push(name_rc);
                        }
                    }
                }
            }
        }

        let mut result = TaggedValue::Null;
        const MAX_FUEL: usize = 10000;
        let mut converged = false;

        for _iter in 0..MAX_FUEL {
            let mut changed = false;

            for entry in children {
                let child_idx = entry.as_u64().ok_or_else(|| {
                    LangError::engine_node(
                        "eval_rec: child index must be a number".to_string(),
                        node,
                    )
                })? as usize;
                let child_node = self.ast.at(AstRef::new(child_idx))?;

                // Handle `return` — exit the region with the return value.
                if self.is_return_node(child_node) {
                    if let Some(rc) = child_node
                        .payload
                        .get("children")
                        .and_then(|c| c.as_array())
                    {
                        if let Some(first) = rc.first() {
                            if let Some(ret_idx) = first.as_u64() {
                                result = self.exec_node_inner(AstRef::new(ret_idx as usize))?;
                                self.scopes.pop_scope();
                                return Ok(result);
                            }
                        }
                    }
                    // `return` with no value → exit with Null.
                    self.scopes.pop_scope();
                    return Ok(TaggedValue::Null);
                }

                // Handle `assign` bindings — evaluate the value expression
                // and update the binding in the current scope. Errors are
                // caught: a binding may reference another not-yet-computed
                // binding (still unbound or Null), which would error. The
                // binding is skipped for this pass and retried next pass.
                if self.is_assign_node(child_node) {
                    if let Some(ac) = child_node
                        .payload
                        .get("children")
                        .and_then(|c| c.as_array())
                    {
                        if ac.len() >= 2 {
                            let var_idx = self
                                .role_child_arena(child_node, "target")?
                                .or_else(|| ac[0].as_u64().map(|i| i as usize))
                                .ok_or_else(|| {
                                    LangError::engine_node(
                                        "eval_rec: assign var index must be a number".to_string(),
                                        node,
                                    )
                                })?;
                            let value_idx = self
                                .role_child_arena(child_node, "value")?
                                .or_else(|| ac[1].as_u64().map(|i| i as usize))
                                .ok_or_else(|| {
                                    LangError::engine_node(
                                        "eval_rec: assign value index must be a number".to_string(),
                                        node,
                                    )
                                })?;
                            let var_name = self.binding_target_name(var_idx, "eval_rec", node)?;

                            match self.exec_node_inner(AstRef::new(value_idx)) {
                                Ok(new_val) => {
                                    self.scopes.define(&var_name, new_val.clone());
                                    result = new_val;
                                }
                                Err(_) => {
                                    // Skip — binding couldn't be computed
                                    // this pass (likely references a
                                    // not-yet-computed binding). It will be
                                    // retried on the next pass.
                                }
                            }
                            continue;
                        }
                    }
                }

                // Non-binding expression — evaluate normally.
                result = self.exec_node_inner(AstRef::new(child_idx))?;
            }

            // End-of-pass fixpoint check. A binding's value can also change
            // via a nested update — `x = y` inside a block updates the
            // nearest enclosing binding (`ScopeStack::assign`), which is
            // how the `while`/`do-while`/`for` desugars rewrite the
            // rec-scope `__continue`/`__xs` slots from inside the
            // `if`/`match` branch blocks. Comparing only at rec-level
            // assign-eval time misses those writes and converges one pass
            // early (or oscillates), so compare each tracked binding's
            // *current* env value against the value it held at the end of
            // the previous pass — and treat any cross-scope write observed
            // this pass (e.g. a body rebinding an outer variable) as a
            // change.
            if self.scopes.enclosing_updated {
                changed = true;
                self.scopes.enclosing_updated = false;
            }
            for name in &binding_names {
                let cur = self
                    .scopes
                    .lookup(name)
                    .cloned()
                    .unwrap_or(TaggedValue::Null);
                let prev = prev_values
                    .get(name.as_ref())
                    .cloned()
                    .unwrap_or(TaggedValue::Null);
                prev_values.insert(Rc::clone(name), cur.clone());
                if !rec_value_eq(&cur, &prev) {
                    changed = true;
                }
            }

            if !changed {
                converged = true;
                break;
            }
        }

        self.scopes.pop_scope();
        if !converged {
            return Err(LangError::engine_node(
                "rec: exceeded fuel limit (10000 iterations) without reaching fixpoint".to_string(),
                node,
            ));
        }
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
        // `scrutinee`/`block` positions resolve by named role, arity-aware:
        // the optional `motive` slot (ADR-0081 optional roles) shifts
        // `block` to index 2 when present. Positional 0 / `len >= 3 ? 2 : 1`
        // is the role-less fallback.
        let scrut_idx = children
            .get(
                self.lang
                    .child_role_index_for(node, "scrutinee")?
                    .unwrap_or(0),
            )
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                LangError::engine_node("match: missing scrutinee child".to_string(), node)
            })? as usize;
        let block_idx = children
            .get(
                self.lang
                    .child_role_index_for(node, "block")?
                    .unwrap_or(if children.len() >= 3 { 2 } else { 1 }),
            )
            .and_then(|v| v.as_u64())
            .ok_or_else(|| LangError::engine_node("match: missing block child".to_string(), node))?
            as usize;

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
            // `pattern`/`body` positions resolve by named role
            // (ADR-0081); positional 0/1 is the role-less fallback.
            let pattern_role = self
                .lang
                .child_role_index(&arm_node.kind, "pattern")?
                .unwrap_or(0);
            let body_role = self
                .lang
                .child_role_index(&arm_node.kind, "body")?
                .unwrap_or(1);
            let pattern_idx = arm_children
                .get(pattern_role)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    LangError::engine_node("match: arm missing pattern child".to_string(), node)
                })? as usize;
            let body_idx = arm_children
                .get(body_role)
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
        // `target`/`impl` positions resolve by named role (ADR-0081);
        // positional 0/1 is the role-less fallback.
        let body_role = self
            .lang
            .child_role_index(&node.kind, "target")?
            .unwrap_or(0);
        let impl_role = self.lang.child_role_index(&node.kind, "impl")?.unwrap_or(1);
        let body_idx = children
            .get(body_role)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| LangError::engine_node("handle: missing body child".to_string(), node))?
            as usize;
        let block_idx = children
            .get(impl_role)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
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
            // `pattern`/`body` (the handler position) resolve by named
            // role (ADR-0081); positional 0/1 is the role-less fallback.
            let pattern_role = self
                .lang
                .child_role_index(&clause_node.kind, "pattern")?
                .unwrap_or(0);
            let body_role = self
                .lang
                .child_role_index(&clause_node.kind, "body")?
                .unwrap_or(1);
            let pattern_idx = clause_children
                .get(pattern_role)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    LangError::engine_node("handle: clause missing pattern child".to_string(), node)
                })? as usize;
            let handler_idx = clause_children
                .get(body_role)
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
        // Bound-arrow record `{__term__, __bound__}` (ADR-0120 Phase 1 —
        // generated by `desugar.data` for `Type.rec`): applying it runs
        // `__term__` on `__bound__ ++ args`, delivered as ONE array
        // argument — the dispatch is a unary arrow over the flat `__in`
        // tuple `[motive, c0.., params.., v]`.
        if let TaggedValue::Obj(map) = callee {
            if let Some(term_val) = map.get("__term__") {
                let mut elems: Vec<TaggedValue> = Vec::new();
                if let Some(bound) = map.get("__bound__") {
                    match bound {
                        TaggedValue::Arr(items) => {
                            elems.extend(items.iter().cloned());
                        }
                        other => elems.push(other.clone()),
                    }
                }
                elems.extend(args.iter().cloned());
                let term_val = term_val.clone();
                let tuple = TaggedValue::Arr(Rc::from(elems));
                return self.apply_app(&term_val, &[tuple]);
            }
        }
        // Closure application — match on TaggedValue::Arrow directly.
        let (params, rest_param, body_idx, captured_env, home, array_pattern) = match callee {
            TaggedValue::Arrow {
                params,
                rest,
                body,
                env,
                home,
                array_pattern,
                ..
            } => (
                params.clone(),
                rest.clone(),
                *body,
                Rc::clone(env),
                *home,
                *array_pattern,
            ),
            _ => {
                return Err(LangError::engine(
                    "callee is not an arrow-value".to_string(),
                ));
            }
        };
        let body_ref = AstRef::new(body_idx);

        // Boundary-crossed arrow: its `body` index refers to a different
        // AST (the one that minted it). Run the body in that home frame,
        // with the home frame's modules, globals, and the arrow's captured
        // env.
        if let Some(frame_id) = home {
            let frame = home_frame(frame_id).ok_or_else(|| {
                LangError::engine(format!("arrow: unknown home frame {}", frame_id))
            })?;
            let home_modules_ref = frame.modules.as_ref();
            let mut home_vm = VmFrame::new(&*frame.ast, &*frame.lang, home_modules_ref, None);
            if let Some(g) = &frame.globals {
                home_vm.scopes.set_globals(Rc::clone(g));
            }
            let local_callee = TaggedValue::Arrow {
                params: params.clone(),
                rest: rest_param.clone(),
                body: body_idx,
                env: captured_env.clone(),
                home: None,
                array_pattern,
                // ADR-0106 item 2 step 1: re-wrap of an existing body.
                body_def: None,
            };
            return home_vm.apply_app(&local_callee, args);
        }
        // Array-pattern arrows (`[x, y] ~> body`) consume a single array
        // argument and destructure it by position.
        if array_pattern {
            if args.len() != 1 {
                return Err(LangError::engine(format!(
                    "array-pattern arrow expects 1 array argument, got {}",
                    args.len()
                )));
            }
            let arr = args[0].as_array().ok_or_else(|| {
                LangError::engine("array-pattern arrow expects an array argument".to_string())
            })?;
            let fixed = params.len();
            if arr.len() < fixed {
                return Err(LangError::engine(format!(
                    "array-pattern arrow expects at least {} element(s), got {}",
                    fixed,
                    arr.len()
                )));
            }
            if rest_param.is_none() && arr.len() != fixed {
                return Err(LangError::engine(format!(
                    "array-pattern arrow expects exactly {} element(s), got {}",
                    fixed,
                    arr.len()
                )));
            }
            let (head, tail) = arr.split_at(fixed);
            let saved_captured = self.scopes.captured_root.take();
            let saved_marks_len = self.scopes.marks.len();
            let saved_bindings_len = self.scopes.bindings.len();
            self.scopes.captured_root = Some(captured_env);
            self.scopes.push_scope();
            for (param, arg) in params.iter().zip(head.iter()) {
                self.scopes.define(param, arg.clone());
            }
            if let Some(rest_name) = &rest_param {
                self.scopes.define(
                    rest_name.as_ref(),
                    TaggedValue::Arr(Rc::from(tail.to_vec())),
                );
            }
            let result = self.exec_node_inner(body_ref);
            self.scopes.bindings.truncate(saved_bindings_len);
            self.scopes.marks.truncate(saved_marks_len);
            self.scopes.captured_root = saved_captured;
            return result;
        }
        // Positional arrows: ADR-0074 multi-param arrows may receive a
        // single `Pair` (product `***`) and destructure it into individual
        // args. `(a, b) ~> body` applied to `Pair(x, y)` binds `a=x, b=y`.
        let actual_args: Vec<TaggedValue> =
            if params.len() > 1 && args.len() == 1 && matches!(args[0], TaggedValue::Pair(_)) {
                flatten_pair(&args[0])
            } else {
                args.to_vec()
            };
        // Rest param (`[a, ...rest] ~> body`): surplus args bind to the rest
        // name as an array, so `args.len() >= params.len()` is required.
        let arity_ok = if rest_param.is_some() {
            actual_args.len() >= params.len()
        } else {
            params.len() == actual_args.len()
        };
        if !arity_ok {
            return Err(LangError::engine(format!(
                "arrow-value expects {} args, got {}",
                params.len(),
                actual_args.len()
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
        for (param, arg) in params.iter().zip(actual_args.iter()) {
            self.scopes.define(param, arg.clone());
        }
        // Bind the rest param to the surplus args as an array.
        if let Some(rest_name) = &rest_param {
            self.scopes.define(
                rest_name.as_ref(),
                TaggedValue::Arr(Rc::from(actual_args[params.len()..].to_vec())),
            );
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
/// `loop_step` (the ergonomic `loop` surface lowering) and `fold_step`
/// (the ADR-0120 `loop` fold node) are handled here — their step arg is a
/// closure evaluated once, then applied per iteration via
/// [`VmFrame::apply_app`]. The remaining derived step ops
/// (`while_step`/`for_step`/`fix_step`) use the [`Instr::DerivedStepOp`]
/// instruction instead, which re-evaluates their cond/body sub-programs
/// per iteration with the current value as incoming.
fn op_is_derived_step(op: &str) -> bool {
    matches!(op, "loop_step" | "fold_step")
}

// ---------------------------------------------------------------------------
// Shared op services (ADR-0044 §7)
// ---------------------------------------------------------------------------

/// The bytecode VM's [`NativeOpCtx`] — the engine-local globals the shared
/// op table needs: the scope stack, AST-node execution, and arrow
/// application. The op *semantics* live in
/// `plugin-engine-exec::native_ops`; this struct only forwards services.
struct VmCtx<'f, 'a, 'h, 'n> {
    frame: &'f mut VmFrame<'a, 'h>,
    node: &'n AstNode,
}

impl NativeOpCtx for VmCtx<'_, '_, '_, '_> {
    fn supports(&self, _service: OpService) -> bool {
        // The VM can provide every declared service: AST children are
        // compiled+executed on demand through `exec_node_inner`.
        true
    }

    fn node_payload(&self) -> Option<&Value> {
        Some(&self.node.payload)
    }

    fn eval_node(&mut self, r: AstRef) -> Result<TaggedValue, LangError> {
        self.frame.exec_node_inner(r)
    }

    fn env_lookup(&mut self, name: &str) -> Option<TaggedValue> {
        self.frame.scopes.lookup(name).cloned()
    }

    fn env_define(&mut self, name: &str, value: TaggedValue) {
        self.frame.scopes.define(name, value);
    }

    fn env_push(&mut self) {
        self.frame.scopes.push_scope();
    }

    fn env_pop(&mut self) {
        self.frame.scopes.pop_scope();
    }

    fn env_capture(&mut self) -> Rc<FxHashMap<Rc<str>, TaggedValue>> {
        Rc::new(self.frame.scopes.capture())
    }

    fn apply_arrow(
        &mut self,
        f: &TaggedValue,
        args: &[TaggedValue],
    ) -> Result<TaggedValue, LangError> {
        self.frame.apply_app(f, args)
    }
}

// ---------------------------------------------------------------------------
// Typed instruction evaluation — semantics delegated to the shared op table
// ---------------------------------------------------------------------------

/// Evaluate a typed arithmetic bytecode (`Add`/`Sub`/`Mul`/`Div`/`Mod`).
///
/// `a` is the first argument (popped second), `b` is the second argument
/// (popped first/top). The instruction is this engine's interpretation of
/// the op; the *semantics* delegate to the shared [`Num`] arithmetic (the
/// same rules as `apply_native_op`'s `add`/`sub`/`mul`/`div`/`mod` arms —
/// int-int preserves integer arithmetic, `Div`/`Mod` error on a zero
/// divisor, `Add` concatenates strings).
fn eval_bin_arith(instr: &Instr, a: TaggedValue, b: TaggedValue) -> Result<TaggedValue, LangError> {
    match instr {
        // `add` is the one arithmetic op that also concatenates strings —
        // shared `bin_add` covers both branches.
        Instr::Add => bin_add(&[a, b]),
        Instr::Sub => Ok(num_value(as_num(&a)? - as_num(&b)?)),
        Instr::Mul => Ok(num_value(as_num(&a)? * as_num(&b)?)),
        Instr::Div | Instr::Mod => {
            let a_n = as_num(&a)?;
            let b_n = as_num(&b)?;
            if b_n.to_f64() == 0.0 {
                return Err(LangError::engine(format!(
                    "{} by zero",
                    if matches!(instr, Instr::Div) {
                        "division"
                    } else {
                        "modulo"
                    }
                )));
            }
            Ok(num_value(if matches!(instr, Instr::Div) {
                a_n / b_n
            } else {
                a_n % b_n
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
/// (popped first/top). `Eq`/`Neq` use [`handl_value::eq`]; the ordered
/// comparisons use the shared [`Num`] ordering — identical to the
/// `apply_native_op` comparison arms.
fn eval_bin_cmp(instr: &Instr, a: TaggedValue, b: TaggedValue) -> Result<TaggedValue, LangError> {
    let ord = |a: &TaggedValue, b: &TaggedValue, f: &dyn Fn(Num, Num) -> bool| {
        Ok(TaggedValue::Bool(f(as_num(a)?, as_num(b)?)))
    };
    match instr {
        Instr::Eq => Ok(TaggedValue::Bool(handl_value::eq(&a, &b))),
        Instr::Neq => Ok(TaggedValue::Bool(!handl_value::eq(&a, &b))),
        Instr::Lt => ord(&a, &b, &|a, b| a < b),
        Instr::Lte => ord(&a, &b, &|a, b| a <= b),
        Instr::Gt => ord(&a, &b, &|a, b| a > b),
        Instr::Gte => ord(&a, &b, &|a, b| a >= b),
        _ => Err(LangError::engine(format!(
            "eval_bin_cmp: not a comparison instruction: {:?}",
            instr
        ))),
    }
}

/// Evaluate a binary [`ArithOp`] (ADR-0103 `Arith` instruction).
///
/// `a` is the first argument (popped second), `b` is the second argument
/// (popped first/top). Semantics delegate to the shared [`Num`]/
/// [`is_truthy`]/[`handl_value::eq`] vocabulary — identical to the typed
/// bytecodes and the `apply_native_op` arms.
fn eval_arith_bin(op: ArithOp, a: TaggedValue, b: TaggedValue) -> Result<TaggedValue, LangError> {
    match op {
        ArithOp::Add => bin_add(&[a, b]),
        ArithOp::Sub => Ok(num_value(as_num(&a)? - as_num(&b)?)),
        ArithOp::Mul => Ok(num_value(as_num(&a)? * as_num(&b)?)),
        ArithOp::Div | ArithOp::Mod => {
            let a_n = as_num(&a)?;
            let b_n = as_num(&b)?;
            if b_n.to_f64() == 0.0 {
                return Err(LangError::engine(format!(
                    "{} by zero",
                    if matches!(op, ArithOp::Div) {
                        "division"
                    } else {
                        "modulo"
                    }
                )));
            }
            Ok(num_value(if matches!(op, ArithOp::Div) {
                a_n / b_n
            } else {
                a_n % b_n
            }))
        }
        ArithOp::Eq => Ok(TaggedValue::Bool(handl_value::eq(&a, &b))),
        ArithOp::Neq => Ok(TaggedValue::Bool(!handl_value::eq(&a, &b))),
        ArithOp::Lt => Ok(TaggedValue::Bool(as_num(&a)? < as_num(&b)?)),
        ArithOp::Le => Ok(TaggedValue::Bool(as_num(&a)? <= as_num(&b)?)),
        ArithOp::Gt => Ok(TaggedValue::Bool(as_num(&a)? > as_num(&b)?)),
        ArithOp::Ge => Ok(TaggedValue::Bool(as_num(&a)? >= as_num(&b)?)),
        ArithOp::And => Ok(TaggedValue::Bool(is_truthy(&a) && is_truthy(&b))),
        ArithOp::Or => Ok(TaggedValue::Bool(is_truthy(&a) || is_truthy(&b))),
        // Unary ops are handled by `eval_arith_unary`.
        ArithOp::Neg | ArithOp::Not => Err(LangError::engine(format!(
            "eval_arith_bin: not a binary ArithOp: {:?}",
            op
        ))),
    }
}

/// Evaluate a unary [`ArithOp`] (`Neg`/`Not`) for the `Arith` instruction.
///
/// Semantics delegate to the shared [`Num`]/[`is_truthy`] vocabulary.
fn eval_arith_unary(op: ArithOp, a: TaggedValue) -> Result<TaggedValue, LangError> {
    match op {
        ArithOp::Neg => Ok(num_value(match as_num(&a)? {
            Num::Int(i) => Num::Int(-i),
            Num::Float(f) => Num::Float(-f),
        })),
        ArithOp::Not => Ok(TaggedValue::Bool(!is_truthy(&a))),
        // Binary ops are handled by `eval_arith_bin`.
        _ => Err(LangError::engine(format!(
            "eval_arith_unary: not a unary ArithOp: {:?}",
            op
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use handl_language::{ArrSource, Ast, AstNode, EvalSchemeDef, Language, SchemeVersion, Term};
    use handl_platform::{EvalContext, Function, Module, ModuleLookup};
    use handl_plugin_engine_exec::ImportResolver;
    use handl_plugin_module::ModuleRegistry;
    use plugin_engine::TermData;
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
    fn eval_standalone(scheme: &Term) -> Result<Value, LangError> {
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
        let s = TermData::op("const", vec![], &ArrSource::Native, Some(json!(42).into()))
            .to_term()
            .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn arr_const_null_when_no_literal() {
        let s = TermData::op("const", vec![], &ArrSource::Native, None)
            .to_term()
            .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, Value::Null);
    }

    // 2. arr proj_env

    #[test]
    fn arr_proj_env_bound_lookup() {
        let s = TermData::op("proj_env", vec![], &ArrSource::Native, Some("x".into()))
            .to_term()
            .unwrap();
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
        let s = TermData::op(
            "proj_env",
            vec![],
            &ArrSource::Native,
            Some("unknown".into()),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, Value::from("unknown"));
    }

    // 3. Seq const >>> id

    #[test]
    fn seq_const_then_id() {
        let s = TermData::seq(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(5).into())),
            TermData::op("id", vec![], &ArrSource::Native, None),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(5));
    }

    // 4. Bind shape

    #[test]
    fn bind_shape_extend_env_then_proj_env() {
        let s = TermData::seq(
            TermData::seq(
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(5).into())),
                TermData::op("extend_env", vec![], &ArrSource::Native, Some("x".into())),
            ),
            TermData::op("proj_env", vec![], &ArrSource::Native, Some("x".into())),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(5));
    }

    // 5. First on a pair

    #[test]
    fn first_id_on_pair() {
        let s = TermData::seq(
            TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(pair(json!(3), json!(7)).into()),
            ),
            TermData::first(TermData::op("id", vec![], &ArrSource::Native, None)),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, pair(json!(3), json!(7)));
    }

    #[test]
    fn first_const_on_pair() {
        let s = TermData::seq(
            TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(pair(json!(3), json!(7)).into()),
            ),
            TermData::first(TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(json!(99).into()),
            )),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, pair(json!(99), json!(7)));
    }

    // 6. Choice on Either

    #[test]
    fn choice_left() {
        let s = TermData::seq(
            TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(either_left(json!(5)).into()),
            ),
            TermData::choice(
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(1).into())),
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2).into())),
            ),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(1));
    }

    #[test]
    fn choice_right() {
        let s = TermData::seq(
            TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(either_right(json!(5)).into()),
            ),
            TermData::choice(
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(1).into())),
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2).into())),
            ),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(2));
    }

    // 7. tag + Choice (Branch lowering)

    #[test]
    fn tag_choice_true_yields_then() {
        let s = TermData::seq(
            TermData::seq(
                TermData::op(
                    "const",
                    vec![],
                    &ArrSource::Native,
                    Some(json!(true).into()),
                ),
                TermData::op("tag", vec![], &ArrSource::Native, None),
            ),
            TermData::choice(
                TermData::op("const", vec![], &ArrSource::Native, Some("yes".into())),
                TermData::op("const", vec![], &ArrSource::Native, Some("no".into())),
            ),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!("yes"));
    }

    #[test]
    fn tag_choice_false_yields_else() {
        let s = TermData::seq(
            TermData::seq(
                TermData::op(
                    "const",
                    vec![],
                    &ArrSource::Native,
                    Some(json!(false).into()),
                ),
                TermData::op("tag", vec![], &ArrSource::Native, None),
            ),
            TermData::choice(
                TermData::op("const", vec![], &ArrSource::Native, Some("yes".into())),
                TermData::op("const", vec![], &ArrSource::Native, Some("no".into())),
            ),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!("no"));
    }

    // 8. Loop causal single-step

    #[test]
    fn loop_causal_single_step() {
        let s = TermData::seq(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(10).into())),
            TermData::loop_feedback(TermData::op("inc_fst", vec![], &ArrSource::Native, None)),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(11));
    }

    // 8b. Loop multi-step (Either-based)

    #[test]
    fn loop_multi_step_either_countdown() {
        let s = TermData::seq(
            TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(pair(json!(42), json!(5)).into()),
            ),
            TermData::loop_feedback(TermData::op(
                "countdown_step",
                vec![],
                &ArrSource::Native,
                None,
            )),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn loop_multi_step_max_iterations_guard() {
        let s = TermData::seq(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(1).into())),
            TermData::loop_feedback(TermData::op(
                "infinite_step",
                vec![],
                &ArrSource::Native,
                None,
            )),
        )
        .to_term()
        .unwrap();
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
                TermData::op(
                    "proj_env",
                    vec![],
                    &ArrSource::Native,
                    Some(Value::String("$payload".into()).into()),
                )
                .to_term()
                .unwrap(),
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
                    TermData::op(
                        op,
                        vec![
                            TermData::op(
                                "proj_child",
                                vec![],
                                &ArrSource::Native,
                                Some(json!(0u64).into()),
                            ),
                            TermData::op(
                                "proj_child",
                                vec![],
                                &ArrSource::Native,
                                Some(json!(1u64).into()),
                            ),
                        ],
                        &ArrSource::Native,
                        None,
                    )
                    .to_term()
                    .unwrap(),
                ))
                .unwrap();
        }
        // if -> Seq(Seq(proj_child(0), tag), Choice(proj_child(1), proj_child(2)))
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.if",
                "if",
                TermData::seq(
                    TermData::seq(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(0u64).into()),
                        ),
                        TermData::op("tag", vec![], &ArrSource::Native, None),
                    ),
                    TermData::choice(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(1u64).into()),
                        ),
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(2u64).into()),
                        ),
                    ),
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        // object -> native `object` prim (reads the node's children payload)
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.object",
                "object",
                TermData::op("object", vec![], &ArrSource::Native, None)
                    .to_term()
                    .unwrap(),
            ))
            .unwrap();
        // function -> closure prim
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.function",
                "function",
                TermData::op("closure", vec![], &ArrSource::Native, None)
                    .to_term()
                    .unwrap(),
            ))
            .unwrap();
        // return -> proj_child(0) (v0.1: `return v` evaluates to `v`)
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.return",
                "return",
                TermData::op(
                    "proj_child",
                    vec![],
                    &ArrSource::Native,
                    Some(json!(0u64).into()),
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        // loop -> Seq(proj_child(0), Loop(loop_step(proj_child(1))))
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.loop",
                "loop",
                TermData::seq(
                    TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(0u64).into()),
                    ),
                    TermData::loop_feedback(TermData::op(
                        "loop_step",
                        vec![TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(1u64).into()),
                        )],
                        &ArrSource::Native,
                        None,
                    )),
                )
                .to_term()
                .unwrap(),
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
        // Term: Seq(const 5, Loop(while_step(is_pos, dec)))
        // is_pos: native op, 0 args, uses incoming (checks > 0)
        // dec: native op, 0 args, uses incoming (subtracts 1)
        let s = TermData::seq(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(5).into())),
            TermData::loop_feedback(TermData::op(
                "while_step",
                vec![
                    TermData::op("is_pos", vec![], &ArrSource::Native, None),
                    TermData::op("dec", vec![], &ArrSource::Native, None),
                ],
                &ArrSource::Native,
                None,
            )),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(0));
    }

    /// `while` loop with zero iterations: cond is immediately false.
    #[test]
    fn while_step_zero_iterations() {
        let s = TermData::seq(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(0).into())),
            TermData::loop_feedback(TermData::op(
                "while_step",
                vec![
                    TermData::op("is_pos", vec![], &ArrSource::Native, None),
                    TermData::op("dec", vec![], &ArrSource::Native, None),
                ],
                &ArrSource::Native,
                None,
            )),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(0));
    }

    /// `for` loop: accumulate 5 increments from 0. `for_loop(inc)` applied
    /// to pair(5, 0).
    #[test]
    fn for_step_accumulate() {
        let s = TermData::seq(
            TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(pair(json!(5), json!(0)).into()),
            ),
            TermData::loop_feedback(TermData::op(
                "for_step",
                vec![TermData::op("inc", vec![], &ArrSource::Native, None)],
                &ArrSource::Native,
                None,
            )),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(5));
    }

    /// `for` loop with zero iterations: counter is 0, return initial.
    #[test]
    fn for_step_zero_iterations() {
        let s = TermData::seq(
            TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(pair(json!(0), json!(99)).into()),
            ),
            TermData::loop_feedback(TermData::op(
                "for_step",
                vec![TermData::op("inc", vec![], &ArrSource::Native, None)],
                &ArrSource::Native,
                None,
            )),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(99));
    }

    /// `fix` loop: repeatedly halve 100 until fixpoint at 0.
    /// `fix_loop(half_floor)` applied to 100.
    #[test]
    fn fix_step_halve_to_zero() {
        let s = TermData::seq(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(100).into())),
            TermData::loop_feedback(TermData::op(
                "fix_step",
                vec![TermData::op("half_floor", vec![], &ArrSource::Native, None)],
                &ArrSource::Native,
                None,
            )),
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(0));
    }

    /// `fix` loop already at fixpoint: 0 → 0.
    #[test]
    fn fix_step_already_at_fixpoint() {
        let s = TermData::seq(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(0).into())),
            TermData::loop_feedback(TermData::op(
                "fix_step",
                vec![TermData::op("half_floor", vec![], &ArrSource::Native, None)],
                &ArrSource::Native,
                None,
            )),
        )
        .to_term()
        .unwrap();
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
                TermData::op(
                    "proj_env",
                    vec![],
                    &ArrSource::Native,
                    Some("$payload".into()),
                )
                .to_term()
                .unwrap(),
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
        let s = TermData::app(
            TermData::op("const", vec![], &ArrSource::Native, Some(closure.into())),
            vec![TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(json!(41).into()),
            )],
        )
        .to_term()
        .unwrap();
        let node = ast.at(AstRef::new(1)).unwrap();
        let instrs = compile(&s, &lang, &ast, node).unwrap();
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        let result = frame.exec_instrs(&instrs, node, None).unwrap();
        assert_eq!(result, json!(41));
    }

    // 10. Arithmetic parity

    #[test]
    fn native_add() {
        let s = TermData::op(
            "add",
            vec![
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2).into())),
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(3).into())),
            ],
            &ArrSource::Native,
            None,
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(5));
    }

    #[test]
    fn native_eq() {
        let s = TermData::op(
            "eq",
            vec![
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2).into())),
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2).into())),
            ],
            &ArrSource::Native,
            None,
        )
        .to_term()
        .unwrap();
        let result = eval_standalone(&s).unwrap();
        assert_eq!(result, json!(true));
    }

    // 11. Unknown op

    #[test]
    fn unknown_native_op_errors() {
        let s = TermData::op("bogus", vec![], &ArrSource::Native, None)
            .to_term()
            .unwrap();
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
        let s = TermData::op(
            "double",
            vec![TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(json!(21).into()),
            )],
            &ArrSource::Foreign("double".into()),
            None,
        )
        .to_term()
        .unwrap();
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
        let s = TermData::op(
            "double",
            vec![TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(json!(21).into()),
            )],
            &ArrSource::Foreign("double".into()),
            None,
        )
        .to_term()
        .unwrap();
        let err = eval_standalone(&s).unwrap_err();
        // ADR-0103: foreign ops now compile to `Ffi`, so the no-modules
        // error is reported as "unregistered ffi" (the `Ffi` handler's
        // fallback message) rather than "unregistered foreign".
        assert!(
            matches!(err, LangError::Engine { ref message, .. } if message.contains("unregistered ffi"))
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
                TermData::op(
                    "add",
                    vec![
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(0u64).into()),
                        ),
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(1u64).into()),
                        ),
                    ],
                    &ArrSource::Native,
                    None,
                )
                .to_term()
                .unwrap(),
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
                TermData::seq(
                    TermData::seq(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(0u64).into()),
                        ),
                        TermData::op("tag", vec![], &ArrSource::Native, None),
                    ),
                    TermData::choice(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(1u64).into()),
                        ),
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(2u64).into()),
                        ),
                    ),
                )
                .to_term()
                .unwrap(),
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
                TermData::seq(
                    TermData::seq(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(0u64).into()),
                        ),
                        TermData::op("tag", vec![], &ArrSource::Native, None),
                    ),
                    TermData::choice(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(1u64).into()),
                        ),
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(2u64).into()),
                        ),
                    ),
                )
                .to_term()
                .unwrap(),
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

        let resolver: Rc<dyn ImportResolver> = Rc::new(StubResolver {
            bindings: json!({"x": 42, "y": "hi"}),
        });

        let mut lang = Language::new();
        // var -> proj_env $payload
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.var",
                "var",
                TermData::op(
                    "proj_env",
                    vec![],
                    &ArrSource::Native,
                    Some(Value::from("$payload").into()),
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        // stmt.import -> import(proj_child(0))
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.stmt.import",
                "stmt.import",
                TermData::op(
                    "import",
                    vec![TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(Value::from(0u64).into()),
                    )],
                    &ArrSource::Native,
                    None,
                )
                .to_term()
                .unwrap(),
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
                TermData::op(
                    "import",
                    vec![TermData::op(
                        "const",
                        vec![],
                        &ArrSource::Native,
                        Some(Value::from("m.hnl").into()),
                    )],
                    &ArrSource::Native,
                    None,
                )
                .to_term()
                .unwrap(),
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

        let resolver: Rc<dyn ImportResolver> = Rc::new(StubResolver {
            bindings: json!({"x": 42 }),
        });

        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.var",
                "var",
                TermData::op(
                    "proj_env",
                    vec![],
                    &ArrSource::Native,
                    Some(Value::from("$payload").into()),
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.block",
                "block",
                TermData::op("eval_block", vec![], &ArrSource::Native, None)
                    .to_term()
                    .unwrap(),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.stmt.import",
                "stmt.import",
                TermData::op(
                    "import",
                    vec![TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(Value::from(0u64).into()),
                    )],
                    &ArrSource::Native,
                    None,
                )
                .to_term()
                .unwrap(),
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

    // --- ADR-0103 Arith hardware-boundary instruction tests ---
    //
    // These construct raw instruction sequences and execute them via
    // `exec_instrs`, verifying the consolidated `Arith(op)` ALU instruction
    // produces results identical to the typed bytecodes / `NativeOp` arms.

    /// Execute a raw instruction sequence against a fresh frame and return
    /// the result `Value`.
    fn eval_instrs(instrs: &[Instr]) -> Value {
        let ast = make_ast(vec![dummy_node()]);
        let lang = Language::new();
        let node = ast.at(AstRef::new(0)).unwrap();
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        frame.exec_instrs(instrs, node, None).unwrap()
    }

    #[test]
    fn arith_add_computes_sum() {
        let instrs = vec![
            Instr::Push(json!(3)),
            Instr::Push(json!(4)),
            Instr::Arith(ArithOp::Add),
            Instr::Return,
        ];
        assert_eq!(eval_instrs(&instrs), json!(7));
    }

    #[test]
    fn arith_sub_computes_difference() {
        let instrs = vec![
            Instr::Push(json!(10)),
            Instr::Push(json!(3)),
            Instr::Arith(ArithOp::Sub),
            Instr::Return,
        ];
        assert_eq!(eval_instrs(&instrs), json!(7));
    }

    #[test]
    fn arith_mul_computes_product() {
        let instrs = vec![
            Instr::Push(json!(3)),
            Instr::Push(json!(4)),
            Instr::Arith(ArithOp::Mul),
            Instr::Return,
        ];
        assert_eq!(eval_instrs(&instrs), json!(12));
    }

    #[test]
    fn arith_eq_returns_true_for_equal() {
        let instrs = vec![
            Instr::Push(json!(3)),
            Instr::Push(json!(3)),
            Instr::Arith(ArithOp::Eq),
            Instr::Return,
        ];
        assert_eq!(eval_instrs(&instrs), json!(true));
    }

    #[test]
    fn arith_lt_returns_true_for_less() {
        let instrs = vec![
            Instr::Push(json!(3)),
            Instr::Push(json!(5)),
            Instr::Arith(ArithOp::Lt),
            Instr::Return,
        ];
        assert_eq!(eval_instrs(&instrs), json!(true));
    }

    #[test]
    fn arith_neg_is_unary() {
        let instrs = vec![
            Instr::Push(json!(5)),
            Instr::Arith(ArithOp::Neg),
            Instr::Return,
        ];
        assert_eq!(eval_instrs(&instrs), json!(-5));
    }

    #[test]
    fn arith_not_is_unary() {
        let instrs = vec![
            Instr::Push(json!(false)),
            Instr::Arith(ArithOp::Not),
            Instr::Return,
        ];
        assert_eq!(eval_instrs(&instrs), json!(true));
    }

    // --- ADR-0103 Ffi hardware-boundary instruction test ---

    #[test]
    fn ffi_calls_registered_foreign_function() {
        let instrs = vec![
            Instr::Push(json!(21)),
            Instr::Ffi {
                module: "test".to_string(),
                name: "double".to_string(),
                n: 1,
            },
            Instr::Return,
        ];
        let ast = make_ast(vec![dummy_node()]);
        let lang = Language::new();
        let node = ast.at(AstRef::new(0)).unwrap();
        let mut frame = VmFrame::new(&ast, &lang, None, None);
        frame.register_ffi("test", "double", |args| {
            let n = args[0].as_i64().ok_or("expected integer")?;
            Ok(Value::from(n * 2))
        });
        let result = frame.exec_instrs(&instrs, node, None).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn ffi_falls_back_to_module_lookup() {
        // When no Ffi registry entry exists, `Ffi { module: "host", name,
        // n }` falls back to the existing `ForeignOp` dispatch via
        // `ModuleLookup` — so a registered host function still resolves.
        let instrs = vec![
            Instr::Push(json!(3)),
            Instr::Ffi {
                module: "host".to_string(),
                name: "double".to_string(),
                n: 1,
            },
            Instr::Return,
        ];
        let ast = make_ast(vec![dummy_node()]);
        let lang = Language::new();
        let node = ast.at(AstRef::new(0)).unwrap();
        let registry = double_registry();
        let mut frame = VmFrame::new(&ast, &lang, Some(&registry), None);
        let result = frame.exec_instrs(&instrs, node, None).unwrap();
        assert_eq!(result, json!(6));
    }
}

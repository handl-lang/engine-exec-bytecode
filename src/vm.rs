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
//! - **Arrow-value** (`App`): `{"__arrow__": true, "params": [...],
//!   "body_def": {"__ast__": <arena>}, "binds": {...}, "handlers": [...]}`
//!   — self-contained serializable data (ADR-0106).
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
//!   `VmFrame::apply_native_v2`, which applies the body closure via
//!   `VmFrame::apply_app`.
//! - `while_step`/`for_step`/`fix_step`: the `DerivedStepOp` instruction
//!   re-evaluates the arg sub-programs per iteration with the current value
//!   (extracted from the loop pair) as incoming, then dispatches to
//!   `VmFrame::apply_native_v2`. See the [`instr`] module
//!   docs for details.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use handl_language::{Ast, AstNode, AstRef, LangError, Language, Term, Value};
use handl_value::{TaggedValue, tagged_to_value};
use hnl_engine_exec::native_ops::{
    self, NativeOpCtx, Num, OpService, as_num, bin_add, build_loop_step_args, is_truthy,
    loop_step_outcome, num_value, step_decision, type_tag,
};
use hnl_engine_exec::pattern_match::{self, PatternMatchCtx};
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
/// [`ModuleLookup`].
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
    #[cfg(test)]
    pub(crate) fn register(&mut self, module: &str, name: &str, f: FfiFn) {
        self.fns.insert((module.to_string(), name.to_string()), f);
    }

    /// Look up the host function for `(module, name)`.
    pub(crate) fn lookup(&self, module: &str, name: &str) -> Option<&FfiFn> {
        self.fns.get(&(module.to_string(), name.to_string()))
    }
}

/// Tag for first-class `arr <op>` dispatch objects — mirrors the
/// tree-walk's `ARR_DISPATCH_TAG` (ADR-0103 value shape).
const ARR_DISPATCH_TAG: &str = "__arr__";

/// Id space base for bytecode-minted `{"__resume__": id}` values — keeps
/// them disjoint from the tree-walk's small sequential ids so a
/// cross-engine `resume` reports "unknown resumption" (see
/// [`VmFrame::resume_next_id`]).
const RESUME_ID_BASE: u64 = 1_000_000_000;

/// A replay journal for multi-shot `resume` (ADR-0158 workstream —
/// `host/project/design/multishot-resume.md`): everything `resume`
/// needs to re-run one `handle` body (delimited re-run model). All
/// captured state is data — the body node + its def-arena slot, the
/// `handle`-entry scope snapshot, and the covering handler value.
/// Mirrors the tree-walk's `ResumeJournal`; `target`/`active_def`
/// replace its `body: TermData`/`ast` pair (the VM evals arena nodes,
/// not term trees).
pub(crate) struct ResumeJournal {
    /// Unique journal instance id — suspensions name the journal they
    /// belong to so nested `handle`s catch only their own.
    id: u64,
    /// The `handle` body node — `handle_dyn`'s `target` child —
    /// resolved against the arena selected by `active_def`.
    target: AstRef,
    /// The `def_arenas` slot live when the journal was minted (`None` =
    /// the frame's `ast`). `apply_arrow_body` switches `active_def` for
    /// the clause call that invokes `resume`, so each re-run restores
    /// the body's own arena.
    active_def: Option<usize>,
    /// Environment snapshot at `handle` entry — each run evals a clone,
    /// so `assign` writes and scope pushes stay inside the run.
    entry_env: ScopeStack,
    /// The handler value (`{impls: {op: impl}}` or `{op: impl}` record).
    handler: TaggedValue,
    /// `(op, result)` entries in perform order: the replay log.
    /// `resume(x)` at suspension index `i` seeds a fresh journal with
    /// `entries[..i] ++ [(op, x)]`.
    entries: RefCell<Vec<(Rc<str>, TaggedValue)>>,
    /// Count of journal-intercepted `perform`s so far this run — the
    /// replay cursor. Fresh runs start at 0 and consume `entries`.
    cursor: Cell<usize>,
}

/// A captured `perform` suspension awaiting `resume` (see
/// [`VmFrame::suspensions`]) — mirrors the tree-walk's
/// `ResumeSuspension`.
pub(crate) struct ResumeSuspension {
    /// The journal active when the perform suspended.
    journal: Rc<ResumeJournal>,
    /// This perform's index in the journal's `entries` order.
    index: usize,
    /// The performed op name.
    op: Rc<str>,
    /// The perform's payload argument.
    arg: TaggedValue,
}

/// Decode a `__resume__` payload to a suspension id — mirrors the
/// tree-walk decoder (integral `Int`/`Float` payloads only).
fn resume_id_of(v: &TaggedValue) -> Option<u64> {
    match v {
        TaggedValue::Int(i) if *i >= 0 => Some(*i as u64),
        TaggedValue::Float(f) if *f >= 0.0 && f.fract() == 0.0 => Some(*f as u64),
        _ => None,
    }
}

/// Remaining-stack budget at which [`stacker::maybe_grow`] moves the next
/// eval step onto a fresh segment. The instruction loop is iterative,
/// but nested node evals and arrow applications still recurse natively
/// (`exec_node_inner` → `exec_instrs_inner` → `apply_app` →
/// `apply_arrow_body` → `exec_node_inner`), so a self-recursive arrow
/// overflows the platform stack at a few thousand calls — same seam as
/// the tree-walk engine's `STACK_RED_ZONE`.
const STACK_RED_ZONE: usize = 4 * 1024 * 1024;

/// Fresh stack segment allocated by [`stacker::maybe_grow`] on overflow
/// (see [`STACK_RED_ZONE`]).
const STACK_SEGMENT: usize = 8 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Scope stack — identical semantics to the tree-walk interpreter
// ---------------------------------------------------------------------------

/// A scope stack: a flat `Vec<(Rc<str>, TaggedValue)>` of bindings with
/// scope markers, searched innermost-first.
///
/// Mirrors the tree-walk's `ScopeStack` exactly: `push_scope` records a
/// mark; lookup searches innermost-first (linear scan from the end);
/// arrow applications install the arrow's residual `binds` as
/// `captured_root`, bind params, evaluate the `body_def` arena, then
/// restore. For the small scopes typical in handler programs (3-10
/// variables), a linear scan is faster than `HashMap` because there is no
/// hashing overhead.
///
/// `Clone`: multi-shot `resume` snapshots the `handle`-entry environment
/// into a replay journal so each `resume` re-run starts from the same
/// scope state (mirrors the tree-walk `entry_env` clone).
#[derive(Clone)]
pub(crate) struct ScopeStack {
    /// Flat list of (name, value) bindings, innermost last.
    pub(crate) bindings: Vec<(Rc<str>, TaggedValue)>,
    /// Stack of mark positions — each `push_scope` records the current
    /// length, each `pop_scope` truncates back to that length.
    pub(crate) marks: Vec<usize>,
    /// Lexical floor: bindings below this index belong to a *caller* frame
    /// (`apply_app` sets it to the pre-call bindings length and restores it
    /// after). `lookup`/`assign`/`capture` only see bindings at or above
    /// the floor, so a callee can neither read nor mutate its caller's
    /// live locals — matching the tree-walk, where an arrow's visible
    /// bindings are its `binds` plus its own scopes only.
    pub(crate) floor: usize,
    /// The applied arrow's residual `binds` for the current invocation.
    /// Searched after `bindings` so user-defined scopes shadow residual
    /// bindings. Stored as `Rc` so arrow calls avoid deep-cloning the
    /// `binds` map.
    pub(crate) captured_root: Option<Rc<FxHashMap<Rc<str>, TaggedValue>>>,
    /// Host-provided globals (e.g. `self` and `__arg_N`), searched after
    /// local/captured bindings so user-defined scopes shadow them. Held in
    /// a cheaply-cloned `Rc` so it is excluded from `capture()` and shared
    /// across closure calls. Mirrors the tree-walk `ScopeStack::globals`.
    pub(crate) globals: Rc<FxHashMap<Rc<str>, TaggedValue>>,
    /// Set when `assign` updates a binding in an *enclosing* scope (below
    /// the innermost mark) with a different value. Read and cleared by
    /// `eval_rec` fixpoint detection (the shared `eval_rec` op) — such writes are
    /// the ADR-0074 feedback wires: the `while`/`do-while`/`for` desugars
    /// update `__continue`/`__xs` (and user bindings like `x = x + 1`)
    /// from inside `if`/`match` branch blocks, and a pass that produces
    /// any such write must not count as converged. Mirrors the tree-walk
    /// `ScopeStack::enclosing_updated`.
    pub(crate) enclosing_updated: bool,
    /// Names bound by enclosing `eval_rec` regions (flat stack, segmented
    /// by `rec_marks`). The `closure` mint keeps these as residual `binds`
    /// on minted arrows rather than substituting them into `body_def`
    /// (ADR-0106): substituting a rec name would embed a stale
    /// current-pass value and break fixpoint convergence.
    pub(crate) rec_names: Vec<Rc<str>>,
    /// Segment marks for `rec_names` — one per active `eval_rec` region.
    pub(crate) rec_marks: Vec<usize>,
}

impl ScopeStack {
    pub(crate) fn new() -> Self {
        Self {
            bindings: Vec::new(),
            marks: vec![0],
            floor: 0,
            captured_root: None,
            globals: Rc::new(FxHashMap::default()),
            enclosing_updated: false,
            rec_names: Vec::new(),
            rec_marks: Vec::new(),
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
        // Search innermost bindings first (reverse linear scan), down to
        // the lexical floor — a caller frame's bindings are not visible.
        for (n, v) in self.bindings[self.floor..].iter().rev() {
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
            if pos < self.floor {
                break;
            }
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
        if self.marks.len() > 1
            && let Some(mark) = self.marks.pop()
        {
            self.bindings.truncate(mark);
        }
    }

    /// Collect all visible bindings into a single flat map.
    ///
    /// Used by the `closure` mint to gather the definition-site bindings
    /// split into `body_def` substitution + residual `binds` (ADR-0106).
    /// Inner scopes shadow outer ones (later entries overwrite earlier).
    /// Includes the enclosing arrow's `binds` (if any) so that nested
    /// arrows retain access to bindings from all enclosing arrow scopes.
    pub(crate) fn capture(&self) -> FxHashMap<Rc<str>, TaggedValue> {
        let mut captured = FxHashMap::default();
        // Include the enclosing arrow's `binds` first (lower priority).
        if let Some(parent_captured) = &self.captured_root {
            for (k, v) in parent_captured.iter() {
                captured.insert(Rc::clone(k), v.clone());
            }
        }
        // Then include current scopes (higher priority, overwrites parent).
        for (n, v) in &self.bindings[self.floor..] {
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

/// Whether a value is arrow-like for `flow` composition purposes (mirrors
/// the tree-walk `is_arrow_like`): a `TaggedValue::Arrow` closure or a
/// `{"__tag__": "Arrow", "first"|"inputs": …}` composite/descriptor record.
fn is_arrow_like(v: &TaggedValue) -> bool {
    match v {
        TaggedValue::Arrow { .. } => true,
        TaggedValue::Obj(m) => {
            m.get("__tag__").and_then(|t| t.as_str()) == Some("Arrow")
                && (m.get("first").is_some() || m.get("inputs").is_some())
        }
        _ => false,
    }
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

/// Extract the implementation entry for `op` from a handler-like value
/// (mirrors the tree-walk `handler_impl`).
///
/// A canonical `handler` value is a tagged object
/// `{"__tag__": "handler", "impls": {"<op>": <arrow or Term>, ...}}`.
/// Surface handler records are plain objects whose keys are the operation
/// names: `{"<op>": <arrow or Term>, ...}`. Returns the value for `op`,
/// or `None` when the value has no implementation for `op`.
fn handler_impl(v: &TaggedValue, op: &str) -> Option<TaggedValue> {
    v.get("impls")
        .and_then(|impls| impls.get(op))
        .or_else(|| v.get(op))
        .cloned()
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
    import_resolver: Option<&'h Rc<dyn hnl_engine_exec::ImportResolver>>,
    /// Deserialized `body_def` arenas carried by applied arrows
    /// (ADR-0106). Applying an arrow pushes its arena here and executes
    /// under `active_def`; the entries keep the arenas alive for the
    /// frame's lifetime so `def_cache` keys stay valid.
    def_arenas: Vec<Rc<Ast>>,
    /// Memo of `body_def` `Rc` pointer -> `def_arenas` slot, so applying
    /// the same arrow repeatedly does not re-deserialize its arena. The
    /// value stores an `Rc::clone` of the definition, so the keyed
    /// address cannot be recycled while the entry lives — a stale hit is
    /// impossible.
    def_arena_keys: FxHashMap<usize, (usize, Rc<TaggedValue>)>,
    /// Per-node v2 compilation cache, keyed by `(arena_slot, node_index)`:
    /// slot `0` is `self.ast`, slots `1..` index into `def_arenas`, so
    /// def-arena node indices never collide with `self.ast` indices.
    def_cache: FxHashMap<(usize, usize), Rc<[Instr]>>,
    /// The `def_arenas` slot whose arena is currently executing, or `None`
    /// while running `self.ast`. `apply_arrow_body` sets/restores it
    /// around the arrow body's execution.
    active_def: Option<usize>,
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
    /// The runtime handler stack (ADR-0076 dynamic extent): `handle_dyn`
    /// pushes the impl value for the body's extent; `arr_dispatch` and
    /// the `IO(sig)` former interpretation resolve ops through it
    /// (innermost-first), matching the tree-walk `handler_stack`.
    handler_stack: Vec<TaggedValue>,
    /// Active replay journals for multi-shot `resume` (ADR-0158
    /// workstream — `host/project/design/multishot-resume.md`). One
    /// entry per `handle` whose handler has a ≥2-ary resumption clause,
    /// innermost last. Frame-local (the VM's journal state is owned,
    /// not thread-local like the tree-walk's — `resume` values minted
    /// here are only callable while this frame is alive).
    journals: Vec<Rc<ResumeJournal>>,
    /// Suspension table: `LangError::Suspended { id }` carries only the
    /// index; the captured state lives here. Entries persist for the
    /// frame's lifetime so an escaped `resume` still runs while the
    /// frame is alive.
    suspensions: FxHashMap<u64, Rc<ResumeSuspension>>,
    /// Monotonic journal/suspension id source, starting at
    /// [`RESUME_ID_BASE`] so bytecode-minted `{"__resume__": id}`
    /// values can never alias the tree-walk's small sequential ids —
    /// a cross-engine `resume` call reports "unknown resumption"
    /// instead of silently resuming the wrong suspension.
    resume_next_id: u64,
}

impl<'a, 'h> VmFrame<'a, 'h> {
    /// Create a new v2 frame with a fresh cache and scope stack.
    pub(crate) fn new(
        ast: &'a Ast,
        lang: &'a Language,
        modules: Option<&'h Rc<dyn handl_platform::ModuleLookup>>,
        import_resolver: Option<&'h Rc<dyn hnl_engine_exec::ImportResolver>>,
    ) -> Self {
        Self {
            ast,
            lang,
            modules,
            import_resolver,
            def_arenas: Vec::new(),
            def_arena_keys: FxHashMap::default(),
            def_cache: FxHashMap::default(),
            active_def: None,
            scopes: ScopeStack::new(),
            scratch_stack: Vec::new(),
            ffi_registry: FfiRegistry::new(),
            handler_stack: Vec::new(),
            journals: Vec::new(),
            suspensions: FxHashMap::default(),
            resume_next_id: RESUME_ID_BASE,
        }
    }

    /// Define a global variable in the root scope (for unit tests).
    #[cfg(test)]
    pub(crate) fn define_global(&mut self, name: &str, value: TaggedValue) {
        self.scopes.define(name, value);
    }

    /// Register a mock foreign function for [`Instr::Ffi`] (for unit tests).
    ///
    /// The function receives the popped argument values (as `serde_json::
    /// Value`) and returns a result value or an error message.
    #[cfg(test)]
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

    /// The arena currently executing: the applied arrow's `body_def`
    /// arena while `active_def` is set, else the frame's own `ast`.
    pub(crate) fn cur_ast(&self) -> &Ast {
        match self.active_def {
            Some(slot) => &self.def_arenas[slot],
            None => self.ast,
        }
    }

    /// Deserialize `def` (a `{"__ast__": <Ast>}` `body_def` value) into a
    /// `def_arenas` slot, memoized by the definition's `Rc` pointer.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the definition is not a well-formed `__ast__`
    /// value or deserializes to an empty arena — the arrow is malformed.
    fn def_arena_slot(&mut self, def: &Rc<TaggedValue>) -> Result<usize, LangError> {
        let key = Rc::as_ptr(def) as usize;
        if let Some(&(slot, _)) = self.def_arena_keys.get(&key) {
            return Ok(slot);
        }
        let ast = handl_language::as_ast(&tagged_to_value(def))
            .filter(|a| !a.is_empty())
            .ok_or_else(|| LangError::engine("arrow: malformed body_def".to_string()))?;
        let slot = self.def_arenas.len();
        self.def_arenas.push(Rc::new(ast));
        self.def_arena_keys.insert(key, (slot, Rc::clone(def)));
        Ok(slot)
    }
}

/// `PatternMatchCtx` impl for the bytecode VM (ADR-0061).
///
/// The VM implements the trait directly: `define_binding` delegates to
/// the scope stack, and `eval_pattern_node` delegates to `exec_node`
/// (which uses `self.ast` internally; the `ast` parameter from the trait
/// is the same AST).
impl<'a, 'h> PatternMatchCtx for VmFrame<'a, 'h> {
    fn define_binding(&mut self, name: &str, value: TaggedValue) {
        self.scopes.define(name, value);
    }

    fn eval_pattern_node(
        &mut self,
        _ast: &Ast,
        node_ref: AstRef,
    ) -> Result<TaggedValue, LangError> {
        self.exec_node_inner(node_ref)
    }

    fn global_value(&self, name: &str) -> Option<TaggedValue> {
        self.scopes.globals.get(name).cloned()
    }

    fn scope_value(&self, name: &str) -> Option<TaggedValue> {
        self.scopes.lookup(name).cloned()
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
    /// [`exec_node`]. Returns a [`TaggedValue`].
    pub(crate) fn exec_node_inner(&mut self, node_ref: AstRef) -> Result<TaggedValue, LangError> {
        // Growable-stack seam (see `STACK_RED_ZONE`): every recursive
        // eval cycle — arrow-body exec via `apply_arrow_body`, child
        // exec via `ctx.eval_node`, `match`/`handle` shells — re-enters
        // here, so checking at this one funnel bounds every level.
        stacker::maybe_grow(STACK_RED_ZONE, STACK_SEGMENT, || {
            self.exec_node_inner_unguarded(node_ref)
        })
    }

    /// The recursive body of [`exec_node_inner`],
    /// run under [`stacker::maybe_grow`].
    fn exec_node_inner_unguarded(&mut self, node_ref: AstRef) -> Result<TaggedValue, LangError> {
        // Resolve the node in the active arena. For a `body_def` arena we
        // clone the `Rc` into a local handle so `node` borrows the handle
        // rather than `self` — no per-node clone on either path.
        let arena_handle;
        let (node, arena): (&AstNode, &Ast) = match self.active_def {
            Some(slot) => {
                arena_handle = Rc::clone(&self.def_arenas[slot]);
                (arena_handle.at(node_ref)?, &arena_handle)
            }
            None => (self.ast.at(node_ref)?, self.ast),
        };

        // Compile-on-demand with cache. `def_arena` programs are keyed by
        // `(slot + 1, index)`; slot 0 is `self.ast`.
        let key = match self.active_def {
            Some(slot) => (slot + 1, node_ref.index()),
            None => (0, node_ref.index()),
        };
        if !self.def_cache.contains_key(&key) {
            let scheme_def = self.lang.eval.get_by_identity(&node.kind);
            let instrs = match scheme_def {
                Some(def) => compile(&def.scheme, self.lang, arena, node)?,
                None => vec![Instr::LoadPayload, Instr::Return],
            };
            self.def_cache.insert(key, Rc::from(instrs));
        }

        // Rc-clone the instruction slice (cheap refcount bump) to avoid
        // borrow conflict with recursive exec_node calls that may insert
        // into the cache.
        let instrs = self
            .def_cache
            .get(&key)
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
    #[cfg(test)]
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
    /// [`exec_instrs`]. Operates on a `Vec<TaggedValue>`
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
        let mut current_incoming = incoming.clone();
        // Save-stack for `SetIncoming`/`RestoreIncoming` pairs so a
        // `Seq`/`Choice` nested inside an `Arr` arg region restores the
        // enclosing region's incoming for later sibling args.
        let mut incoming_save: Vec<Option<TaggedValue>> = Vec::new();
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
                                .to_vec()
                        } else {
                            args
                        };
                    let result = self.apply_app(&callee, &actual_args, node)?;
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
                            .to_vec();
                        self.apply_app(&callee, &actual, node)?
                    } else {
                        self.apply_app(&callee, std::slice::from_ref(&arg0), node)?
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
                    let result = self.apply_app(&callee, &[arg0, arg1], node)?;
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
                    let result = self.apply_app(&callee, &[arg0, arg1, arg2], node)?;
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
                Instr::PushIncoming => {
                    stack.push(current_incoming.clone().unwrap_or(TaggedValue::Null));
                }
                Instr::SetIncoming => {
                    incoming_save.push(current_incoming.take());
                    current_incoming = stack.last().cloned();
                }
                Instr::RestoreIncoming => {
                    current_incoming = incoming_save.pop().flatten();
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
                        // Clone, not take: arg-position incoming-ops and
                        // the enclosing op share the region's incoming
                        // (tree-walk passes the same `incoming` to every
                        // subterm) — consuming it here would starve a
                        // later sibling op in the same region.
                        current_incoming.clone()
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
                Instr::SubProgram(prog) => {
                    // Detached `Seq`/`Choice` arg — run on a fresh stack
                    // with the region's incoming; only the sub-program's
                    // top result returns (see `Instr::SubProgram`).
                    let result = self.exec_instrs_inner(prog, node, current_incoming.clone())?;
                    stack.push(result);
                }
                Instr::ArrDispatch(n) => {
                    // `arr <op>` (`Op { name: "$payload" }`): the op name
                    // lives in the node's payload — a bare string, or the
                    // `name`/`op` field of an object payload (mirrors the
                    // tree-walk `eval_arr` resolution order).
                    let op = node
                        .payload
                        .as_str()
                        .map(str::to_string)
                        .or_else(|| {
                            node.payload
                                .get("name")
                                .and_then(|v| v.as_str())
                                .map(str::to_string)
                        })
                        .or_else(|| {
                            node.payload
                                .get("op")
                                .and_then(|v| v.as_str())
                                .map(str::to_string)
                        })
                        .ok_or_else(|| {
                            LangError::engine_node(
                                "expected string payload for $payload op".to_string(),
                                node,
                            )
                        })?;
                    let mut args = Vec::with_capacity(*n);
                    for _ in 0..*n {
                        args.push(stack.pop().ok_or_else(|| {
                            LangError::engine_node(
                                "stack underflow on ArrDispatch".to_string(),
                                node,
                            )
                        })?);
                    }
                    args.reverse();
                    // The operand: the value the enclosing combinator left
                    // on the stack (a `Seq` predecessor's result or the
                    // program incoming, pushed at entry). Falls back to
                    // `current_incoming` when a `Store` consumed the stack
                    // copy — `extend_env`'s result IS its bound value, so
                    // `Seq{extend_env, arr <op>}` still sees it.
                    let inc = stack.pop().or_else(|| current_incoming.take());
                    if args.is_empty() {
                        match inc {
                            Some(v) => {
                                let result = self.arr_dispatch(&op, &v, node)?;
                                stack.push(result);
                            }
                            None => {
                                // First-class `arr <op>`: a dispatch object
                                // `apply_app` recognizes (mirrors the
                                // tree-walk `__arr__` value shape).
                                let mut map = FxHashMap::<Rc<str>, TaggedValue>::default();
                                map.insert(Rc::from("op"), TaggedValue::Str(Rc::from(op.as_str())));
                                let mut obj = FxHashMap::<Rc<str>, TaggedValue>::default();
                                obj.insert(
                                    Rc::from(ARR_DISPATCH_TAG),
                                    TaggedValue::Obj(Rc::new(map)),
                                );
                                stack.push(TaggedValue::Obj(Rc::new(obj)));
                            }
                        }
                    } else {
                        let result = self.apply_native_v2(&op, &args, node, inc)?;
                        stack.push(result);
                    }
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
                    capture,
                } => {
                    // Mint a self-contained arrow (ADR-0106): `body_def`
                    // is the body subtree extracted into a standalone
                    // arena with every free variable substituted by a
                    // `lit` node holding its definition-site value;
                    // `binds` keeps the residual names that substitution
                    // must not close over (active `eval_rec` binders,
                    // non-serializable resources).
                    //
                    // `capture` is the compile-time free-name superset of
                    // the body: the substitution map is filtered to names
                    // the body can actually reference, so sequential
                    // sibling closures cannot nest exponentially.
                    let mut captured = self.scopes.capture();
                    if let Some(names) = capture {
                        captured.retain(|k, _| names.contains(k));
                    }
                    // The arrow's own parameters are bound at apply time —
                    // they are binders, not free variables. If a param name
                    // collides with an in-scope binding, substituting it
                    // would bake the *enclosing* value into `body_def` and
                    // shadow the argument (shared `closure` op parity).
                    for p in params.iter() {
                        captured.remove(p.as_str());
                    }
                    if let Some(r) = rest {
                        captured.remove(r.as_ref());
                    }
                    let mut subst: std::collections::HashMap<Rc<str>, Value> =
                        std::collections::HashMap::with_capacity(captured.len());
                    let mut binds: FxHashMap<Rc<str>, TaggedValue> = FxHashMap::default();
                    for (name, value) in captured.iter() {
                        if self.scopes.rec_names.contains(name) || !native_ops::arrow_free(value) {
                            binds.insert(Rc::clone(name), value.clone());
                        } else {
                            subst.insert(Rc::clone(name), tagged_to_value(value));
                        }
                    }
                    let sub = self
                        .cur_ast()
                        .instantiate(AstRef::new(*body), self.lang, &subst)
                        .map_err(|e| e.with_node(node))?;
                    stack.push(TaggedValue::Arrow {
                        params: Rc::clone(params),
                        rest: rest.clone(),
                        array_pattern: *array_pattern,
                        body_def: Rc::new(TaggedValue::from(handl_language::ast_value(&sub))),
                        binds: Rc::new(binds),
                        handlers: Rc::from(Vec::new()),
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
    /// `hnl-engine-exec::native_ops` (ADR-0044 §7): this engine supplies
    /// only the globals — the scope stack, AST-node execution, and arrow
    /// application — through [`VmCtx`]. The residual `match` below owns
    /// ops that still require this engine's machinery (imports, module
    /// calls, `match`/`handle`, closures, FFI) and are migrated to the
    /// shared table as their services are declared.
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
            // `closure`, `array`, `object`, and `assign` are owned by the
            // shared op table (native_ops) — EvalNode + Env gated; AST
            // children re-enter through `exec_node_inner` via
            // `VmCtx::eval_node`, and the constructor ops resolve node
            // shape data through the `VmCtx` services
            // (`pattern_role_at`/`literal_child_at`/`closure_params_at`).
            // The compile-time `Instr::Closure`/`Instr::MakeObj` fast
            // paths (compile.rs) are this engine's *interpretation* of
            // the same semantics — they stay.
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
                // Arg array position: when the callee arrives as `incoming`
                // (canonical `call` scheme), the `proj_rest` arg array is
                // `args[0]`; the legacy convention puts it at `args[1]`.
                let arg_src = if incoming.as_ref().and_then(|v| v.as_str()).is_some() {
                    args.first()
                } else {
                    args.get(1)
                };
                // ADR-0158: `perform(op, arg)` under a covering handler
                // dispatches in-flight — consult the runtime handler
                // stack before the `IO` mint; uncovered ops fall
                // through to `PerformFn`'s residual mint.
                if native_ops::is_perform_callee(callee) {
                    let arg_tvs: Vec<TaggedValue> = arg_src
                        .and_then(|v| v.as_array().map(|a| a.to_vec()))
                        .unwrap_or_default();
                    // Multi-shot first: a journal-bound ≥2-ary clause
                    // replays a journaled result or suspends
                    // (`LangError::Suspended`) to the `handle` driver.
                    if let Some(v) = self.journal_perform_or_suspend(&arg_tvs, node)? {
                        return Ok(v);
                    }
                    let mut ctx = VmCtx { frame: self, node };
                    if let Some(v) = native_ops::perform_dispatch(&mut ctx, &arg_tvs)
                        .map_err(|e| e.with_node(node))?
                    {
                        return Ok(v);
                    }
                }
                let modules = self.modules.ok_or_else(|| {
                    LangError::engine_node("call_host: no module registry".to_string(), node)
                })?;
                let func = modules.lookup(callee).ok_or_else(|| {
                    LangError::engine_node(
                        format!("call_host: function '{}' not found", callee),
                        node,
                    )
                })?;
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
                let result = self.apply_app(body_proc, &call_args, node)?;
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
                self.apply_app(step_proc, &call_args, node)
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
            // `bind`, `eval_block`, `eval_rec`, `match`, and `handle`
            // are owned by the shared op table (native_ops) — EvalNode +
            // Env gated (+ `Pattern` for the two orchestration shells).
            // ADR-0053: the data-driven `if → match` desugar rewrites `if`
            // nodes into `match` nodes before eval; the shared
            // `match_shell` handles them here.
            // Phase 9.5: `import "path"` is owned by the shared op table
            // (native_ops) — it resolves the module through
            // `VmCtx::import_resolver` and binds the exports into the
            // current scope via `env_define`.
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
                // When BOTH operands are arrow values (`TaggedValue::Arrow`
                // closures or `{"__tag__": "Arrow"}` composite/descriptor
                // records), `f >>> g` is sequential 1-cell composition, not
                // application — it yields the `Arrow{first, second}`
                // composite (`handl_value::cell_comp_axis(0, ..)`, the `∘₀`
                // step). Mirrors the tree-walk `flow` arm: without this,
                // `f := inc >>> dbl` tried to *apply* `dbl` to the `inc`
                // arrow value (`expected number, got Arrow`).
                if is_arrow_like(input) && is_arrow_like(callee) {
                    let composed = handl_value::cell_comp_axis(0, 1, input, callee)
                        .map_err(|e| LangError::engine_node(e, node))?;
                    return Ok(composed);
                }
                let actual_args: Vec<TaggedValue> = if matches!(input, TaggedValue::Pair(_)) {
                    flatten_pair(input)
                } else {
                    vec![input.clone()]
                };
                if let Some(name) = callee.as_str() {
                    // ADR-0158: `perform` under a covering handler
                    // dispatches in-flight before the `IO` mint.
                    if native_ops::is_perform_callee(name) {
                        // Multi-shot first: journal-bound ≥2-ary clauses
                        // replay or suspend to the `handle` driver.
                        if let Some(v) = self.journal_perform_or_suspend(&actual_args, node)? {
                            return Ok(v);
                        }
                        let mut ctx = VmCtx { frame: self, node };
                        if let Some(v) = native_ops::perform_dispatch(&mut ctx, &actual_args)
                            .map_err(|e| e.with_node(node))?
                        {
                            return Ok(v);
                        }
                    }
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
                    self.apply_app(callee, &actual_args, node)
                }
            }
            // `export` and `record_proj` are owned by the shared table.
            _ => Err(LangError::engine_node(
                format!("unknown v2 native op: {}", op),
                node,
            )),
        }
    }

    /// Dispatch a payload-resolved `arr <op>` against a concrete operand.
    /// Mirrors the tree-walk `arr_dispatch`: a handler implementation
    /// resolved through the runtime handler stack (innermost-first) or
    /// carried by the operand's `impls` (or a same-named field) wins;
    /// otherwise the operand's `Pair` spine is flattened into positional
    /// args for the native op. `handle_dyn` pushes handlers for the
    /// body's dynamic extent; `handle` terms whose target is a *named*
    /// capability still eliminate statically via
    /// `Term::eliminate_handlers`.
    fn arr_dispatch(
        &mut self,
        op: &str,
        incoming: &TaggedValue,
        node: &AstNode,
    ) -> Result<TaggedValue, LangError> {
        // Multi-shot resumption (ADR-0158 workstream): an `arr <op>`
        // dispatch covered by a journal-bound ≥2-ary clause replays a
        // journaled result or suspends — same interception as
        // `perform` (the tree-walk `dispatch_op` arm).
        if let Some(v) = self.journal_perform_or_suspend_op(op, incoming.clone(), node)? {
            return Ok(v);
        }
        let impl_val = self
            .handler_stack
            .iter()
            .rev()
            .find_map(|h| handler_impl(h, op))
            .or_else(|| handler_impl(incoming, op));
        if let Some(impl_val) = impl_val {
            if matches!(impl_val, TaggedValue::Arrow { .. }) {
                return self.apply_app(&impl_val, std::slice::from_ref(incoming), node);
            }
            // Serialized `Term` impl: compile it against this node and run
            // it with the operand as incoming — mirrors the tree-walk's
            // `eval_inner(&impl_term, ..., incoming)` fallback.
            let impl_term: Term =
                serde_json::from_value(tagged_to_value(&impl_val)).map_err(|e| {
                    LangError::engine_node(
                        format!("handler impl for op '{op}' is not a Term or arrow: {e}"),
                        node,
                    )
                })?;
            let instrs = compile(&impl_term, self.lang, self.cur_ast(), node)?;
            return self.exec_instrs_inner(&instrs, node, Some(incoming.clone()));
        }
        let op_args = flatten_pair(incoming);
        self.apply_native_v2(op, &op_args, node, Some(incoming.clone()))
    }

    // ------------------------------------------------------------------
    // Multi-shot resumption (ADR-0158 workstream —
    // `host/project/design/multishot-resume.md`): replay-journal driver.
    // ------------------------------------------------------------------

    /// Next journal/suspension id — bytecode ids start at
    /// [`RESUME_ID_BASE`] so they never alias the tree-walk's small
    /// sequential ids (a cross-engine `resume` reports "unknown
    /// resumption" rather than resuming the wrong suspension).
    fn take_resume_id(&mut self) -> u64 {
        let id = self.resume_next_id;
        self.resume_next_id += 1;
        id
    }

    /// Run `journal`'s `handle` body — the driver behind the multi-shot
    /// [`NativeOpCtx::run_delimited`] branch and `resume` application.
    ///
    /// The journal + a `__journal__`-stamped handler frame are installed
    /// for the body's dynamic extent (in place of the `handle_dyn`
    /// shell's frame — the `run_delimited` override pops the shell's
    /// first), then dropped before the `return`/op clauses apply: the
    /// clauses run outside their own coverage (a `perform` inside a
    /// clause dispatches to *outer* handlers, matching the algebra
    /// reading).
    ///
    /// - Body completes `v` → `h["return"](v)` when a `return` clause
    ///   exists, else `v`.
    /// - Body suspends (`LangError::Suspended`) on this journal →
    ///   invoke the op's clause `(payload, resume)`; the clause result
    ///   is the driver's result — `resume`'s own calls fold the
    ///   `return` clause in through their own runs.
    /// - Suspensions owned by other journals, and ordinary errors,
    ///   re-propagate.
    fn drive_journal(
        &mut self,
        journal: &Rc<ResumeJournal>,
        node: &AstNode,
    ) -> Result<TaggedValue, LangError> {
        self.journals.push(Rc::clone(journal));
        let stamped = native_ops::journal_frame_value(&journal.handler, journal.id);
        self.handler_stack.push(stamped);
        // Each run evals a *clone* of the handle-entry env — scope
        // pushes and `assign` writes stay inside the run — under the
        // arena the body node belongs to.
        let saved_scopes = std::mem::replace(&mut self.scopes, journal.entry_env.clone());
        let saved_def = std::mem::replace(&mut self.active_def, journal.active_def);
        let run_result = self.exec_node_inner(journal.target);
        self.active_def = saved_def;
        self.scopes = saved_scopes;
        self.handler_stack.pop();
        self.journals.pop();
        match run_result {
            Ok(val) => {
                // `return` clause — the algebra's unit map on the pure
                // result (ADR-0158). Applied at every driver completion
                // so a `resume` return includes it.
                if let Some(ret) = handler_impl(&journal.handler, "return") {
                    return self.apply_app(&ret, &[val], node);
                }
                Ok(val)
            }
            Err(e) => {
                if let Some(id) = e.suspension_id()
                    && let Some(susp) = self.suspensions.get(&id).cloned()
                    && susp.journal.id == journal.id
                {
                    let impl_val = handler_impl(&journal.handler, &susp.op).ok_or_else(|| {
                        LangError::engine_node(
                            format!("resume: no impl for suspended op '{}'", susp.op),
                            node,
                        )
                    })?;
                    // The clause gets `(payload, resume)`; its
                    // result is the `handle` result
                    // (deep-handler reading).
                    let k = native_ops::mint_resume_value(id);
                    return self.apply_app(&impl_val, &[susp.arg.clone(), k], node);
                }
                Err(e)
            }
        }
    }

    /// `perform` interception (multi-shot): `args` are the `PerformFn`
    /// argument vector `[op, payload?]`. When the op's covering handler
    /// frame is journal-bound and its clause is ≥2-ary, either replay
    /// the journaled result at the cursor (`Ok(Some(v))`) or suspend
    /// (`Err(LangError::Suspended)`). `Ok(None)` → fall through to
    /// `native_ops::perform_dispatch` (one-shot) or the residual mint.
    fn journal_perform_or_suspend(
        &mut self,
        args: &[TaggedValue],
        node: &AstNode,
    ) -> Result<Option<TaggedValue>, LangError> {
        // Op-name operand spellings mirror `PerformFn`: string, sym, or
        // the `{"sym": <sym>}` object encoding.
        let op: Rc<str> = match args.first() {
            Some(TaggedValue::Str(s)) => Rc::clone(s) as Rc<str>,
            Some(TaggedValue::Sym(s)) => Rc::from(s.name()),
            Some(TaggedValue::Obj(f)) => match f.get("sym") {
                Some(TaggedValue::Sym(s)) => Rc::from(s.name()),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        let arg = args.get(1).cloned().unwrap_or(TaggedValue::Null);
        self.journal_perform_or_suspend_op(&op, arg, node)
    }

    /// The op-level half of [`journal_perform_or_suspend`], shared with
    /// [`arr_dispatch`]: resolve the innermost covering handler frame,
    /// and when its clause is a journal-bound ≥2-ary impl, replay or
    /// suspend.
    fn journal_perform_or_suspend_op(
        &mut self,
        op: &str,
        arg: TaggedValue,
        node: &AstNode,
    ) -> Result<Option<TaggedValue>, LangError> {
        // The innermost covering frame — the same frame
        // `handler_lookup` / `perform_dispatch` would resolve.
        let covering = self
            .handler_stack
            .iter()
            .rev()
            .find(|h| handler_impl(h, op).is_some())
            .cloned();
        let Some(frame) = covering else {
            return Ok(None);
        };
        // `acc` frames use 2-ary impls for the `(payload, s)`
        // convention — not resumptions. Non-arrow and 1-ary impls take
        // the one-shot path.
        if frame.get(native_ops::HANDLER_ACC_KEY).is_some() {
            return Ok(None);
        }
        let Some(impl_val) = handler_impl(&frame, op) else {
            return Ok(None);
        };
        if !native_ops::impl_takes_resumption(&impl_val) {
            return Ok(None);
        }
        // Journal-bound? A frame installed by a journal driver carries
        // `__journal__`. Carried/foreign ≥2-ary impls fall through to
        // the ordinary apply (arity error, as before).
        let Some(TaggedValue::Int(jid)) = frame.get(native_ops::JOURNAL_FRAME_KEY) else {
            return Ok(None);
        };
        let jid = *jid as u64;
        let Some(journal) = self.journals.iter().rev().find(|j| j.id == jid).cloned() else {
            return Ok(None);
        };
        // Replay-or-suspend at this journal's cursor.
        let i = journal.cursor.get();
        journal.cursor.set(i + 1);
        if let Some((logged_op, val)) = journal.entries.borrow().get(i).cloned() {
            if logged_op.as_ref() != op {
                return Err(LangError::engine_node(
                    format!(
                        "resume: replay op mismatch at index {i} — journaled '{logged_op}', \
                         re-ran '{op}' (nondeterministic perform sequence: replay requires \
                         deterministic bodies)"
                    ),
                    node,
                ));
            }
            return Ok(Some(val));
        }
        let id = self.take_resume_id();
        self.suspensions.insert(
            id,
            Rc::new(ResumeSuspension {
                journal,
                index: i,
                op: Rc::from(op),
                arg,
            }),
        );
        Err(LangError::suspended(id))
    }

    /// `resume(x)` — re-run the suspended `handle` body under a fresh
    /// journal seeded with `entries[..index] ++ [(op, x)]`. The body's
    /// completed value (through the `return` clause) is `resume`'s
    /// result; a suspension inside the re-run invokes the covering
    /// clause again and its result is this call's result — copyable,
    /// multi-shot delimited continuation via replay.
    fn resume_suspension(
        &mut self,
        id: u64,
        x: TaggedValue,
        node: &AstNode,
    ) -> Result<TaggedValue, LangError> {
        let susp = self.suspensions.get(&id).cloned().ok_or_else(|| {
            LangError::engine_node(
                format!("resume: unknown resumption {id} — suspension not in the table"),
                node,
            )
        })?;
        let j = &susp.journal;
        let prefix_len = susp.index.min(j.entries.borrow().len());
        let mut entries: Vec<(Rc<str>, TaggedValue)> = j.entries.borrow()[..prefix_len].to_vec();
        entries.push((Rc::clone(&susp.op), x));
        let j2 = Rc::new(ResumeJournal {
            id: self.take_resume_id(),
            target: j.target,
            active_def: j.active_def,
            entry_env: j.entry_env.clone(),
            handler: j.handler.clone(),
            entries: RefCell::new(entries),
            cursor: Cell::new(0),
        });
        self.drive_journal(&j2, node)
    }

    /// Apply an `App` callee: closure application only (ADR-0060).
    ///
    /// Host-function dispatch (string callee) has been moved to the
    /// `call_host` native op. This helper handles first-class `arr <op>`
    /// dispatch objects (`{__arr__: {op}}`) and arrow-value application:
    /// install the arrow's `binds` as `captured_root`, bind params,
    /// evaluate the `body_def` arena.
    fn apply_app(
        &mut self,
        callee: &TaggedValue,
        args: &[TaggedValue],
        node: &AstNode,
    ) -> Result<TaggedValue, LangError> {
        // Growable-stack seam (see `STACK_RED_ZONE`): the apply↔exec
        // recursion passes through here once per arrow application.
        stacker::maybe_grow(STACK_RED_ZONE, STACK_SEGMENT, || {
            self.apply_app_inner(callee, args, node)
        })
    }

    /// The recursive body of [`apply_app`], run under
    /// [`stacker::maybe_grow`].
    fn apply_app_inner(
        &mut self,
        callee: &TaggedValue,
        args: &[TaggedValue],
        node: &AstNode,
    ) -> Result<TaggedValue, LangError> {
        // Multi-shot resumption (ADR-0158 workstream): a `resume` value
        // (`{"__resume__": <id>}`) drives a replay re-run of the
        // suspended `handle` body. Intercepted before the
        // dispatch-object/`__term__`/Arrow arms — the reserved-key
        // convention is the same (`__term__`, `__arr__`).
        if let TaggedValue::Obj(map) = callee
            && let Some(id_val) = map.get(native_ops::RESUME_KEY)
        {
            let id = resume_id_of(id_val).ok_or_else(|| {
                LangError::engine_node(
                    format!("resume: malformed resumption value {id_val:?}"),
                    node,
                )
            })?;
            let x = args.first().cloned().unwrap_or(TaggedValue::Null);
            return self.resume_suspension(id, x, node);
        }
        // First-class `arr <op>` dispatch object: apply the op to the
        // args (the tree-walk `apply_app` `__arr__` arm).
        if let Some(inner) = callee.get(ARR_DISPATCH_TAG) {
            let op = inner.get("op").and_then(|v| v.as_str()).ok_or_else(|| {
                LangError::engine_node("arr: missing op in dispatch object".to_string(), node)
            })?;
            let incoming = args.first().cloned().unwrap_or(TaggedValue::Null);
            return self.arr_dispatch(op, &incoming, node);
        }
        if let TaggedValue::Obj(map) = callee {
            // `Arrow{first, second}` composite record (`>>>` composition,
            // `handl_value::cell_comp_axis(0, ..)`): applying it runs
            // `first` on the args, then `second` on the result —
            // `x ~> second(first(x))`. A `second` that is a bare string
            // is a host-function name and dispatches through the module
            // registry (the tree-walk `apply_app` `Arrow` arm).
            if map.get("__tag__").and_then(|t| t.as_str()) == Some("Arrow")
                && let (Some(first), Some(second)) = (map.get("first"), map.get("second"))
            {
                let mid = self.apply_app(first, args, node)?;
                if let Some(name) = second.as_str() {
                    // ADR-0158: `perform` under a covering handler
                    // dispatches in-flight before the `IO` mint.
                    if native_ops::is_perform_callee(name) {
                        // Multi-shot first: journal-bound ≥2-ary
                        // clauses replay or suspend to the driver.
                        if let Some(v) =
                            self.journal_perform_or_suspend(std::slice::from_ref(&mid), node)?
                        {
                            return Ok(v);
                        }
                        let mut ctx = VmCtx { frame: self, node };
                        if let Some(v) =
                            native_ops::perform_dispatch(&mut ctx, std::slice::from_ref(&mid))
                                .map_err(|e| e.with_node(node))?
                        {
                            return Ok(v);
                        }
                    }
                    let modules = self.modules.ok_or_else(|| {
                        LangError::engine_node("call_host: no module registry".to_string(), node)
                    })?;
                    let func = modules.lookup(name).ok_or_else(|| {
                        LangError::engine_node(
                            format!("call_host: function '{}' not found", name),
                            node,
                        )
                    })?;
                    let call_args: Vec<Value> = [mid].iter().map(tagged_to_value).collect();
                    let mut pctx = handl_platform::EvalContext::new();
                    let val = func
                        .call(&call_args, &mut pctx)
                        .map_err(|e| LangError::engine_node(e.to_string(), node))?;
                    return Ok(TaggedValue::from(val));
                }
                return self.apply_app(second, &[mid], node);
            }
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
                return self.apply_app(&term_val, &[tuple], node);
            }
        }
        // Closure application — match on TaggedValue::Arrow directly. The
        // arrow is self-contained data (ADR-0106): `body_def` is the
        // substituted standalone arena, `binds` the explicit residual
        // bindings, `handlers` the carried deep handlers.
        let (params, rest_param, array_pattern, body_def, binds) = match callee {
            TaggedValue::Arrow {
                params,
                rest,
                array_pattern,
                body_def,
                binds,
                ..
            } => (
                params.clone(),
                rest.clone(),
                *array_pattern,
                Rc::clone(body_def),
                Rc::clone(binds),
            ),
            _ => {
                return Err(LangError::engine(
                    "callee is not an arrow-value".to_string(),
                ));
            }
        };
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
            return self.apply_arrow_body(
                &params,
                rest_param.as_ref(),
                &body_def,
                &binds,
                head,
                tail,
                node,
                callee,
            );
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
        if actual_args.len() < params.len() {
            return Err(LangError::engine(format!(
                "arrow-value expects {} args, got {}",
                params.len(),
                actual_args.len()
            )));
        }
        // Rest params (`[a, ...rest] ~> body`): surplus args bind to the
        // rest name as an array and the call consumes all args at once.
        if let Some(rest_name) = &rest_param {
            let (head, tail) = actual_args.split_at(params.len());
            return self.apply_arrow_body(
                &params,
                Some(rest_name),
                &body_def,
                &binds,
                head,
                tail,
                node,
                callee,
            );
        }
        // Curried application (ADR-0074): apply the first `params.len()`
        // args, evaluate the body, then apply the result to the surplus —
        // `f(1, 5)` on a curried arrow is `(f 1) 5`.
        let (head, mut rest) = actual_args.split_at(params.len());
        let mut val =
            self.apply_arrow_body(&params, None, &body_def, &binds, head, &[], node, callee)?;
        while !rest.is_empty() {
            val = self.apply_app(&val, rest, node)?;
            rest = &[];
        }
        Ok(val)
    }

    /// Apply an arrow body (ADR-0106): execute the arrow's `body_def`
    /// arena under a scope whose `captured_root` is the arrow's residual
    /// `binds` map, with params (and the rest binding) bound in a fresh
    /// scope on top. The caller's bindings are never visible — the lexical
    /// `floor` is set to the pre-call bindings length.
    ///
    /// The `body_def` arena is deserialized once per arrow and memoized in
    /// `def_arenas`; its node indices are compiled into `def_cache` keyed
    /// by `(arena_slot, index)` so they never collide with `self.ast`.
    /// `active_def` selects the arena for the duration of the body —
    /// `cur_ast()` resolves every child/payload read through it.
    #[allow(clippy::too_many_arguments)]
    fn apply_arrow_body(
        &mut self,
        params: &[String],
        rest_param: Option<&Rc<str>>,
        body_def: &Rc<TaggedValue>,
        binds: &Rc<FxHashMap<Rc<str>, TaggedValue>>,
        head: &[TaggedValue],
        tail: &[TaggedValue],
        node: &AstNode,
        self_val: &TaggedValue,
    ) -> Result<TaggedValue, LangError> {
        let slot = self
            .def_arena_slot(body_def)
            .map_err(|e| e.with_node(node))?;
        let body_ref = AstRef::new(self.def_arenas[slot].len() - 1);

        // Arrow-value invocation (ADR-0052): install the residual `binds`
        // as the `captured_root` (a cheap `Rc` clone — no deep HashMap
        // clone), push a fresh scope for params, and evaluate the body.
        // The caller's env is never mutated — we save the pre-call
        // binding/marks lengths, `captured_root`, `active_def`, and the
        // lexical floor, then restore after the body returns. This avoids
        // allocating a fresh `ScopeStack` (two `Vec` allocations) on every
        // closure call — critical for `loop_step` where `apply_app` runs
        // once per iteration.
        let saved_captured = self.scopes.captured_root.take();
        let saved_marks_len = self.scopes.marks.len();
        let saved_bindings_len = self.scopes.bindings.len();
        let saved_floor = self.scopes.floor;
        let saved_def = self.active_def.replace(slot);
        self.scopes.floor = saved_bindings_len;
        self.scopes.captured_root = Some(Rc::clone(binds));
        self.scopes.push_scope();
        // Self-reference markers (mirrors the tree-walk apply): a `binds`
        // entry holding `self_ref_marker()` is rebound to the arrow under
        // application so a self-recursive name resolves to the arrow
        // itself — the marker is serializable data, no cyclic env.
        for (name, v) in binds.iter() {
            if handl_value::is_self_ref_marker(v) {
                self.scopes.define(name, self_val.clone());
            }
        }
        for (param, arg) in params.iter().zip(head.iter()) {
            self.scopes.define(param, arg.clone());
        }
        // Bind the rest param to the surplus args as an array.
        if let Some(rest_name) = rest_param {
            self.scopes.define(
                rest_name.as_ref(),
                TaggedValue::Arr(Rc::from(tail.to_vec())),
            );
        }
        // Deep-handler reinstall (the tree-walk `apply` parity): a
        // `handle f with h` result carries `h` in the arrow's `handlers`
        // — reinstall it for the body's extent so ops inside `f`
        // dispatch through their captured handler.
        let handlers_mark = self.handler_stack.len();
        if let TaggedValue::Arrow { handlers, .. } = self_val {
            self.handler_stack.extend(handlers.iter().cloned());
        }
        let result = self.exec_node_inner(body_ref);
        self.handler_stack.truncate(handlers_mark);
        // Restore: truncate bindings and marks to pre-call state, discarding
        // any scopes the body may have pushed, and restore the captured
        // root, lexical floor, and active arena.
        self.scopes.bindings.truncate(saved_bindings_len);
        self.scopes.marks.truncate(saved_marks_len);
        self.scopes.captured_root = saved_captured;
        self.scopes.floor = saved_floor;
        self.active_def = saved_def;
        result
    }
}

/// Whether a native op uses the `incoming` value (as opposed to only args).
///
/// Ops that use `incoming` with 0 args pop the incoming value from the stack
/// (it's the value flowing through the arrow). Ops that don't use `incoming`
/// (like `closure`) don't pop anything for a 0-arg call.
pub(crate) fn op_uses_incoming(op: &str) -> bool {
    matches!(
        op,
        "id" | "tag"
            | "pair_tag"
            | "truthy"
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
/// `incoming`, before dispatching to `VmFrame::apply_native_v2`.
///
/// `loop_step` (the ergonomic `loop` surface lowering) and `fold_step`
/// (the ADR-0120 `loop` fold node) are handled here — their step arg is a
/// closure evaluated once, then applied per iteration via
/// `VmFrame::apply_app`. The remaining derived step ops
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
/// `hnl-engine-exec::native_ops`; this struct only forwards services.
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

    fn node_payload_at(&mut self, r: AstRef) -> Option<Value> {
        self.frame.cur_ast().at(r).ok().map(|n| n.payload.clone())
    }

    fn child_role_index(&mut self, role: &str) -> Option<usize> {
        self.frame
            .lang
            .child_role_index(&self.node.kind, role)
            .ok()
            .flatten()
    }

    fn child_role_index_for(&mut self, role: &str) -> Option<usize> {
        self.frame
            .lang
            .child_role_index_for(self.node, role)
            .ok()
            .flatten()
    }

    fn role_declared_at(&mut self, r: AstRef, role: &str) -> bool {
        self.frame
            .cur_ast()
            .at(r)
            .ok()
            .and_then(|n| {
                self.frame
                    .lang
                    .child_role_index(&n.kind, role)
                    .ok()
                    .flatten()
            })
            .is_some()
    }

    fn child_role_index_at(&mut self, r: AstRef, role: &str) -> Option<usize> {
        self.frame
            .cur_ast()
            .at(r)
            .ok()
            .and_then(|n| self.frame.lang.child_role_index_for(n, role).ok().flatten())
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

    fn env_assign(&mut self, name: &str, value: TaggedValue) {
        self.frame.scopes.assign(name, value);
    }

    fn env_take_enclosing_updated(&mut self) -> bool {
        std::mem::take(&mut self.frame.scopes.enclosing_updated)
    }

    fn env_capture(&mut self) -> Rc<FxHashMap<Rc<str>, TaggedValue>> {
        Rc::new(self.frame.scopes.capture())
    }

    fn try_pattern(&mut self, pattern: AstRef, scrutinee: &TaggedValue) -> Result<bool, LangError> {
        // `VmFrame` itself is the `PatternMatchCtx` (ADR-0061);
        // `try_pattern` works on `TaggedValue` directly and binds into
        // `self.frame.scopes` via `define_binding`. The arena handle is
        // copied/cloned out first so the `&mut self.frame` argument does
        // not conflict with the arena borrow.
        let arena_handle;
        let ast: &Ast = match self.frame.active_def {
            Some(slot) => {
                arena_handle = Rc::clone(&self.frame.def_arenas[slot]);
                &arena_handle
            }
            None => self.frame.ast,
        };
        pattern_match::try_pattern(ast, pattern, scrutinee, self.frame)
    }

    fn node_kind_at(&mut self, r: AstRef) -> Option<String> {
        self.frame.cur_ast().get(r).map(|n| n.kind.clone())
    }

    fn pattern_role_at(&mut self, r: AstRef, role: &str) -> Option<bool> {
        self.frame
            .cur_ast()
            .get(r)
            .and_then(|n| self.frame.lang.pattern_role_is(&n.kind, role))
    }

    fn literal_child_at(&mut self, parent: AstRef, role: &str, child: AstRef) -> Option<bool> {
        let parent_kind = self.frame.cur_ast().get(parent)?.kind.clone();
        let child_kind = self.frame.cur_ast().get(child)?.kind.clone();
        self.frame
            .lang
            .literal_child(&parent_kind, role, &child_kind)
    }

    fn closure_params_at(
        &mut self,
        r: AstRef,
    ) -> Result<(Vec<String>, Option<String>, bool), LangError> {
        let node = self.frame.cur_ast().at(r)?;
        let (names, rest) = self.frame.lang.param_names(self.frame.cur_ast(), node)?;
        Ok((names, rest, self.frame.lang.is_array_pattern(node)))
    }

    fn closure_body_def(
        &mut self,
        body: AstRef,
        env: &std::collections::HashMap<Rc<str>, Value>,
    ) -> Result<TaggedValue, LangError> {
        // ADR-0106: extract the body into a standalone arena with `env`'s
        // free variables substituted as `lit` nodes, then wrap as
        // `{"__ast__": <Ast>}` — the arrow's self-describing definition.
        let sub = self
            .frame
            .cur_ast()
            .instantiate(body, self.frame.lang, env)?;
        Ok(TaggedValue::from(handl_language::ast_value(&sub)))
    }

    fn rec_bound_names(&self) -> std::collections::HashSet<Rc<str>> {
        self.frame.scopes.rec_names.iter().cloned().collect()
    }

    fn rec_region_enter(&mut self, names: &[Rc<str>]) {
        let mark = self.frame.scopes.rec_names.len();
        self.frame.scopes.rec_marks.push(mark);
        self.frame.scopes.rec_names.extend(names.iter().cloned());
    }

    fn rec_region_exit(&mut self) {
        if let Some(mark) = self.frame.scopes.rec_marks.pop() {
            self.frame.scopes.rec_names.truncate(mark);
        }
    }

    fn apply_arrow(
        &mut self,
        f: &TaggedValue,
        args: &[TaggedValue],
    ) -> Result<TaggedValue, LangError> {
        self.frame.apply_app(f, args, self.node)
    }

    fn import_resolver(&self) -> Option<&dyn hnl_engine_exec::ImportResolver> {
        self.frame.import_resolver.map(|rc| &**rc)
    }

    fn handler_push(&mut self, handler: TaggedValue) {
        self.frame.handler_stack.push(handler);
    }

    fn handler_pop(&mut self) -> Option<TaggedValue> {
        self.frame.handler_stack.pop()
    }

    fn handler_lookup(&self, op: &str) -> Option<TaggedValue> {
        self.frame
            .handler_stack
            .iter()
            .rev()
            .find_map(|h| handler_impl(h, op))
    }

    fn handler_acc(&self, op: &str) -> Option<TaggedValue> {
        native_ops::covering_frame_acc(&self.frame.handler_stack, op)
    }

    fn handler_set_acc(&mut self, op: &str, s: TaggedValue) {
        native_ops::covering_frame_set_acc(&mut self.frame.handler_stack, op, s);
    }

    fn run_delimited(
        &mut self,
        target: AstRef,
        handler: &TaggedValue,
        frame_handler: &TaggedValue,
    ) -> Result<TaggedValue, LangError> {
        // Multi-shot resumption (ADR-0158 workstream —
        // `design/multishot-resume.md`): a ≥2-ary clause
        // `(payload, resume) ~> …` runs the body under a replay journal —
        // covered `perform`s/`arr` dispatches suspend to the driver and
        // `resume` re-runs `target` from this boundary. `acc` frames are
        // excluded: their 2-ary impls are the `(payload, s)` convention,
        // not resumptions. 1-ary-only handlers take the shared one-shot
        // tail.
        let multishot = frame_handler.get(native_ops::HANDLER_ACC_KEY).is_none()
            && native_ops::handler_has_resumption(handler);
        if !multishot {
            return native_ops::run_delimited_default(self, target, handler, frame_handler);
        }
        let journal = Rc::new(ResumeJournal {
            id: self.frame.take_resume_id(),
            target,
            active_def: self.frame.active_def,
            entry_env: self.frame.scopes.clone(),
            handler: handler.clone(),
            entries: RefCell::new(Vec::new()),
            cursor: Cell::new(0),
        });
        // The shell already pushed `frame_handler`; the journal driver
        // installs its own `__journal__`-stamped frame for the extent.
        // This pop *is* the shell-frame pop the `run_delimited`
        // contract requires — the driver pushes/pops its own frame.
        self.frame.handler_stack.pop();
        let val = self.frame.drive_journal(&journal, self.node)?;
        // Residual-`IO` interpretation under the handler + deep-handler
        // carry — the tree-walk multi-shot `Handle` arm's postlude.
        self.frame.handler_stack.push(frame_handler.clone());
        let interpreted = native_ops::interpret_io_former(self, val);
        self.frame.handler_stack.pop();
        let val = interpreted?;
        Ok(native_ops::attach_carried_handler(val, frame_handler))
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
    use handl_plugin_module::ModuleRegistry;
    use hnl_engine::TermData;
    use hnl_engine_exec::ImportResolver;
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
        let s = TermData::op("const", vec![], &ArrSource::Native, Some(json!(42)))
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
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(5))),
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
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(5))),
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
                Some(pair(json!(3), json!(7))),
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
                Some(pair(json!(3), json!(7))),
            ),
            TermData::first(TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(json!(99)),
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
                Some(either_left(json!(5))),
            ),
            TermData::choice(
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(1))),
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2))),
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
                Some(either_right(json!(5))),
            ),
            TermData::choice(
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(1))),
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2))),
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
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(true))),
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
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(false))),
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
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(10))),
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
                Some(pair(json!(42), json!(5))),
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
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(1))),
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
                    Some(Value::String("$payload".into())),
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
                                Some(json!(0u64)),
                            ),
                            TermData::op(
                                "proj_child",
                                vec![],
                                &ArrSource::Native,
                                Some(json!(1u64)),
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
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(0u64))),
                        TermData::op("tag", vec![], &ArrSource::Native, None),
                    ),
                    TermData::choice(
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(1u64))),
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(2u64))),
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
                TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(0u64)))
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
                    TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(0u64))),
                    TermData::loop_feedback(TermData::op(
                        "loop_step",
                        vec![TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(1u64)),
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
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(5))),
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
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(0))),
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
                Some(pair(json!(5), json!(0))),
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
                Some(pair(json!(0), json!(99))),
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
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(100))),
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
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(0))),
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
        // ADR-0106: the arrow carries its body as a serialized standalone
        // arena (`body_def`); the root is the arena's last node.
        let closure = json!({
            "__arrow__": true,
            "params": ["x"],
            "body_def": {
                "__ast__": {
                    "nodes": [
                        { "kind": "var", "span": null, "payload": "x" }
                    ]
                }
            },
        });
        let s = TermData::app(
            TermData::op("const", vec![], &ArrSource::Native, Some(closure)),
            vec![TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(json!(41)),
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
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2))),
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(3))),
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
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2))),
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(2))),
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
                Some(json!(21)),
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
                Some(json!(21)),
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
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(0u64))),
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(1u64))),
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
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(0u64))),
                        TermData::op("tag", vec![], &ArrSource::Native, None),
                    ),
                    TermData::choice(
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(1u64))),
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(2u64))),
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
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(0u64))),
                        TermData::op("tag", vec![], &ArrSource::Native, None),
                    ),
                    TermData::choice(
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(1u64))),
                        TermData::op("proj_child", vec![], &ArrSource::Native, Some(json!(2u64))),
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
                    Some(Value::from("$payload")),
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
                        Some(Value::from(0u64)),
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
                        Some(Value::from("m.hnl")),
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
                    Some(Value::from("$payload")),
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
                        Some(Value::from(0u64)),
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

    // -----------------------------------------------------------------
    // Multi-shot `resume` (ADR-0158 workstream — replay journals)
    // -----------------------------------------------------------------

    /// `proj_env name` — variable lookup (the v2 `var` spelling).
    fn rvar(name: &str) -> Term {
        TermData::op("proj_env", vec![], &ArrSource::Native, Some(name.into()))
            .to_term()
            .expect("TermData -> Term")
    }

    /// `proj_child n` — child-node projection.
    fn rchild(n: u64) -> Term {
        TermData::op(
            "proj_child",
            vec![],
            &ArrSource::Native,
            Some(Value::from(n)),
        )
        .to_term()
        .expect("TermData -> Term")
    }

    /// `arr_native op args` — a native op with no literal operand.
    fn rprim(op: &str, args: Vec<Term>) -> Term {
        Term::arr_native(op, args)
    }

    /// Minimal language for resumption tests (mirrors the cranelift and
    /// tree-walk `resume_lang`): `var`, payload fallback, `binary.add`,
    /// `array`, `idx` (`prim index` — element projection), `call`
    /// (canonical callee routing — string callees go to `call_host`,
    /// values to `App`), `function` (closure op), `object`/`entry`
    /// (handler records), and the dynamic-extent `handle`.
    fn resume_lang() -> Language {
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new("e.var", "var", rvar("$payload")))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.add",
                "binary.add",
                rprim("add", vec![rchild(0), rchild(1)]),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.array",
                "array",
                rprim("array", vec![]),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.index",
                "idx",
                rprim("index", vec![rchild(0), rchild(1)]),
            ))
            .unwrap();
        // eval.call's canonical shape: route the evaluated callee by
        // type — string → `call_host` (host fn, incl. `perform`), value
        // → `App` (arrows, `resume` objects).
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.call",
                "call",
                Term::seq(
                    Term::seq(
                        rchild(0),
                        Term::arr_native_lit("route_on_type", vec![], "string"),
                    ),
                    Term::choice(
                        rprim("call_host", vec![rprim("proj_rest", vec![])]),
                        Term::app(rprim("id", vec![]), vec![rprim("proj_rest", vec![])]),
                    ),
                ),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.function",
                "function",
                rprim("closure", vec![]),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.object",
                "object",
                rprim("object", vec![]),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.handle",
                "handle",
                Term::handle(rchild(0), "", rchild(1)),
            ))
            .unwrap();
        lang
    }

    /// `choose`: `handle (perform(choose, [1,2]) + 100) with
    /// {choose: (opts, resume) ~> [resume(opts[0]), resume(opts[1])]}`
    /// → `[101, 102]` — each `resume` call re-runs the body, feeding the
    /// option back at the `perform` site and continuing through `+100`.
    #[test]
    fn choose_multishot_resume_collects_both_continuations() {
        let engine = crate::BytecodeEngine::new();
        let lang = resume_lang();
        let ast = make_ast(vec![
            AstNode::new("var").with_payload(json!("perform")), // 0: callee
            AstNode::new("var").with_payload(json!("choose")),  // 1: op arg
            AstNode::new("value").with_payload(json!(1)),       // 2
            AstNode::new("value").with_payload(json!(2)),       // 3
            AstNode::new("array").with_payload(json!({"children": [2, 3]})), // 4: [1,2]
            AstNode::new("call").with_payload(json!({"children": [0, 1, 4]})), // 5: perform(choose,[1,2])
            AstNode::new("value").with_payload(json!(100)),                    // 6
            AstNode::new("binary.add").with_payload(json!({"children": [5, 6]})), // 7: BODY
            AstNode::new("var").with_payload(json!("resume")),                 // 8
            AstNode::new("var").with_payload(json!("opts")),                   // 9
            AstNode::new("value").with_payload(json!(0)),                      // 10
            AstNode::new("idx").with_payload(json!({"children": [9, 10]})),    // 11: opts[0]
            AstNode::new("call").with_payload(json!({"children": [8, 11]})), // 12: resume(opts[0])
            AstNode::new("value").with_payload(json!(1)),                    // 13
            AstNode::new("idx").with_payload(json!({"children": [9, 13]})),  // 14: opts[1]
            AstNode::new("call").with_payload(json!({"children": [8, 14]})), // 15: resume(opts[1])
            AstNode::new("array").with_payload(json!({"children": [12, 15]})), // 16: clause body
            AstNode::new("function")
                .with_payload(json!({"params": ["opts", "resume"], "body": 16})), // 17
            AstNode::new("key").with_payload(json!("choose")),               // 18: entry key
            AstNode::new("entry").with_payload(json!({"children": [18, 17]})), // 19
            AstNode::new("object").with_payload(json!({"children": [19]})),  // 20: handler
            AstNode::new("handle").with_payload(json!({"children": [7, 20]})), // 21: root
        ]);
        let req = hnl_engine_exec::EvalRequest::new(ast);
        assert_eq!(
            hnl_engine_exec::EvalEngine::eval(&engine, &lang, &req).unwrap(),
            json!([101, 102])
        );
    }

    /// Nested multi-shot: two `choose`s in one body.
    /// `handle (perform(choose,[1,2]) + perform(choose,[10,20])) with
    /// {choose: (opts, resume) ~> [resume(opts[0]), resume(opts[1])]}`
    /// → `[[11, 21], [12, 22]]` — the second perform suspends inside each
    /// first-level resume, the clause re-runs per nested suspension.
    #[test]
    fn choose_nested_suspensions_are_cartesian() {
        let engine = crate::BytecodeEngine::new();
        let lang = resume_lang();
        let ast = make_ast(vec![
            AstNode::new("var").with_payload(json!("perform")), // 0
            AstNode::new("var").with_payload(json!("choose")),  // 1
            AstNode::new("value").with_payload(json!(1)),       // 2
            AstNode::new("value").with_payload(json!(2)),       // 3
            AstNode::new("array").with_payload(json!({"children": [2, 3]})), // 4: [1,2]
            AstNode::new("call").with_payload(json!({"children": [0, 1, 4]})), // 5: perform(choose,[1,2])
            AstNode::new("value").with_payload(json!(10)),                     // 6
            AstNode::new("value").with_payload(json!(20)),                     // 7
            AstNode::new("array").with_payload(json!({"children": [6, 7]})),   // 8: [10,20]
            AstNode::new("call").with_payload(json!({"children": [0, 1, 8]})), // 9: perform(choose,[10,20])
            AstNode::new("binary.add").with_payload(json!({"children": [5, 9]})), // 10: BODY
            AstNode::new("var").with_payload(json!("resume")),                 // 11
            AstNode::new("var").with_payload(json!("opts")),                   // 12
            AstNode::new("value").with_payload(json!(0)),                      // 13
            AstNode::new("idx").with_payload(json!({"children": [12, 13]})),   // 14: opts[0]
            AstNode::new("call").with_payload(json!({"children": [11, 14]})), // 15: resume(opts[0])
            AstNode::new("value").with_payload(json!(1)),                     // 16
            AstNode::new("idx").with_payload(json!({"children": [12, 16]})),  // 17: opts[1]
            AstNode::new("call").with_payload(json!({"children": [11, 17]})), // 18: resume(opts[1])
            AstNode::new("array").with_payload(json!({"children": [15, 18]})), // 19: clause
            AstNode::new("function")
                .with_payload(json!({"params": ["opts", "resume"], "body": 19})), // 20
            AstNode::new("key").with_payload(json!("choose")),                // 21
            AstNode::new("entry").with_payload(json!({"children": [21, 20]})), // 22
            AstNode::new("object").with_payload(json!({"children": [22]})),   // 23
            AstNode::new("handle").with_payload(json!({"children": [10, 23]})), // 24: root
        ]);
        let req = hnl_engine_exec::EvalRequest::new(ast);
        assert_eq!(
            hnl_engine_exec::EvalEngine::eval(&engine, &lang, &req).unwrap(),
            json!([[11, 21], [12, 22]])
        );
    }

    /// One-shot regression: a 1-ary clause stays on the direct dispatch
    /// path — `handle perform(puts, "hi") with {puts: s ~> s}` returns
    /// the impl's result at the perform site (in-flight dispatch,
    /// ADR-0158). No journal is installed.
    #[test]
    fn perform_one_shot_clause_dispatch_unchanged() {
        let engine = crate::BytecodeEngine::new();
        let lang = resume_lang();
        let ast = make_ast(vec![
            AstNode::new("var").with_payload(json!("perform")), // 0
            AstNode::new("var").with_payload(json!("puts")),    // 1
            AstNode::new("value").with_payload(json!("hi")),    // 2
            AstNode::new("call").with_payload(json!({"children": [0, 1, 2]})), // 3: perform(puts,"hi")
            AstNode::new("var").with_payload(json!("s")),                      // 4: clause body
            AstNode::new("function").with_payload(json!({"params": ["s"], "body": 4})), // 5: 1-ary impl
            AstNode::new("key").with_payload(json!("puts")),                            // 6
            AstNode::new("entry").with_payload(json!({"children": [6, 5]})),            // 7
            AstNode::new("object").with_payload(json!({"children": [7]})),              // 8
            AstNode::new("handle").with_payload(json!({"children": [3, 8]})),           // 9: root
        ]);
        let req = hnl_engine_exec::EvalRequest::new(ast);
        assert_eq!(
            hnl_engine_exec::EvalEngine::eval(&engine, &lang, &req).unwrap(),
            json!("hi")
        );
    }
}

//! V2 bytecode instruction set for the arrow-calculus kernel (ADR-0052).
//!
//! Each [`Term`](handl_language::term::Term) combinator compiles
//! to a flat sequence of [`Instr`] values. The v2 VM executes them against an
//! operand stack and a scope stack, producing values identical to the v2
//! tree-walk reference handler.
//!
//! The ISA covers the six arrow-calculus primitives.
//!
//! ## ISA overview
//!
//! | Instruction          | Stack effect                          | Description |
//!|----------------------|---------------------------------------|-------------|
//! | [`Push`]             | `-> val`                              | Push a literal value. |
//! | [`Load`]             | `-> val`                              | Look up a named variable in the scope stack. |
//! | [`LoadPayloadVar`]   | `-> val`                              | Read `node.payload` as a string, look up that variable (`$payload`). |
//! | [`LoadPayload`]      | `-> val`                              | Push `node.payload` (no-scheme fallback). |
//! | [`EvalChild`]        | `-> val`                              | Evaluate child `n` of the current node (compile-cache + execute). |
//! | [`Store`]            | `val ->`                              | Pop value, push a new scope, bind name (`extend_env`). |
//! | [`StorePayloadName`] | `val ->`                              | Pop value, read `node.payload["name"]`, push scope, bind (`extend_env $payload`). |
//! | [`PopScope`]         | `->`                                  | Pop the top scope (after a `Seq` whose `fst` pushed a scope). |
//! | [`Call`]             | `callee arg... -> result`             | Apply a closure or host fn to `n` arguments (`App`). |
//! | [`ChoiceJump`]       | `either -> val`                       | Pop Either; push its value; jump to left or right target (`Choice`). |
//! | [`Jump`]             | `->`                                  | Unconditional jump. |
//! | [`NativeOp`]         | `arg... -> result`                    | Apply native op `op` to `n` args (`Arr Native`). |
//! | [`ForeignOp`]        | `arg... -> result`                    | Call host function `name` with `n` args (`Arr Foreign`). |
//! | [`FirstOp`]          | `pair -> {__pair__:[result, snd]}`    | Run inner sub-program on pair's first component; re-pair with second (`First`). |
//! | [`LoopCheck`]        | `result -> ...`                       | Inspect loop body result; route single-step vs multi-step (`Loop`). |
//! | [`DerivedStepOp`]   | `pair -> either`                      | Re-evaluate arg sub-programs per iteration; apply derived step op (`while_step`/`for_step`/`fix_step`). |
//! | [`Return`]           | `val ->`                              | Pop and return the top of the operand stack. |
//! | [`EvalNode`]         | `-> val`                              | Evaluate the AST node at an absolute arena index (used by `MakeObj`). |
//! | [`MakeObj`]          | `val... -> obj`                       | Pop `keys.len()` values, construct an object with pre-allocated keys. |
//!
//! ## Combinator → instruction mapping
//!
//! | `Term` combinator | Compiled instructions |
//!|---|---|
//! | `Arr { op: "const", lit: Some(v), .. }` | `Push(v)` |
//! | `Arr { op: "proj_env", lit: Some(name), .. }` | `Load(name)` or `LoadPayloadVar` if name is `"$payload"` |
//! | `Arr { op: "proj_child", lit: Some(idx), .. }` | `EvalChild(idx)` |
//! | `Arr { op: "extend_env", lit: Some(name), .. }` | `Store(name)` or `StorePayloadName` if name is `"$payload"` |
//! | `Arr { op, args, source: Native, .. }` (general) | `compile(args)...`, `NativeOp(op, n)` |
//! | `Arr { op: "object", source: Native, .. }` (static keys) | `EvalNode(val_0)...`, `MakeObj { keys }` |
//! | `Arr { op: "while_step"/"for_step"/"fix_step", args, source: Native, .. }` | `Jump(derived_pos)`, `<arg sub-programs>`, `DerivedStepOp(op, ranges)` |
//! | `Arr { op, args, source: Foreign(name), .. }` | `compile(args)...`, `ForeignOp(name, n)` |
//! | `Seq { fst, snd }` | `compile(fst)`, `compile(snd)`, [`PopScope`] if `fst` pushed a scope |
//! | `First { inner }` | `FirstOp(start, end)`, `<inner sub-program at [start, end)>` |
//! | `Choice { l, r }` | `ChoiceJump(left_target, right_target)`, `compile(l)`, `Jump(end)`, `compile(r)` |
//! | `Loop { body }` | `LoopCheck(loop_back)`, `compile(body)` (laid out as a loop body region) |
//! | `App { callee, args }` | `compile(callee)`, `compile(args)...`, `Call(n)` |
//!
//! ## Derived step ops
//!
//! All four derived step ops from ADR-0052 are supported by the v2 bytecode VM:
//!
//! - **`loop_step`** (the ergonomic `loop` surface lowering): its carried
//!   sub-scheme arg (the body closure) is compiled inline like any other
//!   `Arr.arg`; the `NativeOp` handler recognizes `loop_step` via
//!   [`op_is_derived_step`] and additionally pops the loop pair `(in, state)`
//!   left on the operand stack by the enclosing `Loop`, passing it as
//!   `incoming` to [`VmFrame::apply_native_v2`]. The body closure is then
//!   applied via [`VmFrame::apply_app`] (recursive `exec_node`).
//!
//! - **`while_step`/`for_step`/`fix_step`** (built by
//!   `plugin-lib-stdlib::arrow_iter::{while_loop,for_loop,fix_loop}`):
//!   these carry cond/body sub-schemes that must be re-evaluated **per
//!   iteration** with the current value (extracted from the loop pair) as
//!   incoming. The compiler emits a [`DerivedStepOp`] instruction that
//!   stores each arg as an instruction range `(start, end)`. A `Jump` before
//!   the sub-programs skips them during normal execution; the
//!   `DerivedStepOp` handler re-executes them via [`VmFrame::exec_instrs`]
//!   with the current value as incoming, then dispatches to
//!   [`VmFrame::apply_native_v2`] with the fresh results and the loop pair
//!   as incoming. The produced `Either` is pushed for [`LoopCheck`].

use std::rc::Rc;

use handl_language::Value;

/// An ALU operation selected by the [`Instr::Arith`] hardware-boundary
/// instruction (ADR-0103 §2.4).
///
/// These map directly to CPU arithmetic/logic operations — they are the
/// irreducible arithmetic boundary of the kernel. `Neg` and `Not` are unary
/// (pop one value); all others are binary (pop two values). Semantics mirror
/// the corresponding `NativeOp` arms in [`crate::vm`] exactly, so results are
/// identical to the `NativeOp`/typed-bytecode paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    /// Addition (`add`).
    Add,
    /// Subtraction (`sub`).
    Sub,
    /// Multiplication (`mul`).
    Mul,
    /// Division (`div`).
    Div,
    /// Modulo (`mod`).
    Mod,
    /// Equality (`eq`).
    Eq,
    /// Inequality (`neq`).
    Neq,
    /// Less-than (`lt`).
    Lt,
    /// Less-than-or-equal (`lte`/`Le`).
    Le,
    /// Greater-than (`gt`).
    Gt,
    /// Greater-than-or-equal (`gte`/`Ge`).
    Ge,
    /// Logical and (`and`).
    And,
    /// Logical or (`or`).
    Or,
    /// Logical not (`not`) — unary.
    Not,
    /// Numeric negation (`neg`) — unary.
    Neg,
}

/// A single bytecode instruction in the v2 eval VM's instruction set.
///
/// See the [module-level documentation](self) for the full ISA table,
/// combinator-to-instruction mapping, and the derived-step-op limitation.
#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    /// Push a literal [`Value`] onto the operand stack.
    ///
    /// Emitted by `Arr { op: "const", lit: Some(v), .. }`.
    Push(Value),

    /// Look up a named variable in the scope stack and push its value.
    ///
    /// Emitted by `Arr { op: "proj_env", lit: Some(name), .. }` when the
    /// name is not `"$payload"`. If the variable is unbound, the name itself
    /// is pushed as a string value (matching the v2 tree-walk convention).
    Load(Rc<str>),

    /// Read the current node's `payload` as a string, look up that variable
    /// name in the scope stack, and push the value.
    ///
    /// Emitted by `Arr { op: "proj_env", lit: Some("$payload"), .. }` (the
    /// actual name lives in the node's payload at runtime).
    LoadPayloadVar,

    /// Push the current node's `payload` as a value.
    ///
    /// Used by the no-scheme fallback (a node kind with no eval scheme
    /// returns its payload directly).
    LoadPayload,

    /// Evaluate the `n`-th child of the current node and push the result.
    ///
    /// Emitted by `Arr { op: "proj_child", lit: Some(idx), .. }`. At
    /// runtime, reads `node.payload["children"][n]` to obtain the child's
    /// absolute [`handl_language::AstRef`], then compiles (via the per-node
    /// cache) and executes that child node's v2 bytecode.
    EvalChild(usize),

    /// Like [`EvalChild`], but pushes `null` when the child index is absent
    /// (e.g. the missing `else` of an else-less `if` — a child produced by an
    /// `optional_children` grammar row).
    ///
    /// Emitted by `Arr { op: "proj_child_or", lit: Some(idx), .. }`.
    EvalChildOr(usize),

    /// Pop a value, push a new scope, and bind the given name in that scope.
    ///
    /// Emitted by `Arr { op: "extend_env", lit: Some(name), .. }` when the
    /// name is not `"$payload"`. The matching [`PopScope`] is emitted by the
    /// enclosing `Seq` after its `snd` completes.
    Store(Rc<str>),

    /// Pop a value, read `node.payload["name"]` as the bind name, push a new
    /// scope, and bind that name in the new scope.
    ///
    /// Emitted by `Arr { op: "extend_env", lit: Some("$payload"), .. }` (the
    /// actual bind name lives in the node's payload at runtime). The matching
    /// [`PopScope`] is emitted by the enclosing `Seq`.
    StorePayloadName,

    /// Pop the top scope from the scope stack.
    ///
    /// Emitted after a `Seq`'s `snd` completes, if the `fst` pushed a scope
    /// via `Store`/`StorePayloadName` (`extend_env`).
    PopScope,

    /// Pop `n` argument values, pop the callee value, apply (closure or host
    /// fn), and push the result.
    ///
    /// Emitted by `App { callee, args }`. The callee may be a closure value
    /// (`{"__arrow__": true, ...}`) or a string naming a registered host
    /// function.
    Call(usize),

    /// Fixed-arity variant of [`Call`] for 1-argument applications.
    ///
    /// Avoids the per-call `Vec` allocation in the `Call(1)` path. Emitted by
    /// the compiler when `App` has exactly 1 arg. Produces results identical
    /// to [`Call`]`(1)`.
    Call1,

    /// Fixed-arity variant of [`Call`] for 2-argument applications.
    ///
    /// Avoids the per-call `Vec` allocation in the `Call(2)` path. Emitted by
    /// the compiler when `App` has exactly 2 args. Produces results identical
    /// to [`Call`]`(2)`.
    Call2,

    /// Fixed-arity variant of [`Call`] for 3-argument applications.
    ///
    /// Avoids the per-call `Vec` allocation in the `Call(3)` path. Emitted by
    /// the compiler when `App` has exactly 3 args. Produces results identical
    /// to [`Call`]`(3)`.
    Call3,

    /// Pop an Either value, push its carried value, and jump to `left_target`
    /// if the Either is a left, or `right_target` if it is a right.
    ///
    /// Emitted by `Choice { l, r }`. The Either shape is
    /// `{"__either__": "left"|"right", "value": v}`.
    ChoiceJump(usize, usize),

    /// Unconditionally set the instruction pointer to `target`.
    ///
    /// Emitted by `Choice` to skip the right-branch after the left-branch
    /// executes.
    Jump(usize),

    /// Pop `n` argument values, apply the native op `op`, and push the
    /// result.
    ///
    /// Emitted by `Arr { op, args, source: Native, .. }` for general native
    /// ops (not the lit-bearing special cases, and not the derived step
    /// ops `while_step`/`for_step`/`fix_step` which use [`DerivedStepOp`]).
    /// The native op set matches the v2 tree-walk exactly: arithmetic
    /// (`add`/`sub`/`mul`/`div`/`mod`/`neg`), comparison (`eq`/`neq`/`lt`/
    /// `gt`/`lte`/`gte`), logic (`and`/`or`/`not`), `tag`, `id`, `closure`,
    /// `bind` (env extension), `loop_step` (ergonomic loop surface), and
    /// utility ops (`is_pos`/`dec`/`inc`/`half_floor`/`inc_fst`/
    /// `countdown_step`/`infinite_step`).
    NativeOp(String, usize),

    /// Pop `n` argument values, call the host function `name`, and push the
    /// result.
    ///
    /// Emitted by `Arr { op, args, source: Foreign(name), .. }`. The host
    /// function is looked up in the
    /// [`ModuleLookup`](handl_platform::ModuleLookup) registry.
    ForeignOp(String, usize),

    /// Run the `First` combinator's inner sub-program on the first component
    /// of a pair.
    ///
    /// Pop a pair `{"__pair__": [a, b]}` from the operand stack. Execute the
    /// sub-program at instruction indices `[start, end)` with `a` as the
    /// incoming value (on a fresh operand stack). The sub-program's result
    /// becomes the new first component; `b` is carried through unchanged.
    /// Push `{"__pair__": [result, b]}`. Execution continues at `end`.
    ///
    /// Using a sub-program (rather than inline `Unpair`/`Swap`/`Pair`)
    /// correctly handles inner arrows like `const` that don't consume the
    /// incoming value — the sub-program has its own fresh stack, so
    /// leftovers are discarded and only the top (the result) is returned.
    FirstOp(usize, usize),

    /// Pop two values `a` (top) and `b` (below), push the pair
    /// `{"__pair__": [b, a]}`.
    ///
    /// Used by `Loop { body }` to construct the initial pair
    /// `(incoming, Null)` from the incoming value on the stack and a pushed
    /// `Null` state seed.
    Pair,

    /// Inspect a `Loop` body result and route single-step vs multi-step
    /// execution.
    ///
    /// Pop the body result from the operand stack:
    /// - If it is an Either **left** `{"__either__": "left", "value": v}`:
    ///   push `v` and fall through (loop terminates, `v` is the output).
    /// - If it is an Either **right** `{"__either__": "right", "value": s}`:
    ///   `s` is the new state. Reconstruct the pair `(original_input, s)`,
    ///   push it, and jump to `loop_back` (the start of the body region).
    /// - If it is a pair `{"__pair__": [out, _state]}`: push `out` and fall
    ///   through (single-step mode — backward-compatible).
    ///
    /// A max-iteration counter (100_000) prevents infinite loops.
    LoopCheck(usize),

    /// Derived step op (`while_step`/`for_step`/`fix_step`) with
    /// re-evaluatable arg sub-programs.
    ///
    /// Unlike [`NativeOp`], which evaluates args once before the handler
    /// runs, `DerivedStepOp` stores each arg as an instruction range
    /// `(start, end)` and re-evaluates them **per iteration** with the
    /// current value (extracted from the loop pair) as incoming. This is
    /// required because `while_step`/`for_step`/`fix_step` carry cond/body
    /// sub-schemes that must be re-run with the evolving current value on
    /// each loop iteration — not evaluated once upfront.
    ///
    /// The handler pops the loop pair `(in, state)` from the stack (left
    /// by the enclosing `Loop`), extracts the current value, re-executes
    /// each arg sub-program via [`VmFrame::exec_instrs`] with `Some(current)`
    /// as incoming, then dispatches to [`VmFrame::apply_native_v2`] with
    /// the fresh results. The produced `Either` is pushed for `LoopCheck`.
    ///
    /// (`loop_step` is handled by [`NativeOp`] instead — its body arg
    /// is a closure evaluated once, then applied per iteration via
    /// [`VmFrame::apply_app`].)
    DerivedStepOp(String, Vec<(usize, usize)>),

    /// Pop the top of the operand stack and return it as the program result.
    ///
    /// Every compiled program ends with `Return`.
    Return,

    /// Pop a value, compare its type tag to `tag`, and push
    /// `either_left(value)` if it matches or `either_right(value)` if not
    /// (ADR-0060).
    ///
    /// Emitted by `Arr { op: "route_on_type", lit: Some(tag), .. }`.
    RouteOnType(String),

    // --- Typed arithmetic/comparison bytecodes (hot-path optimization) ---
    //
    // These mirror the `add`/`sub`/`mul`/`div`/`mod`/`eq`/`neq`/`lt`/`lte`/
    // `gt`/`gte` cases of [`NativeOp`] but avoid the per-op `Vec` allocation
    // and string dispatch: they pop two values directly from the operand
    // stack and push the result. Emitted by the compiler for the
    // corresponding `Arr { op, source: Native, .. }` ops. They produce
    // results identical to the `NativeOp` path.
    /// Pop two values, push their sum (`add`).
    Add,
    /// Pop two values, push their difference (`sub`).
    Sub,
    /// Pop two values, push their product (`mul`).
    Mul,
    /// Pop two values, push their quotient (`div`).
    Div,
    /// Pop two values, push their remainder (`mod`).
    Mod,
    /// Pop two values, push whether they are equal (`eq`).
    Eq,
    /// Pop two values, push whether they are not equal (`neq`).
    Neq,
    /// Pop two values, push whether the first is less than the second (`lt`).
    Lt,
    /// Pop two values, push whether the first is <= the second (`lte`).
    Lte,
    /// Pop two values, push whether the first is greater than the second (`gt`).
    Gt,
    /// Pop two values, push whether the first is >= the second (`gte`).
    Gte,

    // --- Object construction bytecode (hot-path optimization) ---
    /// Evaluate the AST node at the given absolute arena index and push the
    /// result.
    ///
    /// Used by [`MakeObj`] to evaluate object entry value nodes. Object entry
    /// values are nested inside `entry` nodes (which are children of the
    /// `object` node), so they cannot be reached via [`EvalChild`] (which
    /// reads `node.payload["children"][n]` — that yields entry nodes, not
    /// value nodes). `EvalNode` takes an absolute arena index instead.
    EvalNode(usize),

    /// Pop `keys.len()` values from the operand stack and construct a
    /// `TaggedValue::Obj` using the pre-allocated `keys`.
    ///
    /// The keys are stored in the instruction itself (allocated once at
    /// compile time as `Rc<[Rc<str>]>`), so they're reused on every
    /// iteration — avoiding per-evaluation `Rc::from(key)` allocation for
    /// object field names. Values are pushed in source order (first key's
    /// value first); the handler pops in reverse to match keys.
    ///
    /// Emitted by the compiler for `Arr { op: "object", source: Native, .. }`
    /// when all entry keys are static (`var` or `string` kind). Dynamic
    /// (computed) keys fall back to [`NativeOp`]`("object", n)`.
    MakeObj {
        /// Pre-allocated object keys (allocated once at compile time).
        keys: Rc<[Rc<str>]>,
    },

    // --- Closure construction bytecode (hot-path optimization) ---
    /// Capture the current environment and push a `TaggedValue::Arrow`
    /// representing a closure (`closure` native op).
    ///
    /// The param names and body node index are pre-extracted at compile
    /// time from the AST node's payload (they're static), so the runtime
    /// handler only needs to call `scopes.capture()` and construct the
    /// `Arrow` value. This avoids the per-call `Vec::with_capacity(n)` for
    /// args, the string `match op` dispatch, and the node payload
    /// extraction performed by the `NativeOp("closure", n)` path.
    ///
    /// Emitted by the compiler for `Arr { op: "closure", source: Native, .. }`
    /// when the params/body can be statically resolved from the node
    /// payload. Falls back to [`NativeOp`]`("closure", n)` otherwise.
    Closure {
        /// Pre-extracted parameter names (allocated once at compile time).
        params: Rc<[String]>,
        /// Pre-extracted rest/variadic param name (`[a, ...rest] ~> body`).
        rest: Option<Rc<str>>,
        /// AST node index for the closure body.
        body: usize,
        /// Whether the params node is an array-destructuring pattern.
        array_pattern: bool,
    },

    // --- ADR-0103 hardware-boundary instructions ---
    //
    // `Arith` and `Ffi` are the two hardware-boundary instructions from the
    // target minimal ISA (ADR-0103 §2.4). `Arith` consolidates the typed
    // arithmetic/comparison bytecodes and the arithmetic `NativeOp` arms into
    // a single ALU instruction; `Ffi` is the host/OS syscall boundary,
    // generalizing `ForeignOp` with a `module` field. The existing
    // `NativeOp`/`ForeignOp`/typed-bytecode variants remain as fallbacks for
    // ops not yet migrated.
    /// Pop the operands and apply the ALU operation `op`, pushing the result.
    ///
    /// Binary ops (`Add`/`Sub`/`Mul`/`Div`/`Mod`/`Eq`/`Neq`/`Lt`/`Le`/`Gt`/
    /// `Ge`/`And`/`Or`) pop two values (`b` on top, `a` below). Unary ops
    /// (`Neg`/`Not`) pop one value. Semantics are identical to the
    /// corresponding typed bytecodes ([`Instr::Add`], [`Instr::Eq`], …) and
    /// `NativeOp` arms. This is the ADR-0103 target form; the dedicated
    /// bytecodes are retained as hot-path optimizations.
    Arith(ArithOp),

    /// Pop `n` argument values, call the host/OS function `module::name`, and
    /// push the result.
    ///
    /// This is the hardware-boundary instruction for effects (ADR-0103
    /// §2.4). The VM looks up `(module, name)` in its foreign-function
    /// registry; if absent, it falls back to the existing `ForeignOp`
    /// dispatch (looking up `name` in the [`ModuleLookup`] registry) so
    /// already-registered host functions keep working. Generalizes
    /// [`Instr::ForeignOp`] with a `module` field for routing to different
    /// host registries (`"host"`, `"io"`, `"fs"`, …).
    ///
    /// [`ModuleLookup`]: handl_platform::ModuleLookup
    Ffi {
        /// The host module name (e.g. `"host"`, `"io"`, `"fs"`).
        module: String,
        /// The function name within `module`.
        name: String,
        /// The number of argument values to pop from the operand stack.
        n: usize,
    },
}

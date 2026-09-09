//! `Term` -> v2 bytecode compilation (ADR-0052 Phase B).
//!
//! The [`compile`] function translates a
//! [`Term`](handl_language::Term) combinator tree into a
//! flat [`Vec<Instr>`](crate::Instr) program. Each combinator maps to a
//! fixed instruction sequence -- see the
//! [`instr`](crate::instr) module for the full mapping table.
//!
//! Every compiled program ends with [`Instr::Return`]. Sub-schemes are
//! inlined (their instructions are appended without a trailing `Return`) so
//! that exactly one value remains on the operand stack when the program
//! completes.
//!
//! ## Scope discipline
//!
//! Only `extend_env` pushes a scope. [`compile_into`] returns `true` when
//! the compiled code pushes a scope that the caller is responsible for
//! popping. `Seq` pops the scope its `fst` reported after `snd` completes
//! (emitting [`Instr::PopScope`]), and propagates whatever `snd` reported.
//! `First`/`Choice`/`Loop`/`App` pop any scope their sub-evaluation pushed
//! immediately after it completes. A bare `extend_env` at the top level is
//! balanced by [`compile`] appending a [`Instr::PopScope`] before
//! `Return`. This mirrors the tree-walk's `did_push` discipline exactly.

use std::rc::Rc;

use handl_language::{ArrSource, Language, Term};
use handl_language::{Ast, AstNode, AstRef, LangError, Value};

use crate::instr::Instr;

/// Compile a [`Term`] into a v2 bytecode program.
///
/// The `ast` and `node` parameters enable compile-time optimizations that
/// inspect the AST node's payload (e.g. pre-extracting static object keys
/// for [`Instr::MakeObj`]).
///
/// # Errors
///
/// Returns [`LangError::Engine`] if a sub-scheme fails to compile. In
/// practice, every well-formed [`Term`] variant compiles successfully.
pub fn compile(
    scheme: &Term,
    lang: &Language,
    ast: &Ast,
    node: &AstNode,
) -> Result<Vec<Instr>, LangError> {
    let mut out = Vec::new();
    let pushed = compile_into(scheme, lang, ast, node, &mut out)?;
    // A bare extend_env (no enclosing Seq to pop it) is balanced here.
    if pushed {
        out.push(Instr::PopScope);
    }
    out.push(Instr::Return);
    Ok(out)
}

/// Recursively append instructions for `scheme` into `out`.
///
/// After execution of the appended instructions, exactly one value remains on
/// the operand stack. No trailing [`Instr::Return`] is emitted -- the
/// caller ([`compile`]) adds it once at the top level.
///
/// Returns `Ok(true)` if the compiled code pushes a scope that the caller is
/// responsible for popping (i.e. `extend_env` was emitted). This mirrors the
/// tree-walk's `did_push` discipline.
fn compile_into(
    scheme: &Term,
    lang: &Language,
    ast: &Ast,
    node: &AstNode,
    out: &mut Vec<Instr>,
) -> Result<bool, LangError> {
    match scheme {
        Term::Arr { value } => {
            out.push(Instr::Push(value.clone()));
            Ok(false)
        }

        Term::Op {
            name: op,
            args,
            source,
            lit,
        } => compile_arr(op, args, source, lit.as_ref(), lang, ast, node, out),

        Term::Seq { fst, snd } => {
            let fst_pushed = compile_into(fst, lang, ast, node, out)?;
            let snd_pushed = compile_into(snd, lang, ast, node, out)?;
            // Pop the scope `fst` reported (e.g. extend_env as fst), matching
            // the tree-walk's Seq scope discipline.
            if fst_pushed {
                out.push(Instr::PopScope);
            }
            // Propagate snd_pushed: an extend_env in snd position (the Bind
            // lowering shape) is popped by the *enclosing* Seq.
            Ok(snd_pushed)
        }

        Term::First { inner } => {
            // FirstOp(start, end) — the inner sub-program runs on a fresh
            // stack with the pair's first component as incoming. This
            // correctly handles inner arrows like `const` that don't consume
            // the incoming (the sub-program's fresh stack discards
            // leftovers; only the top is returned as the result).
            let firstop_pos = out.len();
            out.push(Instr::FirstOp(0, 0)); // placeholder
            let inner_start = out.len();
            let inner_pushed = compile_into(inner, lang, ast, node, out)?;
            if inner_pushed {
                out.push(Instr::PopScope);
            }
            let inner_end = out.len();
            out[firstop_pos] = Instr::FirstOp(inner_start, inner_end);
            Ok(false)
        }

        Term::Choice { l, r } => {
            // ChoiceJump(left_target, right_target) -- placeholder.
            let cj_pos = out.len();
            out.push(Instr::ChoiceJump(0, 0));
            // Left branch.
            let left_target = out.len();
            let l_pushed = compile_into(l, lang, ast, node, out)?;
            if l_pushed {
                out.push(Instr::PopScope);
            }
            let left_jump_pos = out.len();
            out.push(Instr::Jump(0)); // placeholder -- skip right branch
            // Right branch.
            let right_target = out.len();
            let r_pushed = compile_into(r, lang, ast, node, out)?;
            if r_pushed {
                out.push(Instr::PopScope);
            }
            let end = out.len();
            // Patch ChoiceJump and the left-branch Jump.
            out[cj_pos] = Instr::ChoiceJump(left_target, right_target);
            out[left_jump_pos] = Instr::Jump(end);
            Ok(false)
        }

        Term::Loop { body } => {
            // Optimization: when the body is a `loop_step` Arr, hoist the
            // body closure evaluation outside the loop. The closure captures
            // the definition-time env, which doesn't change between iterations
            // (only the loop state changes, passed as an argument). This
            // avoids re-creating the closure (and deep-cloning its captured
            // scope via `capture()`) on every iteration. Mirrors the
            // tree-walk optimization at
            // `plugin-engine-exec-tree-walk/src/lib.rs:402-456`.
            if let Term::Op {
                name: op,
                args,
                source,
                ..
            } = body.as_ref()
            {
                if *source == ArrSource::Native && op == "loop_step" && !args.is_empty() {
                    let body_proc = &args[0];
                    // Compile the body closure once, before the loop. This
                    // evaluates the `function`/`closure` child once,
                    // producing a `TaggedValue::Arrow` on the operand stack.
                    let bp_pushed = compile_into(body_proc, lang, ast, node, out)?;
                    if bp_pushed {
                        out.push(Instr::PopScope);
                    }
                    // Store the pre-computed closure in a synthetic local.
                    // `Store` pushes a scope; we pop it after the loop
                    // (below) so it persists across all iterations.
                    out.push(Instr::Store(Rc::from("__loop_closure__")));

                    // Loop prologue: pair(incoming, Null) on the stack.
                    out.push(Instr::Push(Value::Null));
                    out.push(Instr::Pair);
                    let loop_back = out.len();

                    // Loop body: load the hoisted closure instead of
                    // re-creating it, compile any remaining args, then emit
                    // the `loop_step` native op. The VM's `loop_step`
                    // handler receives the pre-computed closure as args[0]
                    // and the loop pair as `incoming` — identical to the
                    // non-hoisted path, just without the per-iteration
                    // closure creation.
                    out.push(Instr::Load(Rc::from("__loop_closure__")));
                    for arg in &args[1..] {
                        let arg_pushed = compile_into(arg, lang, ast, node, out)?;
                        if arg_pushed {
                            out.push(Instr::PopScope);
                        }
                    }
                    out.push(Instr::NativeOp("loop_step".to_string(), args.len()));
                    out.push(Instr::LoopCheck(loop_back));

                    // Pop the scope pushed by `Store("__loop_closure__")`.
                    // This runs only when the loop terminates (LoopCheck
                    // falls through); the back-edge jumps over it.
                    out.push(Instr::PopScope);
                    return Ok(false);
                }
            }

            // Fallback: original loop compilation (non-loop_step body).
            //
            // Loop layout:
            //   Push(Null)          -- initial state
            //   Pair                -- pair(incoming, Null) on stack
            //   <loop_back>: <body code>
            //   LoopCheck(<loop_back>)
            //
            // The incoming value is on the stack from the enclosing context.
            // We pair it with Null (the initial state seed), then the body
            // consumes the pair and leaves its result. LoopCheck inspects
            // the result and either terminates (left/pair) or loops back
            // (right) with the new state.
            out.push(Instr::Push(Value::Null));
            out.push(Instr::Pair);
            let loop_back = out.len();
            let body_pushed = compile_into(body, lang, ast, node, out)?;
            // The tree-walk pops any scope the body pushed after each
            // iteration. In the VM, we emit PopScope after the body code
            // but before LoopCheck, so it runs each iteration.
            if body_pushed {
                out.push(Instr::PopScope);
            }
            out.push(Instr::LoopCheck(loop_back));
            Ok(false)
        }

        Term::App { callee, args } => {
            let callee_pushed = compile_into(callee, lang, ast, node, out)?;
            if callee_pushed {
                out.push(Instr::PopScope);
            }
            for arg in args {
                let arg_pushed = compile_into(arg, lang, ast, node, out)?;
                if arg_pushed {
                    out.push(Instr::PopScope);
                }
            }
            // Fixed-arity Call variants avoid the per-call `Vec` allocation
            // in the `Call(n)` handler for the common 1/2/3-arg cases.
            match args.len() {
                1 => out.push(Instr::Call1),
                2 => out.push(Instr::Call2),
                3 => out.push(Instr::Call3),
                n => out.push(Instr::Call(n)),
            }
            Ok(false)
        }
        // `handle body with target = impl_` (ADR-0076): eliminate the
        // handler at compile time, then compile the resulting handler-free
        // scheme. Uses `eliminate_and_check` to surface unhandled
        // capabilities as a compile-time error.
        Term::Handle { .. } => {
            let eliminated = handl_language::eliminate_and_check(scheme)
                .map_err(|e| LangError::engine_node(e.to_string(), node))?;
            compile_into(&eliminated, lang, ast, node, out)
        }

        Term::Iter { .. } => Err(LangError::NotImplemented),
        Term::Child { .. } => Err(LangError::NotImplemented),
        Term::Path { .. } => Err(LangError::NotImplemented),
        Term::Equiv { .. } => Err(LangError::NotImplemented),
        Term::CellId { .. } => Err(LangError::NotImplemented),
        Term::Src { .. } => Err(LangError::NotImplemented),
        Term::Tgt { .. } => Err(LangError::NotImplemented),
        Term::CellCompV { .. } => Err(LangError::NotImplemented),
        Term::Whisker { .. } => Err(LangError::NotImplemented),
        Term::Const { .. } => Err(LangError::NotImplemented),
        Term::RecordProj { .. } => Err(LangError::NotImplemented),
        Term::RecordWith { .. } => Err(LangError::NotImplemented),
        Term::Tag { .. } => Err(LangError::NotImplemented),
        Term::Univalence { .. } => Err(LangError::NotImplemented),
        Term::Glue { .. } => Err(LangError::NotImplemented),
        Term::Fibrant { .. } => Err(LangError::NotImplemented),
        Term::JElim { .. } => Err(LangError::NotImplemented),
        Term::Refine { .. } => Err(LangError::NotImplemented),
        Term::Cobind { .. } => Err(LangError::NotImplemented),
        Term::Rewrite { .. } => Err(LangError::NotImplemented),
    }
}

/// Compile an `Arr` combinator. Returns `Ok(true)` if this Arr pushes a scope
/// (`extend_env`), so the caller can emit `PopScope`.
fn compile_arr(
    op: &str,
    args: &[Term],
    source: &ArrSource,
    lit: Option<&Value>,
    lang: &Language,
    ast: &Ast,
    node: &AstNode,
    out: &mut Vec<Instr>,
) -> Result<bool, LangError> {
    // Special-case the lit-bearing ops.
    match op {
        "const" => {
            out.push(Instr::Push(lit.cloned().unwrap_or(Value::Null)));
            return Ok(false);
        }
        "proj_env" => {
            let name = lit.and_then(|v| v.as_str()).ok_or_else(|| {
                LangError::engine_node("proj_env: missing name literal".to_string(), node)
            })?;
            if name == "$payload" {
                out.push(Instr::LoadPayloadVar);
            } else {
                out.push(Instr::Load(Rc::from(name)));
            }
            return Ok(false);
        }
        "proj_child" => {
            let idx = lit.and_then(|v| v.as_u64()).ok_or_else(|| {
                LangError::engine_node("proj_child: missing index literal".to_string(), node)
            })?;
            out.push(Instr::EvalChild(idx as usize));
            return Ok(false);
        }
        "proj_child_or" => {
            // Like `proj_child`, but tolerates an absent child (pushes null).
            let idx = lit.and_then(|v| v.as_u64()).ok_or_else(|| {
                LangError::engine_node("proj_child_or: missing index literal".to_string(), node)
            })?;
            out.push(Instr::EvalChildOr(idx as usize));
            return Ok(false);
        }
        "extend_env" => {
            let name = lit.and_then(|v| v.as_str()).ok_or_else(|| {
                LangError::engine_node("extend_env: missing name literal".to_string(), node)
            })?;
            if name == "$payload" {
                out.push(Instr::StorePayloadName);
            } else {
                out.push(Instr::Store(Rc::from(name)));
            }
            return Ok(true); // pushed a scope -- caller pops
        }
        // ADR-0060: route_on_type takes one arg (the value) and a lit
        // (the type tag string). Compile the arg, then emit RouteOnType.
        "route_on_type" => {
            for arg in args {
                let arg_pushed = compile_into(arg, lang, ast, node, out)?;
                if arg_pushed {
                    out.push(Instr::PopScope);
                }
            }
            let tag = lit.and_then(|v| v.as_str()).ok_or_else(|| {
                LangError::engine_node("route_on_type: missing type tag literal".to_string(), node)
            })?;
            out.push(Instr::RouteOnType(tag.to_string()));
            return Ok(false);
        }
        _ => {}
    }

    // General native or foreign op: compile args, then emit NativeOp/ForeignOp.
    //
    // Derived step ops (`while_step`/`for_step`/`fix_step`) are special:
    // their args (cond/body sub-schemes) must be re-evaluated per iteration
    // with the current value as incoming. We compile the args as sub-programs
    // at known instruction ranges, then emit a `DerivedStepOp` instruction
    // that stores those ranges. A `Jump` before the sub-programs skips them
    // during normal execution (only the `DerivedStepOp` handler re-executes
    // them per iteration via `exec_instrs`).
    if source == &ArrSource::Native && is_derived_step_op_with_sub_schemes(op) {
        // Jump over the arg sub-programs to the DerivedStepOp instruction.
        let jump_pos = out.len();
        out.push(Instr::Jump(0)); // placeholder — patched below
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(args.len());
        for arg in args {
            let start = out.len();
            let arg_pushed = compile_into(arg, lang, ast, node, out)?;
            if arg_pushed {
                out.push(Instr::PopScope);
            }
            let end = out.len();
            ranges.push((start, end));
        }
        let derived_pos = out.len();
        out.push(Instr::DerivedStepOp(op.to_string(), ranges));
        // Patch the Jump to skip the sub-programs.
        out[jump_pos] = Instr::Jump(derived_pos);
        return Ok(false);
    }

    // Hot-path optimization for `object` with static keys: when the object
    // node's payload has a `children` array of `entry` nodes whose `key`
    // role is declared literal (`literal_children`, ADR-0081), pre-extract
    // the keys into `Rc<[Rc<str>]>` and emit `EvalNode` for each value
    // followed by `MakeObj { keys }`. This avoids per-evaluation
    // `Rc::from(key)` allocation for object field names — the keys are
    // allocated once at compile time and reused (refcount bump) on every
    // iteration. Falls back to `NativeOp("object", n)` for dynamic
    // (computed) keys or the `entries` payload shape.
    if source == &ArrSource::Native && op == "object" {
        if let Some(keys) = try_extract_static_object_keys(lang, ast, node)? {
            // Emit `EvalNode(val_idx)` for each entry's value node, in
            // source order. The VM evaluates the node at the given absolute
            // arena index and pushes the result.
            for &val_idx in keys.value_indices.iter() {
                out.push(Instr::EvalNode(val_idx));
            }
            out.push(Instr::MakeObj { keys: keys.keys });
            return Ok(false);
        }
    }

    // Hot-path optimization for `closure`: pre-extract the param names and
    // body node index from the AST node's payload at compile time (they're
    // static), and emit `Closure { params, body }`. At runtime the handler
    // only calls `scopes.capture()` and constructs the `TaggedValue::Arrow` —
    // avoiding the per-call `Vec::with_capacity(n)` for args, the string
    // `match op` dispatch, and the node payload extraction performed by the
    // `NativeOp("closure", n)` path. Falls back to `NativeOp("closure", n)`
    // when the params/body cannot be statically resolved.
    if source == &ArrSource::Native && op == "closure" {
        if let Some(closure) = try_extract_closure_params_body(lang, ast, node)? {
            out.push(Instr::Closure {
                params: closure.params,
                rest: closure.rest,
                body: closure.body,
            });
            return Ok(false);
        }
    }

    for arg in args {
        let arg_pushed = compile_into(arg, lang, ast, node, out)?;
        if arg_pushed {
            out.push(Instr::PopScope);
        }
    }
    match source {
        ArrSource::Native => {
            // Hot-path: emit a typed arithmetic/comparison instruction
            // instead of `NativeOp(op, 2)` for the common binary numeric
            // ops. This avoids the per-op `Vec` allocation and string
            // dispatch in the VM. Results are identical to the `NativeOp`
            // path.
            if let Some(typed) = typed_binary_instr(op) {
                out.push(typed);
            } else {
                out.push(Instr::NativeOp(op.to_string(), args.len()));
            }
        }
        ArrSource::Foreign(name) => {
            out.push(Instr::ForeignOp(name.clone(), args.len()));
        }
        ArrSource::DynamicForeign => {
            // ADR-0060: DynamicForeign ops (e.g. call_host) are dispatched
            // like Native ops — the op handler in apply_native_v2 performs
            // the host-function lookup at runtime.
            out.push(Instr::NativeOp(op.to_string(), args.len()));
        }
        ArrSource::Judgment => {
            return Err(LangError::engine(format!(
                "compile: judgment op '{}' is not supported by the eval engine (ADR-0054: judgment ops are dispatched by the typecheck engine)",
                op
            )));
        }
    }
    Ok(false)
}

/// `while_step`/`for_step`/`fix_step` — derived step ops that carry
/// cond/body sub-schemes as args and need per-iteration re-evaluation.
/// (`loop_step` is NOT here — its body arg is a closure evaluated
/// once, then applied per iteration via `apply_app`.)
fn is_derived_step_op_with_sub_schemes(op: &str) -> bool {
    matches!(op, "while_step" | "for_step" | "fix_step")
}

/// Map a native binary op name to its typed [`Instr`] variant, if any.
///
/// Returns `Some` for the hot arithmetic/comparison ops (`add`/`sub`/`mul`/
/// `div`/`mod`/`eq`/`neq`/`lt`/`lte`/`gt`/`gte`). These are emitted as
/// dedicated instructions instead of `NativeOp(op, 2)` to avoid the per-op
/// `Vec` allocation and string dispatch in the VM. All other native ops
/// fall through to `NativeOp`.
fn typed_binary_instr(op: &str) -> Option<Instr> {
    Some(match op {
        "add" => Instr::Add,
        "sub" => Instr::Sub,
        "mul" => Instr::Mul,
        "div" => Instr::Div,
        "mod" => Instr::Mod,
        "eq" => Instr::Eq,
        "neq" => Instr::Neq,
        "lt" => Instr::Lt,
        "lte" => Instr::Lte,
        "gt" => Instr::Gt,
        "gte" => Instr::Gte,
        _ => return None,
    })
}

/// The result of extracting static keys from an `object` node: the
/// pre-allocated key strings and the absolute arena indices of the
/// corresponding value nodes.
struct StaticObjectKeys {
    keys: Rc<[Rc<str>]>,
    value_indices: Vec<usize>,
}

/// Try to extract static keys from an `object` AST node's `children` payload.
///
/// The object node's `payload["children"]` is an array of entry-node arena
/// indices. Each entry node has `payload["children"] = [key_idx, val_idx]`.
/// A key is static when the `entry` shape declares the key child's kind in
/// its `literal_children` list for the `key` role (ADR-0081 — e.g.
/// `var`/`string`; the payload string is used directly as the key — see
/// the `object` handler in `vm.rs`). Computed keys (e.g. `[expr]:`
/// desugared to `paren`) cause this function to return `None`, so the
/// compiler falls back to `NativeOp("object", n)`.
///
/// The `var`/`string` kind fallback below is redundant when the `lower`
/// pass has normalized `string` keys to `var` (`BytecodeEngine::with_lowering`),
/// and when the `entry` shape declares `literal_children` for `key`. It is
/// retained only for the non-lowering path and for role-less `entry` shapes.
///
/// Returns `None` if the payload shape is not `children` (e.g. the `entries`
/// rich-payload shape) or if any key is dynamic.
fn try_extract_static_object_keys(
    lang: &Language,
    ast: &Ast,
    node: &AstNode,
) -> Result<Option<StaticObjectKeys>, LangError> {
    // Only the `children` (Pratt-parsed) payload shape is supported.
    // The `entries` (rich) shape has literal values, not child indices.
    let children = match node.payload.get("children").and_then(|c| c.as_array()) {
        Some(c) => c,
        None => return Ok(None),
    };
    let start = match lang.child_role_index(&node.kind, "entries")? {
        Some(i) => i,
        None => 0,
    };

    let mut keys: Vec<Rc<str>> = Vec::with_capacity(children.len().saturating_sub(start));
    let mut value_indices: Vec<usize> = Vec::with_capacity(children.len().saturating_sub(start));

    for entry in children.iter().skip(start) {
        let entry_idx = match entry.as_u64() {
            Some(idx) => idx as usize,
            None => return Ok(None),
        };
        let entry_node = ast.at(AstRef::new(entry_idx))?;
        let entry_children = match entry_node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
        {
            Some(c) => c,
            None => return Ok(None),
        };
        let key_role = lang.child_role_index(&entry_node.kind, "key")?.unwrap_or(0);
        let val_role = lang
            .child_role_index(&entry_node.kind, "value")?
            .unwrap_or(1);
        let key_idx = match entry_children.get(key_role).and_then(|v| v.as_u64()) {
            Some(idx) => idx as usize,
            None => return Ok(None),
        };
        let val_idx = match entry_children.get(val_role).and_then(|v| v.as_u64()) {
            Some(idx) => idx as usize,
            None => return Ok(None),
        };

        let key_node = ast.at(AstRef::new(key_idx))?;
        // Static keys are declared literal for the `key` role by the entry
        // shape (`literal_children`, ADR-0081). The key string is the node's
        // payload. Computed keys (e.g. `paren` from `[expr]:` desugaring)
        // fall back to `NativeOp`. The `var`/`string` fallback is for the
        // non-lowering path and role-less shapes (see doc comment above).
        let is_literal_key = lang
            .literal_child(&entry_node.kind, "key", &key_node.kind)
            .unwrap_or_else(|| matches!(key_node.kind.as_str(), "var" | "string"));
        let key_str: String = if is_literal_key {
            match key_node.payload.as_str() {
                Some(s) => s.to_string(),
                None => return Ok(None),
            }
        } else {
            // Dynamic (computed) key — fall back to NativeOp.
            return Ok(None);
        };

        // Reject reserved keys (mirrors the `object` native op's check).
        if handl_language::is_reserved_value_key(&key_str) {
            return Err(LangError::engine(format!(
                "object: key '{}' is reserved",
                key_str
            )));
        }

        keys.push(Rc::from(key_str.as_str()));
        value_indices.push(val_idx);
    }

    Ok(Some(StaticObjectKeys {
        keys: Rc::from(keys),
        value_indices,
    }))
}

/// The result of extracting static closure params and body from a `function`
/// node: the pre-allocated param-name strings and the body node's absolute
/// arena index.
struct StaticClosure {
    params: Rc<[String]>,
    /// Rest/variadic param name (`[a, ...rest] ~> body`), if declared.
    rest: Option<Rc<str>>,
    body: usize,
}

/// Try to extract the param names and body node index from a `function`
/// (closure) AST node's payload at compile time.
///
/// Mirrors the two payload shapes handled by the `closure` native op in
/// `vm.rs`:
/// - **Rich**: `{"params": ["n", "a", ...], "body": <node_idx>}` — the
///   params are literal strings and `body` is a node index.
/// - **Pratt-parsed**: `{"children": [param_idx, body_idx]}` — `param_idx`
///   points to a node that is either a single `var` (one param) or a
///   `paren`/`array` of `var` children (multiple params); `body_idx` points
///   to the body node.
///
/// Returns `Ok(None)` if the payload shape is unrecognized or a param node
/// lacks a string payload, so the compiler falls back to
/// `NativeOp("closure", n)`. Returns `Err` only for AST lookup failures
/// (malformed arena indices), matching the runtime handler's behavior.
fn try_extract_closure_params_body(
    lang: &Language,
    ast: &Ast,
    node: &AstNode,
) -> Result<Option<StaticClosure>, LangError> {
    let payload = &node.payload;

    // Rich payload shape: `{"params": [...], "body": <idx>}`.
    if let (Some(p), Some(b)) = (
        payload.get("params").and_then(|v| v.as_array()),
        payload.get("body").and_then(|v| v.as_u64()),
    ) {
        let mut names: Vec<String> = Vec::with_capacity(p.len());
        for v in p {
            let s = match v.as_str() {
                Some(s) => s,
                None => return Ok(None),
            };
            names.push(s.to_string());
        }
        let rest = payload.get("rest").and_then(|v| v.as_str()).map(Rc::from);
        return Ok(Some(StaticClosure {
            params: Rc::from(names),
            rest,
            body: b as usize,
        }));
    }

    // Pratt-parsed payload shape: `{"children": [param_idx, body_idx]}`.
    if let Some(children) = payload.get("children").and_then(|c| c.as_array()) {
        let param_idx = match children.first().and_then(|v| v.as_u64()) {
            Some(idx) => idx as usize,
            None => return Ok(None),
        };
        let body_idx = match children.get(1).and_then(|v| v.as_u64()) {
            Some(idx) => idx as usize,
            None => return Ok(None),
        };
        let param_node = ast.at(AstRef::new(param_idx))?;
        // Collection-vs-single params is a shape question (ADR-0081): a
        // shape declaring a variadic role (e.g. `params`) is a param list;
        // the legacy `paren`/`array` kinds remain only as the role-less
        // fallback.
        // `param_names` resolves fixed params plus an optional rest param
        // (`...name`) data-driven via `pattern_role`/variadic metadata
        // (ADR-0081). `Err` (unresolvable param shape) falls back to the
        // runtime `NativeOp("closure")` path like `Ok(None)`.
        let (names, rest) = match lang.param_names(ast, param_node) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        return Ok(Some(StaticClosure {
            params: Rc::from(names),
            rest: rest.map(Rc::from),
            body: body_idx,
        }));
    }

    // Unrecognized payload shape — fall back to NativeOp.
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use handl_language::Term;
    use handl_language::{Ast, AstNode, Language};
    use serde_json::json;

    /// A dummy node used by compilation tests (the node is not inspected).
    fn dummy_node() -> AstNode {
        AstNode::new("test")
    }

    /// A dummy AST containing a single dummy node, for compilation tests.
    fn dummy_ast() -> Ast {
        let mut ast = Ast::new();
        ast.push(dummy_node());
        ast
    }

    // --- Arr compilation tests ---

    #[test]
    fn compile_const() {
        let s = Term::arr_native_lit("const", vec![], json!(42));
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::Push(json!(42)), Instr::Return]);
    }

    #[test]
    fn compile_const_no_lit() {
        let s = Term::arr_native("const", vec![]);
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::Push(Value::Null), Instr::Return]);
    }

    #[test]
    fn compile_proj_env_named() {
        let s = Term::arr_native_lit("proj_env", vec![], "x");
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::Load("x".into()), Instr::Return]);
    }

    #[test]
    fn compile_proj_env_payload() {
        let s = Term::arr_native_lit("proj_env", vec![], "$payload");
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::LoadPayloadVar, Instr::Return]);
    }

    #[test]
    fn compile_proj_child() {
        let s = Term::arr_native_lit("proj_child", vec![], json!(2));
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::EvalChild(2), Instr::Return]);
    }

    #[test]
    fn compile_extend_env_named() {
        let s = Term::arr_native_lit("extend_env", vec![], "x");
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![Instr::Store("x".into()), Instr::PopScope, Instr::Return]
        );
    }

    #[test]
    fn compile_extend_env_payload() {
        let s = Term::arr_native_lit("extend_env", vec![], "$payload");
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![Instr::StorePayloadName, Instr::PopScope, Instr::Return]
        );
    }

    #[test]
    fn compile_native_op() {
        let s = Term::arr_native(
            "add",
            vec![
                Term::arr_native_lit("proj_child", vec![], json!(0)),
                Term::arr_native_lit("proj_child", vec![], json!(1)),
            ],
        );
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::EvalChild(0),
                Instr::EvalChild(1),
                Instr::Add,
                Instr::Return,
            ]
        );
    }

    #[test]
    fn compile_foreign_op() {
        let s = Term::arr_foreign(
            "double",
            vec![Term::arr_native_lit("proj_child", vec![], json!(0))],
        );
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::EvalChild(0),
                Instr::ForeignOp("double".into(), 1),
                Instr::Return,
            ]
        );
    }

    // --- Seq compilation tests ---

    #[test]
    fn compile_seq_no_scope() {
        let s = Term::seq(
            Term::arr_native_lit("const", vec![], json!(5)),
            Term::arr_native("id", vec![]),
        );
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::Push(json!(5)),
                Instr::NativeOp("id".into(), 0),
                Instr::Return,
            ]
        );
    }

    #[test]
    fn compile_seq_with_extend_env_pops_scope() {
        // Seq(Seq(const 5, extend_env "x"), proj_env "x") -- the inner Seq's
        // fst (const) does not push a scope. The inner Seq's snd (extend_env)
        // pushes a scope, which propagates to the outer Seq. The outer Seq
        // pops after its snd (proj_env) completes.
        let s = Term::seq(
            Term::seq(
                Term::arr_native_lit("const", vec![], json!(5)),
                Term::arr_native_lit("extend_env", vec![], "x"),
            ),
            Term::arr_native_lit("proj_env", vec![], "x"),
        );
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::Push(json!(5)),    // 0: const 5
                Instr::Store("x".into()), // 1: extend_env "x"
                Instr::Load("x".into()),  // 2: proj_env "x"
                Instr::PopScope,          // 3: pop scope (outer Seq)
                Instr::Return,            // 4
            ]
        );
    }

    // --- First compilation tests ---

    #[test]
    fn compile_first() {
        let s = Term::first(Term::arr_native("id", vec![]));
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        // FirstOp(1, 2) — inner sub-program at index 1..2 is the `id` op.
        assert_eq!(
            instrs,
            vec![
                Instr::FirstOp(1, 2),
                Instr::NativeOp("id".into(), 0),
                Instr::Return,
            ]
        );
    }

    // --- Choice compilation tests ---

    #[test]
    fn compile_choice() {
        let s = Term::choice(
            Term::arr_native_lit("const", vec![], json!(1)),
            Term::arr_native_lit("const", vec![], json!(2)),
        );
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::ChoiceJump(1, 3), // 0: route left/right
                Instr::Push(json!(1)),   // 1: left branch
                Instr::Jump(4),          // 2: skip right
                Instr::Push(json!(2)),   // 3: right branch
                Instr::Return,           // 4: end
            ]
        );
    }

    // --- Loop compilation tests ---

    #[test]
    fn compile_loop() {
        let s = Term::loop_feedback(Term::arr_native("inc_fst", vec![]));
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::Push(Value::Null),             // 0: initial state
                Instr::Pair,                          // 1: pair(incoming, Null)
                Instr::NativeOp("inc_fst".into(), 0), // 2: body (loop_back = 2)
                Instr::LoopCheck(2),                  // 3: check + loop back
                Instr::Return,                        // 4
            ]
        );
    }

    // --- App compilation tests ---

    #[test]
    fn compile_app() {
        let s = Term::app(
            Term::arr_native_lit("proj_env", vec![], "f"),
            vec![Term::arr_native_lit("const", vec![], json!(41))],
        );
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::Load("f".into()),
                Instr::Push(json!(41)),
                Instr::Call1,
                Instr::Return,
            ]
        );
    }

    #[test]
    fn compile_app_no_args() {
        let s = Term::app(Term::arr_native_lit("proj_env", vec![], "f"), vec![]);
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![Instr::Load("f".into()), Instr::Call(0), Instr::Return]
        );
    }

    // --- Branch lowering (Seq + tag + Choice) ---

    #[test]
    fn compile_branch_lowering() {
        // Seq(Seq(const true, tag), Choice(const "yes", const "no"))
        let s = Term::seq(
            Term::seq(
                Term::arr_native_lit("const", vec![], json!(true)),
                Term::arr_native("tag", vec![]),
            ),
            Term::choice(
                Term::arr_native_lit("const", vec![], "yes"),
                Term::arr_native_lit("const", vec![], "no"),
            ),
        );
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::Push(json!(true)),         // 0: const true
                Instr::NativeOp("tag".into(), 0), // 1: tag
                Instr::ChoiceJump(3, 5),          // 2: route left/right
                Instr::Push(json!("yes")),        // 3: left (then)
                Instr::Jump(6),                   // 4: skip right
                Instr::Push(json!("no")),         // 5: right (else)
                Instr::Return,                    // 6: end
            ]
        );
    }
}

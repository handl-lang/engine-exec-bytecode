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
use plugin_engine::TermData;

use crate::instr::{ArithOp, Instr};

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
///
/// This is the `&Term` entry point: it converts the scheme to [`TermData`]
/// (ADR-0103 Phase 6) and delegates to [`compile_into_td`], which dispatches
/// on the `__tag__` field instead of matching the `Term` enum.
fn compile_into(
    scheme: &Term,
    lang: &Language,
    ast: &Ast,
    node: &AstNode,
    out: &mut Vec<Instr>,
) -> Result<bool, LangError> {
    let td = TermData::from_term(scheme);
    compile_into_td(&td, lang, ast, node, out)
}

/// The [`TermData`]-driven counterpart of [`compile_into`].
///
/// Dispatches on [`TermData::tag`] and reads variant fields via the
/// `get`/`get_str`/`get_term`/`get_term_list` accessors. Derived forms
/// (`CellId`/`Src`/`Tgt`/`CellCompV`/`Whisker`/`Const`/`RecordProj`) are
/// rewritten to their `Op`/`Arr` lowering using [`TermData`] constructors and
/// recompiled recursively, so no `Term` enum value is constructed for them.
fn compile_into_td(
    td: &TermData,
    lang: &Language,
    ast: &Ast,
    node: &AstNode,
    out: &mut Vec<Instr>,
) -> Result<bool, LangError> {
    match td.tag() {
        "Arr" => {
            let value = td
                .get("value")
                .cloned()
                .expect("Arr term missing 'value' field");
            out.push(Instr::Push(value));
            Ok(false)
        }

        "Op" => {
            let op = td
                .get_str("name")
                .expect("Op term missing 'name' field")
                .to_string();
            let args = td.get_term_list("args");
            let source = td
                .arr_source()
                .expect("Op term missing or malformed 'source' field");
            // `lit` is stored as `null` when `None`; treat Null as absent.
            let lit = td.get("lit").filter(|v| !v.is_null());
            compile_arr(&op, &args, &source, lit, lang, ast, node, out)
        }

        "Seq" => {
            let fst = td.get_term("fst").expect("Seq term missing 'fst' field");
            let snd = td.get_term("snd").expect("Seq term missing 'snd' field");
            // ADR-0120 Phase 5: `Seq{axis: k}` is `∘ₖ` — composition along
            // the k-th globular axis. Axis 0 on 1-cells (`dim ≤ 1`) is
            // ordinary sequential composition; `k < dim` pastes n-cells at
            // the value level — both operands see the same incoming, which
            // is exactly op-argument evaluation, so the paste lowers to the
            // shared `cell_comp_axis` op with `[axis, fst, snd]` as args.
            let fst_dim = fst.globular_dim();
            match td.axis().nat_literal() {
                // `∘₀` on 1-cells — sequential composition (`>>>`).
                Some(0) if fst_dim <= 1 => {
                    let fst_pushed = compile_into_td(&fst, lang, ast, node, out)?;
                    // fst's result is now the current incoming — snd's
                    // arg-position incoming-ops (`PushIncoming`) read it
                    // from the `current_incoming` register, which the
                    // frame `incoming` param doesn't track.
                    out.push(Instr::SetIncoming);
                    let snd_pushed = compile_into_td(&snd, lang, ast, node, out)?;
                    // Pop the scope `fst` reported (e.g. extend_env as fst),
                    // matching the tree-walk's Seq scope discipline.
                    if fst_pushed {
                        out.push(Instr::PopScope);
                    }
                    out.push(Instr::RestoreIncoming);
                    // Propagate snd_pushed: an extend_env in snd position
                    // (the Bind lowering shape) is popped by the *enclosing*
                    // Seq.
                    return Ok(snd_pushed);
                }
                // A dynamic axis on 1-cell operands could resolve to 0 —
                // sequential composition — which is not expressible as
                // value-level composition.
                None if fst_dim <= 1 => {
                    return Err(LangError::engine_node(
                        "Seq: non-literal composition axis on 1-cell operands (dim ≤ 1) is not supported"
                            .to_string(),
                        node,
                    ));
                }
                // Pasting regime: every other case lowers to the shared
                // `cell_comp_axis` op — including `k >= dim(fst)` and
                // literal `k > 0` on declared 1-cells. The op resolves the
                // *effective* dimension at runtime (`max(declared,
                // cell_dim(fst), cell_dim(snd))`): a `Cell`/`__dim__`
                // record pastes at its own height, and `k >= n` on plain
                // values is the shared executor's dimension error. The
                // declared dimension is baked into the op name because
                // `NativeOp` cannot carry `lit` (the `route_on_type`
                // convention).
                _ => {
                    return compile_into_td(
                        &TermData::op(
                            format!("cell_comp_axis:{fst_dim}"),
                            vec![td.axis(), fst, snd],
                            &ArrSource::Native,
                            None,
                        ),
                        lang,
                        ast,
                        node,
                        out,
                    );
                }
            }
        }

        "First" => {
            let inner = td
                .get_term("inner")
                .expect("First term missing 'inner' field");
            // FirstOp(start, end) — the inner sub-program runs on a fresh
            // stack with the pair's first component as incoming. This
            // correctly handles inner arrows like `const` that don't consume
            // the incoming (the sub-program's fresh stack discards
            // leftovers; only the top is returned as the result).
            let firstop_pos = out.len();
            out.push(Instr::FirstOp(0, 0)); // placeholder
            let inner_start = out.len();
            let inner_pushed = compile_into_td(&inner, lang, ast, node, out)?;
            if inner_pushed {
                out.push(Instr::PopScope);
            }
            let inner_end = out.len();
            out[firstop_pos] = Instr::FirstOp(inner_start, inner_end);
            Ok(false)
        }

        "Choice" => {
            let l = td.get_term("l").expect("Choice term missing 'l' field");
            let r = td.get_term("r").expect("Choice term missing 'r' field");
            // ChoiceJump(left_target, right_target) -- placeholder.
            let cj_pos = out.len();
            out.push(Instr::ChoiceJump(0, 0));
            // Left branch. `SetIncoming` makes the routed value the arm's
            // incoming (arg-position ops read the `current_incoming`
            // register, not the stack top); `RestoreIncoming` before the
            // merge so later sibling args see the enclosing incoming.
            let left_target = out.len();
            out.push(Instr::SetIncoming);
            let l_pushed = compile_into_td(&l, lang, ast, node, out)?;
            if l_pushed {
                out.push(Instr::PopScope);
            }
            out.push(Instr::RestoreIncoming);
            let left_jump_pos = out.len();
            out.push(Instr::Jump(0)); // placeholder -- skip right branch
            // Right branch.
            let right_target = out.len();
            out.push(Instr::SetIncoming);
            let r_pushed = compile_into_td(&r, lang, ast, node, out)?;
            if r_pushed {
                out.push(Instr::PopScope);
            }
            out.push(Instr::RestoreIncoming);
            let end = out.len();
            // Patch ChoiceJump and the left-branch Jump.
            out[cj_pos] = Instr::ChoiceJump(left_target, right_target);
            out[left_jump_pos] = Instr::Jump(end);
            Ok(false)
        }

        "Loop" => {
            let body = td.get_term("body").expect("Loop term missing 'body' field");
            // Optimization: when the body is a `loop_step` Arr, hoist the
            // body closure evaluation outside the loop. The closure captures
            // the definition-time env, which doesn't change between iterations
            // (only the loop state changes, passed as an argument). This
            // avoids re-creating the closure (and deep-cloning its captured
            // scope via `capture()`) on every iteration. Mirrors the
            // tree-walk optimization at
            // `plugin-engine-exec-tree-walk/src/lib.rs:402-456`.
            if body.tag() == "Op" {
                let op = body.get_str("name").unwrap_or("");
                let source = body.arr_source().unwrap_or(ArrSource::Native);
                let args = body.get_term_list("args");
                if source == ArrSource::Native
                    && matches!(op, "loop_step" | "fold_step")
                    && !args.is_empty()
                {
                    let body_proc = &args[0];
                    // Compile the body closure once, before the loop. This
                    // evaluates the `function`/`closure` child once,
                    // producing a `TaggedValue::Arrow` on the operand stack.
                    let bp_pushed = compile_into_td(body_proc, lang, ast, node, out)?;
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
                    // the step native op. The VM's `loop_step`/`fold_step`
                    // handlers receive the pre-computed closure as args[0]
                    // and the loop pair as `incoming` — identical to the
                    // non-hoisted path, just without the per-iteration
                    // closure creation.
                    out.push(Instr::Load(Rc::from("__loop_closure__")));
                    for arg in &args[1..] {
                        let arg_pushed = compile_into_td(arg, lang, ast, node, out)?;
                        if arg_pushed {
                            out.push(Instr::PopScope);
                        }
                    }
                    out.push(Instr::NativeOp(op.to_string(), args.len()));
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
            let body_pushed = compile_into_td(&body, lang, ast, node, out)?;
            // The tree-walk pops any scope the body pushed after each
            // iteration. In the VM, we emit PopScope after the body code
            // but before LoopCheck, so it runs each iteration.
            if body_pushed {
                out.push(Instr::PopScope);
            }
            out.push(Instr::LoopCheck(loop_back));
            Ok(false)
        }

        "App" => {
            let callee = td
                .get_term("callee")
                .expect("App term missing 'callee' field");
            let args = td.get_term_list("args");
            let callee_pushed = compile_into_td(&callee, lang, ast, node, out)?;
            if callee_pushed {
                out.push(Instr::PopScope);
            }
            for arg in &args {
                let arg_pushed = compile_into_td(arg, lang, ast, node, out)?;
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
        // capabilities as a compile-time error. `eliminate_and_check`
        // operates on the `Term` enum, so the `TermData` is converted back
        // via [`TermData::to_term`] for this one case.
        "Handle" => {
            let term = td.to_term().map_err(|e| LangError::engine_node(e, node))?;
            let eliminated = term.eliminate_handlers();
            compile_into(&eliminated, lang, ast, node, out)
        }

        "Iter" => Err(LangError::NotImplemented),
        "Child" => Err(LangError::NotImplemented),
        "Path" => Err(LangError::NotImplemented),
        "Equiv" => Err(LangError::NotImplemented),
        // HoTT/cell combinators lower to native `cell2_*` ops. The lowering
        // is expressed with [`TermData`] constructors and recompiled, so no
        // `Term` enum value is built.
        "CellId" => {
            let arrow = td
                .get_term("arrow")
                .expect("CellId term missing 'arrow' field");
            compile_into_td(
                &TermData::op("cell2_id", vec![arrow], &ArrSource::Native, None),
                lang,
                ast,
                node,
                out,
            )
        }
        "Src" => {
            let cell = td.get_term("cell").expect("Src term missing 'cell' field");
            compile_into_td(
                &TermData::op("cell2_src", vec![cell], &ArrSource::Native, None),
                lang,
                ast,
                node,
                out,
            )
        }
        "Tgt" => {
            let cell = td.get_term("cell").expect("Tgt term missing 'cell' field");
            compile_into_td(
                &TermData::op("cell2_tgt", vec![cell], &ArrSource::Native, None),
                lang,
                ast,
                node,
                out,
            )
        }
        "CellCompV" => {
            let first = td
                .get_term("first")
                .expect("CellCompV term missing 'first' field");
            let second = td
                .get_term("second")
                .expect("CellCompV term missing 'second' field");
            compile_into_td(
                &TermData::op(
                    "cell2_comp_v",
                    vec![first, second],
                    &ArrSource::Native,
                    None,
                ),
                lang,
                ast,
                node,
                out,
            )
        }
        "Whisker" => {
            let arrow = td
                .get_term("arrow")
                .expect("Whisker term missing 'arrow' field");
            let cell = td
                .get_term("cell")
                .expect("Whisker term missing 'cell' field");
            let op = match td.get_str("side").unwrap_or("") {
                "Left" => "cell2_whisker_left",
                "Right" => "cell2_whisker_right",
                other => {
                    return Err(LangError::engine_node(
                        format!("Whisker: unknown side '{other}'"),
                        node,
                    ));
                }
            };
            compile_into_td(
                &TermData::op(op, vec![arrow, cell], &ArrSource::Native, None),
                lang,
                ast,
                node,
                out,
            )
        }
        "Const" => {
            let value = td
                .get("value")
                .cloned()
                .expect("Const term missing 'value' field");
            compile_into_td(&TermData::arr(value), lang, ast, node, out)
        }
        "RecordProj" => {
            let record = td
                .get_term("record")
                .expect("RecordProj term missing 'record' field");
            let field = td
                .get_str("field")
                .expect("RecordProj term missing 'field' field")
                .to_string();
            compile_into_td(
                &TermData::op(
                    "record_proj",
                    vec![record, TermData::arr(Value::String(field))],
                    &ArrSource::Native,
                    None,
                ),
                lang,
                ast,
                node,
                out,
            )
        }
        "RecordWith" => Err(LangError::NotImplemented),
        "Tag" => Err(LangError::NotImplemented),
        "Univalence" => Err(LangError::NotImplemented),
        "Glue" => Err(LangError::NotImplemented),
        "Fibrant" => Err(LangError::NotImplemented),
        "JElim" => Err(LangError::NotImplemented),
        "Refine" => Err(LangError::NotImplemented),
        "Cobind" => Err(LangError::NotImplemented),
        "Rewrite" => Err(LangError::NotImplemented),
        other => Err(LangError::engine_node(
            format!("compile: unknown Term tag '{other}'"),
            node,
        )),
    }
}

/// Compile an `Arr` combinator. Returns `Ok(true)` if this Arr pushes a scope
/// (`extend_env`), so the caller can emit `PopScope`.
///
/// `args` are the op's sub-schemes as [`TermData`] values (ADR-0103 Phase 6);
/// each is compiled recursively via [`compile_into_td`].
fn compile_arr(
    op: &str,
    args: &[TermData],
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
                let arg_pushed = compile_op_arg(arg, lang, ast, node, out)?;
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
            let arg_pushed = compile_into_td(arg, lang, ast, node, out)?;
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
            // Free-variable capture set: the body subtree's name
            // candidates, computed once here so the VM can filter
            // `scopes.capture()` to the bindings the body can actually
            // reference. `None` (unresolvable subtree) keeps the full env.
            let capture = handl_language::subtree_name_candidates(
                |r| ast.get(r).map(|n| n.payload.clone()),
                AstRef::new(closure.body),
            )
            .map(Rc::new);
            out.push(Instr::Closure {
                params: closure.params,
                rest: closure.rest,
                body: closure.body,
                array_pattern: closure.array_pattern,
                capture,
            });
            return Ok(false);
        }
    }

    for arg in args {
        let arg_pushed = compile_op_arg(arg, lang, ast, node, out)?;
        if arg_pushed {
            out.push(Instr::PopScope);
        }
    }
    match source {
        ArrSource::Native => {
            // `$payload` ops (`arr <op>` surface form): the op name lives
            // in the node's payload at runtime, so it cannot be baked into
            // a `NativeOp`. Emit `ArrDispatch`, which resolves the name and
            // dispatches through the handler-impl → native path (mirrors
            // the tree-walk `eval_arr`/`arr_dispatch`).
            if op == "$payload" {
                out.push(Instr::ArrDispatch(args.len()));
            // Hot-path: emit a typed arithmetic/comparison instruction
            // instead of `NativeOp(op, 2)` for the common binary numeric
            // ops. This avoids the per-op `Vec` allocation and string
            // dispatch in the VM. Results are identical to the `NativeOp`
            // path. Gated on exact arity: the typed instrs pop a fixed
            // operand count, so n-ary applications (`sub(10,3,2)`) must
            // fall through to `NativeOp`, which left-folds via
            // `native_ops` — the shared n-ary convention.
            } else if args.len() == 2 && typed_binary_instr(op).is_some() {
                out.push(typed_binary_instr(op).expect("checked"));
            } else if arith_arity(op) == Some(args.len()) {
                // ADR-0103: migrate the remaining arithmetic/logic ops
                // (`neg`/`and`/`or`/`not`) that currently go through
                // `NativeOp` to the consolidated `Arith` hardware-boundary
                // instruction. Results are identical to the `NativeOp` path.
                out.push(Instr::Arith(arith_instr(op).expect("checked")));
            } else {
                out.push(Instr::NativeOp(op.to_string(), args.len()));
            }
        }
        ArrSource::Foreign(name) => {
            // ADR-0103: emit the `Ffi` hardware-boundary instruction instead
            // of `ForeignOp`. The VM's `Ffi` handler looks up `(module,
            // name)` in its foreign-function registry, falling back to the
            // existing `ModuleLookup` dispatch (treating `name` as the
            // qualified/flat host name) so already-registered host functions
            // keep working. `ForeignOp` is retained as a fallback variant.
            out.push(Instr::Ffi {
                module: "host".to_string(),
                name: name.clone(),
                n: args.len(),
            });
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

/// Compile a single `Arr`/`Op` argument. A bare 0-arg native op that uses
/// `incoming` (`id`, `inc`, `truthy`, ...) in arg position sees the
/// enclosing op's incoming — the region's current incoming — per the
/// tree-walk's arg evaluation (`eval_arr` evaluates each arg with the
/// `Arr`'s incoming). Emitting a bare `NativeOp(op, 0)` there would
/// instead pop the previous arg's pushed result and underflow the
/// enclosing op, so emit [`Instr::PushIncoming`] first — the `NativeOp`
/// then pops that copy and applies the op.
fn compile_op_arg(
    arg: &TermData,
    lang: &Language,
    ast: &Ast,
    node: &AstNode,
    out: &mut Vec<Instr>,
) -> Result<bool, LangError> {
    if arg.tag() == "Op"
        && arg.get_term_list("args").is_empty()
        && arg.get("lit").filter(|v| !v.is_null()).is_none()
        && matches!(arg.arr_source(), Some(ArrSource::Native))
        && crate::vm::op_uses_incoming(arg.get_str("name").unwrap_or(""))
    {
        out.push(Instr::PushIncoming);
        out.push(Instr::NativeOp(
            arg.get_str("name").unwrap_or("").to_string(),
            0,
        ));
        return Ok(false);
    }
    compile_into_td(arg, lang, ast, node, out)
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

/// Map a native op name to an [`ArithOp`] for the ADR-0103 `Arith`
/// hardware-boundary instruction, if the op is one of the arithmetic/logic
/// ops not already covered by [`typed_binary_instr`].
///
/// Returns `Some` for `neg` (unary), `and`/`or` (binary logic), and `not`
/// (unary). These currently go through `NativeOp`; emitting `Arith`
/// instead begins the migration to the target minimal ISA. The common
/// binary numeric ops (`add`/`sub`/`mul`/…) remain on the dedicated typed
/// bytecodes for now (hot-path optimization); they will migrate to `Arith`
/// in a later step. All other native ops fall through to `NativeOp`.
fn arith_instr(op: &str) -> Option<ArithOp> {
    Some(match op {
        "neg" => ArithOp::Neg,
        "and" => ArithOp::And,
        "or" => ArithOp::Or,
        "not" => ArithOp::Not,
        _ => return None,
    })
}

/// The exact operand count an [`ArithOp`] instruction pops — `neg`/`not`
/// are unary, `and`/`or` binary. `None` for ops without an `Arith` form.
/// N-ary applications (`and(t,t,f)`) fall through to `NativeOp`, which
/// folds via `native_ops`.
fn arith_arity(op: &str) -> Option<usize> {
    Some(match op {
        "neg" | "not" => 1,
        "and" | "or" => 2,
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
    /// Whether the params node is an array-destructuring pattern.
    array_pattern: bool,
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
            array_pattern: false,
        }));
    }

    // Pratt-parsed payload shape: `{"children": [param_idx, body_idx]}`.
    // `params`/`body` child positions resolve by named role (ADR-0081);
    // positional 0/1 is the role-less fallback.
    if let Some(children) = payload.get("children").and_then(|c| c.as_array()) {
        let params_role = lang
            .child_role_index(&node.kind, "params")
            .ok()
            .flatten()
            .unwrap_or(0);
        let body_role = lang
            .child_role_index(&node.kind, "body")
            .ok()
            .flatten()
            .unwrap_or(1);
        let param_idx = match children.get(params_role).and_then(|v| v.as_u64()) {
            Some(idx) => idx as usize,
            None => return Ok(None),
        };
        let body_idx = match children.get(body_role).and_then(|v| v.as_u64()) {
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
        let array_pattern = lang.is_array_pattern(param_node);
        return Ok(Some(StaticClosure {
            params: Rc::from(names),
            rest: rest.map(Rc::from),
            body: body_idx,
            array_pattern,
        }));
    }

    // Unrecognized payload shape — fall back to NativeOp.
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use handl_language::{ArrSource, Ast, AstNode, Language};
    use plugin_engine::TermData;
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
        let s = TermData::op("const", vec![], &ArrSource::Native, Some(json!(42).into()))
            .to_term()
            .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::Push(json!(42)), Instr::Return]);
    }

    #[test]
    fn compile_const_no_lit() {
        let s = TermData::op("const", vec![], &ArrSource::Native, None)
            .to_term()
            .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::Push(Value::Null), Instr::Return]);
    }

    #[test]
    fn compile_proj_env_named() {
        let s = TermData::op("proj_env", vec![], &ArrSource::Native, Some("x".into()))
            .to_term()
            .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::Load("x".into()), Instr::Return]);
    }

    #[test]
    fn compile_proj_env_payload() {
        let s = TermData::op(
            "proj_env",
            vec![],
            &ArrSource::Native,
            Some("$payload".into()),
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::LoadPayloadVar, Instr::Return]);
    }

    #[test]
    fn compile_proj_child() {
        let s = TermData::op(
            "proj_child",
            vec![],
            &ArrSource::Native,
            Some(json!(2).into()),
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(instrs, vec![Instr::EvalChild(2), Instr::Return]);
    }

    #[test]
    fn compile_extend_env_named() {
        let s = TermData::op("extend_env", vec![], &ArrSource::Native, Some("x".into()))
            .to_term()
            .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![Instr::Store("x".into()), Instr::PopScope, Instr::Return]
        );
    }

    #[test]
    fn compile_extend_env_payload() {
        let s = TermData::op(
            "extend_env",
            vec![],
            &ArrSource::Native,
            Some("$payload".into()),
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![Instr::StorePayloadName, Instr::PopScope, Instr::Return]
        );
    }

    #[test]
    fn compile_native_op() {
        let s = TermData::op(
            "add",
            vec![
                TermData::op(
                    "proj_child",
                    vec![],
                    &ArrSource::Native,
                    Some(json!(0).into()),
                ),
                TermData::op(
                    "proj_child",
                    vec![],
                    &ArrSource::Native,
                    Some(json!(1).into()),
                ),
            ],
            &ArrSource::Native,
            None,
        )
        .to_term()
        .unwrap();
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
        let s = TermData::op(
            "double",
            vec![TermData::op(
                "proj_child",
                vec![],
                &ArrSource::Native,
                Some(json!(0).into()),
            )],
            &ArrSource::Foreign("double".into()),
            None,
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        // ADR-0103: foreign ops now compile to the `Ffi` hardware-boundary
        // instruction (module "host") instead of `ForeignOp`. The VM's `Ffi`
        // handler falls back to `ModuleLookup` dispatch, so behavior is
        // identical.
        assert_eq!(
            instrs,
            vec![
                Instr::EvalChild(0),
                Instr::Ffi {
                    module: "host".to_string(),
                    name: "double".to_string(),
                    n: 1,
                },
                Instr::Return,
            ]
        );
    }

    // --- Seq compilation tests ---

    #[test]
    fn compile_seq_no_scope() {
        let s = TermData::seq(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(5).into())),
            TermData::op("id", vec![], &ArrSource::Native, None),
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::Push(json!(5)),
                Instr::SetIncoming, // snd's incoming = fst's result
                Instr::NativeOp("id".into(), 0),
                Instr::RestoreIncoming,
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
        let s = TermData::seq(
            TermData::seq(
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(5).into())),
                TermData::op("extend_env", vec![], &ArrSource::Native, Some("x".into())),
            ),
            TermData::op("proj_env", vec![], &ArrSource::Native, Some("x".into())),
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::Push(json!(5)),    // 0: const 5
                Instr::SetIncoming,       // 1: inner Seq snd incoming = 5
                Instr::Store("x".into()), // 2: extend_env "x"
                Instr::RestoreIncoming,   // 3: inner Seq done
                Instr::SetIncoming,       // 4: outer Seq snd incoming
                Instr::Load("x".into()),  // 5: proj_env "x"
                Instr::PopScope,          // 6: pop scope (outer Seq)
                Instr::RestoreIncoming,   // 7: outer Seq done
                Instr::Return,            // 8
            ]
        );
    }

    // --- First compilation tests ---

    #[test]
    fn compile_first() {
        let s = TermData::first(TermData::op("id", vec![], &ArrSource::Native, None))
            .to_term()
            .unwrap();
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
        let s = TermData::choice(
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(1).into())),
            TermData::op("const", vec![], &ArrSource::Native, Some(json!(2).into())),
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::ChoiceJump(1, 5),  // 0: route left/right
                Instr::SetIncoming,       // 1: left arm incoming = routed val
                Instr::Push(json!(1)),    // 2: left branch
                Instr::RestoreIncoming,   // 3
                Instr::Jump(8),           // 4: skip right
                Instr::SetIncoming,       // 5: right arm incoming = routed val
                Instr::Push(json!(2)),    // 6: right branch
                Instr::RestoreIncoming,   // 7
                Instr::Return,            // 8: end
            ]
        );
    }

    // --- Loop compilation tests ---

    #[test]
    fn compile_loop() {
        let s = TermData::loop_feedback(TermData::op("inc_fst", vec![], &ArrSource::Native, None))
            .to_term()
            .unwrap();
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
        let s = TermData::app(
            TermData::op("proj_env", vec![], &ArrSource::Native, Some("f".into())),
            vec![TermData::op(
                "const",
                vec![],
                &ArrSource::Native,
                Some(json!(41).into()),
            )],
        )
        .to_term()
        .unwrap();
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
        let s = TermData::app(
            TermData::op("proj_env", vec![], &ArrSource::Native, Some("f".into())),
            vec![],
        )
        .to_term()
        .unwrap();
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
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::Push(json!(true)),         // 0: const true
                Instr::SetIncoming,               // 1: inner Seq snd incoming
                Instr::NativeOp("tag".into(), 0), // 2: tag
                Instr::RestoreIncoming,           // 3: inner Seq done
                Instr::SetIncoming,               // 4: outer Seq snd incoming = Either
                Instr::ChoiceJump(6, 10),         // 5: route left/right
                Instr::SetIncoming,               // 6: left arm incoming = routed val
                Instr::Push(json!("yes")),        // 7: left (then)
                Instr::RestoreIncoming,           // 8
                Instr::Jump(13),                  // 9: skip right
                Instr::SetIncoming,               // 10: right arm incoming
                Instr::Push(json!("no")),         // 11: right (else)
                Instr::RestoreIncoming,           // 12
                Instr::RestoreIncoming,           // 13: outer Seq done
                Instr::Return,                    // 14: end
            ]
        );
    }

    // --- ADR-0103 Arith compilation ---

    #[test]
    fn compile_arith_and_emits_arith_instr() {
        // `and` currently goes through `NativeOp`; ADR-0103 migrates it to
        // the consolidated `Arith(ArithOp::And)` hardware-boundary
        // instruction.
        let s = TermData::op(
            "and",
            vec![
                TermData::op(
                    "proj_child",
                    vec![],
                    &ArrSource::Native,
                    Some(json!(0).into()),
                ),
                TermData::op(
                    "proj_child",
                    vec![],
                    &ArrSource::Native,
                    Some(json!(1).into()),
                ),
            ],
            &ArrSource::Native,
            None,
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::EvalChild(0),
                Instr::EvalChild(1),
                Instr::Arith(ArithOp::And),
                Instr::Return,
            ]
        );
    }

    #[test]
    fn compile_arith_neg_emits_arith_instr() {
        // `neg` is unary; it migrates to `Arith(ArithOp::Neg)`.
        let s = TermData::op(
            "neg",
            vec![TermData::op(
                "proj_child",
                vec![],
                &ArrSource::Native,
                Some(json!(0).into()),
            )],
            &ArrSource::Native,
            None,
        )
        .to_term()
        .unwrap();
        let instrs = compile(&s, &Language::new(), &dummy_ast(), &dummy_node()).unwrap();
        assert_eq!(
            instrs,
            vec![
                Instr::EvalChild(0),
                Instr::Arith(ArithOp::Neg),
                Instr::Return,
            ]
        );
    }
}

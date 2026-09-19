//! # handl-plugin-engine-exec-bytecode
//!
//! The **bytecode eval engine** for the eval operation (ADR-0044 §4, §7).
//!
//! This engine compiles [`handl_language::Term`] (the v2 arrow-calculus
//! kernel, ADR-0052) data to a bytecode representation and executes it via a
//! stack-based VM. It shares the same `Term` vocabulary as the
//! [`handl_plugin_engine_exec_tree_walk`](handl_plugin_engine_exec_tree_walk)
//! reference interpreter; conformance tests verify both produce **identical
//! results**.
//!
//! ## Compilers erase data dispatch (ADR-0044 §7)
//!
//! The tree-walk pays for per-node scheme lookup (`lang.eval[node.kind]` →
//! `Term`) on every execution. The bytecode engine **compiles this lookup
//! away**: at compile time, every node kind's scheme is resolved to a fixed
//! instruction sequence. The compiled artifact has no hash lookups, no string
//! matching, no `lang.eval` access at execution time.
//!
//! A per-node compilation cache (compile each node's scheme once, cache by
//! [`AstRef`](handl_language::AstRef)) is used to demonstrate this. When
//! [`Instr::EvalChild`] runs, it looks up the child's `AstRef` in the cache;
//! on a miss, it compiles the child's scheme (one-time `lang.eval` lookup)
//! and caches the result. On subsequent executions, the cached instructions
//! are reused — no data dispatch occurs. Full ahead-of-time compilation of
//! the whole AST is a future refinement.
//!
//! ## Bytecode ISA
//!
//! See the [`instr`] module for the full instruction set documentation. The
//! ISA covers all six `Term` combinators (`Arr`, `Seq`, `First`, `Choice`,
//! `Loop`, `App`); see the [`instr`] module's combinator-to-instruction
//! mapping table for details.
//!
//! ## Conventions (matched to the tree-walk)
//!
//! - `proj_child(n)` = nth child of current node: reads
//!   `node.payload["children"][n]` to get the absolute `AstRef`, then
//!   recurses.
//! - `proj_env("$payload")` reads `node.payload` as the variable name.
//! - `extend_env("$payload")` reads `node.payload["name"]` as the bind name.
//! - No-scheme fallback: if a node kind has no eval scheme, returns
//!   `node.payload` directly.
//! - Closure representation: `Value::Object` with
//!   `{"__arrow__": true, "params": [...], "body": <arena_idx>, "env": {...}}`,
//!   created by the `"closure"` native op.
//! - Native ops: `add, sub, mul, div, mod, neg, eq, neq, lt, gt, lte, gte,
//!   and, or, not, tag, id, closure, bind` — identical semantics to the
//!   tree-walk.
//! - Scope: `Vec<HashMap<String, Value>>` scope stack; `extend_env` pushes a
//!   child scope; lookup searches innermost-first. Closure calls swap in the
//!   captured env, bind params, eval body, restore.

#![warn(missing_docs)]

pub mod compile;
pub mod instr;
mod vm;

pub use compile::compile;
pub use instr::{ArithOp, Instr};

use handl_language::{Ast, AstRef, LangError, Language, SchemeVersion, Value};
use handl_plugin_engine_exec::{EvalEngine, EvalRequest};
use handl_plugin_engine_exec_tree_walk::{EvalFrame as TreeWalkEvalFrame, register_home_frame};
use handl_plugin_engine_transform::native::{NativeRewrite, NativeRewriteRegistry, RewriteCtx};
use handl_plugin_engine_transform_rewrite::RewriteEngine;
use handl_value::TaggedValue;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use vm::VmFrame;

/// The bytecode eval engine.
///
/// Compiles [`Term`](handl_language::Term) (the v2 arrow-calculus kernel)
/// data to bytecode and executes via a stack-based VM. Construct with
/// [`BytecodeEngine::new`] and optionally pass a module registry via
/// [`BytecodeEngine::with_modules`].
///
/// The engine checks `lang.scheme_version` against [`SchemeVersion::v2()`]
/// (exact equality) and rejects mismatched versions.
#[derive(Clone)]
pub struct BytecodeEngine {
    /// The module registry for `Foreign` dispatch (may be `None`).
    modules: Option<Rc<dyn handl_platform::ModuleLookup>>,
    /// The import resolver for `import "path"` statements (Phase 9.5).
    import_resolver: Option<Rc<dyn handl_plugin_engine_exec::ImportResolver>>,
    /// Global variables injected by the host (e.g. `self` -> language value).
    globals: HashMap<String, Value>,
    /// Run the `lower` transform pass before evaluation.
    lowering: bool,
}

impl BytecodeEngine {
    /// Create a new bytecode engine with no module registry.
    pub fn new() -> Self {
        Self {
            modules: None,
            import_resolver: None,
            globals: HashMap::new(),
            lowering: false,
        }
    }

    /// Create a bytecode engine with a module registry attached.
    ///
    /// The module registry is used for
    /// [`Term::Arr`](handl_language::Term::Arr) with
    /// [`ArrSource::Foreign`](handl_language::ArrSource::Foreign) dispatch.
    pub fn with_modules(modules: Rc<dyn handl_platform::ModuleLookup>) -> Self {
        Self {
            modules: Some(modules),
            import_resolver: None,
            globals: HashMap::new(),
            lowering: false,
        }
    }

    /// Opt in to the `lower` transform pass before evaluation.
    pub fn with_lowering(mut self) -> Self {
        self.lowering = true;
        self
    }

    /// Inject a global variable into the root scope (builder style).
    /// ADR-0041 Phase 3: `self` is an ambient variable, not a function call.
    pub fn with_global(mut self, name: &str, value: Value) -> Self {
        self.globals.insert(name.to_string(), value);
        self
    }
}

impl Default for BytecodeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BytecodeEngine {
    /// Shared core of [`EvalEngine::eval`] and [`EvalEngine::eval_shared`]:
    /// install the `ast`/`lang`/`modules` in the home-frame thread-locals,
    /// compile/execute the arena root, and return a `Value`.
    fn eval_ast_rc(&self, lang: Rc<Language>, ast: Rc<Ast>) -> Result<Value, LangError> {
        self.eval_ast_rc_with_args(lang, ast, &[])
    }

    /// Shared core of [`EvalEngine::eval_shared_with_call_args`] (ADR-0106):
    /// like [`eval_ast_rc`](Self::eval_ast_rc), but merges `call_args`
    /// directly into the call frame's `ScopeStack::globals` map — without
    /// cloning the engine via [`with_global`](Self::with_global).
    ///
    /// The call args live in the same `globals` field as host-provided
    /// globals (e.g. `self`), so they are resolvable by name lookup but
    /// excluded from closure capture ("no env capture") and never leak
    /// between calls.
    fn eval_ast_rc_with_args(
        &self,
        lang: Rc<Language>,
        ast: Rc<Ast>,
        call_args: &[(String, Value)],
    ) -> Result<Value, LangError> {
        if ast.is_empty() {
            return Ok(Value::Null);
        }
        // Term version check -- the engine supports v2 only.
        if !SchemeVersion::v2().supports(&lang.scheme_version) {
            return Err(LangError::engine(format!(
                "unsupported scheme version: {} (engine supports v2)",
                lang.scheme_version
            )));
        }

        // Optional lowering pass.  The `lower` pass canonicalizes backend-
        // specific AST shapes (e.g. object entry key kinds) before the VM
        // compiles and executes the program.  It is disabled by default to
        // preserve existing behavior; opt in with `with_lowering`.
        let (lowered, root_ref) = if self.lowering {
            let root = AstRef::new(ast.len() - 1);
            let engine = RewriteEngine::new()
                .with_native_registry(Arc::new(lower_natives()) as Arc<dyn NativeRewrite>);
            let out = engine.run_pass(&lang, &ast, root, "lower")?;
            (Some(out.ast), out.root)
        } else {
            (None, AstRef::new(ast.len() - 1))
        };
        let ast_ref: &Ast = lowered.as_ref().unwrap_or(&ast);

        // Host-provided globals (e.g. `self`, `__arg_N`) merged with the
        // per-call argument bindings. Both live in the `ScopeStack::globals`
        // map so they are resolvable via lookup but not captured into
        // closures. Per-call args are inserted after engine globals so they
        // shadow same-named ambient globals for this eval only.
        let globals_rc: Rc<FxHashMap<Rc<str>, TaggedValue>> = Rc::new({
            let mut globals: FxHashMap<Rc<str>, TaggedValue> = self
                .globals
                .iter()
                .map(|(k, v)| (Rc::from(k.as_str()), TaggedValue::from(v.clone())))
                .collect();
            for (name, val) in call_args {
                globals.insert(Rc::from(name.as_str()), TaggedValue::from(val.clone()));
            }
            globals
        });

        let eval_frame = TreeWalkEvalFrame {
            ast: Rc::clone(&ast),
            lang: Rc::clone(&lang),
            modules: self.modules.clone(),
            globals: Some(Rc::clone(&globals_rc)),
            import_resolver: self.import_resolver.clone(),
        };

        eval_frame.enter(|| {
            let mut frame = VmFrame::new(
                ast_ref,
                &lang,
                self.modules.as_ref(),
                self.import_resolver.as_ref(),
            );
            frame.set_globals(Rc::clone(&globals_rc));
            frame.exec_node(root_ref)
        })
    }
}

impl EvalEngine for BytecodeEngine {
    fn eval(&self, lang: &Language, req: &EvalRequest) -> Result<Value, LangError> {
        self.eval_ast_rc(Rc::new(lang.clone()), Rc::new(req.ast.clone()))
    }

    /// Borrow the caller's `Rc<Ast>` arena instead of deep-cloning it into
    /// an [`EvalRequest`] — the VM only reads the arena, so per-call
    /// re-evaluation of a cached AST (e.g. `plugin-lib-handl`'s
    /// `HandlFunction` bridge) costs zero clones.
    fn eval_shared(&self, lang: &Language, ast: &Rc<Ast>) -> Result<Value, LangError> {
        self.eval_ast_rc(Rc::new(lang.clone()), Rc::clone(ast))
    }

    /// Bytecode override (ADR-0106): bind `args` directly into the call
    /// frame's `ScopeStack::globals` alongside host-provided globals, without
    /// cloning the engine or persisting the args into [`BytecodeEngine::globals`].
    ///
    /// This mirrors the tree-walk engine's zero-clone path: args are
    /// resolvable by name lookup, excluded from closure capture, and scoped to
    /// this single `eval_shared` call.
    fn eval_shared_with_call_args(
        &self,
        lang: &Language,
        ast: &Rc<Ast>,
        args: &[(String, Value)],
    ) -> Result<Value, LangError> {
        self.eval_ast_rc_with_args(Rc::new(lang.clone()), Rc::clone(ast), args)
    }

    fn box_clone(&self) -> Box<dyn EvalEngine> {
        Box::new(self.clone())
    }

    fn with_modules(&self, modules: Rc<dyn handl_platform::ModuleLookup>) -> Box<dyn EvalEngine> {
        let mut engine = self.clone();
        engine.modules = Some(modules);
        Box::new(engine)
    }

    fn with_global(&self, name: &str, val: Value) -> Box<dyn EvalEngine> {
        Box::new(self.clone().with_global(name, val))
    }

    fn with_import_resolver(
        &self,
        resolver: Rc<dyn handl_plugin_engine_exec::ImportResolver>,
    ) -> Box<dyn EvalEngine> {
        let mut engine = self.clone();
        engine.import_resolver = Some(resolver);
        Box::new(engine)
    }

    /// Seal a module's exported bindings before they cross into the
    /// importer's scope (Phase 9.5 `import "path"`).
    ///
    /// Registers the module's AST as a home frame and stamps any unhomed
    /// arrow values so a later engine can route application back here.
    fn seal_module_value(
        &self,
        ast: &Ast,
        lang: &Language,
        value: Value,
    ) -> Result<Value, LangError> {
        if !contains_unhomed_arrow(&value) {
            return Ok(value);
        }
        let globals: Option<HashMap<String, Value>> = if self.globals.is_empty() {
            None
        } else {
            Some(self.globals.clone())
        };
        let frame_id = register_home_frame(
            Rc::new(ast.clone()),
            Rc::new(lang.clone()),
            self.modules.clone(),
            globals,
            self.import_resolver.clone(),
        );
        let mut value = value;
        stamp_home(&mut value, frame_id);
        Ok(value)
    }
}

/// Alias for the bytecode eval engine (ADR-0052 big-bang cutover).
///
/// The v2 engine is now canonical under the name [`BytecodeEngine`]. This
/// alias preserves the historical `BytecodeEngineV2` name for external
/// consumers that have not yet been updated.
pub type BytecodeEngineV2 = BytecodeEngine;

// ---------------------------------------------------------------------------
// CompileEngine / CompiledArtifact impls (ADR-0047 §3)
// ---------------------------------------------------------------------------

/// A compiled bytecode artifact. Holds the [`Language`] and [`Ast`] it was
/// compiled from. Execution lazily compiles each node's scheme (via a
/// per-node cache) at run time, because [`Instr::EvalChild`] resolves child
/// nodes using `lang` + `ast`.
#[derive(Clone)]
pub struct BytecodeArtifact {
    /// The language the program was compiled under.
    lang: Language,
    /// The AST the program was compiled from.
    ast: Ast,
}

impl handl_runtime::CompiledArtifact for BytecodeArtifact {
    fn run(
        &self,
        modules: Option<Rc<dyn handl_platform::ModuleLookup>>,
    ) -> Result<Value, LangError> {
        if self.ast.is_empty() {
            return Ok(Value::Null);
        }
        let root_ref = AstRef::new(self.ast.len() - 1);
        // Compiled artifacts do not yet carry an import resolver; if the
        // compiled program uses `import`, the op will report "no import
        // resolver configured".
        let mut frame = VmFrame::new(&self.ast, &self.lang, modules.as_ref(), None);
        frame.exec_node(root_ref)
    }

    fn box_clone(&self) -> Box<dyn handl_runtime::CompiledArtifact> {
        Box::new(self.clone())
    }
}

/// The bytecode compile engine. Produces [`BytecodeArtifact`] impls from an
/// [`Ast`] (ADR-0047 §3). Registered under the target name `"bytecode"`.
#[derive(Clone, Default)]
pub struct BytecodeCompileEngine;

impl BytecodeCompileEngine {
    /// Create a new bytecode compile engine.
    pub fn new() -> Self {
        Self
    }
}

impl handl_runtime::CompileEngine for BytecodeCompileEngine {
    fn compile(
        &self,
        lang: &Language,
        ast: &Ast,
        _target: &str,
    ) -> Result<Box<dyn handl_runtime::CompiledArtifact>, LangError> {
        if !SchemeVersion::v2().supports(&lang.scheme_version) {
            return Err(LangError::engine(format!(
                "compile: unsupported scheme version: {} (engine supports v2)",
                lang.scheme_version
            )));
        }
        // Child nodes are compiled lazily at execution time via the VM's
        // per-node cache, so the artifact only needs to carry `lang` + `ast`.
        Ok(Box::new(BytecodeArtifact {
            lang: lang.clone(),
            ast: ast.clone(),
        }))
    }

    fn box_clone(&self) -> Box<dyn handl_runtime::CompileEngine> {
        Box::new(self.clone())
    }
}

/// Native rewrites for the `lower` pass.
///
/// These are backend-specific normalizations that the canonical transform
/// engine cannot express without an escape hatch (e.g. copying a child
/// payload into a newly-created leaf).  They are registered only when the
/// `BytecodeEngine` opts in to `with_lowering`.
pub(crate) fn lower_natives() -> NativeRewriteRegistry {
    let mut reg = NativeRewriteRegistry::new();
    reg.register(
        "lower.object_key",
        Box::new(|ctx: &mut RewriteCtx<'_>| {
            // `key`/`value` child positions resolve by named role
            // (ADR-0081); positional 0/1 is the role-less fallback.
            let key_role = ctx
                .lang()
                .child_role_index(&ctx.node().kind, "key")
                .ok()
                .flatten()
                .unwrap_or(0);
            let val_role = ctx
                .lang()
                .child_role_index(&ctx.node().kind, "value")
                .ok()
                .flatten()
                .unwrap_or(1);
            let key_idx = ctx.child_ref(key_role).ok_or_else(|| {
                LangError::engine("lower.object_key: entry missing key child".to_string())
            })?;
            let key_node = ctx.ast().get(AstRef::new(key_idx)).ok_or_else(|| {
                LangError::engine("lower.object_key: key child out of bounds".to_string())
            })?;
            if key_node.kind != "string" {
                return Ok(ctx.pass_through());
            }
            let val_idx = ctx.remap_child(val_role).ok_or_else(|| {
                LangError::engine("lower.object_key: entry missing value child".to_string())
            })?;
            let new_key = ctx.push_leaf("var", key_node.payload.clone());
            Ok(AstRef::new(ctx.push_node("entry", vec![new_key, val_idx])))
        }),
    );
    reg
}

/// Whether `value` contains an arrow value (`{"__arrow__": true, ...}`)
/// that carries no `__home__` frame id — i.e. an arrow whose `body` index
/// refers to the ambient eval's AST and must be stamped before it crosses
/// a runtime boundary.
fn contains_unhomed_arrow(value: &Value) -> bool {
    match value {
        Value::Array(a) => a.iter().any(contains_unhomed_arrow),
        Value::Object(o) => {
            if o.get("__arrow__").and_then(|v| v.as_bool()) == Some(true) {
                return o.get("__home__").and_then(|v| v.as_u64()).is_none()
                    || o.values().any(contains_unhomed_arrow);
            }
            o.values().any(contains_unhomed_arrow)
        }
        _ => false,
    }
}

/// Stamp every arrow value inside `value` that lacks a `__home__` frame id
/// with `frame_id` — recursively, including arrows nested inside a
/// closure's captured `env`. Arrows that already carry a `__home__` keep it.
fn stamp_home(value: &mut Value, frame_id: u64) {
    match value {
        Value::Array(a) => {
            for v in a.iter_mut() {
                stamp_home(v, frame_id);
            }
        }
        Value::Object(o) => {
            if o.get("__arrow__").and_then(|v| v.as_bool()) == Some(true)
                && o.get("__home__").and_then(|v| v.as_u64()).is_none()
            {
                o.insert("__home__".to_string(), Value::from(frame_id));
            }
            for v in o.values_mut() {
                stamp_home(v, frame_id);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use handl_language::{ArrSource, AstNode, EvalSchemeDef, Language, Value};
    use handl_platform::{EvalContext, Function, Module, ModuleLookup};
    use handl_plugin_engine_exec::EvalRequest;
    use handl_plugin_module::ModuleRegistry;
    use plugin_engine::TermData;
    use serde_json::json;

    /// Build a simple AST for testing.
    fn make_ast(nodes: Vec<AstNode>) -> handl_language::Ast {
        let mut ast = handl_language::Ast::new();
        for n in nodes {
            ast.push(n);
        }
        ast
    }

    /// A host function that doubles its integer argument.
    struct DoubleHostFn;

    impl Function for DoubleHostFn {
        fn call(
            &self,
            args: &[Value],
            _ctx: &mut EvalContext,
        ) -> Result<Value, handl_platform::EvalError> {
            let n = args[0].as_i64().unwrap();
            Ok(Value::from(n * 2))
        }
    }

    /// A module with "double" registered.
    struct ConformanceModule;

    impl Module for ConformanceModule {
        fn name(&self) -> &str {
            "conformance"
        }
        fn function(&self, name: &str) -> Option<&dyn Function> {
            if name == "double" {
                Some(&DoubleHostFn)
            } else {
                None
            }
        }
    }

    /// Build a module registry with the `double` function registered.
    fn conformance_registry() -> Rc<dyn ModuleLookup> {
        Rc::new(ModuleRegistry::new().with(ConformanceModule))
    }

    // --- BytecodeEngine tests ---

    #[test]
    fn engine_empty_ast_returns_null() {
        let engine = BytecodeEngine::new();
        let lang = Language::new();
        let req = EvalRequest::new(handl_language::Ast::new());
        let result = engine.eval(&lang, &req).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn engine_no_scheme_returns_payload() {
        let engine = BytecodeEngine::new();
        let lang = Language::new();
        let ast = make_ast(vec![AstNode::new("lit").with_payload(json!(42))]);
        let req = EvalRequest::new(ast);
        let result = engine.eval(&lang, &req).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn engine_lit_scheme() {
        let engine = BytecodeEngine::new();
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.const",
                "const",
                TermData::op("const", vec![], &ArrSource::Native, Some(json!(99).into()))
                    .to_term()
                    .unwrap(),
            ))
            .unwrap();
        let ast = make_ast(vec![AstNode::new("const")]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(99));
    }

    #[test]
    fn engine_add() {
        let engine = BytecodeEngine::new();
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
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(5));
    }

    #[test]
    fn engine_let_binding() {
        let engine = BytecodeEngine::new();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.let",
                "let",
                TermData::seq(
                    TermData::seq(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(0u64).into()),
                        ),
                        TermData::op(
                            "extend_env",
                            vec![],
                            &ArrSource::Native,
                            Some("$payload".into()),
                        ),
                    ),
                    TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(1u64).into()),
                    ),
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(42)),
            AstNode::new("var").with_payload(json!("x")),
            AstNode::new("let").with_payload(json!({"name": "x", "children": [0, 1]})),
        ]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(42));
    }

    #[test]
    fn engine_if_true() {
        let engine = BytecodeEngine::new();
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
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(1));
    }

    #[test]
    fn engine_if_false() {
        let engine = BytecodeEngine::new();
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
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(2));
    }

    #[test]
    fn engine_foreign() {
        let engine = BytecodeEngine::with_modules(conformance_registry());
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.dbl",
                "dbl",
                TermData::op(
                    "double",
                    vec![TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(0u64).into()),
                    )],
                    &ArrSource::Foreign("double".into()),
                    None,
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(21)),
            AstNode::new("dbl").with_payload(json!({"children": [0]})),
        ]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(42));
    }

    #[test]
    fn engine_foreign_no_host_errors() {
        let engine = BytecodeEngine::new();
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.dbl",
                "dbl",
                TermData::op(
                    "double",
                    vec![TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(0u64).into()),
                    )],
                    &ArrSource::Foreign("double".into()),
                    None,
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(21)),
            AstNode::new("dbl").with_payload(json!({"children": [0]})),
        ]);
        let req = EvalRequest::new(ast);
        let err = engine.eval(&lang, &req).unwrap_err();
        assert!(matches!(err, LangError::Engine { .. }));
    }

    #[test]
    fn engine_nested_let_add() {
        let engine = BytecodeEngine::new();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.let",
                "let",
                TermData::seq(
                    TermData::seq(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(0u64).into()),
                        ),
                        TermData::op(
                            "extend_env",
                            vec![],
                            &ArrSource::Native,
                            Some("$payload".into()),
                        ),
                    ),
                    TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(1u64).into()),
                    ),
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
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
            AstNode::new("lit").with_payload(json!(3)),
            AstNode::new("lit").with_payload(json!(4)),
            AstNode::new("var").with_payload(json!("x")),
            AstNode::new("var").with_payload(json!("y")),
            AstNode::new("binary.add").with_payload(json!({"children": [2, 3]})),
            AstNode::new("let").with_payload(json!({"name": "y", "children": [1, 4]})),
            AstNode::new("let").with_payload(json!({"name": "x", "children": [0, 5]})),
        ]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(7));
    }

    #[test]
    fn engine_version_mismatch_errors() {
        let engine = BytecodeEngine::new();
        let mut lang = Language::new();
        lang.scheme_version = SchemeVersion::new("v3");
        let ast = make_ast(vec![AstNode::new("lit").with_payload(json!(42))]);
        let req = EvalRequest::new(ast);
        let err = engine.eval(&lang, &req).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported scheme version"));
    }

    #[test]
    fn engine_default_constructor() {
        let engine = BytecodeEngine::default();
        let lang = Language::new();
        let ast = make_ast(vec![AstNode::new("lit").with_payload(json!(0))]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(0));
    }

    #[test]
    fn engine_trait_dispatch_via_dyn() {
        let engine: Box<dyn EvalEngine> = Box::new(BytecodeEngine::new());
        let lang = Language::new();
        let ast = make_ast(vec![AstNode::new("lit").with_payload(json!(99))]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(99));
    }

    #[test]
    fn engine_closure_call() {
        let engine = BytecodeEngine::new();
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.fn",
                "function",
                TermData::op("closure", vec![], &ArrSource::Native, None)
                    .to_term()
                    .unwrap(),
            ))
            .unwrap();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.call",
                "call",
                TermData::app(
                    TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(0u64).into()),
                    ),
                    vec![TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(1u64).into()),
                    )],
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("var").with_payload(json!("x")),
            AstNode::new("lit").with_payload(json!(1)),
            AstNode::new("binary.add").with_payload(json!({"children": [0, 1]})),
            AstNode::new("function").with_payload(json!({"params": ["x"], "body": 2})),
            AstNode::new("lit").with_payload(json!(5)),
            AstNode::new("call").with_payload(json!({"children": [3, 4]})),
        ]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(6));
    }

    #[test]
    fn engine_closure_captures_env() {
        let engine = BytecodeEngine::new();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.let",
                "let",
                TermData::seq(
                    TermData::seq(
                        TermData::op(
                            "proj_child",
                            vec![],
                            &ArrSource::Native,
                            Some(json!(0u64).into()),
                        ),
                        TermData::op(
                            "extend_env",
                            vec![],
                            &ArrSource::Native,
                            Some("$payload".into()),
                        ),
                    ),
                    TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(1u64).into()),
                    ),
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.fn",
                "function",
                TermData::op("closure", vec![], &ArrSource::Native, None)
                    .to_term()
                    .unwrap(),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.call",
                "call",
                TermData::app(
                    TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(0u64).into()),
                    ),
                    vec![TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(1u64).into()),
                    )],
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        // let x = 10 in let f = fn(y) { x + y } in f(5) → 15
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(10)),
            AstNode::new("var").with_payload(json!("x")),
            AstNode::new("var").with_payload(json!("y")),
            AstNode::new("binary.add").with_payload(json!({"children": [1, 2]})),
            AstNode::new("function").with_payload(json!({"params": ["y"], "body": 3})),
            AstNode::new("lit").with_payload(json!(5)),
            AstNode::new("call").with_payload(json!({"children": [4, 5]})),
            AstNode::new("let").with_payload(json!({"name": "f", "children": [4, 6]})),
            AstNode::new("let").with_payload(json!({"name": "x", "children": [0, 7]})),
        ]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(15));
    }

    #[test]
    fn engine_foreign_host_present_but_fn_not_found_errors() {
        // Module registry is registered but doesn't have the requested function.
        let engine = BytecodeEngine::with_modules(conformance_registry());
        let mut lang = Language::new();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.triple",
                "triple",
                TermData::op(
                    "triple",
                    vec![TermData::op(
                        "proj_child",
                        vec![],
                        &ArrSource::Native,
                        Some(json!(0u64).into()),
                    )],
                    &ArrSource::Foreign("triple".into()),
                    None,
                )
                .to_term()
                .unwrap(),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(21)),
            AstNode::new("triple").with_payload(json!({"children": [0]})),
        ]);
        let req = EvalRequest::new(ast);
        let err = engine.eval(&lang, &req).unwrap_err();
        let msg = format!("{}", err);
        // ADR-0103: foreign ops now compile to `Ffi`, so the not-found
        // error is reported as "unregistered ffi" (the `Ffi` handler's
        // fallback message) rather than "unregistered foreign".
        assert!(msg.contains("unregistered ffi"));
    }

    #[test]
    fn engine_bind() {
        // `x = 5 in x + 1` → bind { name: "x", children: [lit(5), add(x, 1)] }
        // The bind node's scheme is Arr { op: "bind", source: Native }.
        let engine = BytecodeEngine::new();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.bind",
                "bind",
                TermData::op("bind", vec![], &ArrSource::Native, None)
                    .to_term()
                    .unwrap(),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(5)),   // 0
            AstNode::new("lit").with_payload(json!(1)),   // 1
            AstNode::new("var").with_payload(json!("x")), // 2
            AstNode::new("binary.add").with_payload(json!({"children": [2, 1]})), // 3
            AstNode::new("bind").with_payload(json!({"name": "x", "children": [0, 3]})), // 4
        ]);
        let req = EvalRequest::new(ast);
        assert_eq!(engine.eval(&lang, &req).unwrap(), json!(6));
    }

    #[test]
    fn lower_object_key_bypasses_string_kind_heuristic() {
        use handl_language::{
            AstShapeDef, ChildRoleSpec, TransformCondition, TransformDef, TransformRule,
        };

        let mut lang = Language::new();
        lang.ast
            .merge(AstShapeDef::new("ast.entry", "entry").with_children(vec![
                ChildRoleSpec {
                    role: "key".into(),
                    variadic: false,
                },
                ChildRoleSpec {
                    role: "value".into(),
                    variadic: false,
                },
            ]))
            .unwrap();
        lang.transforms
            .merge(
                TransformDef::new(
                    "transform.lower.entry",
                    "lower",
                    "entry",
                    TransformRule::Native("lower.object_key".into()),
                )
                .with_when(TransformCondition::KindAt {
                    index: None,
                    role: Some("key".into()),
                    kind: "string".into(),
                }),
            )
            .unwrap();

        let ast = make_ast(vec![
            AstNode::new("string").with_payload(json!("x")),
            AstNode::new("value").with_payload(json!({"value": 1})),
            AstNode::new("entry").with_payload(json!({"children": [0, 1]})),
        ]);
        let root = AstRef::new(ast.len() - 1);
        let engine = RewriteEngine::new()
            .with_native_registry(Arc::new(super::lower_natives()) as Arc<dyn NativeRewrite>);
        let out = engine.run_pass(&lang, &ast, root, "lower").unwrap();

        let out_entry = out.ast.at(out.root).unwrap();
        let children = out_entry
            .payload
            .get("children")
            .unwrap()
            .as_array()
            .unwrap();
        let key_idx = children[0].as_u64().unwrap() as usize;
        let key_node = out.ast.at(AstRef::new(key_idx)).unwrap();
        assert_eq!(key_node.kind, "var");
        assert_eq!(key_node.payload, json!("x"));
    }
}

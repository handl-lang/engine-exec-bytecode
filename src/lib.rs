//! # handl-plugin-engine-exec-bytecode
//!
//! The **bytecode eval engine** for the eval operation (ADR-0044 §4, §7).
//!
//! This engine compiles [`handl_language::Scheme`] (the v2 arrow-calculus
//! kernel, ADR-0052) data to a bytecode representation and executes it via a
//! stack-based VM. It shares the same `Scheme` vocabulary as the
//! [`handl_plugin_engine_exec_tree_walk`](handl_plugin_engine_exec_tree_walk)
//! reference interpreter; conformance tests verify both produce **identical
//! results**.
//!
//! ## Compilers erase data dispatch (ADR-0044 §7)
//!
//! The tree-walk pays for per-node scheme lookup (`lang.eval[node.kind]` →
//! `Scheme`) on every execution. The bytecode engine **compiles this lookup
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
//! ISA covers all six `Scheme` combinators (`Arr`, `Seq`, `First`, `Choice`,
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
pub use instr::Instr;

use handl_language::{Ast, AstRef, LangError, Language, SchemeVersion, Value};
use handl_plugin_engine_exec::{EvalEngine, EvalRequest};
use handl_value::TaggedValue;
use rustc_hash::FxHashMap;
use std::rc::Rc;

use vm::VmFrame;

/// The bytecode eval engine.
///
/// Compiles [`Scheme`](handl_language::Scheme) (the v2 arrow-calculus kernel)
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
    globals: FxHashMap<Rc<str>, TaggedValue>,
}

impl BytecodeEngine {
    /// Create a new bytecode engine with no module registry.
    pub fn new() -> Self {
        Self {
            modules: None,
            import_resolver: None,
            globals: FxHashMap::default(),
        }
    }

    /// Create a bytecode engine with a module registry attached.
    ///
    /// The module registry is used for
    /// [`Scheme::Arr`](handl_language::Scheme::Arr) with
    /// [`ArrSource::Foreign`](handl_language::ArrSource::Foreign) dispatch.
    pub fn with_modules(modules: Rc<dyn handl_platform::ModuleLookup>) -> Self {
        Self {
            modules: Some(modules),
            import_resolver: None,
            globals: FxHashMap::default(),
        }
    }

    /// Inject a global variable into the root scope (builder style).
    /// ADR-0041 Phase 3: `self` is an ambient variable, not a function call.
    pub fn with_global(mut self, name: &str, value: Value) -> Self {
        self.globals
            .insert(Rc::from(name), TaggedValue::from(value));
        self
    }
}

impl Default for BytecodeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalEngine for BytecodeEngine {
    fn eval(&self, lang: &Language, req: &EvalRequest) -> Result<Value, LangError> {
        let ast = &req.ast;
        if ast.is_empty() {
            return Ok(Value::Null);
        }
        // Scheme version check -- the engine supports v2 only.
        if !SchemeVersion::v2().supports(&lang.scheme_version) {
            return Err(LangError::engine(format!(
                "unsupported scheme version: {} (engine supports v2)",
                lang.scheme_version
            )));
        }
        let root_ref = AstRef::new(ast.len() - 1);
        let mut frame = VmFrame::new(
            ast,
            lang,
            self.modules.as_ref(),
            self.import_resolver.as_ref(),
        );
        for (name, val) in &self.globals {
            frame.define_global(name, val.clone());
        }
        frame.exec_node(root_ref)
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

#[cfg(test)]
mod tests {
    use super::*;
    use handl_language::{AstNode, EvalSchemeDef, Language, Scheme, Value};
    use handl_platform::{EvalContext, Function, Module, ModuleLookup};
    use handl_plugin_engine_exec::EvalRequest;
    use handl_plugin_module::ModuleRegistry;
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
                Scheme::arr_native_lit("const", vec![], json!(99)),
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
                Scheme::arr_native_lit("proj_env", vec![], "$payload"),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.let",
                "let",
                Scheme::seq(
                    Scheme::seq(
                        Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                        Scheme::arr_native_lit("extend_env", vec![], "$payload"),
                    ),
                    Scheme::arr_native_lit("proj_child", vec![], json!(1u64)),
                ),
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
                Scheme::arr_foreign(
                    "double",
                    vec![Scheme::arr_native_lit("proj_child", vec![], json!(0u64))],
                ),
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
                Scheme::arr_foreign(
                    "double",
                    vec![Scheme::arr_native_lit("proj_child", vec![], json!(0u64))],
                ),
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
                Scheme::arr_native_lit("proj_env", vec![], "$payload"),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.let",
                "let",
                Scheme::seq(
                    Scheme::seq(
                        Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                        Scheme::arr_native_lit("extend_env", vec![], "$payload"),
                    ),
                    Scheme::arr_native_lit("proj_child", vec![], json!(1u64)),
                ),
            ))
            .unwrap();
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
                Scheme::arr_native("closure", vec![]),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.var",
                "var",
                Scheme::arr_native_lit("proj_env", vec![], "$payload"),
            ))
            .unwrap();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.call",
                "call",
                Scheme::app(
                    Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                    vec![Scheme::arr_native_lit("proj_child", vec![], json!(1u64))],
                ),
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
                Scheme::arr_native_lit("proj_env", vec![], "$payload"),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.let",
                "let",
                Scheme::seq(
                    Scheme::seq(
                        Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                        Scheme::arr_native_lit("extend_env", vec![], "$payload"),
                    ),
                    Scheme::arr_native_lit("proj_child", vec![], json!(1u64)),
                ),
            ))
            .unwrap();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.fn",
                "function",
                Scheme::arr_native("closure", vec![]),
            ))
            .unwrap();
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.call",
                "call",
                Scheme::app(
                    Scheme::arr_native_lit("proj_child", vec![], json!(0u64)),
                    vec![Scheme::arr_native_lit("proj_child", vec![], json!(1u64))],
                ),
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
                Scheme::arr_foreign(
                    "triple",
                    vec![Scheme::arr_native_lit("proj_child", vec![], json!(0u64))],
                ),
            ))
            .unwrap();
        let ast = make_ast(vec![
            AstNode::new("lit").with_payload(json!(21)),
            AstNode::new("triple").with_payload(json!({"children": [0]})),
        ]);
        let req = EvalRequest::new(ast);
        let err = engine.eval(&lang, &req).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unregistered foreign"));
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
                Scheme::arr_native_lit("proj_env", vec![], "$payload"),
            ))
            .unwrap();
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
        lang.eval
            .merge(EvalSchemeDef::new(
                "e.bind",
                "bind",
                Scheme::arr_native("bind", vec![]),
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
}

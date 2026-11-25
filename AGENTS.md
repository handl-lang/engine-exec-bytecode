# exec-bytecode

`hnl-exec-bytecode` — the bytecode eval engine for the eval operation:
compiles node schemes to bytecode and executes them on a small VM.

## Layout

| path | role |
| --- | --- |
| `src/{compile,instr,vm}.rs` | scheme→bytecode compiler, instruction set, VM |
| `tests/` | engine tests |
| `xtask/` | vendored gate crate (`fmt`, `clippy`, `test`, `doc`, `gate`) |
| `docs/` | engine docs |
| `workflows/gate.yaml` | per-push gate — `handl-gates` rust campaign via bupha |

## Contract

- A concrete `ExecEngine`/`EvalEngine` implementation over the
  `hnl-exec` abstraction; checked against `exec-tree-walk` (the
  reference oracle).
- Deps resolve from git — a plain clone test-runs as-is.

## Verification

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
```

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->

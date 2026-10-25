# engine-exec-bytecode

The bytecode eval engine: compile `Term` to bytecode, run it on a stack VM.

## What it is

Compiles `Term` (the v2 arrow-calculus kernel, ADR-0052) to a bytecode
representation and executes it on a stack-based VM. It shares the `Term`
vocabulary with `engine-exec-tree-walk`, and conformance tests require both to
produce **identical results** (ADR-0044 §4, §7).

## Modules
- `vm` — the stack machine: frames, resumable `perform`, accumulated handlers (ADR-0158)
- `compile` — `Term` → bytecode, erasing per-node data dispatch


## Start here

- **Reference** — [public API](/engine-exec-bytecode/reference/)
- **How-to** — [task recipes](/engine-exec-bytecode/how-to/)
- **Explanation** — [design and rationale](/engine-exec-bytecode/explanation/)

No tutorials: this repo is a building block used from Rust. End-to-end
walkthroughs live with the product that composes it — see the
[Handl platform](/handl/).

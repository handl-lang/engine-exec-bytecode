# engine-exec-bytecode

`hnl-engine-exec-bytecode` — The **bytecode eval engine** for the eval operation (ADR-0044 §4, §7).

## Development

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
```

`ci.sh` is the gate (fmt → clippy → test → doc). Deps resolve from git, so no
sibling checkouts are needed; a plain clone test-runs as-is.

## License

MIT.

## Documentation

Published site (Diátaxis): <https://handl-lang.github.io/docs/engine-exec-bytecode/>

- [Reference](https://handl-lang.github.io/docs/engine-exec-bytecode/reference/) — public API surface
- [How-to](https://handl-lang.github.io/docs/engine-exec-bytecode/how-to/) — task recipes
- [Explanation](https://handl-lang.github.io/docs/engine-exec-bytecode/explanation/) — design and rationale

Decision records live on the `project` branch under `project/adrs/`.

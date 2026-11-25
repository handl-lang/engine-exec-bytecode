# exec-bytecode

`hnl-exec-bytecode` — the bytecode eval engine for the eval operation.

An `exec-*` crate: a concrete eval engine that compiles node schemes to
bytecode and executes them on a small VM.

## Development

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
```

The repo's `xtask/` crate provides the gate tasks (`fmt`, `clippy`, `test`, `doc`,
`check`, `gate`); `workflows/gate.yaml` is the gate entry. Deps resolve from git, so no
sibling checkouts are needed; a plain clone test-runs as-is.

## License

MIT.

## Documentation

<https://handl-lang.github.io/docs/exec-bytecode/>

# engine-exec-bytecode

`hnl-engine-exec-bytecode` — the bytecode eval engine for the eval operation.

An `engine-*` crate: a concrete eval engine that compiles node schemes to
bytecode and executes them on a small VM.

## Development

```bash
cargo xtask gate        # fmt --check → clippy -D warnings → test → doc
```

`cargo xtask` wraps the repo's `xtask/` crate (`fmt`, `clippy`, `test`, `doc`,
`check`, `gate`); `ci.sh` is the gate entry. Deps resolve from git, so no
sibling checkouts are needed; a plain clone test-runs as-is.

## License

MIT.

## Documentation

<https://handl-lang.github.io/docs/engine-exec-bytecode/>

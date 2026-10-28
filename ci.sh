#!/usr/bin/env bash
# ci.sh — the gate for this repo (bupha shell runner; no GitHub Actions).
#
#   ci.sh                       full gate: fmt → clippy → test → doc
#   ci.sh fmt|clippy|test|doc   one stage
#
# Deps resolve from git (the manifest carries no [patch] tables), so this runs
# in a bare clone or in a bupha worktree of the pushed sha with no sibling
# checkouts. Local sibling builds are an opt-in dev convenience via a
# gitignored .cargo/config.toml (see automation/scripts/dev-link.sh).
set -uo pipefail
cd "$(dirname "$0")"

stage="${1:-all}"
fail=0
note() { printf '\n== %s\n' "$*"; }
failed() { echo "   FAILED: $*" >&2; fail=1; }

run_fmt()    { note fmt;    cargo fmt --check || failed 'cargo fmt --check'; }
run_clippy() { note clippy; cargo clippy --all-targets -- -D warnings || failed 'cargo clippy'; }
run_test()   { note test;   cargo test || failed 'cargo test'; }
run_doc()    { note doc;    RUSTDOCFLAGS='-D warnings' cargo doc --no-deps || failed 'cargo doc'; }

case "$stage" in
    all) run_fmt; run_clippy; run_test; run_doc ;;
    fmt | clippy | test | doc) "run_$stage" ;;
    *) echo 'usage: ci.sh [all|fmt|clippy|test|doc]' >&2; exit 2 ;;
esac

if [ "$fail" -ne 0 ]; then echo -e '\nci: FAILED' >&2; exit 1; fi
echo -e '\nci: ok'

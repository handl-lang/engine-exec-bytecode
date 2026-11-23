# Explanation

## Why this crate exists

The tree-walk pays for `lang.eval[node.kind]` on every node. Compiling to
bytecode moves that dispatch to compile time, which is where the speedup comes
from — but only if the VM reproduces the interpreter exactly, which the shared
conformance tests enforce.

## Where it sits

This crate belongs to the `exec-*` family. Plugins, kits and hosts reach it
through the kit's operation slot, so the engine can be swapped without touching a
language definition.

See the [reference](/exec-bytecode/reference/) for the API and the
[how-to guides](/exec-bytecode/how-to/) for tasks.

# How-to guides

Task recipes. Each assumes sibling checkouts of this repo and of the platform
repos it patches.

## Add an instruction

Add the opcode, its compile case and its VM case, then extend the conformance suite so the tree-walk engine stays the oracle.

## Debug a divergence

Run the failing case on both engines and diff the produced values: a mismatch means the compile step erased semantics the VM does not restore.

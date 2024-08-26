********************
Signedness semantics
********************
Both signless and (un)signed representation of integer types in the high level
IR will work. (un)signed types might be more easier to maintain at this stage.
Here are some motivations for it:

- ``sol.emit``'s non indexed args, ``sol.contract``'s interface function args
  etc. uses the ABI tuple encoder (decoder also needs to validate the smaller
  ints in 256-bit allocations) which involves extending smaller ints to 256-bit
  ints.  Using (un)signed types removes the need for clunky attributes to track
  the sign of integral ssa variables.
- The AST to MLIR lowering's cast generation will be simplified as it doesn't
  need to track the AST type system to decide the type of cast.
- Arithmetic ops, compare ops, ``sol.map`` (when extending the key operand to
  256-bit) etc. need to be sign aware. Every such operation has to track the
  sign (like LLVM-IR instructions) with signless ints.

We should move back to the signless representation iff we see a strong
motivation for it.

Relevant info
^^^^^^^^^^^^^
- https://nondot.org/sabre/LLVMNotes/TypeSystemChanges.txt
- https://www.npopov.com/2021/06/02/Design-issues-in-LLVM-IR.html#canonicalization
- https://github.com/llvm/clangir/issues/62

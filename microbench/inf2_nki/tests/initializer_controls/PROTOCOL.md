# Independent explicit-engine fill controls

These synthetic controls characterize `nisa.memset` on SBUF, separately for
explicit `vector` and `gpsimd` engines. Each has eight distinct literal fills,
each followed by a write to a separate output slice, so all results must be
correct and observable. This is not an operator benchmark or agent evaluation.

The generator fixes 14 engine/geometry combinations before collection: five
fit geometries and two independent validation geometries for each engine. Run
two independent Explorer captures per combination, in the generator's fixed
order, with serialized device use. Preserve generated source, compiler output,
device correctness, exact instruction tables, and all failures. Do not select
shapes or remove points based on measured values.

For a data point to enter a fit, both captures must pass the numerical check,
and each of the eight source fill lines must map to exactly one positive-time
instruction on the requested engine, with a consistent opcode. A missing source
mapping, extra instructions, engine change, or disagreement in opcode marks
the entire point unusable; no unrelated opcode or whole-engine active time may
be substituted. Fit only the declared fit set. Report the held-out errors and
whether the fitted ordering agrees with the held-out ordering; validate any
end-to-end scheduling use separately before enabling it. Do not generalize
these controls to `engine.unknown`, PSUM, other dtypes, shapes outside the
measured range, or Trainium2.

The controls have been compiled on the CPU for one shape per engine before
this registration; no device profiles or timing were examined during design.

# Experimental full-overwrite version scenario

The optional `CostModel(full_overwrite_version_scenario=True, ...)` replays a
source trace with two alternating physical storage identifiers for each SBUF
allocation that is fully overwritten. This is a hypothetical schedule, not a
prediction that the compiler will allocate exactly two physical buffers. It is
disabled by default and requires an explicit initializer calibration.

The transformation requires one grid, complete allocation metadata, dense
2-D FP32 SBUF roots, whole-root access runs, and no in-place read/write or SBUF
aliases. It rejects unsupported source operations, uninitialized reads,
insufficient metadata, durations already overridden, and a conservative
all-roots-live SBUF capacity excess. It does not model compiler scratch space,
compiler transformations, or hardware allocation lifetimes. Versioned roots
continue to carry their original element ranges; readers use the latest written
version and writes wrap to the first slot after two definitions.

In one six-pair development check on repeated initializers, enabling this
scenario alone did not change the final predicted speeds: the global completion
calibration dominated the modeled makespan. Disabling global completion instead
produced equal parent and candidate makespans. Neither scenario is validated as
a whole-kernel timing model. The simulator's existing decision-reliability
guard must remain in force; do not use these outputs to veto or rank candidates
until a separately registered prospective validation supports that use.

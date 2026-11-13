# Independent zero-value validation of the fixed initializer fit

The earlier fit used nonzero constants 1 through 8, and its two held-out shape
checks have already been observed. This follow-up freezes six new controls
before capturing zero: for each explicit engine, use geometries `(128,128)`,
`(128,2048)` and `(64,512)`. Each has one zero-valued fill followed by seven
nonzero fills and observable writes. The seven nonzero fills remain an audit
control; they are not refit. Take two Explorer captures per geometry and retain
all failures.

Use the exact earlier coefficients without refitting. Require the same
source-line/engine/opcode audit and numerical correctness as the earlier
experiment. For each engine, all three zero-fill geometries must have a
positive audited instruction duration and relative prediction error at most
30%; otherwise the zero-value extension is not eligible. The outcome does
not by itself enable the scheduler: separately registered end-to-end dependency
and ranking checks remain required. No extrapolation to `engine.unknown`,
PSUM, other dtypes, devices, or unchecked shapes is authorized.

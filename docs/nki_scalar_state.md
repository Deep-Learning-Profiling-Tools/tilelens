# Scalar reduction state in the beta2 interpreter

The interpreter keeps a 128-lane float32 reduction-register vector per logical kernel program. A register is undefined until that program initializes it. Explicit accumulation from undefined registers raises an error; state is not carried between launches or grid programs.

Independent NeuronCore-v2 controls establish these reducing activation transitions: `reset_reduce` replaces the current sum, `reduce` adds to it, reducing `idle` replaces it, and `reset` writes zero. Register updates occur even without `reduce_res`. Plain activations preserve the registers. These semantics are based on local device probes rather than assuming API prose describes every compiler/device version identically; cross-generation state equivalence is not established.

`Tracer(scalar_state_metadata=True)` enables companion outputs plus logical register read/write lane ranges and command metadata. The default trace schema is unchanged. This metadata describes source semantics. It adds no fake storage allocations or instructions, and it does not claim that one source activation is one physical instruction. In particular, the separate accumulator readout phase and its internal ordering/readiness are not priced by this feature.

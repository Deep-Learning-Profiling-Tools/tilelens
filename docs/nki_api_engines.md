# Opt-in hardware engine tags

`Tracer(hardware_engine_metadata=True)` uses the recorded hardware engine for convenience `nisa.reciprocal` (Vector) and explicit `nisa.tensor_scalar` engine requests. An activation whose function is reciprocal remains Scalar. Later writes clear previous overrides. Numerical results are unchanged; the default tracer retains its legacy tags until model promotion.

Compiler-selected `engine.unknown` tensor-scalar operations retain legacy inference: this option does not establish compiler engine selection for unknown requests. Correct engine identity also does not establish dedicated reciprocal instruction timing or implicit setup lowering. Those require independent calibration before a timing model is promoted.

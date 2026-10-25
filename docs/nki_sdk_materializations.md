# SDK preparation metadata

`Tracer(sdk_setup_metadata=True)` records the implicit constant-vector preparation performed by the native NeuronCore-v2 activation wrapper. Numeric execution and source-operation counts are unchanged. The default trace schema remains unchanged.

For activation functions other than copy and reciprocal, Python scalar bias/scale operands are materialized as FP32 SBUF vectors of shape `(P, 1)`. Metadata preserves the input role and exact FP32 value bits. Tensor operands require no such preparation. The implementation follows the SDK wrapper's scalar-constant test rather than inferring an operation from a kernel pattern. Non-v2 targets and explicit `scale=None` are marked unsupported by this evidence path.

The metadata describes SDK-level preparation. It neither asserts final physical instruction retention nor assigns compiler addresses, lifetimes, issue/readiness times, or queue placement. In particular, the compiler can hoist these constants ahead of other source operations. An eventual command model must validate that ordering separately.

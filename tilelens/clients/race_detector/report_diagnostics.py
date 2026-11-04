"""Explanations attached to already-established race witnesses.

This module consumes only report facts. It must not construct constraints,
change witnesses, or decide whether an access pair conflicts.
"""

from __future__ import annotations


def _located_source(site: tuple[str, int, str] | None) -> bool:
    return (
        isinstance(site, (tuple, list))
        and len(site) >= 2
        and isinstance(site[0], str)
        and bool(site[0])
        and isinstance(site[1], int)
        and site[1] > 0
    )


def append_missing_fence_diagnostic(
    reason: str,
    *,
    same_instance: bool,
    distinct_operations: bool,
    first_seq: int,
    second_seq: int,
    fenced: bool,
    first_site: tuple[str, int, str] | None,
    second_site: tuple[str, int, str] | None,
    first_mode: str,
    second_mode: str,
    pre_exit: bool = False,
) -> str:
    """Explain a located cross-operation witness under source fence order.

    Unknown/equal sequence numbers and pre-exit representatives do not
    identify two ordered source operations. Duplicate lanes and cross-instance
    witnesses need different explanations. Preserve their original reasons.
    """
    if (
        not same_instance
        or not distinct_operations
        or pre_exit
        or first_seq < 0
        or second_seq < 0
        or first_seq == second_seq
        or fenced
        or not _located_source(first_site)
        or not _located_source(second_site)
    ):
        return reason
    assert first_site is not None
    assert second_site is not None
    # Report canonicalization uses event IDs, which need not follow source
    # order. Explain the missing fence in source order without reordering
    # the public report endpoints or its RAW/WAR classification.
    if first_seq > second_seq:
        first_site, second_site = second_site, first_site
        first_mode, second_mode = second_mode, first_mode
    return (
        f"{reason}. Missing source fence: the {first_mode} at "
        f"{first_site[0]}:{first_site[1]} and the {second_mode} at "
        f"{second_site[0]}:{second_site[1]} have no captured fence that orders "
        "them. Their reported overlap is unordered under the tile-level "
        "memory model. Review the required ordering; Triton's "
        "tl.debug_barrier() is an option only with appropriate scope and "
        "uniform participation. Compiler-inserted barriers may add ordering; "
        "this diagnosis alone proves neither a real-execution race nor a "
        "complete repair."
    )

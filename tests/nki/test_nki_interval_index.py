import math
import random

from triton_viz.tools.nki_interval_index import IntervalHistory


def subtract(entries,lo,hi):
    out=[]
    for entry in entries:
        a,b,*rest=entry
        if not (lo<b and a<hi):out.append(entry);continue
        if a<lo:out.append((a,min(b,lo),*rest))
        if hi<b:out.append((max(a,hi),b,*rest))
    return out


def test_randomized_operations_match_original_ordered_list():
    for seed in range(50):
        rng=random.Random(seed);slow=[];index=IntervalHistory()
        for step in range(700):
            lo=rng.randrange(0,200);hi=rng.randrange(lo+1,220);op=rng.randrange(4)
            if op==0:
                entry=(lo,hi,step,'vector',None);slow.append(entry);index.append(entry)
            elif op==1:
                expected=[e for e in slow if lo<e[1] and e[0]<hi]
                assert index.overlaps(lo,hi)==expected
            elif op==2:
                slow=subtract(slow,lo,hi);index.subtract(lo,hi)
            else:
                slow=[e for e in slow if not (lo<=e[0] and e[1]<=hi)]
                index.discard_contained(lo,hi)
            assert index.overlaps(-math.inf,math.inf)==slow


def test_infinite_ranges_and_nan_fallback_match_linear_semantics():
    index=IntervalHistory();entries=[(0.0,math.inf,1),(3.0,8.0,2)]
    for e in entries:index.append(e)
    assert index.overlaps(4.0,5.0)==entries
    index.subtract(2.0,4.0);entries=subtract(entries,2.0,4.0)
    assert index.overlaps(-math.inf,math.inf)==entries
    index.append((math.nan,5.0,3));entries.append((math.nan,5.0,3))
    assert index.overlaps(0.0,6.0)==[e for e in entries if 0.0<e[1] and e[0]<6.0]
    index.subtract(1.0,7.0);entries=subtract(entries,1.0,7.0)
    assert index.overlaps(-math.inf,math.inf)==[e for e in entries if -math.inf<e[1] and e[0]<math.inf]


def test_scheduler_outputs_match_original_for_random_overlapping_histories():
    import copy
    from triton_viz.tools.nki_cost_model import CostModel,simulate
    for seed in range(25):
        rng=random.Random(seed);events=[]
        for i in range(180):
            src,dst=rng.randrange(1,6),rng.randrange(1,6)
            a=rng.randrange(0,80);b=a+rng.randrange(1,24)
            c=rng.randrange(0,80);d=c+rng.randrange(1,24)
            if i%3:
                event={'seq':i,'op':'transfer','engine':'dma','bytes':max(b-a,d-c),
                       'src_ptr':src,'src_storage':src,'src_ranges':[[a,b]],
                       'dst_ptr':dst,'dst_storage':dst,'dst_ranges':[[c,d]],
                       'src_version':rng.choice([None,0,1]),'dst_version':rng.choice([None,0,1])}
            else:
                event={'seq':i,'op':'compute','engine':'vector','elements':32,
                       'input_storages':[src],'input_ptrs':[src],'input_ranges':[[a,b]],
                       'input_versions':[rng.choice([None,0,1])],
                       'output_storage':dst,'output_ptr':dst,'output_range':[c,d],
                       'output_version':rng.choice([None,0,1])}
            events.append(event)
        model=CostModel(cross_engine_sync_ns=7,dma_queue_count=2)
        expected=simulate(copy.deepcopy(events),model,use_interval_index=False).as_dict()
        actual=simulate(copy.deepcopy(events),model,use_interval_index=True).as_dict()
        assert actual==expected


def test_degenerate_intervals_preserve_original_list_behavior():
    index = IntervalHistory()
    entries = [(0, 10, 1), (5, 5, 2), (8, 3, 3)]
    for entry in entries:
        index.append(entry)
    for lo, hi in [(0, 10), (4, 4), (9, 2)]:
        assert index.overlaps(lo, hi) == [e for e in entries if lo < e[1] and e[0] < hi]
    index.discard_contained(4, 6)
    entries = [e for e in entries if not (4 <= e[0] and e[1] <= 6)]
    assert index.overlaps(0, 10) == [e for e in entries if 0 < e[1] and e[0] < 10]


def test_pop_overlaps_and_all_match_linear_for_random_histories():
    for seed in range(50):
        rng = random.Random(seed)
        index = IntervalHistory()
        entries = []
        for step in range(500):
            lo = rng.randrange(0, 100)
            hi = rng.randrange(lo + 1, 120)
            if rng.randrange(3):
                item = (lo, hi, step)
                entries.append(item)
                index.append(item)
            else:
                expected = [entry for entry in entries if lo < entry[1] and entry[0] < hi]
                entries = [entry for entry in entries if not (lo < entry[1] and entry[0] < hi)]
                assert index.pop_overlaps(lo, hi) == expected
        assert index.pop_all() == entries
        assert index.pop_all() == []

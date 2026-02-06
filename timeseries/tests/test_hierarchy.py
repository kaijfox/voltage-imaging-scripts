import pytest
from imaging_scripts.timeseries.rois import ProcessROIId


def test_multiple_node_kinds_parsing():
    pid = ProcessROIId.from_string("1.b2")
    assert pid.parts == ("1", "2")
    assert pid.meta["kind"] == "branch"

def test_short_name_generation():
    pid = ProcessROIId(parts=("1", "2"), meta={"kind": "branch"})
    assert pid.short() == "1.b2"

def test_short_name_with_all_ids():
    all_ids = [ProcessROIId(parts=("1", "2"), meta={"kind": "branch"})]
    pid = ProcessROIId(parts=("1", "3"), meta={"kind": "branch"})
    assert pid.short(all_ids=all_ids) == "u1.b3"

def test_default_kind():
    pid = ProcessROIId.from_string("1.2.3")
    assert pid.meta["kind"] == "unk"

def test_unknown_kind_handling():
    pid = ProcessROIId.from_string("x1")
    assert pid.parts == ("x1",)
    assert pid.meta["kind"] == "unk"

def test_resolve_kind():
    all_ids = [ProcessROIId(parts=("1", "2"), meta={"kind": "branch"})]
    kind = ProcessROIId.resolve_kind(all_ids, ("1", "2"))
    assert kind == "branch"

def test_long_name_generation():
    pid = ProcessROIId(parts=("1", "2"), meta={"kind": "branch"})
    assert pid.long() == "1.branch2"

def test_long_name_with_all_ids():
    all_ids = [
        ProcessROIId(parts=("1",), meta={"kind": "branch"}),
        ProcessROIId(parts=("1", "2"), meta={"kind": "unk"}),
        ProcessROIId(parts=("1", "3"), meta={"kind": "branch"})
    ]
    pid = all_ids[2]
    assert pid.long(all_ids=all_ids) == "branch1.branch3"

def test_long_name_unknown():
    all_ids = [
        ProcessROIId(parts=("1", "3"), meta={"kind": "branch"})
    ]
    pid = all_ids[0]
    assert pid.long(all_ids=all_ids) == "unk1.branch3"
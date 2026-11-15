import pytest

from shardline.planner import KnapsackPartitioner


def test_every_layer_assigned_within_capacity():
    layer_sizes = [100, 200, 150, 300, 250, 180]
    worker_vram = {"worker_0": 500, "worker_1": 500, "worker_2": 500}

    assignment = KnapsackPartitioner.partition(layer_sizes, worker_vram)

    assert sorted(i for layers in assignment.values() for i in layers) == list(range(6))
    for worker, layers in assignment.items():
        assert sum(layer_sizes[i] for i in layers) <= worker_vram[worker]


def test_single_worker_gets_all_layers_in_order():
    assignment = KnapsackPartitioner.partition([100, 200, 150], {"worker_0": 500})
    assert assignment["worker_0"] == [0, 1, 2]


def test_insufficient_memory_raises():
    with pytest.raises(ValueError):
        KnapsackPartitioner.partition([100, 200, 150], {"worker_0": 100})


@pytest.mark.xfail(reason="greedy placement interleaves layers across workers", strict=True)
def test_each_worker_gets_a_contiguous_range():
    assignment = KnapsackPartitioner.partition([100] * 12, {"A": 600, "B": 400, "C": 400})
    for layers in assignment.values():
        if layers:
            assert layers == list(range(layers[0], layers[-1] + 1))

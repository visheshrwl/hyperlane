"""Layer placement across workers."""

from __future__ import annotations


class KnapsackPartitioner:
    """Greedy placement by remaining memory.

    Known defect: the layers a worker receives are not contiguous, so the result
    is not a valid pipeline (see tests/test_planner.py). To be replaced by a
    contiguous bottleneck-minimising planner.
    """

    @staticmethod
    def partition(layer_sizes: list[int], worker_vram: dict[str, int]) -> dict[str, list[int]]:
        workers = list(worker_vram)
        assignment: dict[str, list[int]] = {worker: [] for worker in workers}
        loads = dict.fromkeys(workers, 0)

        for layer_index, layer_size in enumerate(layer_sizes):
            best = max(workers, key=lambda worker: worker_vram[worker] - loads[worker])
            if worker_vram[best] - loads[best] < layer_size:
                raise ValueError(
                    f"Layer {layer_index} (size {layer_size}) does not fit in any worker"
                )
            assignment[best].append(layer_index)
            loads[best] += layer_size

        return assignment

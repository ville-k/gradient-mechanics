from gradient_mechanics.data import gpu_iterator


def test_gpu_iterator():
    iterator = iter(range(10))
    gpu_iter = gpu_iterator.GPUIterator(
        iterator=iterator,
        gpu_device=0,
        gpu_prefetch_factor=1,
        transforms=[],
    )
    count = 0
    for _ in gpu_iter:
        count += 1
    assert count == 10

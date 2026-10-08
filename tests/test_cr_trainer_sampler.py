"""E6: the training sampler order follows meta.seed and stays rank-complementary."""

import pytest
from torch.utils.data.distributed import DistributedSampler

from src.train_patch import make_train_sampler


class Sized:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


def order(seed, epoch, world_size=1, rank=0, length=64):
    sampler = make_train_sampler(Sized(length), world_size, rank, seed)
    sampler.set_epoch(epoch)
    return list(sampler)


def test_same_seed_and_epoch_repeat_order():
    assert order(1234, 25) == order(1234, 25)


def test_different_seed_changes_order():
    assert order(1234, 25) != order(5678, 25)


def test_epoch_still_reshuffles_within_a_seed():
    assert order(1234, 25) != order(1234, 26)


def test_seed_zero_reproduces_previous_unseeded_default():
    legacy = DistributedSampler(Sized(64), num_replicas=1, rank=0, shuffle=True)
    legacy.set_epoch(7)
    assert order(0, 7) == list(legacy)


@pytest.mark.parametrize('world_size', [2, 4])
def test_ranks_with_shared_seed_are_complementary(world_size):
    shards = [order(1234, 25, world_size, rank) for rank in range(world_size)]
    flat = [index for shard in shards for index in shard]
    assert len(flat) == len(set(flat)) == 64
    assert set(flat) == set(range(64))
    other_seed = [order(5678, 25, world_size, rank) for rank in range(world_size)]
    assert shards != other_seed

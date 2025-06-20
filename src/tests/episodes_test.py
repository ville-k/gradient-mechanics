from gradient_mechanics.data import episodes

import pytest


def test_even_samples_episode_length_and_stride_one():
    episode_generator = episodes.EpisodeGenerator(
        sample_count=4, episode_length=1, episode_stride=1
    )
    assert episode_generator.episode_count() == 4
    assert [0] == episode_generator.episode_samples_indices(0)
    assert [1] == episode_generator.episode_samples_indices(1)
    assert [2] == episode_generator.episode_samples_indices(2)
    assert [3] == episode_generator.episode_samples_indices(3)


def test_odd_samples_episode_length_and_stride_one():
    episode_generator = episodes.EpisodeGenerator(
        sample_count=3, episode_length=1, episode_stride=1
    )
    assert episode_generator.episode_count() == 3
    assert [0] == episode_generator.episode_samples_indices(0)
    assert [1] == episode_generator.episode_samples_indices(1)
    assert [2] == episode_generator.episode_samples_indices(2)


def test_even_samples_episode_length_and_stride_two():
    episode_generator = episodes.EpisodeGenerator(
        sample_count=4, episode_length=2, episode_stride=2
    )
    assert episode_generator.episode_count() == 2

    assert [0, 1] == episode_generator.episode_samples_indices(0)
    assert [2, 3] == episode_generator.episode_samples_indices(1)


def test_odd_samples_episode_length_and_stride_two():
    episode_generator = episodes.EpisodeGenerator(
        sample_count=3, episode_length=2, episode_stride=2
    )
    assert episode_generator.episode_count() == 1
    assert [0, 1] == episode_generator.episode_samples_indices(0)


def test_even_samples_episode_length_two_stride_one():
    episode_generator = episodes.EpisodeGenerator(
        sample_count=4, episode_length=2, episode_stride=1
    )
    assert episode_generator.episode_count() == 3

    assert [0, 1] == episode_generator.episode_samples_indices(0)
    assert [1, 2] == episode_generator.episode_samples_indices(1)
    assert [2, 3] == episode_generator.episode_samples_indices(2)


def test_odd_samples_episode_length_two_stride_one():
    episode_generator = episodes.EpisodeGenerator(
        sample_count=3, episode_length=2, episode_stride=1
    )
    assert episode_generator.episode_count() == 2
    assert [0, 1] == episode_generator.episode_samples_indices(0)
    assert [1, 2] == episode_generator.episode_samples_indices(1)


def test_episode_index_out_of_range():
    episode_generator = episodes.EpisodeGenerator(
        sample_count=10, episode_length=2, episode_stride=2
    )
    with pytest.raises(IndexError):
        episode_generator.episode_samples_indices(episode_generator.episode_count())


def test_episode_index_negative_raises_error():
    episode_generator = episodes.EpisodeGenerator(
        sample_count=10, episode_length=2, episode_stride=2
    )
    with pytest.raises(IndexError):
        episode_generator.episode_samples_indices(-1)

class EpisodeGenerator:
    def __init__(self, sample_count: int, episode_length: int, episode_stride: int):
        self.sample_count = sample_count
        self.episode_length = episode_length
        self.episode_stride = episode_stride

    def episode_count(self):
        if self.episode_length == self.episode_stride:
            return self.sample_count // self.episode_stride

        return (self.sample_count - self.episode_length) // self.episode_stride + 1

    def episode_samples_indices(self, episode_index: int):
        if episode_index < 0:
            raise IndexError("Episode index cannot be negative.")
        if episode_index >= self.episode_count():
            raise IndexError("Episode index out of range.")
        start_index = episode_index * self.episode_stride
        end_index = start_index + self.episode_length
        return list(range(start_index, end_index))

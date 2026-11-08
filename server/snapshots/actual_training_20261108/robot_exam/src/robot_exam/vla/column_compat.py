"""LeRobot v2.1 column lists without decoding unrelated HF image features."""


class ColumnCompatView:
    """Keep row access intact and project a named column before iteration.

    Recent datasets.Column iterates *all* features when a custom transform is
    present. LeRobot asks for timestamp/episode_index on initialization; without
    projection that can decode a thousand dual-camera images at once and exhaust
    local RAM. This view changes neither stored data nor installed libraries.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, key):
        return getattr(self.dataset, key)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(key, str):
            return list(self.dataset.select_columns([key])[key])
        return self.dataset[key]

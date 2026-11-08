from robot_exam.vla.column_compat import ColumnCompatView


class LazyFeatureDataset:
    """Model the relevant custom-transform column behavior of datasets4."""
    def __init__(self, features=None, decoded=None):
        self.features = features or {'timestamp': [0., .2], 'episode_index': [0, 0], 'image': ['a', 'b']}
        self.decoded = decoded if decoded is not None else []

    def __len__(self):
        return 2

    def select_columns(self, names):
        return LazyFeatureDataset({key: self.features[key] for key in names}, self.decoded)

    def __getitem__(self, key):
        if isinstance(key, str):
            def lazy_column():
                if 'image' in self.features:
                    raise MemoryError('Unrelated transformed images were decoded')
                yield from self.features[key]
            return lazy_column()
        self.decoded.append(key)
        return {name: values[key] for name, values in self.features.items()}


def test_numeric_initialization_does_not_decode_dual_image_payloads():
    raw = LazyFeatureDataset()
    view = ColumnCompatView(raw)
    assert view['timestamp'] == [0., .2]
    assert view['episode_index'] == [0, 0]
    assert raw.decoded == []


def test_rows_and_length_retain_the_original_dataset_semantics():
    raw = LazyFeatureDataset()
    view = ColumnCompatView(raw)
    assert len(view) == 2
    assert view[1] == {'timestamp': .2, 'episode_index': 0, 'image': 'b'}
    assert raw.decoded == [1]

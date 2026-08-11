"""Per-benchmark dataset loaders.

Each loader exposes ``read_sample(json_path, test_id, m, n)`` returning the image
paths and metadata for one puzzle. The pipeline imports the loader named by the
config's ``dataset`` parameter, so adding a benchmark is a new module here.
"""

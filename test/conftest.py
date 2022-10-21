import os
import tempfile

import pytest


@pytest.fixture
def hdf5_file():
    """Generate a temporary hdf5 file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "file.hdf5")
        yield file

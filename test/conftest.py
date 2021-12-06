import os
import tempfile

import pytest

REMOVE_STORE = True


@pytest.fixture
def hdf5_file():
    """Generate a temporary hdf5 file."""
    store_file = tempfile.mkstemp(suffix=".hdf5")[1]
    yield store_file
    if REMOVE_STORE:
        if os.path.exists(store_file):
            os.remove(store_file)

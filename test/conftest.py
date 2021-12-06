import os
import tempfile

import pytest

# whether to remove storage files after a test has finished
REMOVE_STORE = True


@pytest.fixture
def hdf5_file():
    """Generate a temporary hdf5 file."""
    store_file = tempfile.mkstemp(suffix=".hdf5")[1]
    try:
        yield store_file
    finally:
        if REMOVE_STORE and os.path.exists(store_file):
            os.remove(store_file)

import petabtests
import pypesto

import sys
import os
import pytest
from _pytest.outcomes import Skipped
import logging

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

def test_petab_suite():
    n_success = n_skipped = 0
    for case in petabtests.CASES_LIST:
        try:
            execute_case(case)
            n_success += 1
        except Skipped:
            n_skipped += 1
        except Exception as e:
            # run all despite failures
            logger.error(f"Case {case} failed.")
            logger.error(e)

    logger.info(f"{n_success} / {len(petabtests.CASES_LIST)} successful, "
                f"{n_skipped} skipped")
    if n_success != len(petabtests.CASES_LIST):
        sys.exit(1)

    
def execute_case(case):
    """Wrapper for _execute_case for handling test outcomes"""
    try:
        _execute_case(case)
    except Exception as e:
        if isinstance(e, NotImplementedError) \
                or "Timepoint-specific parameter overrides" in str(e):
            logger.info(
                f"Case {case} expectedly failed. Required functionality is "
                f"not implemented: {e}")
            pytest.skip(str(e))
        else:
            raise e


def _execute_case(case):
    """Run a single PEtab test suite case"""
    case = petabtests.test_id_str(case)
    logger.debug(f"Case {case}")

    # load
    case_dir = os.path.join(petabtests.CASES_DIR, case)

    # import petab problem
    yaml_file = os.path.join(case_dir, petabtests.problem_yaml_name(case))
    importer = pypesto.PetabImporter.from_yaml(yaml_file)

    obj = importer.create_objective()

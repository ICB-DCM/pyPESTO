import os
import sys
import pathlib
import unittest


class Flake8Test(unittest.TestCase):

    def test_flake8(self):
        path = pathlib.Path(__file__)
        cmd = (
            "cd {path}; {exec} -m flake8 --exclude={exclude}"
            .format(path=path.parent.parent,
                    exec=sys.executable,
                    exclude="build,doc,example")
        )
        output = os.popen(cmd).readlines()
        for msg in output:
            print(msg.rstrip())
        self.assertTrue(output == [])

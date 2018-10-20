import subprocess
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
        r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        stdout = r.stdout.decode()
        stderr = r.stderr.decode()
        print(stdout, stderr, sep='\n')
        self.assertTrue(stdout == "")
        self.assertTrue(stderr == "")

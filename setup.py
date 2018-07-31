from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), "pesto", "version.py")) as f:
    version = f.read().split('\n')[0].split('=')[-1].strip(' ').strip('"')

setup(name='pypesto',
      version=version,
      description="Parameter EStimation TOolbox",
      author="The pyPESTO developers",
      author_email="yannik.schaelte@gmail.com",
      url="https://github.com/icb-dcm/pyPESTO",
      packages=find_packages(exclude=["example*", "test*"])
     )

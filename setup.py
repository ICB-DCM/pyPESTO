from setuptools import setup, find_packages
import os


# extract version
with open(os.path.join(os.path.dirname(__file__),
          "pypesto", "version.py")) as f:
    version = f.read().split('\n')[0].split('=')[-1].strip(' ').strip('"')


# read a file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# project metadata
setup(name='pypesto',
      version=version,
      description="python Parameter EStimation TOolbox",
      long_description=read('README.md'),
      long_description_content_type="text/markdown",
      author="The pyPESTO developers",
      author_email="yannik.schaelte@gmail.com",
      url="https://github.com/icb-dcm/pypesto",
      packages=find_packages(exclude=["doc*", "test*"]),
      install_requires=['numpy>=1.15.1',
                        'scipy>=1.1.0',
                        'pandas>=0.23.4',
                        'matplotlib>=2.2.3',
                        'seaborn>=0.10.0',
                        'cloudpickle>=0.7.0',
                        'h5py>=2.10.0',
                        'tqdm>=4.46.0'],
      tests_require=['pytest>=5.4.2',
                     'flake8>=3.7.2',
                     'gitpython>=3.1.2'],
      extras_require={'amici': ['amici>=0.11.1'],
                      'petab': ['petab>=0.1.7'],
                      'ipopt': ['ipopt>=0.1.9'],
                      'dlib': ['dlib>=19.19.0'],
                      'pyswarm': ['pyswarm>=0.6'],
                      'pymc3': ['arviz>=0.8.1, <0.9.0',
                                'theano>=1.0.4',
                                'packaging>=20.0',
                                'pymc3>=3.8, <3.9.2']},
      python_requires='>=3.6',
      )

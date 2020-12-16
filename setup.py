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
                        'scipy>=1.4.1',
                        'pandas>=0.23.4',
                        'matplotlib>=2.2.3',
                        'seaborn>=0.10.0',
                        'cloudpickle>=0.7.0',
                        'h5py>=3.1.0',
                        'tqdm>=4.46.0',
                        'gitpython>=3.1.7'],
      tests_require=['pytest>=5.4.2',
                     'flake8>=3.7.2',
                     'gitpython>=3.1.2',
                     'petabtests'],
      extras_require={'amici': ['amici>=0.11.9'],
                      'petab': ['petab>=0.1.11'],
                      'ipopt': ['ipopt>=0.1.9'],
                      'dlib': ['dlib>=19.19.0'],
                      'nlopt': ['nlopt>=2.6.2'],
                      'pyswarm': ['pyswarm>=0.6'],
                      'cmaes': ['cma>=3.0.3'],
                      'fides': ['fides>=0.2.0'],
                      'pymc3': ['arviz>=0.8.1, <0.9.0',
                                'theano>=1.0.4',
                                'packaging>=20.0',
                                'pymc3>=3.8, <3.9.2'],
                      'test': ['pytest>=5.4.3',
                               'pytest-cov>=2.10.0'],
                      'docs': ['sphinx>=3.1.0',
                               'nbsphinx>=0.7.0',
                               'nbconvert>=5.6.1',
                               'sphinx-rtd-theme>=0.4.3'],
                      'quality': ['flake8>=3.8.3',
                                  'flake8-bugbear>=20.1.4'
                                  'flake8-bandit>=2.1.2'
                                  'flake8-print>=3.1.4'
                                  'flake8-comprehensions>=3.2.3'],
                      },
      python_requires='>=3.6',
      )

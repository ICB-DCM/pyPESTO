from setuptools import find_packages
from distutils.core import setup, Extension
from distutils import sysconfig
import os
from amici import amici_path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def getModelSources():
    """Get list of source files for the amici base library"""
    import glob
    import re
    modelSources = glob.glob('*.cpp')
    try:
        modelSources.remove('main.cpp')
    except ValueError:
        pass
    return modelSources


def getAmiciLibs():
    """Get list of libraries for the amici base library"""
    return ['amici', 
            'sundials', 'suitesparse'
            #'sundials_nvecserial', 'sundials_cvodes', 'sundials_idas',
            #'klu', 'colamd', 'btf', 'amd', 'suitesparseconfig'
            ]

# Find HDF5
import pkgconfig
h5pkgcfg = pkgconfig.parse("hdf5")

cxx_flags = ['-std=c++11']
#linker_flags = ['${BLAS_LIBRARIES}']
linker_flags = []
if 'ENABLE_GCOV_COVERAGE' in os.environ and os.environ['ENABLE_GCOV_COVERAGE'] == 'TRUE':
    cxx_flags.extend(['-g', '-O0',  '--coverage'])
    linker_flags.append('--coverage')

libraries = [*getAmiciLibs(),
             'cblas',# TODO generic BLAS
             'hdf5_hl_cpp', 'hdf5_hl', 'hdf5_cpp', 'hdf5'] 
sources = ['swig/model_conversion_reaction.i', *getModelSources()]

    

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for
# C++ to fix warnings.
cfg_vars = sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


# Build shared object
model_module = Extension('model_conversion_reaction/_model_conversion_reaction',
                         sources=sources,
                         include_dirs=[os.getcwd(),
                                       os.path.join(amici_path, 'include'), 
                                       os.path.join(amici_path, 'ThirdParty/sundials/include'),
                                       os.path.join(amici_path, 'ThirdParty/SuiteSparse/include'),
                                       *h5pkgcfg['include_dirs']
                                       ],
                         libraries = libraries,
                         library_dirs=[
                             *h5pkgcfg['library_dirs'],
                             os.path.join(amici_path, 'libs')],
                         swig_opts=['-c++', '-modern', '-outdir', 'model_conversion_reaction',
                                    '-I%s' % os.path.join(amici_path, 'swig'),
                                    '-I%s' % os.path.join(amici_path, 'include'),],
                         extra_compile_args=cxx_flags,
                         extra_link_args=linker_flags
                         )

# Install
setup(
    name='model_conversion_reaction',
    version='0.1.0',
    description='AMICI-generated module for model model_conversion_reaction',
    url='https://github.com/ICB-DCM/AMICI',
    author='model-author-todo',
    author_email='model-author-todo',
    #license = 'BSD',
    ext_modules=[model_module],
    packages=find_packages(),
    install_requires=[],
    python_requires='>=3',
    package_data={
    },
    zip_safe = False,
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        #'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)

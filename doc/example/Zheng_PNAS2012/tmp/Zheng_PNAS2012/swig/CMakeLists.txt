cmake_minimum_required(VERSION 3.8) # swig_add_library

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

SET(CMAKE_SWIG_FLAGS "")
SET_SOURCE_FILES_PROPERTIES(${PROJECT_NAME}.i PROPERTIES CPLUSPLUS ON)

# swig does not use INTERFACE_INCLUDE_DIRS of linked libraries, so add manually
get_target_property(AMICI_INCLUDE_DIRS Upstream::amici INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${AMICI_INCLUDE_DIRS} ..)

SWIG_ADD_LIBRARY(${PROJECT_NAME} TYPE MODULE LANGUAGE python SOURCES ${PROJECT_NAME}.i)

SWIG_LINK_LIBRARIES(${PROJECT_NAME}
	${PYTHON_LIBRARIES} 
        model)

# configure module setup script
SET(SETUP_PY_IN ${Amici_DIR}/model_setup.template.py)
SET(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_target(install-python
        DEPENDS _${PROJECT_NAME}
        COMMAND python ${SETUP_PY_OUT} install)

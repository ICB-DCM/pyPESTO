project(Zheng_PNAS2012)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)
set(MY_CXX_FLAGS -Wall -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable)
foreach(FLAG ${MY_CXX_FLAGS})
    unset(CUR_FLAG_SUPPORTED CACHE)
    CHECK_CXX_COMPILER_FLAG(${FLAG} CUR_FLAG_SUPPORTED)
    if(${CUR_FLAG_SUPPORTED})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
    endif()
endforeach(FLAG)

find_package(Amici HINTS ${CMAKE_CURRENT_LIST_DIR}/../../build)

set(MODEL_DIR ${CMAKE_CURRENT_LIST_DIR})

set(SRC_LIST_LIB Zheng_PNAS2012_J.cpp 
Zheng_PNAS2012_JB.cpp 
Zheng_PNAS2012_JDiag.cpp 
Zheng_PNAS2012_JSparse.cpp 
Zheng_PNAS2012_JSparseB.cpp 
Zheng_PNAS2012_Jv.cpp 
Zheng_PNAS2012_JvB.cpp 
Zheng_PNAS2012_Jy.cpp 
Zheng_PNAS2012_dJydsigmay.cpp 
Zheng_PNAS2012_dJydy.cpp 
Zheng_PNAS2012_dwdp.cpp 
Zheng_PNAS2012_dwdx.cpp 
Zheng_PNAS2012_dxdotdp.cpp 
Zheng_PNAS2012_dydx.cpp 
Zheng_PNAS2012_dydp.cpp 
Zheng_PNAS2012_dsigmaydp.cpp 
Zheng_PNAS2012_qBdot.cpp 
Zheng_PNAS2012_sigmay.cpp 
Zheng_PNAS2012_sxdot.cpp 
Zheng_PNAS2012_w.cpp 
Zheng_PNAS2012_x0.cpp 
Zheng_PNAS2012_x0_fixedParameters.cpp 
Zheng_PNAS2012_sx0.cpp 
Zheng_PNAS2012_sx0_fixedParameters.cpp 
Zheng_PNAS2012_xBdot.cpp 
Zheng_PNAS2012_xdot.cpp 
Zheng_PNAS2012_y.cpp 
${MODEL_DIR}/wrapfunctions.cpp
)

add_library(${PROJECT_NAME} ${SRC_LIST_LIB})
add_library(model ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    
target_link_libraries(${PROJECT_NAME}
    PUBLIC Upstream::amici
)

set(SRC_LIST_EXE main.cpp)

add_executable(simulate_${PROJECT_NAME} ${SRC_LIST_EXE})
    
target_link_libraries(simulate_${PROJECT_NAME} ${PROJECT_NAME})

if($ENV{ENABLE_GCOV_COVERAGE})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
endif()

## SWIG
option(ENABLE_SWIG "Build swig/python library?" ON)
if(ENABLE_SWIG)
    if(NOT(${CMAKE_VERSION} VERSION_LESS 3.8))
        add_subdirectory(swig)
    else()
        message(WARNING "Unable to build SWIG interface, upgrade CMake to >=3.8.")
    endif()
endif()


# <Export cmake configuration>
include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
export(EXPORT ${PROJECT_NAME}Targets FILE ${PROJECT_NAME}Config.cmake
    NAMESPACE Upstream::
    )
# </Export cmake configuration>


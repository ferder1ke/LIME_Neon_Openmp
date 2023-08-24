
cmake_minimum_required(VERSION 3.12)
project(lime_processing)

set(CMAKE_CXX_STANDARD 14)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
# set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
# set(OpenMP_CXX_LIB_NAMES "libomp")
# set(OpenMP_omp_LIBRARY "/opt/homebrew/Cellar/libomp/16.0.6/lib")
#set(OpenMP_libomp_LIBRARY "/opt/homebrew/Cellar/libomp/16.0.6/lib/libomp.dylib")

find_package( OpenCV REQUIRED )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_BINDING=0")
#find_package(OpenMP REQUIRED)
include_directories(/opt/homebrew/opt/libomp/16.0.6/include)
link_directories(/opt/homebrew/opt/libomp/16.0.6/lib)
# if (OPENMP_FOUND)
#     message("OPENMP FOUND")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# else ()
#     message(FATAL_ERROR "OpenMP Not Found!")
# endif ()


include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
)
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_executable(lime lime_opt.cpp)
target_link_libraries(lime ${OpenCV_LIBS})








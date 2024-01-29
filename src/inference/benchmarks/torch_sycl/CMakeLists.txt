cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inference_torch_sycl_example)

option(USE_CUDA "Running with CUDA backend for Torch" OFF)
if(USE_CUDA)
add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

option(IPEX_BACKEND "Running with IPEX backend for Torch" OFF)
if(IPEX_BACKEND)
add_definitions(-DUSE_XPU)
find_library(IPEX_LIB intel-ext-pt-gpu PATHS ${INTEL_EXTENSION_FOR_PYTORCH_PATH}/lib NO_DEFAULT_PATH REQUIRED)
set(TORCH_LIBS ${TORCH_LIBS} ${IPEX_LIB})
include_directories(SYSTEM ${INTEL_EXTENSION_FOR_PYTORCH_PATH}/include)
endif(IPEX_BACKEND)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wl,--no-as-needed")
set(TORCH_LIBS ${TORCH_LIBRARIES})

add_executable(inference_torch_sycl inference_torch_sycl.cpp)
target_link_libraries(inference_torch_sycl ${TORCH_LIBS})

set_property(TARGET inference_torch_sycl PROPERTY CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
project(inference_torch_sycl_example)

option(USE_CUDA "Running with CUDA backend for Torch" OFF)
if(USE_CUDA)
add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wl,--no-as-needed")
set(TORCH_LIBS ${TORCH_LIBRARIES})

option(USE_XPU "Running with IPEX backend for Torch" OFF)
option(INTEL_EXTENSION_FOR_PYTORCH_PATH "Link Intel Extension for Torch" OFF)
if(USE_XPU)
add_definitions(-DUSE_XPU)
find_library(IPEX_LIB intel-ext-pt-gpu PATHS ${INTEL_EXTENSION_FOR_PYTORCH_PATH}/lib NO_DEFAULT_PATH REQUIRED)
set(TORCH_LIBS ${TORCH_LIBS} ${IPEX_LIB})
include_directories(SYSTEM ${INTEL_EXTENSION_FOR_PYTORCH_PATH}/include)
endif(USE_XPU)

add_executable(inference_torch_sycl_cpy inference_torch_sycl_copy.cpp)
add_executable(inference_torch_sycl_ptr inference_torch_sycl_pointer.cpp)
add_executable(test_zero_copy test_zero_copy.cpp)

target_link_libraries(inference_torch_sycl_cpy ${TORCH_LIBS})
target_link_libraries(inference_torch_sycl_ptr ${TORCH_LIBS})
target_link_libraries(test_zero_copy ${TORCH_LIBS})

set_property(TARGET inference_torch_sycl_cpy PROPERTY CXX_STANDARD 17)
set_property(TARGET inference_torch_sycl_ptr PROPERTY CXX_STANDARD 17)
set_property(TARGET test_zero_copy PROPERTY CXX_STANDARD 17)

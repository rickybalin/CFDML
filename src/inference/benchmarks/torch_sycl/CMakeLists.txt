cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inference_torch_sycl_example)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wl,--no-as-needed")
set(TORCH_LIBS ${TORCH_LIBRARIES})

option(INTEL_EXTENSION_FOR_PYTORCH_PATH "Link Intel Extension for Torch" OFF)
if(INTEL_EXTENSION_FOR_PYTORCH_PATH)
find_library(IPEX_LIB intel-ext-pt-gpu PATHS ${INTEL_EXTENSION_FOR_PYTORCH_PATH}/lib NO_DEFAULT_PATH REQUIRED)
set(TORCH_LIBS ${TORCH_LIBS} ${IPEX_LIB})
include_directories(SYSTEM ${INTEL_EXTENSION_FOR_PYTORCH_PATH}/include)
endif(INTEL_EXTENSION_FOR_PYTORCH_PATH)

add_executable(inference_torch_sycl inference_torch_sycl.cpp)
target_link_libraries(inference_torch_sycl ${TORCH_LIBS})

set_property(TARGET inference_torch_sycl PROPERTY CXX_STANDARD 17)

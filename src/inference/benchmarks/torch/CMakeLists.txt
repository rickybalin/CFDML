cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inference_torch_example)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wl,--no-as-needed")

add_executable(inference_torch inference_torch.cpp)
target_link_libraries(inference_torch "${TORCH_LIBRARIES}")

set_property(TARGET inference_torch PROPERTY CXX_STANDARD 14)

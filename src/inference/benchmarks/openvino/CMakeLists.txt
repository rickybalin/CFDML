cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(inference_openvino_sycl_example)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
set(ov_link_libraries openvino::runtime)

add_executable(inference_openvino_sycl inference_openvino_sycl.cpp)
target_link_libraries(inference_openvino_sycl ${ov_link_libraries} -lOpenCL)

set_property(TARGET inference_openvino_sycl PROPERTY CXX_STANDARD 17)


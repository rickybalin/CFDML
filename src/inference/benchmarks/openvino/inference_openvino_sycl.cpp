#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <numeric>
#include "sycl/sycl.hpp"
#include <unistd.h>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

const int N_SAMPLES = 2048;
const int N_INPUTS = 6;
const int N_OUTPUTS = 6;
const int INPUTS_SIZE = N_SAMPLES*N_OUTPUTS;

int main(int argc, const char* argv[]) 
{
  if (argc < 3) {
        std::cerr << "Usage: inference_torch <model_path> <device_str>" << std::endl;
        return argc;
  }

  // Parse input arguments
  char model_path[256], device_str[10];
  strcpy(model_path, argv[1]); 
  strcpy(device_str, argv[2]);

  // Print some information about OpenVINO and start the runtime
  std::cout << "Running with " << ov::get_openvino_version() << "\n\n";
  ov::Core core;
  std::vector<std::string> availableDevices = core.get_available_devices();
  bool found_device = false;
  for (auto&& device : availableDevices) {
    if (strcmp(device.c_str(),device_str)==0) {
      std::cout << "Found device " << device << " \n\n";
      found_device = true;
    }
  }
  if (not found_device) {
    std::cout << "Input device not found \n";
    std::cout << "Available devices are: \n";
    for (auto&& device : availableDevices) {
      std::cout << device << std::endl;
    }
    return -1;
  }

  // Load the model
  std::shared_ptr<ov::Model> model = core.read_model(model_path);
  std::cout << "Loaded model \n\n";

  // Create the input data on the host
  std::vector<float> inputs(INPUTS_SIZE);
  srand(123456789);
  for (int i=0; i<INPUTS_SIZE; i++) {
    inputs[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  } 
  std::cout << "Generated input data on the host \n";
  for (int i=0; i<5; i++) {
    for (int j=0; j<N_INPUTS; j++) {
      std::cout << inputs[i*N_INPUTS+j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Move input data to the device with SYCL
  auto selector = sycl::cpu_selector_v;
  if (strcmp(device_str,"CPU")!=0) {
    selector = sycl::gpu_selector_v;
  }
  sycl::queue Q(selector, sycl::property::queue::in_order{}); // oneDNN needs in order queues
  std::cout << "SYCL running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n\n";
  float *d_inputs = sycl::malloc_device<float>(INPUTS_SIZE, Q); 
  Q.memcpy((void *) d_inputs, (void *) inputs.data(), INPUTS_SIZE*sizeof(float)); 
  Q.wait();

  // Share the SYCL queue and context with the GPU plugin and compile the model
  auto queue = sycl::get_native<sycl::backend::opencl>(Q);
  auto remote_context = ov::intel_gpu::ocl::ClContext(core, queue);
  auto compiled_model = core.compile_model(model, remote_context,
                                           ov::hint::inference_precision("f32"));
  
  // Convert input array to OpenVINO Tensor
  ov::element::Type input_type = ov::element::f32;
  ov::Shape input_shape = {N_SAMPLES, N_INPUTS};
  //ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, d_inputs);
  auto input_tensor = remote_context.create_tensor(input_type, input_shape, (void *) d_inputs); 

  // Run inference in a loop and time it
  int niter = 50;
  ov::InferRequest infer_request = compiled_model.create_infer_request();
  std::vector<std::chrono::milliseconds::rep> times;
  infer_request.set_input_tensor(input_tensor);
  for (int i=0; i<niter; i++) {
    usleep(100000); // sleep a little emulating simulation work

    auto tic = std::chrono::high_resolution_clock::now();
    infer_request.infer();
    auto toc = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    times.push_back(time);
  }
  double mean_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  std::cout << "Performed inference\n";
  printf("Mean inference time %4.2f milliseconds \n\n", mean_time);  

  // Output the predicted Torch tensor
  ov::Tensor output_tensor = infer_request.get_output_tensor();
  std::cout << "Size of output tensor " << output_tensor.get_shape() << std::endl;
  std::cout << "Predicted tensor is : \n";
  for (int i=0; i<5; i++) {
    for (int j=0; j<N_INPUTS; j++) {
      std::cout << output_tensor.data<float>()[i*N_INPUTS+j] << " ";
    }
    std::cout << "\n";
  }

  return 0;
}

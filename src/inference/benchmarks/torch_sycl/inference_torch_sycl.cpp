#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <numeric>
#include "sycl/sycl.hpp"
#include <unistd.h>

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

  torch::DeviceType device;
  if (strcmp(device_str,"cpu")==0) {
    device = torch::kCPU;
  } else if (strcmp(device_str,"cuda")==0) {
    if (torch::cuda::is_available()) {
      device = torch::kCUDA;
    } else {
      std::cout << "CUDA device not found, setting device to CPU \n\n";
      strcpy(device_str, "cpu");
      device = torch::kCPU;
    }
  } else if (strcmp(device_str,"xpu")==0) {
    device = torch::kXPU;
    /*
    if (torch::xpu::is_available()) {
      device = torch::kXPU;
    } else {
      std::cout << "XPU device not found, setting device to CPU \n\n";
      strcpy(device_str, "cpu");
      device = torch::kCPU;
    }
    */
  } else {
    std::cout << "Input device not found, setting device to CPU \n\n";
    strcpy(device_str, "cpu");
    device = torch::kCPU;
  }

  // Initialize and load the model
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(model_path);
    std::cout << "Loaded the model\n";
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  model.to(torch::Device(device));
  std::cout << "Model offloaded to " << device_str << " device \n\n";

  // Create the input data on the host
  //float *inputs = new float[2048*6]();
  std::vector<float> inputs(3276800*6);
  srand(12345);
  for (int i=0; i<3276800*6; i++) {
    inputs[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  } 
  std::cout << "Generated input data on the host \n\n";
  sleep(10);

  // Move input data to the device
  /*auto selector;
  if (strcmp(device_str,"cuda")==0 or strcmp(device_str,"xpu")==0) {
    selector = sycl::gpu_selector_v;
  } else {
    selector = sycl::cpu_selector_v;
  }*/
  sycl::queue Q(sycl::gpu_selector_v);
  std::cout << "SYCL running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";
  sleep(10);
  sycl::buffer<float, 1> inputs_buf((sycl::range<1>(3276800*6)));
  std::cout << "created buffer\n";
  sleep(10);
  Q.submit([&](sycl::handler &cgh) {
    sycl::accessor inputs_acc{inputs_buf, cgh, sycl::read_write};
    cgh.copy(inputs.data(), inputs_acc);
  });
  std::cout << "after Q.submit\n";
  sleep(10);

  // Convert input array to Torch tensor
  // NOTE: the pointer to the input data must reside on same device as
  //       the specified Torch device
  auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device);
  torch::Tensor input_tensor = torch::from_blob(inputs.data(), {3276800,6}, options);
  //torch::Tensor input_tensor = torch::from_blob((void *) inputs_acc, {3276800,6}, options);
  assert(input_tensor.dtype() == torch::kFloat32);
  assert(input_tensor.device().type() == device);
  std::cout << "Converted input data to Torch tesor on " << device_str << " device \n\n";

  // Run inference in a loop and time it
  int niter = 50;
  torch::Tensor output;
  std::vector<std::chrono::milliseconds::rep> times;
  for (int i=0; i<niter; i++) {
    auto tic = std::chrono::high_resolution_clock::now();
    output = model.forward({input_tensor}).toTensor();
    auto toc = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    times.push_back(time);
  }
  double mean_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  std::cout << "Performed inference\n";
  printf("Mean inference time %4.2f milliseconds \n\n", mean_time);  

  // Output the predicted Torch tensor
  std::cout << "Predicted tensor is : \n";
  std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/10) << '\n';

  return 0;
}

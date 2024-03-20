#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <numeric>
#include "sycl/sycl.hpp"
#include <unistd.h>
#ifdef USE_XPU
#include <ipex.h> // only needed for xpu::dpcpp::fromUSM below, but deprecated
#endif

const int N_SAMPLES = 2048;
const int N_INPUTS = 6;
const int INPUTS_SIZE = N_SAMPLES*N_INPUTS;

int main(int argc, const char* argv[])
{
  if (argc < 2) {
        std::cerr << "Usage: test_zero_copy <device_str>" << std::endl;
        return argc;
  }

  // Parse input arguments
  char device_str[10];
  strcpy(device_str, argv[1]);

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
  if (strcmp(device_str,"cuda")==0 or strcmp(device_str,"xpu")==0) {
    selector = sycl::gpu_selector_v;
  }  
  sycl::queue Q(selector);
  std::cout << "SYCL running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n\n";
  float *d_inputs = sycl::malloc_device<float>(INPUTS_SIZE, Q);
  Q.memcpy((void *) d_inputs, (void *) inputs.data(), INPUTS_SIZE*sizeof(float));
  Q.wait();

  // Convert input array to Torch tensor
  auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device);
  torch::Tensor input_tensor;
#ifdef USE_CUDA
  input_tensor = torch::from_blob(d_inputs, {N_SAMPLES,N_INPUTS}, options); // XPU devices not supported yet for this function
#elif USE_XPU
  input_tensor = at::from_blob(d_inputs, {N_SAMPLES,N_INPUTS}, nullptr, at::device(device).dtype(torch::kFloat32), device).to(device); // NOT a zero-copy operation and creates a new Torch tensor
#endif
  assert(input_tensor.dtype() == torch::kFloat32);
  assert(input_tensor.device().type() == device);
  std::cout << "Converted input data to Torch tensor on " << device_str << " device \n\n";

  // Change value of first few elements of Torch tensor
  for (int i=0; i<1; i++) {
    for (int j=0; j<N_INPUTS; j++) {
      input_tensor[i][j] = 1.2345;
    }
  }
  std::cout << "Changed first row of Torch tensor to 1.2345 \n\n";

  // Copy SYCL input tensor to host and print
  float tmp[INPUTS_SIZE];
  Q.memcpy((void *) tmp, (void *) d_inputs, INPUTS_SIZE*sizeof(float));
  Q.wait();
  std::cout << "SYCL array is : \n";
  for (int i=0; i<5; i++) {
    for (int j=0; j<N_INPUTS; j++) {
      std::cout << tmp[i*N_INPUTS+j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Free memory
  sycl::free(d_inputs, Q);

  return 0;
}

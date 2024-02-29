#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <numeric>
#include "sycl/sycl.hpp"
#include <unistd.h>

const int N_SAMPLES = 2048;
const int N_INPUTS = 6;
const int N_OUTPUTS = 6;
const int INPUTS_SIZE = N_SAMPLES*N_INPUTS;
const int OUTPUTS_SIZE = N_SAMPLES*N_OUTPUTS;

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

  // Pre-allocate the output array on device and fill with ones
  float *d_outputs = sycl::malloc_device<float>(OUTPUTS_SIZE, Q);
  Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(OUTPUTS_SIZE, [=](sycl::id<1> idx) {
                        d_outputs[idx] = 1.2345;
      });
  });
  Q.wait();

  // Convert input array to Torch tensor
  // NOTE: the pointer to the input data must reside on same device as
  //       the specified Torch device
  auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device);
  torch::Tensor input_tensor;
#ifdef USE_CUDA
  input_tensor = torch::from_blob(d_inputs, {N_SAMPLES,N_INPUTS}, options); // XPU devices not supported yet for this function
#elif USE_XPU  
  input_tensor = at::from_blob(d_inputs, {N_SAMPLES,N_INPUTS}, nullptr, at::device(device).dtype(torch::kFloat32), device).to(device);
#endif
  assert(input_tensor.dtype() == torch::kFloat32);
  assert(input_tensor.device().type() == device);
  std::cout << "Converted input data to Torch tensor on " << device_str << " device \n\n";

  // Convert output array to Torch tensor
  torch::Tensor output_tensor;
#ifdef USE_CUDA
  output_tensor = torch::from_blob(d_outputs, {N_SAMPLES,N_OUTPUTS}, options); // XPU devices not supported yet for this function
#elif USE_XPU
  output_tensor = at::from_blob(d_outputs, {N_SAMPLES,N_INPUTS}, nullptr, at::device(device).dtype(torch::kFloat32), device).to(device);
#endif
  assert(output_tensor.dtype() == torch::kFloat32);
  assert(output_tensor.device().type() == device);
  std::cout << "Converted output data to Torch tensor on " << device_str << " device \n";
  for (int i=0; i<5; i++) {
    for (int j=0; j<N_OUTPUTS; j++) {
      std::cout << output_tensor[i][j].item() << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Run inference in a loop and time it
  torch::NoGradGuard no_grad; // equivalent to "with torch.no_grad():" in PyTorch
  int niter = 50;
  torch::Tensor output;
  std::vector<std::chrono::milliseconds::rep> times;
  for (int i=0; i<niter; i++) {
    usleep(100000); // sleep a little emulating simulation work

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
  std::cout << "Predicted Torch tensor is : \n";
  for (int i=0; i<5; i++) {
    for (int j=0; j<N_OUTPUTS; j++) {
      std::cout << output[i][j].item() << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Copy the output Torch tensor to the SYCL pointer
  //output_tensor_ptr = output.contiguous().data_ptr();
  float outputs[OUTPUTS_SIZE];
  output_tensor.copy_(output);
  Q.memcpy((void *) outputs, (void *) d_outputs, OUTPUTS_SIZE*sizeof(float)); 
  Q.wait();
  std::cout << "Predicted SYCL array is : \n";
  for (int i=0; i<5; i++) {
    for (int j=0; j<N_OUTPUTS; j++) {
      std::cout << outputs[i*N_OUTPUTS+j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Free memory
  sycl::free(d_inputs, Q);
  sycl::free(d_outputs, Q);

  return 0;
}

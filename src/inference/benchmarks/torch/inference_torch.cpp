#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

int main(int argc, const char* argv[]) {
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
  // Upload model to GPU
  model.to(torch::Device(device));
  std::cout << "Model offloaded to " << device_str << " device \n\n";

  // Create the input Torch tensor
  auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device);
  torch::Tensor input_tensor = torch::rand({2048,6}, options);
  assert(input_tensor.dtype() == torch::kFloat32);
  assert(input_tensor.device().type() == device);
  std::cout << "Created the input tesor on " << device_str << " device \n\n";

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

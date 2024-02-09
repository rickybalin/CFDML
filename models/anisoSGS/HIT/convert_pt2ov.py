# Simple Python script that convert the PyTorch model to the OpenVINO IR

import openvino as ov
import openvino.properties.hint as hints
import torch
import numpy as np

model = torch.jit.load("NNmodel_HIT_jit.pt")
torch.manual_seed(12345)
input_data = torch.rand(2048,6)

ov_model = ov.convert_model(model, example_input=input_data)
ov.save_model(ov_model, "NNmodel_HIT.xml", compress_to_fp16=False)

import intel_extension_for_pytorch as ipex
device = 'xpu'
model.to(device)
output_torch = model(input_data.to(device))

core = ov.runtime.Core()
config = {hints.inference_precision: 'f32'}
compiled_model = core.compile_model(model="NNmodel_HIT.xml", device_name="GPU", config=config)
output_ov = torch.tensor(compiled_model(input_data)[0])

print(f"Torch predicted tensor sample: {output_torch[0]}")
print(f"OpenVINO predicted tensor sample: {output_ov[0]}")
assert np.allclose(output_torch.detach().cpu().numpy(),output_ov.detach().cpu().numpy()), \
    "Torch and OpenVINO predicted tensors are different"

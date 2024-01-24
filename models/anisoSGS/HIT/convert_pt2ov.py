# Simple Python script that convert the PyTorch model to the OpenVINO IR

import openvino as ov
import torch

model = torch.jit.load("NNmodel_HIT_jit.pt")
input_data = torch.rand(2048,6)

ov_model = ov.convert_model(model, example_input=input_data)
ov.save_model(ov_model, "NNmodel_HIT.xml", compress_to_fp16=False)

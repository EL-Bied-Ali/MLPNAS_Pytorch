import onnx
import torch

import torch.nn as nn
from brevitas.nn import QuantIdentity, QuantLinear
from brevitas.quant import SignedBinaryWeightPerTensorConst, SignedBinaryActPerTensorConst

from platform import python_version



from tabnanny import verbose
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil



import brevitas.onnx as bo
from brevitas.quant_tensor import QuantTensor
import numpy as np

import json


# print("Python version: ", python_version())
# print("Project directory: ", os.getcwd())


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Target device: " + str(device))


class FinnCompiler(object):
    
    def __init__(self, model, fps, mvau_max):
        
        self.model = model
        self.fps = fps
        self.mvau_max = mvau_max

        self.model_for_export = MNIST_BNN(model)

    def to_onnx(self, model):
        ready_model_filename = "mnist_bnn_mlp.onnx"
        input_shape = (1, 28*28)

        # create a QuantTensor instance to mark input as bipolar during export
        input_a = np.random.randint(0, 256, size=input_shape).astype(np.float32)
        scale = 1.0
        input_t = torch.from_numpy(input_a * scale)
        input_qt = QuantTensor(
            input_t, scale=torch.tensor(scale), bit_width=torch.tensor(8), signed=False
        )

        #Move to CPU before export
        model.cpu()

        # Export to ONNX
        bo.export_finn_onnx(model, export_path=ready_model_filename, input_t=input_qt)

        print("Model saved to %s" % ready_model_filename)

        return ready_model_filename
    
    def compile_model(self):

        model_file = self.to_onnx(self.model_for_export)

        estimates_output_dir = "mnist_bnn_mlp_output_estimates_only"

        #Delete previous run results if exist
        if os.path.exists(estimates_output_dir):
            shutil.rmtree(estimates_output_dir)
            print("Previous run results deleted!")


        cfg_estimates = build.DataflowBuildConfig(
            output_dir          = estimates_output_dir,
            mvau_wwidth_max     = self.mvau_max,
            target_fps          = self.fps,
            synth_clk_period_ns = 10.0,
            fpga_part           = "xc7z020clg400-1",
            steps               = build_cfg.estimate_only_dataflow_steps,
            generate_outputs=[
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            ]
        )


        build.build_dataflow_cfg(model_file, cfg_estimates)

        metrics = read_json_dict(estimates_output_dir + "/report/estimate_network_performance.json")
    
        # metrics['estimated_latency_ns']
        return metrics
        



def read_json_dict(filename):
    with open(filename, "r") as f:
        ret = json.load(f)
    return ret



class MNIST_BNN(nn.Module):
    def __init__(self, model):
        super(MNIST_BNN, self).__init__()
        self.model = model
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0], device=x.device)  
        out = self.model(x)
        return out





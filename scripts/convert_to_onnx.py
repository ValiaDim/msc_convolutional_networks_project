import argparse
import os

import torch
import onnx

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(net, output_name):
    input = torch.randn(1, 3, 256, 456)
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

    output_path = "../optimized_models/onnx"
    os.makedirs(output_path, exist_ok=True)

    torch.onnx.export(net, input, os.path.join(output_path, output_name), verbose=True, input_names=input_names,
                      output_names=output_names, do_constant_folding=True, export_params=True, opset_version=10)
    model = onnx.load(os.path.join(output_path, output_name))
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    onnx.save(model, (os.path.join(output_path, os.path.splitext(output_name)[0] + "-dynamic.onnx")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation.onnx',
                        help='name of output model in ONNX format')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path)
    net.eval()
    load_state(net, checkpoint)

    convert_to_onnx(net, args.output_name)

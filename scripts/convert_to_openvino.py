import argparse
import sys
import os
import logging as log

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore


class openvino_network:
    def __init__(self, args):
        self.arguments = args
        self.net, self.ie = self.create_openvino_inference_engine()
        self.ensure_supported_layers()  # todo handle
        self.exec_net, self.out_blob, self.input_blob, self.input_name, self.input_info_name, self.input_shape = self.prepare_inference()

    def create_openvino_inference_engine(self):
        model_xml = "/home/valia/dev/lightweight-human-pose-estimation.pytorch/optimized_models/openVINO/human-pose-estimation.xml"
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        ie = IECore()
        net = IENetwork(model=model_xml, weights=model_bin)
        return net, ie

    def ensure_supported_layers(self):
        supported_layers = self.ie.query_network(self.net, 'CPU')
        not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]

        if len(not_supported_layers) != 0:
            log.error('...The following layers are not supported by the device.{}'.format(', '.join(not_supported_layers)))
            return False
        assert len(self.net.inputs.keys()) == 1, 'The application supports single input topologies.'
        # assert len(net.outputs) == 1, 'The application supports single output topologies'
        return True

    def prepare_inference(self):
        exec_net = self.ie.load_network(network=self.net, device_name='CPU')
        input_blob = next(iter(self.net.inputs))
        out_blob = self.net.outputs.keys()
        self.net.batch_size = 1
        input_name = ''
        input_info_name = ''
        for input_key in self.net.inputs:
            if len(self.net.inputs[input_key].layout) == 4:
                input_name = input_key
                self.net.inputs[input_key].precision = 'U8'
            elif len(self.net.inputs[input_key].layout) == 2:
                input_info_name = input_key
                self.net.inputs[input_key].precision = 'FP32'
                if self.net.inputs[input_key].shape[1] != 3 and self.net.inputs[input_key].shape[1] != 6 or self.net.inputs[input_key].shape[
                    0] != 1:
                    log.error('Invalid input info. Should be 3 or 6 values length.')
        input_shape = self.net.inputs[input_blob].shape
        print("input shape should be {}".format(input_shape))
        return exec_net, out_blob, input_blob, input_name, input_info_name, input_shape

    def prepare_image_for_infer(self, image):
        print(image.shape[:-1])
        if image.shape[:-1] != (self.input_shape[3], self.input_shape[2]):
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def infer(self, image):
        res = self.exec_net.infer(inputs={self.input_blob: image})
        res_output = []
        for idx, output_t in enumerate(self.out_blob):
            res_output.append(res[output_t])
        return res_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    #parser.add_argument('-i', '--ifile', type=str, required=True,
    #                    help='Required. Filename of the image to load and classify')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Required. Path to the model to use for classification. Should end in .xml')
    parser.add_argument('-o', '--ofile', type=str, required=False,
                        help='Optional. Filename to write the annotated image to', default=None)
    parser.add_argument('-l', '--labels', type=str, required=False,
                        help='Optional. Filename of the class id to label mappings', default=None)
    parser.add_argument('-d', '--device', type=str, required=False,
                        help='Optional. Specify the target device to infer on: CPU, GPU, MYRIAD or HETERO.', default='CPU')
    parser.add_argument('-x', '--extension', type=str, required=False,
                        help='Optional. Extension for custom layers.', default=None)

    args = parser.parse_args()
    args = vars(args)

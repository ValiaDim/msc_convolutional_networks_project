import onnx
import argparse
import os
import numpy as np
from onnx_tf.backend import prepare
import tensorflow as tf


def convert_to_pb(args):
    onnx_model = onnx.load(args.checkpoint_path)  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    output_path = "../optimized_models/tensorflow"
    os.makedirs(output_path, exist_ok=True)
    tf_rep.export_graph(path=output_path)  # export the model
    os.rename(os.path.join("../optimized_models/tensorflow", "saved_model.pb"),
              os.path.join("../optimized_models/tensorflow", args.output_name))


def convert_to_tflite(args):
    ## TFLite Conversion
    # Before conversion, fix the model input size
    frozen_model_path = "../optimized_models/tensorflow"
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(frozen_model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    open("../optimized_models/tensorflow/converted_model.tflite", "wb").write(tflite_model)

    ## TFLite Interpreter to check input shape
    interpreter = tf.lite.Interpreter(model_path="../optimized_models/tensorflow/converted_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the onnx checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation.pb',
                        help='name of output model in tensorflow frozen model format')
    args = parser.parse_args()

    # convert_to_pb(args)
    convert_to_tflite(args)
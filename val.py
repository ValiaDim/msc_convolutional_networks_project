import argparse
import cv2
import json
import math
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import onnxruntime as rt
import tensorflow as tf
import numpy
import torch

from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state


def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def infer(net, img, scales, base_height, stride, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256,
          mode="pytorch"):
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
        if mode == "pytorch":
            tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            stages_output = net(tensor_img)
        elif mode == "onnx":
            input_name = net.get_inputs()[0].name
            label_names = [output.name for output in net.get_outputs()]
            padded_img = np.transpose(padded_img, (2, 0, 1))
            padded_img = np.expand_dims(padded_img, axis=0)

            stages_output = net.run(label_names, {input_name: padded_img.astype(numpy.float32)})
        elif mode == "tflite":
            # Get input and output tensors.
            input_details = net.get_input_details()
            output_details = net.get_output_details()

            # Test model on random input data.
            input_shape = input_details[0]['shape']

            net.resize_tensor_input(
                input_details[0]['index'], (1, 3, base_height, padded_img.shape[1]))
            net.allocate_tensors()
            input_data = np.transpose(padded_img, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            input_details = net.get_input_details()

            net.set_tensor(input_details[0]['index'], input_data.astype('float32'))

            net.invoke()
            stages_output = []
            for idx, output_t in enumerate(output_details):
                stages_output.append(net.get_tensor(output_t['index']))

        stage2_heatmaps = stages_output[-2]
        if mode == "pytorch":
            heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        else:
            heatmaps = np.transpose(stage2_heatmaps.squeeze(), (1, 2, 0))

        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        if mode == "pytorch":
            pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        else:
            pafs = np.transpose(stage2_pafs.squeeze(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs


def infer_onnx(session, img, scales, base_height, stride, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    input_name = session.get_inputs()[0].name
    label_names = [output.name for output in session.get_outputs()]
    # pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        # tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        #stages_output = net(tensor_img)
        padded_img = np.transpose(padded_img, (2, 0, 1))
        padded_img = np.expand_dims(padded_img, axis=0)

        stages_output = session.run(label_names, {input_name: padded_img.astype(numpy.float32)})

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs


def evaluate_tflite(labels, output_name, images_folder, interpreter, multiscale=False, visualize=False):
    # net = net.cuda().eval()
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []

    for idx, sample in enumerate(tqdm(dataset)):
        file_name = sample['file_name']
        img = sample['img']

        avg_heatmaps, avg_pafs = infer_onnx(interpreter, img, scales, base_height, stride)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey(0)
            if key == 27:  # esc
                return

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)


def evaluate_onnx(labels, output_name, images_folder, session, multiscale=False, visualize=False):
    # net = net.cuda().eval()
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []

    for idx, sample in enumerate(tqdm(dataset)):
        file_name = sample['file_name']
        img = sample['img']

        avg_heatmaps, avg_pafs = infer_onnx(session, img, scales, base_height, stride)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey(0)
            if key == 27:  # esc
                return

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)


def evaluate_core(labels, output_name, images_folder, net, multiscale=False, visualize=False, mode="pytorch"):
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []

    for idx, sample in enumerate(tqdm(dataset)):
        file_name = sample['file_name']
        img = sample['img']

        avg_heatmaps, avg_pafs = infer(net, img, scales, base_height, stride, mode=mode)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey()
            if key == 27:  # esc
                return

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)


def evaluate(args):
    if args.mode == "pytorch":
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(args.checkpoint_path)
        load_state(net, checkpoint)
        net = net.cuda().eval()
        evaluate_core(args.labels, args.output_name, args.images_folder, net, args.multiscale, args.visualize, args.mode)
    elif args.mode == "onnx":
        # https://pypi.org/project/onnxruntime-gpu/0.1.4/
        sess = rt.InferenceSession(
            "/home/valia/dev/lightweight-human-pose-estimation.pytorch/optimized_models/onnx/human-pose-estimation-dynamic.onnx",
            providers=['CUDAExecutionProvider'])
        evaluate_core(args.labels, args.output_name, args.images_folder, sess, args.multiscale, args.visualize, args.mode)
    elif args.mode == "tflite":
        interpreter = tf.lite.Interpreter(model_path="/home/valia/dev/lightweight-human-pose-estimation.pytorch/optimized_models/tensorflow/converted_model.tflite")
        interpreter.allocate_tensors()
        evaluate_core(args.labels, args.output_name, args.images_folder, interpreter, args.multiscale, args.visualize, args.mode)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    framework_options = ["pytorch", "onnx", "tensorflow", "tflite"]
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    parser.add_argument('--mode', choices=framework_options)
    args = parser.parse_args()

    evaluate(args)


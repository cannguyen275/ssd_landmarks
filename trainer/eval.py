import utils
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import torchvision.ops as ops
import torch.nn.functional as F
from utils import box_processing as box_utils


def get_detections(img_batch, model, config, score_threshold=0.5, iou_threshold=0.5):
    cpu_device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        img_batch = img_batch.type(torch.cuda.FloatTensor)
        scores, boxes, landmarks = model(img_batch)
        scores = F.softmax(scores, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            boxes, config.priors, config.center_variance, config.size_variance)
        # landmarks = box_utils.decode_landm(landmarks, config.priors, config.center_variance,
        #                                    config.size_variance)
        boxes = box_utils.center_form_to_corner_form(boxes)
        # boxes = boxes[0]
        # scores = scores[0]
        # landmarks = landmarks[0]

        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        # landmarks = landmarks.to(cpu_device)

        picked_boxes = []
        picked_labels = []
        # picked_landmarks = []

        for class_index in range(1, scores.size(2)):
            probs = scores[:, :, class_index]
            mask = probs > score_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            # subset_landmarks = landmarks[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            # landmark_probs = torch.cat([subset_landmarks, probs.reshape(-1, 1)], dim=1)
            box_probs, landmark_probs = box_utils.nms(box_probs, None, None,
                                                      score_threshold=score_threshold,
                                                      iou_threshold=iou_threshold,
                                                      sigma=0.5,
                                                      top_k=2000,
                                                      candidate_size=200)
            picked_boxes.append(box_probs)
            # picked_landmarks.append(landmark_probs)
            picked_labels.extend([class_index] * box_probs.size(0))

        if not picked_boxes:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_boxes = torch.cat(picked_boxes)
        # picked_landmarks = torch.cat(picked_landmarks)
        picked_boxes[:, 0] *= 300
        picked_boxes[:, 1] *= 300
        picked_boxes[:, 2] *= 300
        picked_boxes[:, 3] *= 300
        # picked_landmarks[:, 0:10:2] *= 300
        # picked_landmarks[:, 1:10:2] *= 300

        return picked_boxes[:, :4], None, None, None


def compute_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    # (N, K) ndarray of overlap between boxes and query_boxes
    return torch.from_numpy(intersection / ua)


def evaluate(val_data, model, config, threshold=0.5):
    recall = 0.
    precision = 0.
    # for i, data in tqdm(enumerate(val_data)):
    for idx, (images, targets) in enumerate(val_data):
        print(idx)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        picked_boxes, _, _, _ = get_detections(images, model, config)
        recall_iter = 0.
        precision_iter = 0.

        for j, boxes in enumerate(picked_boxes):
            annot_boxes = targets[j]
            annot_boxes = annot_boxes[annot_boxes[:, 0] != -1]

            if boxes.shape[0] == 0 and annot_boxes.shape[0] == 0:
                continue
            elif boxes.shape[0] == 0 and annot_boxes.shape[0] != 0:
                recall_iter += 0.
                precision_iter += 1.
                continue
            elif boxes.shape[0] != 0 and annot_boxes.shape[0] == 0:
                recall_iter += 1.
                precision_iter += 0.
                continue

            overlap = ops.boxes.box_iou(annot_boxes, boxes.cuda())

            # compute recall
            max_overlap, _ = torch.max(overlap, dim=1)
            mask = max_overlap > threshold
            detected_num = mask.sum().item()
            recall_iter += detected_num / annot_boxes.shape[0]

            # compute precision
            max_overlap, _ = torch.max(overlap, dim=0)
            mask = max_overlap > threshold
            true_positives = mask.sum().item()
            precision_iter += true_positives / boxes.shape[0]

        recall += recall_iter / len(picked_boxes)
        precision += precision_iter / len(picked_boxes)

    return recall / len(val_data), precision / len(val_data)

import numpy as np
import cv2
import json
import os


def get_gt(path):
    path_images = []
    words = []
    lines = list(open(path, 'r').readlines())
    labels = []
    flag = False
    for line in lines:
        line = line.rstrip()
        if line.startswith('#') or line.startswith('/'):
            if flag == False:
                flag = True
            else:
                words.append(labels)
                labels = []
            image_name = line[2:]
            path_images.append(image_name)
        else:
            label = [float(x) for x in line.split(' ')]
            labels.append(label)
    words.append(labels)
    return path_images, words


def get_predict(filename):
    annotations = np.zeros((0, 15))
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for infors in content[2:]:
        infor = np.array([float(x) for x in infors.split(',')])
        infor = np.expand_dims(infor, 0)
        annotations = np.append(annotations, infor, axis=0)

    return annotations


def intersect(a, b):
    '''Determine the intersection of two rectangles'''
    rect = (0, 0, 0, 0)
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    # Do we have a valid intersection?
    if r1 > r0 and c1 > c0:
        rect = (r0, c0, r1, c1)
    return rect


def union(a, b):
    r0 = min(a[0], b[0])
    c0 = min(a[1], b[1])
    r1 = max(a[2], b[2])
    c1 = max(a[3], b[3])
    return (r0, c0, r1, c1)


def area(a):
    '''Computes rectangle area'''
    width = a[2] - a[0]
    height = a[3] - a[1]
    return abs(width * height)


def draw_detections(img, box, text, color=(255, 0, 0)):
    # draw2 = np.copy(img)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
    cx = box[0]
    cy = box[1] - 20
    cv2.putText(img, text, (cx, cy),
                cv2.FONT_HERSHEY_TRIPLEX, 1, color)
    return img


def is_negative(list_input):
    for x in list_input:
        if x < 0:
            return True
    return False


def analyze(image_name, detections, ground_truth, iou_th=0.6):
    gt_to_detection = {}
    detection_to_gt = {}
    tp = 0
    tp_e2e = 0
    tp_e2e_ed1 = 0
    gt_e2e = 0

    gt_matches = np.zeros(len(ground_truth))
    gt_matches_ed1 = np.zeros(len(ground_truth))
    data_out = []
    data_out.append('# ' + image_name)
    for i in range(0, len(ground_truth)):
        image_gt = ground_truth[i]
        print(image_gt)
        gt_xywh = [image_gt[0], image_gt[1], image_gt[2], image_gt[3]]
        gt_bbox = [image_gt[0], image_gt[1], image_gt[0] + image_gt[2], image_gt[1] + image_gt[3]]
        gt_lands = [image_gt[4], image_gt[5], image_gt[7], image_gt[8], image_gt[10], image_gt[11], image_gt[13],
                    image_gt[14], image_gt[16], image_gt[17]]
        if is_negative(gt_xywh) or is_negative(gt_lands):
            continue
        if gt_lands[0] > 0:
            data_out.append(image_gt)
            continue
        flag = False
        for dets in detections:
            det_bbox = [dets[0], dets[1], dets[2], dets[3]]
            det_xywh = [dets[0], dets[1], dets[2] - dets[0], dets[3] - dets[1]]
            det_lands = [dets[5], dets[6], 0, dets[7], dets[8], 0, dets[9], dets[10], 0,
                         dets[11], dets[12], 0, dets[13], dets[14], 0, 0]
            inter = intersect(det_bbox, gt_bbox)  # Intersection of predicted and GT bounding-boxes
            uni = union(det_bbox, gt_bbox)  # Union of predicted and GT bounding-boxes
            ratio = area(inter) / float(area(uni))  # IoU measure between predicted and GT bounding-boxes
            print("mark 3", ratio)
            if ratio > iou_th:
                print("mark 2", ratio)
                print(gt_xywh + det_lands)
                data_out.append(det_xywh + det_lands)
                print(data_out)
                flag = True
                break
        if not flag:
            print('mark3')
            data_out.append(image_gt)
    return data_out


if __name__ == "__main__":
    image_path = "/media/can/Data/Dataset/WiderFace/widerface/train/images"
    txt_path = "helper/widerface_txt/widerface/train/images"
    images_name, data = get_gt("helper/label.txt")
    new_label = []
    for index, name in enumerate(images_name):
        print("\n", name)
        # image = cv2.imread(os.path.join(image_path, name))
        gt = data[index]
        predict_filePath = os.path.join(txt_path, name[:-4] + ".txt")
        # Load predict file
        predict = get_predict(predict_filePath)
        data_new = analyze(name, predict, gt)
        new_label.append(data_new)

        debug = False
        if debug:
            img_debug = image.copy()
            for infor in data_new[1:]:
                b = list(map(int, infor))
                cv2.rectangle(img_debug, (b[0], b[1]), (b[2] + b[0], b[3] + b[1]), (0, 0, 255), 2)
                # landms
                cv2.circle(img_debug, (b[4], b[5]), 1, (0, 0, 255), 4)
                cv2.circle(img_debug, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_debug, (b[10], b[11]), 1, (255, 0, 255), 4)
                cv2.circle(img_debug, (b[13], b[14]), 1, (0, 255, 0), 4)
                cv2.circle(img_debug, (b[16], b[17]), 1, (255, 0, 0), 4)
            for value in gt:
                b = list(map(int, value))
                cv2.rectangle(image, (b[0], b[1]), (b[2] + b[0], b[3] + b[1]), (0, 0, 255), 2)
                # landms
                cv2.circle(image, (b[4], b[5]), 1, (0, 0, 255), 4)
                cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(image, (b[10], b[11]), 1, (255, 0, 255), 4)
                cv2.circle(image, (b[13], b[14]), 1, (0, 255, 0), 4)
                cv2.circle(image, (b[16], b[17]), 1, (2500, 0, 0), 4)
            for box in predict:
                b = list(map(int, box))
                cv2.rectangle(img_debug, (b[0], b[1]), (b[2], b[3]), (0, 0, 0), 1)
                # landms
                cv2.circle(img_debug, (b[5], b[6]), 1, (0, 0, 0), 2)
                cv2.circle(img_debug, (b[7], b[8]), 1, (0, 0, 0), 2)
                cv2.circle(img_debug, (b[9], b[10]), 1, (0, 0, 0), 2)
                cv2.circle(img_debug, (b[11], b[12]), 1, (0, 0, 0), 2)
                cv2.circle(img_debug, (b[13], b[14]), 1, (0, 0, 0), 2)
            imgshow = np.hstack((image, img_debug))
            cv2.imshow("Test", imgshow)
            cv2.waitKey()
    with open('label_remake.txt', 'w') as f:
        for infor in new_label:
            for index, ele in enumerate(infor):
                if index == 0:
                    f.write("%s\n" % ele)
                    continue
                f.write("%s\n" % " ".join(str(e) for e in ele))

import sys

from model.mb_ssd_lite_f19 import create_mb_ssd_lite_f19, create_mb_ssd_lite_f19_predictor
from model.mb_ssd_lite_f38 import create_mb_ssd_lite_f38, create_mb_ssd_lite_f38_predictor
from model.mb_ssd_lite_f38_face import create_mb_ssd_lite_f38_face, create_mb_ssd_lite_f38_person_predictor
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd, create_rfb_tiny_mb_ssd_predictor
import timeit
from utils.misc import Timer
# from torchscope import scope
import argparse
import cv2
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector predictor With Pytorch')
parser.add_argument("--net_type", default="mb2-ssd-lite_f38", type=str,
                    help='mb2-ssd-lite_f19, mb2-ssd-lite_f38, rfb_tiny_mb2_ssd')
parser.add_argument('--model_path',
                    default='checkpoint/mb2-ssd-lite_f38/mb2-ssd-lite_f38-epoch-299-train_loss-7.06.pth',
                    help='model weight')
parser.add_argument('--label_path', default='utils/labels/head.txt',
                    help='class names lable')
parser.add_argument('--result_path', default='detect_results',
                    help='result path to save')
parser.add_argument('--test_path', default="/media/can/Data/Dataset/WiderFace/widerface/val/images/0--Parade",
                    help='path of folder test')
parser.add_argument('--test_device', default="cpu", type=str, help='cuda:0 or cpu')
args = parser.parse_args()


def load_model():
    class_names = [name.strip() for name in open(args.label_path).readlines()]
    if args.net_type == 'mb2-ssd-lite_f19':
        net = create_mb_ssd_lite_f19(len(class_names), is_test=True)
        net.load(args.model_path)
        predictor = create_mb_ssd_lite_f19_predictor(net, candidate_size=200)
    elif args.net_type == 'mb2-ssd-lite_f38':
        net = create_mb_ssd_lite_f38(len(class_names), is_test=True, device=args.test_device)
        predictor = create_mb_ssd_lite_f38_predictor(net, candidate_size=2000)
        net.load(args.model_path)
    elif args.net_type == 'mb2-ssd-lite_f38_person':
        net = create_mb_ssd_lite_f38_face(len(class_names), is_test=True, )
        predictor = create_mb_ssd_lite_f38_person_predictor(net, candidate_size=2000)
        net.load(args.model_path)
    elif args.net_type == 'rfb_tiny_mb2_ssd':
        net = create_rfb_tiny_mb_ssd(len(class_names), is_test=True, device=args.test_device)
        net.load(args.model_path)
        predictor = create_rfb_tiny_mb_ssd_predictor(net, candidate_size=5000, device=args.test_device)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    # scope(net, (3, 300, 300))
    return predictor


if __name__ == "__main__":
    tt_time = 0
    predictor = load_model()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    listdir = os.listdir(args.test_path)
    sum = 0

    for image_path in listdir:
        # orig_image = cv2.imread("/home/can/Desktop/0_Parade_Parade_0_628.jpg")
        orig_image = cv2.imread(os.path.join(args.test_path, image_path))
        new_image = orig_image
        # print(orig_image.shape)
        # orig_image = cv2.resize(orig_image, (640,480))

        old_size = orig_image.shape[:2]
        old_ratio = old_size[0] / old_size[1]
        # print(old_ratio)#0.5625
        new_height = old_size[0]
        new_width = old_size[1]
        # print(new_height)#540
        if old_ratio > 1:
            new_width = int(new_height)
            # print(new_height,new_width)

        elif old_ratio < 1:
            new_height = int(new_width)

        # im = cv2.resize(im, (new_width, new_height))

        delta_h = new_height - old_size[0]
        delta_w = new_width - old_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_image = cv2.copyMakeBorder(orig_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=color)

        # new_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        import time

        boxes, landmarks, probs, labels = predictor.predict(new_image, 2000, 0.5)
        # print(landmarks)
        probs = probs.numpy()
        sum += boxes.size(0)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            box = box.numpy()
            landmark = landmarks[i, :]
            cv2.rectangle(new_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(new_image, str(probs[i]), (int(box[0]), int(box[1] + 20)), cv2.FONT_HERSHEY_DUPLEX, 0.3,
                        (255, 255, 255))
            cv2.putText(new_image, str(labels[i]), (int(box[0] - 30), int(box[1] - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.3,
                        (255, 255, 255))
            cv2.circle(new_image, (int(landmark[0]), int(landmark[1])), 1, (0, 0, 255), 4)
            cv2.circle(new_image, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 255), 4)
            cv2.circle(new_image, (int(landmark[4]), int(landmark[5])), 1, (255, 0, 255), 4)
            cv2.circle(new_image, (int(landmark[6]), int(landmark[7])), 1, (0, 255, 0), 4)
            cv2.circle(new_image, (int(landmark[8]), int(landmark[9])), 1, (255, 0, 0), 4)
        cv2.putText(new_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.imshow("test", new_image)
        # cv2.waitKey()
        # new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(args.result_path, image_path), new_image)
        print(f"Found {len(probs)} object. The output image is {args.result_path}")

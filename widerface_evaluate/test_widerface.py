import sys

sys.path.append('/home/quannm/github/ssd_landmarks')
from model.mb_ssd_lite_f19 import create_mb_ssd_lite_f19, create_mb_ssd_lite_f19_predictor
from model.mb_ssd_lite_f38 import create_mb_ssd_lite_f38, create_mb_ssd_lite_f38_predictor
from model.mb_ssd_lite_f38_face import create_mb_ssd_lite_f38_face, create_mb_ssd_lite_f38_person_predictor
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd, create_rfb_tiny_mb_ssd_predictor

from utils.misc import Timer
# from torchscope import scope
import argparse
import cv2
import sys
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector predictor With Pytorch')
parser.add_argument("--net_type", default="mb2-ssd-lite_f38", type=str,
                    help='mb2-ssd-lite_f19, mb2-ssd-lite_f38, rfb_tiny_mb2_ssd')
parser.add_argument('--model_path',
                    default='checkpoint/mb2-ssd-lite_f38/mb2-ssd-lite_f38-epoch-299-train_loss-7.06.pth',
                    help='model weight')
parser.add_argument('--label_path', default='utils/labels/face.txt',
                    help='class names lable')
parser.add_argument('--result_path', default='detect_results', help='result path to save')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str,
                    help='Dir to save txt results')
parser.add_argument('--dataset_folder', default="/media/can/Data/Dataset/WiderFace/widerface/val/images",
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
        print("load weight done ")
    elif args.net_type == 'mb2-ssd-lite_f38_person':
        net = create_mb_ssd_lite_f38_face(len(class_names), is_test=True, device=args.test_device)
        predictor = create_mb_ssd_lite_f38_person_predictor(net, candidate_size=2000)
        net.load(args.model_path)
    elif args.net_type == 'rfb_tiny_mb2_ssd':
        net = create_rfb_tiny_mb_ssd(len(class_names), is_test=True, device=args.test_device)
        # net.load(args.model_path)
        predictor = create_rfb_tiny_mb_ssd_predictor(net, candidate_size=5000, device=args.test_device)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    # scope(net, (3, 300, 300))
    return predictor


if __name__ == '__main__':
    predictor = load_model()

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "/wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing begin
    for i, img_name in enumerate(test_dataset):
        print(i)
        image_path = testset_folder + img_name
        orig_image = cv2.imread(image_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, landmarks, labels, probs = predictor.predict(image, 2000, 0.5)
        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(boxes)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for index, box in enumerate(boxes):
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(probs[index])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)


        # save image
        if args.save_image:
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                landmark = landmarks[i, :]
                cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(orig_image, str(probs[i]), (int(box[0]), int(box[1] + 20)), cv2.FONT_HERSHEY_DUPLEX, 0.3,
                            (255, 255, 255))
                cv2.circle(orig_image, (int(landmark[0]), int(landmark[1])), 1, (0, 0, 255), 4)
                cv2.circle(orig_image, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 255), 4)
                cv2.circle(orig_image, (int(landmark[4]), int(landmark[5])), 1, (255, 0, 255), 4)
                cv2.circle(orig_image, (int(landmark[6]), int(landmark[7])), 1, (0, 255, 0), 4)
                cv2.circle(orig_image, (int(landmark[8]), int(landmark[9])), 1, (255, 0, 0), 4)
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, orig_image)

from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from models.efficientface import EfficientFace
from config import cfg_efficient
from utils.utils import decode_output

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default="weights/efficientface-d0/efficientface-d0_50.pth",
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--image_size', default=512,
                    help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str,
                    help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--dataset_folder', default='./data/widerface/test/images/', type=str, help='dataset path')
args = parser.parse_args()


def load_model(args, model):
    checkpoint = torch.load(args.trained_model, map_location='cuda:0')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('fpn'):
            continue
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def resize_image(image, common_size=512):
    height, width, _ = image.shape
    if height > width:
        scale = common_size / height
        resized_height = common_size
        resized_width = int(width * scale)
    else:
        scale = common_size / width
        resized_height = int(height * scale)
        resized_width = common_size

    image = cv2.resize(image, (resized_width, resized_height))
    cv2.imshow("12", image)
    new_image = np.zeros((common_size, common_size, 3))
    new_image[0:resized_height, 0:resized_width] = image
    return new_image


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    model = EfficientFace(cfg=cfg_efficient, phase='test')
    net = load_model(args, model)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_face_test_filelist.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # cv2.imshow("test raw", img_raw)
        # img = resize_image(img_raw)
        # cv2.imshow("Test", img)
        # cv2.waitKey()
        img_raw = np.float32(img_raw)

        img_meta = aspectaware_resize_padding(img_raw, width=args.image_size, height=args.image_size)
        img = img_meta[0]
        # cv2.imwrite(os.path.join('debug', os.path.basename(img_name)), img)
        img = img - (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device).type(torch.cuda.FloatTensor)
        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        device = torch.device('cpu')
        loc = loc.to(device)
        conf = conf.to(device)
        landms = landms.to(device)
        _t['misc'].tic()
        dets = decode_output(loc[0], conf[0], landms[0], cfg_efficient=cfg_efficient)
        _t['misc'].toc()
        new_w, new_h, old_w, old_h, padding_w, padding_h = img_meta[1:]
        dets[:, :4:2] = dets[:, :4:2] / (new_w / old_w)
        dets[:, 1:4:2] = dets[:, 1:4:2] / (new_h / old_h)
        dets[:, 5::2] = dets[:, 5::2] / (new_w / old_w)
        dets[:, 6::2] = dets[:, 6::2] / (new_h / old_h)
        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images,
                                                                                     _t['forward_pass'].average_time,
                                                                                     _t['misc'].average_time))

        # save image
        if args.save_image:
            for b in dets:
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)

import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import torch.nn.functional as F
import skimage.transform
import torchvision.transforms as transforms


class FaceDataset(data.Dataset):
    def __init__(self, root_path, file_name, preproc, target_transform=None):
        super(FaceDataset, self).__init__()
        self.path_images, self.labels = self.read_file(root_path, file_name)
        self.preproc = preproc

        self.target_transform = target_transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, idx):
        img = cv2.imread(self.path_images[idx])

        labels = self.labels[idx]

        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[7]  # l1_x
            annotation[0, 7] = label[8]  # l1_y
            annotation[0, 8] = label[10]  # l2_x
            annotation[0, 9] = label[11]  # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        debug = False
        if debug:
            img_debug = img.copy()
            for index, b in enumerate(annotations):
                b = [int(x) for x in b.tolist()]
                cv2.rectangle(img_debug, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

                # landms
                cv2.circle(img_debug, (b[4], b[5]), 1, (0, 0, 255), 4)
                cv2.circle(img_debug, (b[6], b[7]), 1, (0, 255, 255), 4)
                cv2.circle(img_debug, (b[8], b[9]), 1, (255, 0, 255), 4)
                cv2.circle(img_debug, (b[10], b[11]), 1, (0, 255, 0), 4)
                cv2.circle(img_debug, (b[12], b[13]), 1, (255, 0, 0), 4)

            name = "test_data.jpg"
            cv2.imwrite(name, img_debug)

        if self.preproc is not None:
            img, target = self.preproc(img, target)
        truths = target[:, :4]
        labels = target[:, -1]
        landms = target[:, 4:14]
        # TODO write landms to target_transforms
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return torch.from_numpy(img), target

    @staticmethod
    def read_file(root_path, file_name):
        path_images = []
        words = []
        file_name = os.path.join('/'.join(root_path.split('/')[:-1]), file_name)
        lines = list(open(file_name, 'r').readlines())
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
                path_images.append(os.path.join(root_path, image_name))
            else:
                label = [float(x) for x in line.split(' ')]
                labels.append(label)
        words.append(labels)
        return path_images, words

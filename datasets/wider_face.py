import os
import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
from imgaug import augmenters as iaa


class FaceDataset(data.Dataset):
    def __init__(self, root_path, file_name, preproc, target_transform=None):
        super(FaceDataset, self).__init__()
        self.path_images, self.labels = self.read_file(root_path, file_name)
        self.preproc = preproc
        self.augment = ImgAugTransform()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, idx):
        img_raw = cv2.imread(self.path_images[idx])
        img = self.augment(img_raw)
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
        debug = True
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

            img_show = np.hstack((img_debug, img_raw))
            cv2.imshow("test", img_show)
            cv2.waitKey()

        if self.preproc is not None:
            img, target = self.preproc(img, target)
        truths = target[:, :4]
        labels = target[:, -1]
        landms = target[:, 4:14]
        # TODO write landms to target_transforms
        # if self.target_transform:
        #     boxes,landms, labels = self.target_transform(boxes,landms,labels)
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


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.15, iaa.Grayscale(alpha=(0.05, 1.0))),
            iaa.Sometimes(0.2,
                          iaa.OneOf([
                              iaa.Sometimes(0.25, iaa.MotionBlur(k=5, angle=[-45, 45])),
                              iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.3)))
                          ])),
            iaa.Sometimes(0.2,
                          iaa.OneOf([
                              iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                              iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                              iaa.EdgeDetect(alpha=(0.0, 1.0)),
                              iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0)),
                              iaa.Canny(
                                  alpha=(0.0, 0.5),
                                  colorizer=iaa.RandomColorsBinaryImageColorizer(
                                      color_true=255,
                                      color_false=0
                                  )
                              )
                          ])),
            iaa.Sometimes(0.2,
                          iaa.OneOf([
                              iaa.GammaContrast((0.5, 2.0)),
                              iaa.GammaContrast((0.5, 2.0), per_channel=True),
                              iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                              iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
                              iaa.LogContrast(gain=(0.6, 1.4)),
                              iaa.LinearContrast((0.4, 1.6)),
                              iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                              iaa.CLAHE(clip_limit=(1, 10)),
                              iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization()),
                              iaa.Alpha((0.0, 1.0), iaa.HistogramEqualization()),
                              iaa.pillike.Autocontrast(cutoff=(0, 15.0)),
                          ])),
            iaa.Sometimes(0.2,
                          iaa.OneOf([
                              iaa.MultiplyBrightness(),
                              iaa.Add((-40, 40)),
                              iaa.Multiply((0.5, 1.5)),
                              iaa.Multiply((0.5, 1.5), per_channel=0.5),
                              iaa.Sometimes(0.15, iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)))
                          ])),
            iaa.Sometimes(0.15,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.Dropout2d(p=0.1),
                                     iaa.SaltAndPepper(0.1),
                                     ])),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(100)),
                                     iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0), upscale_method="linear"),
                                     iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(5)),
                                     iaa.BlendAlphaSomeColors(iaa.AveragePooling(7), from_colorspace="BGR"),
                                     iaa.BlendAlphaHorizontalLinearGradient(
                                         iaa.TotalDropout(1.0),
                                         min_value=0.2, max_value=0.8),
                                     iaa.BlendAlphaHorizontalLinearGradient(
                                         iaa.AveragePooling(11),
                                         start_at=(0.0, 1.0), end_at=(0.0, 1.0)),

                                     ])),
            iaa.Sometimes(0.15,
                          iaa.OneOf([
                              iaa.Clouds(),
                              iaa.Fog(),
                              iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
                              iaa.Rain(speed=(0.1, 0.3))
                          ])),
            # iaa.Sometimes(0.15,
            #               iaa.OneOf([
            #                   iaa.Cartoon(),
            #                   iaa.Cartoon(blur_ksize=3, segmentation_size=1.0,
            #                               saturation=2.0, edge_prevalence=1.0),
            #                   iaa.Superpixels(p_replace=0.5, n_segments=64),
            #                   iaa.Superpixels(p_replace=(0.1, 1.0), n_segments=(16, 128)),
            #                   iaa.UniformVoronoi(250, p_replace=0.9, max_size=None),
            #               ])),
        ], random_order=True)

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


if __name__ == "__main__":
    from datasets.data_augment import preproc

    loader = FaceDataset(root_path=os.path.join('/home/can/AI_Camera/EfficientFaceNet/data/widerface/train/images'), file_name='label.txt',
                                preproc=preproc(300, (127, 127, 127)))
    print(len(loader))
    for i in range(0, len(loader)):
        print("\n****")
        print(i)
        a = loader.__getitem__(i)
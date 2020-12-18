import os
import os.path
import torch
import cv2
import numpy as np
import skimage.transform
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from imgaug import augmenters as iaa


class FaceDataset(data.Dataset):
    def __init__(self, root_path, file_name, preproc, target_transform=None):
        super(FaceDataset, self).__init__()
        self.path_images, self.labels = self.read_file(root_path, file_name)
        self.preproc = preproc
        self.target_transform = target_transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, idx):
        img_raw = cv2.imread(self.path_images[idx])
        labels_image = self.labels[idx]

        annotations = np.zeros((0, 15))
        if len(labels_image) == 0:
            return annotations
        for idx, label in enumerate(labels_image):
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
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)

        if self.preproc is not None:
            img, target = self.preproc(img_raw, target)
        boxes = target[:, :4]
        labels_image = target[:, -1]
        landms = target[:, 4:14]
        if self.target_transform:
            boxes, landms, labels_image = self.target_transform(boxes, landms, labels_image)
        return torch.from_numpy(img), boxes, landms, labels_image

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


class ValDataset(data.Dataset):
    def __init__(self, txt_path, flip=False):
        self.imgs_path = []
        self.words = []
        self.transform = transforms.Compose([
            Resizer(),
            PadToSquare(),
            SubtractMeans(np.array([127, 127, 127])),
            # lambda img, boxes=None, labels=None: (img / 128.0, boxes, labels),
            lambda sample: {'img': (sample['img'] / 128.0), 'annot': sample['annot']},
            ToTensor()
        ])
        self.flip = flip
        self.batch_count = 0
        self.img_size = 300

        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = self.words[index]
        annotations = np.zeros((0, 4))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 4))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            annotations = np.append(annotations, annotation, axis=0)

        sample = {'img': img, 'annot': annotations}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample['img'], sample['annot']

    def __len__(self):
        return len(self.imgs_path)

    def _load_annotations(self, index):
        labels = self.words[index]
        annotations = np.zeros((0, 4))

        if len(labels) == 0:
            return annotations

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 4))
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            annotations = np.append(annotations, annotation, axis=0)

        return annotations


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    boxes, landms, labels = [], [], []
    imgs = []
    for _, sample in enumerate(batch):
        # for _, tup in enumerate(sample):
        #     if torch.is_tensor(tup):
        #         imgs.append(tup)
        #     elif isinstance(tup, type(np.empty(0))):
        #         annos = torch.from_numpy(tup).float()
        #         targets.append(annos)
        if isinstance(sample, np.ndarray):
            continue
        imgs.append(sample[0])
        boxes.append(sample[1])
        landms.append(sample[2])
        labels.append(sample[3])
    return torch.stack(imgs, 0), torch.stack(boxes, 0), torch.stack(landms, 0), torch.stack(labels, 0)


def detection_collate_valid(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    boxes, landms, labels = [], [], []
    imgs = []
    for _, sample in enumerate(batch):
        # for _, tup in enumerate(sample):
        #     if torch.is_tensor(tup):
        #         imgs.append(tup)
        #     elif isinstance(tup, type(np.empty(0))):
        #         annos = torch.from_numpy(tup).float()
        #         targets.append(annos)
        if isinstance(sample, np.ndarray):
            continue
        imgs.append(sample[0])
        boxes.append(sample[1])
    return torch.stack(imgs, 0), boxes


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.15, iaa.Grayscale(alpha=(0.05, 1.0))),
            # iaa.Sometimes(0.2,
            #               iaa.OneOf([
            #                   iaa.Sometimes(0.25, iaa.MotionBlur(k=5, angle=[-45, 45])),
            #                   # iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.3)))
            #               ])),
            # iaa.Sometimes(0.2,
            #               iaa.OneOf([
            #                   iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
            #                   iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
            #                   iaa.EdgeDetect(alpha=(0.0, 1.0)),
            #                   iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0)),
            # iaa.Canny(
            #     alpha=(0.0, 0.5),
            #     colorizer=iaa.RandomColorsBinaryImageColorizer(
            #         color_true=255,
            #         color_false=0
            #     )
            # )
            # ])),
            iaa.Sometimes(0.2,
                          iaa.OneOf([
                              iaa.GammaContrast((0.5, 2.0)),
                              # iaa.GammaContrast((0.5, 2.0), per_channel=True),
                              iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                              # iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
                              # iaa.LogContrast(gain=(0.6, 1.4)),
                              # iaa.LinearContrast((0.4, 1.6)),
                              # iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                              # iaa.CLAHE(clip_limit=(1, 10)),
                              # iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization()),
                              # iaa.Alpha((0.0, 1.0), iaa.HistogramEqualization()),
                              iaa.pillike.Autocontrast(cutoff=(0, 15.0)),
                          ])),
            iaa.Sometimes(0.2,
                          iaa.OneOf([
                              # iaa.MultiplyBrightness(),
                              # iaa.Add((-60, 60)),
                              # iaa.Multiply((0.1, 2.1)),
                              iaa.Multiply((0.5, 1.5), per_channel=0.5),
                              iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-60, 60))
                          ])),
            iaa.Sometimes(0.15,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     # iaa.Dropout2d(p=0.1),
                                     iaa.SaltAndPepper(0.1),
                                     ])),
            # iaa.Sometimes(0.25,
            #               iaa.OneOf([iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(100)),
            #                          iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0), upscale_method="linear"),
            #                          iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(5)),
            #                          iaa.BlendAlphaSomeColors(iaa.AveragePooling(7), from_colorspace="BGR"),
            #                          iaa.BlendAlphaHorizontalLinearGradient(
            #                              iaa.TotalDropout(1.0),
            #                              min_value=0.2, max_value=0.8),
            #                          iaa.BlendAlphaHorizontalLinearGradient(
            #                              iaa.AveragePooling(11),
            #                              start_at=(0.0, 1.0), end_at=(0.0, 1.0)),
            #
            #                          ])),
            # iaa.Sometimes(0.15,
            #               iaa.OneOf([
            #                   iaa.Clouds(),
            #                   # iaa.Fog(),
            #                   iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
            #                   iaa.Rain(speed=(0.1, 0.3))
            #               ])),
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


class Resizer(object):
    def __call__(self, sample, input_size=300):
        image, annots = sample['img'], sample['annot']

        rows, cols, _ = image.shape
        long_side = max(rows, cols)
        scale = input_size / long_side

        # resize image
        resized_image = skimage.transform.resize(image, (
            int(rows * input_size / long_side), int(cols * input_size / long_side)))
        resized_image = resized_image * 255

        assert (resized_image.shape[0] == input_size or resized_image.shape[
            1] == input_size), 'resized image size not {}'.format(input_size)

        if annots.shape[1] > 4:
            annots = annots * scale
        else:
            annots[:, :4] = annots[:, :4] * scale

        return {'img': resized_image, 'annot': annots}


class PadToSquare(object):
    def __call__(self, sample, input_size=300):
        image, annots = sample['img'], sample['annot']
        rows, cols, _ = image.shape
        dim_diff = np.abs(rows - cols)

        # relocate bbox annotations
        if rows == input_size:
            diff = input_size - cols
            annots[:, 0] = annots[:, 0] + diff / 2
            annots[:, 2] = annots[:, 2] + diff / 2
        elif cols == input_size:
            diff = input_size - rows
            annots[:, 1] = annots[:, 1] + diff / 2
            annots[:, 3] = annots[:, 3] + diff / 2
        if annots.shape[1] > 4:
            ldm_mask = annots[:, 4] > 0
            if rows == input_size:
                diff = input_size - cols
                annots[ldm_mask, 4::2] = annots[ldm_mask, 4::2] + diff / 2
            elif cols == input_size:
                diff = input_size - rows
                annots[ldm_mask, 5::2] = annots[ldm_mask, 5::2] + diff / 2

        # pad image
        img = torch.from_numpy(image)
        img = img.permute(2, 0, 1)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if rows <= cols else (pad1, pad2, 0, 0)

        padded_img = F.pad(img, pad, "constant", value=0)
        padded_img = padded_img.permute(1, 2, 0)

        annots = torch.from_numpy(annots)

        return {'img': padded_img.numpy(), 'annot': annots}


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = image.astype(np.float32)
        image -= self.mean
        return {'img': image.astype(np.float32), 'annot': annots}


class ToTensor(object):
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), 'annot': annots}


if __name__ == "__main__":
    from datasets.data_augment import preproc
    from module.ssd import MatchPrior
    from model.config import mb_ssd_lite_f38_config

    config = mb_ssd_lite_f38_config
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, config.iou_threshold)
    loader = FaceDataset(root_path=os.path.join('/media/can/Data/Dataset/WiderFace/widerface/train/images'),
                         file_name='label_remake.txt',
                         preproc=preproc(300, (127, 127, 127)),
                         target_transform=target_transform)
    print(len(loader))
    for i in range(0, len(loader)):
        # print("\n****")
        # print(i)
        a = loader.__getitem__(i)

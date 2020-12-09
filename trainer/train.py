import logging
import sys
import os
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from datasets.data_loader import _DataLoader
from module.ssd import MatchPrior
from datasets.data_preprocessing import TrainAugmentation, TestTransform
from torch.utils.data import DataLoader
from utils.loss import FocalLoss
import torch
from datasets.data_augment import preproc
from datasets.wider_face import FaceDataset, detection_collate
from utils.misc import Timer, freeze_net_layers
from utils.argument import _argument

timer = Timer()

args = _argument()


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    running_landmark_loss = 0.0
    training_loss = 0.0
    for i, data in enumerate(loader):
        print(".", end="", flush=True)
        images, boxes, landmarks_gt, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        landmarks_gt = landmarks_gt.to(device)

        optimizer.zero_grad()
        confidence, locations, landmarks = net(images)
        regression_loss, classification_loss, landmark_loss = criterion(confidence, locations, landmarks, labels, boxes,
                                                                        landmarks_gt)
        loss = regression_loss + classification_loss + landmark_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        running_landmark_loss += landmark_loss.item()
        if i and i % args.debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            avg_lmd_loss = running_landmark_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"train_avg_loss: {avg_loss:.4f}, " +
                f"train_reg_loss: {avg_reg_loss:.4f}, " +
                f"train_cls_loss: {avg_clf_loss:.4f}, " +
                f"train_lmd_loss: {avg_lmd_loss:.4f}, "
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            running_landmark_loss = 0.0
            training_loss = avg_loss

    return training_loss


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    running_landmark_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, landmarks_gt, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        landmarks_gt = landmarks_gt.to(device)
        num += 1
        with torch.no_grad():
            confidence, locations, landmarks = net(images)
            regression_loss, classification_loss, landmark_loss = criterion(confidence, locations, landmarks, labels,
                                                                            boxes, landmarks_gt)
            loss = regression_loss + classification_loss + landmark_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        running_landmark_loss += landmark_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num, running_landmark_loss / num


def data_loader(config):
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, config.iou_threshold)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")

    loader = FaceDataset(root_path=os.path.join('/media/can/Data/Dataset/WiderFace/widerface/train/images'),
                         file_name='label_remake.txt',
                         preproc=preproc(300, (127, 127, 127)),
                         target_transform=target_transform)
    logging.info("Train dataset size: {}".format(len(loader)))
    train_loader = DataLoader(loader, args.batch_size, num_workers=args.num_workers, shuffle=True,
                              collate_fn=detection_collate)
    if args.valid:
        # TODO: add validation dataset
        pass

    else:
        return train_loader


def create_network(create_net, device, num_classes=2):
    logging.info("Build network.")
    net = create_net(num_classes, device=device)
    # print(net)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(device)

    # criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
    #                          center_variance=0.1, size_variance=0.2, device=DEVICE)
    criterion = FocalLoss()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    return net, criterion, optimizer, scheduler

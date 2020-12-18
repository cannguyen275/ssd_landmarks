import parser
import sys
import os
import logging
import sys
import itertools
import torch

sys.path.append('/home/quannm/github/ssd_landmarks/')
# from torchscope import scope
# from torchsummary import summary
from utils.loss import MultiboxLoss, FocalLoss
from utils.argument import _argument
from trainer.train import train, test, data_loader, create_network
from model.mb_ssd_lite_f38 import create_mb_ssd_lite_f38
from model.config import mb_ssd_lite_f38_config
from model.mb_ssd_lite_f38_face import create_mb_ssd_lite_f38_face
from model.config import mb_ssd_lite_f38_face_config
from model.mb_ssd_lite_f19 import create_mb_ssd_lite_f19
from model.config import mb_ssd_lite_f19_config
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd
from model.config import rfb_tiny_mb_ssd_config
from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from eval import evaluate


# sys.path.append('/home/quannm/ssd_landmarks/')


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Train():
    '''
    The class to training
    '''

    def __init__(self):
        self.args = _argument()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        self.net, self.criterion, self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.config = self.get_model()
        self.dir_path = os.path.join(self.args.checkpoint_folder, self.args.net)
        self.config.priors = self.config.priors.to(self.device)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def get_model(self):
        timer = Timer()
        logging.info(self.args)

        if self.args.net == 'mb2-ssd-lite_f19':
            create_net = create_mb_ssd_lite_f19
            config = mb_ssd_lite_f19_config
        elif self.args.net == 'mb2-ssd-lite_f38':
            create_net = create_mb_ssd_lite_f38
            config = mb_ssd_lite_f38_config
        elif self.args.net == 'mb2-ssd-lite_f38_face':
            create_net = create_mb_ssd_lite_f38_face
            config = mb_ssd_lite_f38_face_config
        elif self.args.net == 'rfb_tiny_mb2_ssd':
            create_net = create_rfb_tiny_mb_ssd
            config = rfb_tiny_mb_ssd_config
        else:
            logging.fatal("The net type is wrong.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        train_loader, valid_loader = data_loader(config)
        net, criterion, optimizer, scheduler = create_network(create_net, self.device)
        return net, criterion, optimizer, scheduler, train_loader, valid_loader, config

    def training(self):
        print(self.dir_path)
        writer = SummaryWriter()
        for epoch in range(0, self.args.num_epochs):
            self.scheduler.step()
            # training_loss, avg_reg_loss, avg_clf_loss, avg_lmd_loss, learning_rate = train(self.train_loader, self.net,
            #                                                                                self.criterion,
            #                                                                                self.optimizer,
            #                                                                                device=self.device,
            #                                                                                debug_steps=self.args.debug_steps,
            #                                                                                epoch=epoch)
            # writer.add_scalar('model/regression_loss', avg_reg_loss, epoch)
            # writer.add_scalar('model/classification_loss', avg_clf_loss, epoch)
            # writer.add_scalar('model/landmark_loss', avg_lmd_loss, epoch)
            # writer.add_scalar('model/train_loss', training_loss, epoch)
            # writer.add_scalar('model/learning_rate', learning_rate, epoch)
            if epoch % self.args.validation_epochs == 0 or epoch == self.args.num_epochs - 1:
                if self.args.valid:
                    recall, precision = evaluate(self.val_loader, self.net, config=self.config)
                    logging.info(
                        f"Epoch: {epoch}, " +
                        f"recall: {recall:.4f}, " +
                        f"precision {precision:.4f}")
                    writer.add_scalar('model/precision', precision, epoch)
                    writer.add_scalar('model/recall', recall, epoch)
                    model_path = os.path.join(self.dir_path,
                                              f"{self.args.net}-epoch-{epoch}-train_loss-{round(training_loss, 2)}-recall-{round(recall, 2)}-precision-{round(precision, 2)}.pth")
                else:
                    model_path = os.path.join(self.dir_path,
                                              f"{self.args.net}-epoch-{epoch}-train_loss-{round(training_loss, 2)}.pth")
                self.net.is_test = True
                self.net.save(model_path)
                logging.info(f"Saved model {self.dir_path}")


if __name__ == '__main__':
    Train().training()

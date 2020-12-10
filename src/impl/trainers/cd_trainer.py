import os
import os.path as osp
from random import randint
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from skimage import io
from tqdm import tqdm

from core.trainer import Trainer
from utils.data_utils import (
    to_array, to_pseudo_color, 
    normalize_8bit,
    quantize_8bit as quantize
)
from utils.utils import mod_crop, HookHelper
from utils.metrics import (AverageMeter, Precision, Recall, Accuracy, F1Score)


class CDTrainer(Trainer):
    def __init__(self, settings):
        super().__init__(settings['model'], settings['dataset'], 'NLL', settings['optimizer'], settings)
        self.tb_on = (hasattr(self.logger, 'log_path') or self.debug) and self.ctx['tb_on']
        if self.tb_on:
            # Initialize tensorboard
            if hasattr(self.logger, 'log_path'):
                tb_dir = self.path(
                    'log', 
                    osp.join('tb', osp.splitext(osp.basename(self.logger.log_path))[0], '.'), 
                    name='tb', 
                    auto_make=True, 
                    suffix=False
                )
            else:
                tb_dir = self.path(
                    'log', 
                    osp.join('tb', 'debug', '.'), 
                    name='tb', 
                    auto_make=True, 
                    suffix=False
                )
                for root, dirs, files in os.walk(self.gpc.get_dir('tb'), False):
                    for f in files:
                        os.remove(osp.join(root, f))
                    for d in dirs:
                        os.rmdir(osp.join(root, d))
            self.tb_writer = SummaryWriter(tb_dir)
            self.logger.show_nl("\nTensorboard logdir: {}".format(osp.abspath(self.gpc.get_dir('tb'))))
            self.tb_intvl = int(self.ctx['tb_intvl'])
            
            # Global steps
            self.train_step = 0
            self.eval_step = 0

        # Whether to save network output
        self.out_dir = self.ctx['out_dir']
        self.save = (self.ctx['save_on'] or self.out_dir) and not self.debug

        self.val_iters = self.ctx['val_iters']
            
    def init_learning_rate(self):
        # Set learning rate adjustment strategy
        if self.ctx['lr_mode'] == 'const':
            return self.lr
        else:
            def _simple_scheduler_step(self, epoch, acc):
                self.scheduler.step()
                return self.scheduler.get_lr()[0]
            def _scheduler_step_with_acc(self, epoch, acc):
                self.scheduler.step(acc)
                # Only return the lr of the first param group
                return self.optimizer.param_groups[0]['lr']
            lr_mode = self.ctx['lr_mode']
            if lr_mode == 'step':
                self.scheduler = lr_scheduler.StepLR( 
                    self.optimizer, self.ctx['step'], gamma=0.5
                )
                self.adjust_learning_rate = partial(_simple_scheduler_step, self)
            elif lr_mode == 'exp':
                self.scheduler = lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=0.9
                )
                self.adjust_learning_rate = partial(_simple_scheduler_step, self)
            elif lr_mode == 'plateau':
                if self.load_checkpoint:
                    self.logger.warn("The old state of the lr scheduler will not be restored.")
                self.scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', factor=0.5, threshold=1e-4
                )
                self.adjust_learning_rate = partial(_scheduler_step_with_acc, self)
                return self.optimizer.param_groups[0]['lr']
            else:
                raise NotImplementedError

            if self.start_epoch > 0:
                # Restore previous state
                # FIXME: This will trigger pytorch warning "Detected call of `lr_scheduler.step()` 
                # before `optimizer.step()`" in pytorch 1.1.0 and later.
                # Perhaps I should store the state of scheduler to a checkpoint file and restore it from disk.
                last_epoch = self.start_epoch
                while self.scheduler.last_epoch < last_epoch:
                    self.scheduler.step()
            return self.scheduler.get_lr()[0]

    def train_epoch(self, epoch):
        losses = AverageMeter()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)
        
        self.model.train()
        
        for i, (t1, t2, tar) in enumerate(pb):
            t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
            
            show_imgs_on_tb = self.tb_on and (i%self.tb_intvl == 0)
            
            prob = self.model(t1, t2)
            
            loss = self.criterion(prob, tar)
            
            losses.update(loss.item(), n=self.batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_train, losses.val, losses.avg)

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

            if self.tb_on:
                # Write to tensorboard
                self.tb_writer.add_scalar("Train/loss", losses.val, self.train_step)
                if show_imgs_on_tb:
                    self.tb_writer.add_image("Train/t1_picked", normalize_8bit(t1.detach()[0]), self.train_step)
                    self.tb_writer.add_image("Train/t2_picked", normalize_8bit(t2.detach()[0]), self.train_step)
                    self.tb_writer.add_image("Train/labels_picked", tar[0].unsqueeze(0), self.train_step)
                    self.tb_writer.flush()
                self.train_step += 1

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = AverageMeter()
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(), Recall(), F1Score(), Accuracy())

        self.model.eval()

        with torch.no_grad():
            for i, (name, t1, t2, tar) in enumerate(pb):
                if self.is_training and i >= self.val_iters:
                    # This saves time
                    pb.close()
                    self.logger.warn("Evaluation ends early.")
                    break
                t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)

                prob = self.model(t1, t2)

                loss = self.criterion(prob, tar)
                losses.update(loss.item(), n=self.batch_size)

                # Convert to numpy arrays
                cm = to_array(torch.argmax(prob[0], 0)).astype('uint8')
                tar = to_array(tar[0]).astype('uint8')

                for m in metrics:
                    m.update(cm, tar)

                desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_eval, losses.val, losses.avg)
                for m in metrics:
                    desc += " {} {:.4f} ({:.4f})".format(m.__name__, m.val, m.avg)

                pb.set_description(desc)
                self.logger.dump(desc)

                if self.tb_on:
                    self.tb_writer.add_image("Eval/t1", normalize_8bit(t1[0]), self.eval_step)
                    self.tb_writer.add_image("Eval/t2", normalize_8bit(t2[0]), self.eval_step)
                    self.tb_writer.add_image("Eval/labels", quantize(tar), self.eval_step, dataformats='HW')
                    prob = quantize(to_array(torch.exp(prob[0,1])))
                    self.tb_writer.add_image("Eval/prob", to_pseudo_color(prob), self.eval_step, dataformats='HWC')
                    self.tb_writer.add_image("Eval/cm", quantize(cm), self.eval_step, dataformats='HW')
                    self.eval_step += 1
                
                if self.save:
                    self.save_image(name[0], quantize(cm), epoch)

        if self.tb_on:
            self.tb_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            self.tb_writer.add_scalars("Eval/metrics", {m.__name__.lower(): m.avg for m in metrics}, self.eval_step)

        return metrics[2].avg   # F1-score

    def save_image(self, file_name, image, epoch):
        file_path = osp.join(
            'epoch_{}'.format(epoch),
            self.out_dir,
            file_name
        )
        out_path = self.path(
            'out', file_path,
            suffix=not self.ctx['suffix_off'],
            auto_make=True,
            underline=True
        )
        return io.imsave(out_path, image)

    # def __del__(self):
    #     if self.tb_on:
    #         self.tb_writer.close()
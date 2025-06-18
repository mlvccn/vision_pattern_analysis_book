import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import argparse

import data_loader
from utils import init_weights
from .tester import Tester

# 此处为简化版
class LightningSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hparams.act_thresh = np.linspace(
            self.hparams.act_thresh_params[0],
            self.hparams.act_thresh_params[1],
            self.hparams.act_thresh_params[2])
        self.hparams.tIoU_thresh = np.arange(*self.hparams.tIoU_thresh_params)

        self.net = HAMNet(hparams)
        self.tester = Tester(self.hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve")
        parser.add_argument("--model_name", type=str, default="dev0.5_anet")
        parser.add_argument("--rat", type=int, default=20, help="topk value")
        parser.add_argument("--rat2", type=int, default=5, help="topk value")
        parser.add_argument("--beta", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--num_segments", type=int, default=80)
        return parser

    def train_dataloader(self):
        dataset = data_loader.Dataset(self.hparams, mode='train', sampling=self.hparams.sampling)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers)
        return dataloader

    def test_dataloader(self):
        dataset = data_loader.Dataset(self.hparams, mode='test', sampling='all')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers)
        self.class_dict = dataset.class_dict
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr, weight_decay=0.001)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        total_loss, tqdm_dict = self._loss_fn(batch)
        self.log_dict(tqdm_dict, prog_bar=False)
        return total_loss

    def test_step(self, batch, batch_idx):
        self.tester.eval_one_batch(batch, self, self.class_dict)

    def test_epoch_end(self, outputs):
        mAP = self.tester.final(logger=self.logger.experiment, class_dict=self.class_dict)
        mAP = torch.tensor(mAP).to(self.device)
        self.log("mAP", mAP, prog_bar=True)
        self.tester = Tester(self.hparams)

    def _loss_fn(self, batch):
        features, labels, segm, vid_name, _ = batch
        element_logits, atn_supp, atn_drop, element_atn = self.net(features)
        # 关键损失计算逻辑（省略部分细节）
        total_loss = (self.hparams.lm_1 * loss_1 + self.hparams.lm_2 * loss_2 +
                      self.hparams.alpha * loss_norm + self.hparams.beta * loss_guide)
        return total_loss, tqdm_dict

    def topkloss(self, element_logits, labels, is_back=True, rat=8):
        # 关键topk损失计算逻辑（省略部分细节）
        return milloss, topk_ind
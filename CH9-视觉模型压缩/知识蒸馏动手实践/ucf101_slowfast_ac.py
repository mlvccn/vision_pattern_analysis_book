import argparse
import os
import torch
import numpy as np
import math
import torch.nn as nn
from dataset.ucf101 import get_dataset
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, OPT_PATH, get_cfg_custom, MODEL_TO_CKPTS
import tqdm
import torch.nn.functional as F
from thop import profile
import random
from distillers.KD import KD
from distillers.DKD import DKD
from distillers.CrossKD import CrossKD
from collections import Counter
checkpoint_path='/log'
train_state_path='/log'
def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(
            checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(
            train_state_path, 'checkpoint-{}_optimizer.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
    return start_epoch


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num,distiller):
    net.train()
    optimizer.zero_grad()

    total_loss=0.0
    total_correct=0
    
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (input, target,index) in enumerate(pbar):
            input = input.float().cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            diff=hisloss[index].mean()
            output,ce_loss, kd_loss =  distiller(input,target)
            loss = ce_loss+ diff*kd_loss

            total_loss+=loss.item()

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(output, dim=1)
            correct = (predictions == target).sum().item()
            total_correct += correct
            pbar.set_description(f"Epoch {epoch}, KDLoss: {kd_loss.item():.4f}, CEloss:{ce_loss.item():.4f},Accuracy: {correct / cfg.CONFIG.TRAIN.BATCH_SIZE:.4f}")
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / len(data_loader.dataset)
    print(f"Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {avg_accuracy}")
    return avg_loss, avg_accuracy

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def validate_rgb(val_loader, net, top1, top5):
    with torch.no_grad():
        with tqdm.tqdm(val_loader, total=len(val_loader), ncols=0) as pbar:
            for n_iter, (input, target,index) in enumerate(pbar):
                input = input.float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output,_ = net(input)
                fusion = F.softmax(output, dim=1)

                prec1, prec5 = accuracy(fusion, target, topk=(1, 5))
                top1.update(prec1.item())
                top5.update(prec5.item())

def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states
def save_model(epoch, model, optimizer):
    torch.save(model.module.state_dict(),
               os.path.join(args.adv_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(args.adv_path, 'checkpoint-{}_optimizer.ckpt'.format(epoch)))
def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N')
    parser.add_argument('--model', type=str, default='videoST', help='videoST | slowfast_resnet101 | tpn_resnet101.')
    parser.add_argument('--file_prefix', type=str, default='')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, 'UCF-{}'.format(args.model, args.file_prefix))
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    args = arg_parse()
    gpu_id = [0,1,2,3]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    print (args)
    set_seed(3407)
    best_acc=0.0
    # Loading Cfg
    cfg_path = CONFIG_PATHS[args.model]
    cfg = get_cfg_custom(cfg_path, args.batch_size)
    cfg.CONFIG.MODEL.PRETRAINED = True
    print(cfg)
    model= get_model(cfg)
    model.fc = nn.Linear(2304, 101)

    cfg_patht=CONFIG_PATHS[cfg.CONFIG.MODEL.TEACHER]
    cfgt = get_cfg_custom(cfg_patht, args.batch_size)
    ckpt_patht = MODEL_TO_CKPTS[cfg.CONFIG.MODEL.TEACHER]
    teacher= get_model(cfgt)
    teacher.fc = nn.Linear(2304, 101)
    teacher.load_state_dict(torch.load('../checkpoint/slowfast_resnet101.ckpt'))
    model = nn.DataParallel(model, device_ids=gpu_id).cuda()
    teacher = nn.DataParallel(teacher, device_ids=gpu_id).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.CONFIG.TRAIN.LR,
                                momentum=0.9,
                                weight_decay=cfg.CONFIG.TRAIN. W_DECAY)

    # Loading Dataset
    val_loader = get_dataset(cfg.CONFIG.DATA.VAL_ANNO_PATH, args.batch_size)
    train_loader=get_dataset(cfg.CONFIG.DATA.TRAIN_ANNO_PATH, args.batch_size)
    choose_loader=get_dataset(cfg.CONFIG.DATA.TRAIN_ANNO_PATH, 48,False)
    #Init Distiller
    distiller=KD(model,teacher)
    #distiller=DKD(model,teacher)
    #istiller=CrossKD(model,teacher
    resume=0
    start_epoch = resume_training(resume, model, optimizer)
    epoch_step_num = len(train_loader)
    select=torch.zeros(len(train_loader.dataset))
    hisloss=torch.zeros(len(train_loader.dataset))
    featurelist = [torch.zeros(2304).cuda() for _ in range(len(train_loader.dataset))]
    #Start Training
    for i in range(start_epoch, cfg.CONFIG.TRAIN.EPOCH_NUM+ 1):
        model.train()
        run_one_epoch(i, model, optimizer, train_loader, len( train_loader),distiller)
        top1_rgb = AverageMeter()
        top5_rgb = AverageMeter()
        model.eval()
        validate_rgb(val_loader, model, top1_rgb, top5_rgb)
        print('* RGB Prec@1 {top1_rgb.avg:.3f} Prec@5 {top5_rgb.avg:.3f}'.format(top1_rgb=top1_rgb, top5_rgb=top5_rgb))

        if best_acc<top1_rgb.avg:
            save_model(i, model, optimizer)
            best_acc= top1_rgb.avg







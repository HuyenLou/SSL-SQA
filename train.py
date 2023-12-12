import argparse 
import logging
import shutil
import time
import os
import math
import random

import numpy as np
import torch
import torch.nn.functional as F  
import torch.optim as optim 

from utils import right_pad, AverageMeter, accuracy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_data
from model import SSL_SQAmodel
from conformer.model import Conformer

logger = logging.getLogger(__name__)

sampling_rate = 16000
train_path = ""
val_path = ""

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def main():
    parser = argparse.ArgumentParser('Pytorch SER Fimatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help = 'id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=9, 
                        help='nume of workers')
    parser.add_argument('--num-labeled', type=int, default=4,
                        help='number of labeled data')
    parser.add_argument('--total-steps', default=30000, type=int,
                        help='--number of total stpos to run')
    parser.add_argument('--eval-step', default= 3000, type=int,
                        help='number of eval steps')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')    
    parser.add_argument("--batch-size", default=2, type=int,
                        help='train batch size')
    parser.add_argument('--lr', '--learning-rate', default=1.5e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (classification data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of classification loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")   
    
    args = parser.parse_args()
    
    
        
    def collate_fn(batch):
        max_lengh_1 = 0
        max_lengh_2 =0
        for x in batch:
            if max_lengh_1 < len(x[0][0]):
                max_lengh_1 = len(x[0][0])
            if max_lengh_2 < len(x[1][0]):
                max_lengh_2 = len(x[1][0])
            
        speech_array_1 = torch.stack([right_pad(batch[i][1], max_lengh_1) for i in range(len(batch))])
        speech_array_2 = torch.stack([right_pad(batch[i][0], max_lengh_2) for i in range(len(batch))])
        
        subaudio_pair = []
        labels = []
        for i in range(len(batch)):
            for j in range(len(batch)):
                if i == j :
                    subaudio_pair.append(torch.cat(speech_array_1[i], speech_array_2[j]))
                    labels.append(1)
                else:
                    subaudio_pair.append(torch.cat(speech_array_1[i], speech_array_2[j]))
                    labels.append(0)
        return {
            "pairs": torch.stack(subaudio_pair[i] for i in range(len(subaudio_pair))),
            "labels": torch.tensor(labels),
        }
                    
                    
    def create_model(input_dim):
        model = SSL_SQAmodel(input_dim = input_dim,
                                num_heads=4,
                                ffn_dim=128,
                                num_layers=4,
                                depthwise_conv_kernel_size=31)
        return model
    
    device = torch.device("cuda", args.gpu_id)
    args.world_size = 1
    args.n_gpu  = torch.cuda.device_count()
    
    args.device = device
    
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

    if args.seed is not None:
        set_seed(args)   
        
    train_dataset, val_dataset = get_data(train_path, val_path)
    
    train_sampler = RandomSampler
    
    trainloader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True, collate_fn = collate_fn)

    val_loader = DataLoader(
        val_dataset,
        sampler=train_sampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True, collate_fn = collate_fn)   
    
    model = create_model()
    
    model.to(args.device)
    
    args.epochs = math.ceil(args.total_steps / args.eval_steps)
    
    no_decay = ["bias", "bn"]
    
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)   
    
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, trainloader, val_loader,
          model, optimizer, scheduler)      
    
def train(args, trainloader, val_loader, model, optimizer, scheduler):
    if args.amp:
        from apex import amp
        
    global best_acc
    test_accs = []
    end = time.time()
    
    train_iter = iter(trainloader)
    
    model.train()   
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()  
        
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        
        for batch_idx in range(args.eval_step):
            dict_input_target = next(train_iter)
            inputs = dict_input_target["pairs"]
            targets = dict_input_target["labels"]
            
            data_time.update(time.time() - end)
            batch_size = inputs.shape[0]
            
            targets= targets.to(args.device) 
            
            logits = model(inputs.squeeze(dim=1).to(args.device))
            
            loss = F.cross_entropy(logits, targets, reduction="mean")
            
            if args.apm:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            losses.updated(loss.item())
            
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()
                
        if not args.no_progress:
            p_bar.close()        
        
        test_loss, test_acc = test(args, val_loader, model, epoch)  
          
        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
        
        best_acc = max(test_acc, best_acc)
        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        
        args.writer.close()
        
def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    end = time.time()
    if not args.no_progress:
        test_loader = tqdm(test_loader)
        
    model.eval()
    with torch.no_grad():
        for batch_idx, dict_input_target in enumerate(test_loader):
            data_time.update(time.time() - end)
            
            inputs = dict_input_target["pairs"]
            targets = dict_input_target["labels"]
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(torch.squeeze(inputs, dim=1))
            loss = F.cross_entropy(outputs, targets)

            prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top2: {top2:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                ))
        if not args.no_progress:
            test_loader.close()
            
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    
    model.train()
    return losses.avg, top1.avg

if __name__ == '__main__':
    main()
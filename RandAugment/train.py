# sbatch batch.sh "RandAugment/train.py -c confs/best_cnn.yaml --tag runs"
# sbatch batch.sh "RandAugment/train.py -c confs/best_cnn.yaml --tag runs -d saved_data/gmaxup_cifar-10-batches-py-orig-1l-1s-randaug_sim-6.05.20-15:50 -n randaug_sim"
# sbatch batch.sh "RandAugment/train.py -c confs/best_cnn.yaml --tag runs -d saved_data/gmaxup_cifar-randaug_cache -n randaug_cache"

import sys
sys.path.append("/sailhome/acai21/mo_workspace/gmaxup-augmentation")

import itertools
import json
import logging
import math
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from RandAugment.common import get_logger
from RandAugment.data import get_dataloaders
from RandAugment.lr_scheduler import adjust_learning_rate_resnet
from RandAugment.metrics import accuracy, Accumulator
from RandAugment.networks import get_model, num_class
from warmup_scheduler import GradualWarmupScheduler

from RandAugment.common import add_filehandler
from RandAugment.smooth_ce import SmoothCrossEntropyLoss

import data_loading
import numpy as np
import datetime
from data_loading import DatasetFromTupleList

logger = get_logger('RandAugment')
logger.setLevel(logging.INFO)


def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch'])) # eg. [train 0024/0025]

    metrics = Accumulator()
    cnt = 0

    for steps, (data, label) in enumerate(loader):
        first_train = desc_default == "train" and epoch == 1
        first_valid = desc_default == "valid" and epoch == 5
        first_test = desc_default == "*test" and epoch == 5

        if steps == 0 and (first_train or first_valid or first_test):
            data_loading.show_images(writer, data, 128, desc_default)

        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)

        if optimizer:
            loss.backward()
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        # eg. [00:13<00:00, 29.15it/s, loss=0.753, top1=0.736, top5=0.982, lr=0.001]
        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

        # if scheduler is not None:
        #     scheduler.step(epoch - 1 + float(steps) / len(loader))

        del preds, loss, top1, top5, data, label

    # if tqdm_disable:
    #     if optimizer:
    #         logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
    #     else:
    #         logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def build_save_str(args):
    optional_tokens = [] 
    if "gmaxup_cifar" in args.dataroot:
        optional_tokens.append("gmaxup")
    if args.name:
        optional_tokens.append(args.name)

    optional_str = ""
    if len(optional_tokens):
        for token in optional_tokens:
            optional_str += "{}-".format(token)

    return '{}e-{}-{}{}'.format(
        C.get()['epoch'], 
        C.get()['aug'], # augmentation string
        optional_str, # optional string 
        datetime.datetime.now().strftime("%-m.%d.%y-%H:%M:%s:%f"))


def train_and_eval(tag, dataroot, save_str, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False):
    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    # _, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold)
    trainloader, validloader, testloader_ = data_loading.build_wrapped_dl(C.get()['aug'], dataroot)
    print("len(trainloader): {}".format(len(trainloader)))
    print("len(validloader): {}".format(len(validloader)))
    print("len(testloader_): {}".format(len(testloader_)))

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']))

    criterion = nn.CrossEntropyLoss()
    # lb_smooth = C.get()['optimizer'].get('label_smoothing', 0.0)
    # if lb_smooth > 0.0:
    #     criterion = SmoothCrossEntropyLoss(lb_smooth)
    # else:
    #     criterion = nn.CrossEntropyLoss()

    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    elif C.get()['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=C.get()['lr'],
            weight_decay=C.get()['optimizer'].get('decay', 0.0),
        )
    # else:
    #     raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    # if C.get()['optimizer'].get('lars', False):
    #     from torchlars import LARS
    #     optimizer = LARS(optimizer)
    #     logger.info('*** LARS Enabled.')

    # scheduler = None
    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    elif lr_scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    from torch.utils.tensorboard import SummaryWriter
    writers = [SummaryWriter(log_dir='./runs/{}/{}'.format(save_str, x)) for x in ['train', 'valid', 'test']]
    
    result = OrderedDict()
    epoch_start = 1

    logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
    if only_eval:
        logger.warning('model checkpoint not found. only-evaluation mode is off.')
    only_eval = False
    # if save_path and os.path.exists(save_path):
    #     logger.info('%s file found. loading...' % save_path)
    #     data = torch.load(save_path)
    #     if 'model' in data or 'state_dict' in data:
    #         key = 'model' if 'model' in data else 'state_dict'
    #         logger.info('checkpoint epoch@%d' % data['epoch'])
    #         if not isinstance(model, DataParallel):
    #             model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
    #         else:
    #             model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
    #         optimizer.load_state_dict(data['optimizer'])
    #         if data['epoch'] < C.get()['epoch']:
    #             epoch_start = data['epoch']
    #         else:
    #             only_eval = True
    #     else:
    #         model.load_state_dict({k: v for k, v in data.items()})
    #     del data
    # else:
    #     logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
    #     if only_eval:
    #         logger.warning('model checkpoint not found. only-evaluation mode is off.')
    #     only_eval = False

    # if only_eval:
    #     logger.info('evaluation only+')
    #     model.eval()
    #     rs = dict()
    #     rs['train'] = run_epoch(model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0])
    #     rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1])
    #     rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])
    #     for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
    #         if setname not in rs:
    #             continue
    #         result['%s_%s' % (key, setname)] = rs[setname][key]
    #     result['epoch'] = 0
    #     return result

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, 
            desc_default='train', epoch=epoch, writer=writers[0], verbose=True, scheduler=scheduler)
        model.eval()

        # if math.isnan(rs['train']['loss']):
        #     raise Exception('train loss is NaN.')

        if epoch % 5 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None, 
                desc_default='valid', epoch=epoch, writer=writers[1], verbose=True)
            rs['test'] = run_epoch(model, testloader_, criterion, None, 
                desc_default='*test', epoch=epoch, writer=writers[2], verbose=True)

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                # reporter(
                #     loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                #     loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                # )

                # save checkpoint
                if save_path:
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path.replace('.pth', '_e%d_top1_%.3f_%.3f' % (epoch, rs['train']['top1'], rs['test']['top1']) + '.pth'))

    del model

    result['top1_test'] = best_top1
    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('-d', '--dataroot', type=str, default='cifar-10-batches-py', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--cv-ratio', type=float, default=0.15)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('-n', '--name', 
        help = 'Add optional string to save files.', 
        type=str)
    args = parser.parse_args()

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    if args.save:
        add_filehandler(logger, args.save.replace('.pth', '') + '.log')

    # logger.info(json.dumps(C.get().conf, indent=4))

    import time
    t = time.time()
    print("Using GPU: {}".format(torch.cuda.is_available()))

    save_str = build_save_str(args)
    logger.info(save_str)
    result = train_and_eval(args.tag, args.dataroot, save_str, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, only_eval=args.only_eval, metric='test')
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    # logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)

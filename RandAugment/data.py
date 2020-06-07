import logging
import os

import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler
from torch.utils.data.dataset import ConcatDataset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C

from RandAugment.augmentations import *
from RandAugment.common import get_logger
from RandAugment.imagenet import ImageNet

from RandAugment.augmentations import Lighting, RandAugment

from pathlib import Path
import pickle
import data_loading

logger = get_logger('RandAugment')
logger.setLevel(logging.INFO)

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def get_dataloaders(dataset, batch, dataroot, split=0.15, split_idx=0):
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train = transforms.Compose([
            # randaugment 
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    logger.debug('augmentation: %s' % C.get()['aug'])

    if C.get()['aug'] == 'randaugment':
        transform_train.transforms.insert(0, RandAugment(C.get()['randaug']['N'], C.get()['randaug']['M']))
        print("Rand augment.")
    elif C.get()['aug'] == 'none':
        print("No rand augment.")
    else:
        raise ValueError('not found augmentations. %s' % C.get()['aug'])

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if dataset == 'cifar10':
        dataroot = Path(__file__).parent.parent.parent / "saved_data"
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)

    train_sampler = None
    if split > 0.0: # 0 by default 
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)
            print("len(train_idx): {}".format(len(train_idx)))
            print("len(valid_idx): {}".format(len(valid_idx)))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    print("len(total_trainset): {}".format(len(total_trainset)))
    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=32, pin_memory=True,
        sampler=train_sampler, drop_last=True)

    # augmented_batch = []
    # for i, (images, labels) in enumerate(trainloader):
    #     if i % 50 == 0: 
    #         print("{} batches dumped.".format(i))
    #     for sample_num, (x, y) in enumerate(zip(images, labels)):
    #         if i == 0 and sample_num == 0:
    #             print(type(x))
    #             print(type(y))
    #         augmented_batch.append((x.cpu(), y.cpu()))

    # print("finished iterating")
    # print(len(augmented_batch))

    # augmented_cifar10_ds = data_loading.DatasetFromTupleList(augmented_batch)
    # print("generated dataset")
    # print(len(augmented_cifar10_ds))

    # pickle.dump(augmented_cifar10_ds,
    #     open("gmaxup_cifar-vanilla_randaugment", 'wb'), 
    #     protocol=pickle.HIGHEST_PROTOCOL)

    # print("DUMP COMPLETE")

    print("len(trainloader): {}".format(len(trainloader)))
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=16, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=32, pin_memory=True,
        drop_last=False
    )
    return train_sampler, trainloader, validloader, testloader


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

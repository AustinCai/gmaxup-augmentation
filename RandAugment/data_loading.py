import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import pickle
from torch.utils.data import Dataset
import time

# import util
# from util import Constants
# from util import Objects
# from util import BasicTransforms
import augmentations
import sys
from torch.utils.tensorboard import SummaryWriter

class Objects:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Constants:
    save_str = "untitled_save_str"

    cifar10_dim = (32, 32, 3)

    batch_size = 128
    learning_rate = 1e-3
    out_channels = 10
    model_str = "best_cnn"

    randaugment_n = 1
    randaugment_m = 2

    dataset_str = "cifar10"

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

class BasicTransforms:
    pil_image_to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
        )
    none = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            + [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)] 
        )
    randaugment = transforms.Compose(
            [augmentations.RandAugment(Constants.randaugment_n, Constants.randaugment_m)]
            + [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            + [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
        )

# =======================================================================================================


def show_images(writer, images, batch_size, title="Images", verbose=False):
    '''Displays the input images passed in through train_dl, both to the console
    and to tensorboard. 
    '''
    start_time = time.time()

    images = images.view(batch_size, Constants.cifar10_dim[2], \
        Constants.cifar10_dim[0], Constants.cifar10_dim[1]) 
    norm_min, norm_max = -1, 1
    img_grid = torchvision.utils.make_grid(images, normalize=True, range=(norm_min, norm_max))
    if verbose: 
        print("In visualize.show_images(title={}).".format(title))
        print("    images.shape: {}.".format(images.shape))
        print("    img_grid.shape: {}.".format(img_grid.shape))

    # format_and_show(img_grid, one_channel=False)
    writer.add_image(title, img_grid)

    print("    visualize.show_images() completed in {} seconds.".format(time.time() - start_time))


def build_cifar10_ds(dataset_root_path="saved_data", 
    train_transform=BasicTransforms.pil_image_to_tensor, 
    test_transform=BasicTransforms.pil_image_to_tensor):
    '''Loads and returns training and test datasets, applying the provided 
    transform functions. Because cifar10 is a built-in dataset, only the 
    dataset root path must be specified. 
    '''
    
    print(Path(__file__).parent.parent.resolve().parent.resolve() / dataset_root_path)
    train_ds = torchvision.datasets.CIFAR10(
        Path(__file__).parent.parent.resolve().parent / dataset_root_path, 
        train=True, transform=train_transform, download=False)
    valid_test_ds = torchvision.datasets.CIFAR10(
        Path(__file__).parent.parent.resolve().parent / dataset_root_path, 
        train=False, transform=test_transform, download=False)
    return train_ds, valid_test_ds


def build_mnist_ds(dataset_root_path="saved_data", 
    train_transform=BasicTransforms.pil_image_to_tensor,
    test_transform=BasicTransforms.pil_image_to_tensor):
    '''Loads and returns training and test datasets, applying the provided 
    transform functions. Because mnist is a built-in dataset, only the
    dataset root path must be specified. 
    '''
    train_ds = torchvision.datasets.MNIST(
        Path(__file__).parent.parent.parent / dataset_root_path, 
        train=True, transform=train_transform, download=False)
    valid_test_ds = torchvision.datasets.MNIST(
        Path(__file__).parent.parent.parent / dataset_root_path, 
        train=False, transform=test_transform, download=False)
    return train_ds, valid_test_ds


def build_custom_cifar10_ds(dataset_path, 
    test_transform=BasicTransforms.pil_image_to_tensor):

    print(Path(__file__).parent.parent.resolve().parent / dataset_path)

    print("before pickle load")
    file = open(Path(__file__).parent.parent.resolve().parent / dataset_path, 'rb')
    print(file)
    train_ds = pickle.load(file)
    file.close()
    print("after pickle load")

    _, valid_test_ds = build_cifar10_ds(test_transform = test_transform)

    return train_ds, valid_test_ds

def build_dl(augmentation_str, dataset_str, train_valid_split=0.85, shuffle=True, verbose=False): 
    '''Constructs and loads training and test dataloaders, which can be iterated 
    over to return one batch at a time. 
    '''
    transform = BasicTransforms.pil_image_to_tensor if augmentation_str == "none" \
        else getattr(BasicTransforms, augmentation_str)

    # built-in datasets
    if dataset_str == "mnist":
        train_ds, valid_test_ds = build_mnist_ds("saved_data", transform)           
    elif dataset_str == "cifar-10-batches-py":
        train_ds, valid_test_ds = build_cifar10_ds("saved_data", transform)

    # custom datasets
    elif "gmaxup_cifar" in dataset_str:
        train_ds, valid_test_ds = build_custom_cifar10_ds(dataset_str)
    else:
        raise Exception("Invalid dataset path {}".format(dataset_str))


    valid_dl = []
    if train_valid_split < 1.0: 
        train_ds, valid_ds = torch.utils.data.random_split(
            train_ds, [int(len(train_ds)*train_valid_split), len(train_ds) - int(len(train_ds)*train_valid_split)])

        valid_dl = torch.utils.data.DataLoader(
            valid_ds, batch_size=Constants.batch_size, shuffle=shuffle, drop_last=True, num_workers=32, pin_memory=True)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=Constants.batch_size, shuffle=shuffle, drop_last=True, num_workers=32, pin_memory=True)

    test_dl = torch.utils.data.DataLoader(
        valid_test_ds, batch_size=Constants.batch_size, shuffle=shuffle, drop_last=True, num_workers=32, pin_memory=True)

    if verbose:
        print("In data_loading.build_dl() with dataset \'{}\' and augmentation \'{}\'.".format(
            dataset_str, augmentation_str))
        print("    len(train_ds): {}, len(valid_ds): {}, len(test_ds): {}.".format(
            len(train_ds), len(valid_ds), len(test_ds)))
        print("    len(train_dl): {}, len(valid_dl): {}, len(test_dl): {}.".format(
            len(train_dl), len(valid_dl), len(test_dl)))
        
    return train_dl, valid_dl, test_dl


class WrappedDataLoader:
    '''Wrapper that applies func to the wrapped dataloader.''' 
    def __init__(self, dataloader, reshape=False):
        self.dataloader = dataloader
        self.reshape = reshape
        self.dev = Objects.dev

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        return self.next()

    def __iter__(self):
        for batch in iter(self.dataloader):
            x_batch, y_batch = batch
            if self.reshape:
                yield (x_batch.view(x_batch.shape[0], -1).to(self.dev), y_batch.to(self.dev))
            else:
                yield(x_batch.to(self.dev), y_batch.to(self.dev))


def wrap_dl(train_dl, valid_dl, test_dl, reshape, verbose=False):
    '''Creates two versions of training and test dataloaders: one the resizes 
    inputs and one that doesn't. The resized inputs are passed to the model,
    while the un-resized inputs are displayed as images on tensorboard. 
    '''
    train_dl = WrappedDataLoader(train_dl, reshape=reshape)
    valid_dl = WrappedDataLoader(valid_dl, reshape=reshape)
    test_dl = WrappedDataLoader(test_dl, reshape=reshape)

    if verbose:
        print("In data_loading.wrap_dl().")
        print("    reshape: {}".format(reshape))
        print("    respective lengths of train_dl, valid_dl, test_dl: " 
              + "{}, {}, {}.".format(len(train_dl), len(valid_dl), len(test_dl)))

    return train_dl, valid_dl, test_dl


def build_wrapped_dl(augmentation, dataset, verbose=False):
    train_dl, valid_dl, test_dl = build_dl(
        augmentation, dataset, verbose=verbose)
    train_dlr, valid_dlr, test_dlr = wrap_dl(
        train_dl, valid_dl, test_dl, not "cnn" in Constants.model_str, verbose)

    return train_dlr, valid_dlr, test_dlr


# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================

def load_data_by_path(path):
    f = open(path, 'rb')
    data_label_dict = pickle.load(f, encoding='bytes')
    f.close()

    images = data_label_dict[b'data'] # img = 10000 x 3072
    labels = data_label_dict[b'labels']

    return images, labels


class DatasetFromTupleList(Dataset):
    def __init__(self, tuple_list):
        self.samples = tuple_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# helpers for testing ===================================================================================
# =======================================================================================================

def test():

    writer = SummaryWriter(log_dir='./runs/randaug_cache')

    trainloader, _, _ = build_dl("randaugment", "cifar-10-batches-py", 1.00, shuffle=False)

    augmented_batch = []
    for i, (images, labels) in enumerate(trainloader):
        for sample_num, (x, y) in enumerate(zip(images, labels)):
            x_tens = x.squeeze().clone().cpu()
            augmented_batch.append((x_tens, y.item()))

    augmented_cifar10_ds = DatasetFromTupleList(augmented_batch)

    with open(Path(__file__).parent.parent.resolve().parent / "saved_data" / "gmaxup_cifar-randaug_cache", 'wb') as handle:
        pickle.dump(augmented_cifar10_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("DUMP COMPLETE")


if __name__ == "__main__":
    test()

import os
import pdb
import torch
import torchvision
from torchvision.datasets.folder import default_loader
from engine.datasets import dataset_classes
from engine.tools.utils import load_json


def get_few_shot_setup_name(train_shot, seed):
    """Get the name for a few-shot setup.
    """
    return f"shot_{train_shot}-seed_{seed}"


class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)


class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        img = self.transform(default_loader(item['impath']))

        output = {
            "img": img,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
        }

        return output


def get_few_shot_benchmark(data_dir,
                           indices_dir,
                           dataset,
                           train_shot,
                           seed):
    # Check if the dataset is supported
    assert dataset in dataset_classes
    few_shot_index_file = os.path.join(
        indices_dir, dataset, f"{get_few_shot_setup_name(train_shot, seed)}.json")
    assert os.path.exists(few_shot_index_file), f"Few-shot data does not exist at {few_shot_index_file}."
    benchmark = dataset_classes[dataset](data_dir)
    few_shot_dataset = load_json(few_shot_index_file)
    return {
        'train': few_shot_dataset['train']['data'],
        'val': few_shot_dataset['val']['data'],
        'test': benchmark.test,
        'lab2cname': benchmark.lab2cname,
        'classnames': benchmark.classnames,
    }


class TestDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = self.transform(default_loader(item['impath']))

        return img, item['label']

def get_testset(args, transform):
    if args.dataset in dataset_classes:
        benchmark = dataset_classes[args.dataset](args.data_dir)
        return TestDatasetWrapper(benchmark.test, transform)
    elif args.dataset == 'food101_test':
        return torchvision.datasets.Food101(root='/data3/zhiqiul/datasets', split='test', transform=transform, download=True)
    elif args.dataset == 'dtd_test':
        return torchvision.datasets.DTD(root='/data3/zhiqiul/datasets', split='test', transform=transform, download=True)
    else:
        raise NotImplementedError()

def get_label_map(args, dataset_name):
    if dataset_name in ['imagenet_a', 'imagenet_r']:
        benchmark = dataset_classes[dataset_name](args.data_dir)
        return benchmark.label_map
    else:
        return None
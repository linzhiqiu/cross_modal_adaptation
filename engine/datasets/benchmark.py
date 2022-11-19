import os
import random
from collections import defaultdict

from engine.tools.utils import check_isfile, load_json, save_as_json, listdir_nohidden


def read_split(filepath, path_prefix):
    '''Read train/val/test split from a json file.'''
    def _convert(items):
        '''Convert a list of items to a list of dict.'''
        lst = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            check_isfile(impath)
            item = {'impath': impath,
                    'label': int(label),
                    'classname': classname}
            lst.append(item)
        return lst

    print(f"Reading split from {filepath}")
    split = load_json(filepath)
    train = _convert(split["train"])
    val = _convert(split["val"])
    test = _convert(split["test"])

    return train, val, test


def split_trainval(trainval, p_val=0.2):
    '''Random split train+val into train and val.'''
    p_trn = 1 - p_val
    print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
    tracker = defaultdict(list)
    for idx, item in enumerate(trainval):
        label = item['label']
        tracker[label].append(idx)

    train, val = [], []
    for label, idxs in tracker.items():
        n_val = round(len(idxs) * p_val)
        assert n_val > 0
        random.shuffle(idxs)
        for n, idx in enumerate(idxs):
            item = trainval[idx]
            if n < n_val:
                val.append(item)
            else:
                train.append(item)

    return train, val


def save_split(train, val, test, filepath, path_prefix):
    '''Save train/val/test split to a json file.'''
    def _extract(items):
        '''Extract a list of dict to a list of tuples.'''
        lst = []
        for item in items:
            impath = item['impath']
            label = item['label']
            classname = item['classname']
            impath = impath.replace(path_prefix, "")
            if impath.startswith("/"):
                impath = impath[1:]
            lst.append((impath, label, classname))
        return lst

    train = _extract(train)
    val = _extract(val)
    test = _extract(test)

    split = {"train": train, "val": val, "test": test}

    save_as_json(split, filepath)
    print(f"Saved split to {filepath}")


def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
    # The data are supposed to be organized into the following structure
    # =============
    # images/
    #     dog/
    #     cat/
    #     horse/
    # =============
    categories = listdir_nohidden(image_dir)
    categories = [c for c in categories if c not in ignored]
    categories.sort()

    p_tst = 1 - p_trn - p_val
    print(
        f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

    def _collate(ims, y, c):
        items = []
        for im in ims:
            # is already 0-based
            item = {'impath': im, 'label': y, 'classname': c}
            items.append(item)
        return items

    train, val, test = [], [], []
    for label, category in enumerate(categories):
        category_dir = os.path.join(image_dir, category)
        images = listdir_nohidden(category_dir)
        images = [os.path.join(category_dir, im) for im in images]
        random.shuffle(images)
        n_total = len(images)
        n_train = round(n_total * p_trn)
        n_val = round(n_total * p_val)
        n_test = n_total - n_train - n_val
        assert n_train > 0 and n_val > 0 and n_test > 0

        if new_cnames is not None and category in new_cnames:
            category = new_cnames[category]

        train.extend(
            _collate(images[:n_train], label, category))
        val.extend(
            _collate(images[n_train: n_train + n_val], label, category))
        test.extend(
            _collate(images[n_train + n_val:], label, category))

    return train, val, test



def get_num_classes(data_source):
    """Count number of classes.

    Args:
        data_source (list): a list of Datum objects.
    """
    label_set = set()
    for item in data_source:
        label_set.add(item['label'])
    return max(label_set) + 1


def get_lab2cname(data_source):
    """Get a label-to-classname mapping (dict).

    Args:
        data_source (list): a list of dict.
    """
    container = set()
    for item in data_source:
        container.add((item['label'], item['classname']))
    mapping = {label: classname for label, classname in container}
    labels = list(mapping.keys())
    labels.sort()
    classnames = [mapping[label] for label in labels]
    return mapping, classnames


def split_dataset_by_label(data_source):
    """Split a dataset, into class-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of dict.
    """
    items = defaultdict(list)
    indices = defaultdict(list)

    for idx, item in enumerate(data_source):
        items[item['label']].append(item)
        indices[item['label']].append(idx)

    return items, indices


def sample_few_shot_dataset(data_source, num_shots, repeat=False):
    """Sample a few-shot dataset from a dataset.
    """
    few_shot_dataset = {
        'data': [],
        'indices': [],
    }
    all_items, all_indices = split_dataset_by_label(data_source)

    for label, items in all_items.items():
        item_indices = list(range(len(items)))
        if len(items) >= num_shots:
            sampled_item_indices = random.sample(item_indices, num_shots)
        else:
            if repeat:
                sampled_item_indices = random.choices(
                    item_indices, k=num_shots)
            else:
                sampled_item_indices = item_indices

        sampled_indices = [all_indices[label][idx]
                           for idx in sampled_item_indices]
        sampled_items = [items[idx] for idx in sampled_item_indices]
        few_shot_dataset['data'].extend(sampled_items)
        few_shot_dataset['indices'].extend(sampled_indices)
    return few_shot_dataset


def generate_fewshot_dataset(
    train, val, num_shots=16, max_val_shots=4, repeat=False
):
    """Generate a few-shot dataset (for the training/val set).

    Args:
        train: a list of train samples.
        val: a list of val samples.
        num_shots (int): number of train samples per class.
        max_val_shots (int): maximum number of val samples per class.
        repeat (bool): repeat images if needed (default: False).

    Returns:
        A tuple of (few-shot train, few-shot val).
    """
    assert num_shots >= 1
    print(f"Creating a {num_shots}-shot train set")
    few_shot_train = sample_few_shot_dataset(
        train, num_shots, repeat=repeat)
    num_val_shots = min(max_val_shots, num_shots)
    print(f"Creating a {num_val_shots}-shot validation set")
    few_shot_val = sample_few_shot_dataset(
        val, num_val_shots, repeat=repeat)

    return {
        'train': few_shot_train,
        'val': few_shot_val,
    }


class Benchmark(object):
    """A benchmark that contains 
    1) training data
    2) validation data
    3) test data
    """

    dataset_name = "" # e.g. imagenet, etc.

    def __init__(self, train=None, val=None, test=None):
        self.train = train  # labeled training data source
        self.val = val  # validation data source
        self.test = test  # test data source
        self.num_classes = get_num_classes(train)
        self.lab2cname, self.classnames = get_lab2cname(train)

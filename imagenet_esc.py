from copy import deepcopy
import os
import torch
from engine.optimizer.default import HYPER_DICT
from train import get_hyperparams_str
torch.set_num_threads(4)
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from engine.tools.utils import makedirs, set_random_seed
from engine.config import default
from engine.datasets.utils import TensorDataset
# from engine.model.head import make_classifier_head
from engine.model.logit import LogitHead
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from features import get_text_features_path, get_image_features_path, get_test_features_path

RESULT_DIR = "./imagenet_esc_results/"

ESC_DIR = f"{default.DATA_DIR}/esc-50/"

SHOTS = [1, 2, 4] # for both image and audio

SPLITS = [0, 1, 2, 3, 4] # for both image and audio; make sure imagenet has seed 4 and 5  featuresavailable

TASKS = ['image', 'audio'] # classification task to perform

clip_encoder = "RN50" # AudioCLIP uses a frozen RN50 model
image_layer_idx = 0
text_layer_idx = 0
text_augmentation = "classname"
image_augmentation = "none"
classifier_head = "linear"
logit = 4.60517
hyperparams_audio = "audio"
result_dir = os.path.join(RESULT_DIR, hyperparams_audio)

makedirs(result_dir)


def get_zero_shot_weights(text_dataset, num_classes, in_features):
    # Caveat: Only support text_dataset with 1-D text features. 
    # Need to modify if you want to partial finetuning the text encoder
    weights = torch.zeros(num_classes, in_features)
    count = torch.zeros(num_classes)
    for i in range(len(text_dataset)):
        label = text_dataset.label_tensor[i]
        weights[label] += F.normalize(text_dataset.input_tensor[i], dim=0)
        count[label] += 1
    weights /= count.unsqueeze(1)
    # normalize the weights
    weights.data = F.normalize(weights, dim=1)
    return weights

def make_classifier_head(classifier_head,
                         clip_encoder,
                         classifier_init,
                         zeroshot_dataset,
                         bias=False):
    assert classifier_head in AVAI_HEADS
    if clip_encoder == 'ViT-B/16':
        in_features = 512
    elif clip_encoder == 'RN50':
        in_features = 1024

    num_classes = int(zeroshot_dataset.label_tensor.max()) + 1

    linear_head = nn.Linear(in_features, num_classes, bias=bias)
    if classifier_init == 'zeroshot':
        assert zeroshot_dataset.input_tensor.shape[1] == in_features
        linear_head.weight.data = get_zero_shot_weights(
            zeroshot_dataset, num_classes, in_features)
    
    if classifier_head == 'linear':
        head = linear_head
    else:
        raise ValueError(f"Invalid head: {classifier_head}")
    return head, num_classes, in_features

CLASS_MAP = {
    'imagenet_27': {
        'dataset': 'imagenet',
        'class_map': {
            7: "rooster",  # rooster
            8: "hen",  # hen
            19: "chirping_birds",  # chickadee
            31: "frog",  # tree frog
            175: "dog",  # Otterhound
            285: "cat",  # Egyptian cat
            308: "insects",  # fly
            312: "crickets",  # cricket
            341: "pig",  # pig
            349: "sheep",  # big-horn sheep
            404: "airplane",  # airliner
            466: "train",  # high-speed train
            473: "can_opening",  # can opener
            491: "chainsaw",  # chainsaw
            497: "church_bells",  # church bells
            508: "keyboard_typing",  # computer keyboard
            530: "clock_alarm",  # digital clock
            556: "crackling_fire",  # fire screen
            673: "mouse_click",  # computer mouse
            861: "toilet_flush",  # toilet seat
            882: "vacuum_cleaner",  # vacuum cleaner
            892: "clock_tick",  # wall clock
            896: "water_drops",  # sink
            897: "washing_machine",  # washing machine
            898: "drinking_sipping",  # water bottle
            899: "pouring_water",  # water jug
            977: "sea_waves",  # sandbar
        }
    },
    'imagenet_19': {
        'dataset': 'imagenet',
        'class_map': {
            7: "rooster",  # rooster
            8: "hen",  # hen
            19: "chirping_birds",  # chickadee
            31: "frog",  # tree frog
            175: "dog",  # Otterhound
            285: "cat",  # Egyptian cat
            308: "insects",  # fly
            312: "crickets",  # cricket
            341: "pig",  # pig
            349: "sheep",  # big-horn sheep
            404: "airplane",  # airliner
            466: "train",  # high-speed train
            491: "chainsaw",  # chainsaw
            508: "keyboard_typing",  # computer keyboard
            530: "clock_alarm",  # digital clock
            673: "mouse_click",  # computer mouse
            882: "vacuum_cleaner",  # vacuum cleaner
            892: "clock_tick",  # wall clock
            897: "washing_machine",  # washing machine
        }
    },
}

# del CLASS_MAP['imagenet_27']
# del CLASS_MAP['imagenet_19']

def train(logit_head, 
          image_loader, val_loader, audio_loader, test_loader,
          optimizer, scheduler, criterion, iters,
          eval_freq=100, device="cuda"):
    if image_loader is None and audio_loader is None:
        raise ValueError("Both image_loader and audio_loader are None")
    if image_loader is not None:
        image_loader_iter = iter(image_loader)
    else:
        image_loader_iter = None
    if audio_loader is not None:
        audio_loader_iter = iter(audio_loader)
    else:
        audio_loader_iter = None

    best_val_dict = {
        "iter": None,
        "val_acc": None,
        "image_encoder": None,
        "logit_head": None,
    }

    for i in range(iters):
        logit_head.train()
        if image_loader_iter is not None:
            try:
                image_feature, image_label = next(image_loader_iter)
            except StopIteration:
                image_loader_iter = iter(image_loader)
                image_feature, image_label = next(image_loader_iter)
            image_feature = image_feature.to(device)
            image_label = image_label.to(device)
        else:
            image_feature = None
        
        if audio_loader_iter is not None:
            try:
                audio_feature, audio_label = next(audio_loader_iter)
            except StopIteration:
                audio_loader_iter = iter(audio_loader)
                audio_feature, audio_label = next(audio_loader_iter)
            audio_feature = audio_feature.to(device)
            audio_label = audio_label.to(device)
        else:
            audio_feature = None
        
        if image_feature is not None and audio_feature is not None:
            feature = torch.cat([image_feature, audio_feature], dim=0)
            label = torch.cat([image_label, audio_label], dim=0)
        elif image_feature is not None:
            feature = image_feature
            label = image_label
        elif audio_feature is not None:
            feature = audio_feature
            label = audio_label
        else:
            raise ValueError("Both image_feature and audio_feature are None")

        optimizer.zero_grad()
        logit = logit_head(feature)
        loss = criterion(logit, label)
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % eval_freq == 0:
            val_acc = validate(logit_head, val_loader, device=device)
            test_acc = validate(logit_head, test_loader, device=device)
            if best_val_dict["val_acc"] is None or val_acc > best_val_dict["val_acc"]:
                best_val_dict["iter"] = i
                best_val_dict["val_acc"] = val_acc
                best_val_dict['test_acc'] = test_acc
                best_val_dict["logit_head"] = deepcopy(logit_head.state_dict())
    
    val_acc = validate(logit_head, val_loader, device=device)
    test_acc = validate(logit_head, test_loader, device=device)
    print(f"Best val acc: {best_val_dict['val_acc']:.4f} at iter {best_val_dict['iter']} with test acc {best_val_dict['test_acc']:.4f}")
    return best_val_dict


def validate(logit_head, val_loader, device="cuda"):
    logit_head.eval()
    val_acc = 0
    val_count = 0.
    for image_feature, image_label in val_loader:
        image_feature = image_feature.to(device)
        image_label = image_label.to(device)
        # import pdb; pdb.set_trace()
        logit = logit_head(image_feature)
        pred = torch.argmax(logit, dim=1)
        val_acc += torch.sum(pred == image_label).item()
        val_count += image_label.size(0)
    val_acc /= val_count
    return val_acc


def evaluate(clip_encoder, classifier_head, logit, zero_shot_dataset, test_dataset):
    # Create the zero-shot model and evaluate test accuracy
    head, _, _ = make_classifier_head(
        classifier_head,
        clip_encoder,
        "zeroshot", # meaning zero-shot initialization here
        zero_shot_dataset,
    )
    eval_head = LogitHead(
        head,
        logit_scale=logit
    ).cuda().eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_acc = validate(eval_head, test_loader, device="cuda")
    test_acc = float(test_acc)
    print(f"Test Acc: {test_acc}")
    return test_acc


def transform_features(features, sorted_classes):
    # first filter out the classes that are not in sorted_classes
    new_features = None
    new_labels = None
    for i in range(features['features'].shape[0]):
        if features['labels'][i] in sorted_classes:
            if new_features is None:
                new_features = features['features'][i].unsqueeze(0)
                new_labels = features['labels'][i].unsqueeze(0)
            else:
                new_features = torch.cat((new_features, features['features'][i].unsqueeze(0)), 0)
                new_labels = torch.cat((new_labels, features['labels'][i].unsqueeze(0)), 0)

    # transform new_labels to be the index of sorted_classes
    for i in range(len(new_labels)):
        new_labels[i] = sorted_classes.index(new_labels[i])
    features['features'] = new_features
    features['labels'] = new_labels
    return features


def transform_audio_features(features, class_map, sorted_classes):
    labelname_to_label = {}
    for image_class, labelname in class_map.items():
        labelname_to_label[labelname] = sorted_classes.index(image_class)
    
    # first filter out the classes that are not in sorted_classes
    new_audio_feature = {
        'features': None,
        'labels': None,
    }
    new_features = None
    new_labels = []
    for i in range(features['features'].shape[0]):
        if features['labelnames'][i] in labelname_to_label:
            if new_features is None:
                new_features = features['features'][i].unsqueeze(0)
                new_labels = [labelname_to_label[features['labelnames'][i]]]
            else:
                new_features = torch.cat((new_features, features['features'][i].unsqueeze(0)), 0)
                new_labels = new_labels + [labelname_to_label[features['labelnames'][i]]]

    new_audio_feature['features'] = new_features
    new_audio_feature['labels'] = torch.Tensor(new_labels).long()
    return new_audio_feature


def take_indices(features, indices):
    new_features = {
        'features': features['features'][indices],
        'labels': features['labels'][indices],
    }
    return new_features


def construct_few_shot_dataset(train_features, shot_num):
    assert shot_num >= 1 and shot_num <= 4
    train_indices = []
    val_indices = []
    labels = train_features['labels'].unique()
    for label in labels:
        label_indices = torch.where(train_features['labels'] == label)[0]
        train_indices = train_indices + label_indices[:shot_num].tolist()
        val_indices = val_indices + label_indices[shot_num:2*shot_num].tolist()
    assert len(train_indices) == len(labels) * shot_num
    assert len(val_indices) == len(labels) * shot_num
    new_train_features = {
        'train': take_indices(train_features, train_indices),
        'val': take_indices(train_features, val_indices),
    }
    return new_train_features


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    audio_features_path = os.path.join(ESC_DIR, 'features.pt')
    assert os.path.exists(audio_features_path), \
        f"Audio features not found at {audio_features_path}. Please run python audio_features.py under audioclip/."
    audio_features = torch.load(audio_features_path)

    for dataset in CLASS_MAP:
        print(f"Dataset: {dataset}")
        result_dict_path = os.path.join(result_dir, f"{dataset}_result_all.pt")
        
        if os.path.exists(result_dict_path):
            continue
        else:
            result_dict = {
                task: {
                    'zero_shot': {},
                    'linear_prob': {},
                    'audiovisual_linear_prob': {}, # task modality + audio or image
                    'plustext_linear_prob': {}, # task modality + text
                    'allmodal_linear_prob': {}, # all three modality
                }
                for task in TASKS
            }
        
        original_dataset = CLASS_MAP[dataset]['dataset']
        class_map = CLASS_MAP[dataset]['class_map']
        sorted_classes = sorted(list(class_map.keys()))

        
        for task in TASKS:
            dataset_dir = os.path.join(result_dir, dataset)
            makedirs(dataset_dir)
            task_dir = os.path.join(dataset_dir, task)
            makedirs(task_dir)

            for split_index in SPLITS:
                split_dir = os.path.join(task_dir, f"split_{split_index}")
                makedirs(split_dir)

                ### Construct test set
                if task == 'image':
                    # load image testset
                    test_features_path = get_test_features_path(
                        original_dataset,
                        default.FEATURE_DIR,
                        clip_encoder,
                        image_layer_idx
                    )
                    assert os.path.exists(test_features_path), \
                        f"Image features not found at {test_features_path}. Please run python features.py"
                    test_features = torch.load(test_features_path)

                    test_features = transform_features(test_features, sorted_classes)
                elif task == 'audio':
                    test_features = {
                        'features': None,
                        'labels': None,
                    }
                    for test_split_index in SPLITS:
                        if test_split_index != split_index:
                            transformed_audio_features = transform_audio_features(
                                audio_features[test_split_index], class_map, sorted_classes)
                            if test_features['features'] is None:
                                test_features['features'] = transformed_audio_features['features']
                                test_features['labels'] = transformed_audio_features['labels']
                            else:
                                test_features['features'] = torch.cat(
                                    (test_features['features'], transformed_audio_features['features']))
                                test_features['labels'] = torch.cat(
                                    (test_features['labels'], transformed_audio_features['labels']))

                test_dataset = TensorDataset(
                    test_features['features'],
                    test_features['labels']
                )
                print(f"{task} dataset has {len(test_dataset)} test examples ({dataset})")


                # 1: zero-shot-classifier with text
                result_dict[task]['zero_shot'][split_index] = {}
                text_features_path = get_text_features_path(
                    original_dataset,
                    default.FEATURE_DIR,
                    clip_encoder,
                    text_layer_idx,
                    text_augmentation
                )
                assert os.path.exists(text_features_path), \
                    f"Text features not found at {text_features_path}. Please run python features.py"
                
                text_features = torch.load(text_features_path)
                text_features['features'] = torch.nn.functional.normalize(text_features['features'], dim=1)
                text_features = transform_features(text_features, sorted_classes)
                text_dataset = TensorDataset(
                    text_features['features'], text_features['labels']
                )
                test_acc = evaluate(clip_encoder, classifier_head, logit, text_dataset, test_dataset)
                result_dict[task]['zero_shot'][split_index]['text'] = test_acc
                print(f"Zero-shot-text-classifier for {task} classification with template {text_augmentation}: {test_acc} ({dataset}-{split_index})")
                
                for shot_num in SHOTS:
                    ### Construct train set for {task} modality
                    if task == 'image':
                        other = 'audio'
                        train_features_path = get_image_features_path(
                            original_dataset,
                            shot_num,
                            split_index+1,
                            default.FEATURE_DIR,
                            clip_encoder,
                            image_layer_idx,
                            "none", # meaning center crop
                        )
                        assert os.path.exists(train_features_path), \
                            f"Image features not found at {train_features_path}. Please run python features.py"
                        train_features = torch.load(train_features_path)
                        train_features['train'] = transform_features(train_features['train'], sorted_classes)
                        train_features['val'] = transform_features(train_features['val'], sorted_classes)
                    elif task == 'audio':
                        other = 'image'
                        train_features = transform_audio_features(
                            audio_features[split_index], class_map, sorted_classes)
                        train_features = construct_few_shot_dataset(train_features, shot_num)
                    
                    # normalize both train and val features
                    train_features['train']['features'] = torch.nn.functional.normalize(train_features['train']['features'], dim=1)
                    train_features['val']['features'] = torch.nn.functional.normalize(train_features['val']['features'], dim=1)
                    train_dataset = TensorDataset(
                        train_features['train']['features'], train_features['train']['labels'])
                    val_dataset = TensorDataset(
                        train_features['val']['features'], train_features['val']['labels'])
                    print(f"{task} dataset has {len(train_dataset)} train examples and {len(val_dataset)} val examples ({dataset}-{split_index})")

                    result_dict[task]['zero_shot'][split_index][shot_num] = {}
                    
                    # 2: zero-shot-classifier with {task} modality
                    test_acc = evaluate(clip_encoder, classifier_head, logit, train_dataset, test_dataset)
                    result_dict[task]['zero_shot'][split_index][shot_num][task] = test_acc
                    print(f"Zero-shot-{task}-classifier for {task} classification: {test_acc} ({dataset}-{split_index})")

                    result_dict[task]['zero_shot'][split_index][shot_num][other] = {}
                    
                    # 3: linear classifier with {task}-modal or cross-modal classifier
                    for seed_idx in SPLITS:
                        if seed_idx >= 0:
                            print("Setting fixed seed: {}".format(seed_idx))
                            set_random_seed(seed_idx)

                        if task == 'image':
                            ### Construct a one-shot audio dataset
                            other_features = transform_audio_features(
                                audio_features[seed_idx], class_map, sorted_classes)
                            other_features = construct_few_shot_dataset(other_features, 1)['train']
                        elif task == 'audio':
                            ### Load a one-shot image dataset
                            other_features_path = get_image_features_path(
                                original_dataset,
                                1,
                                seed_idx+1,
                                default.FEATURE_DIR,
                                clip_encoder,
                                image_layer_idx,
                                "none",
                            )
                            assert os.path.exists(other_features_path), \
                                f"Image features not found at {other_features_path}. Please run python features.py"
                            other_features = torch.load(other_features_path)
                            other_features = transform_features(other_features['train'], sorted_classes)
                        
                        # normalize other features
                        other_features['features'] = torch.nn.functional.normalize(other_features['features'], dim=1)

                        all_features = {}
                        # concatenate other with text features
                        all_features['features'] = torch.cat(
                            (other_features['features'], text_features['features']), dim=0)
                        all_features['labels'] = torch.cat(
                            (other_features['labels'], text_features['labels']), dim=0)
                        
                        other_dataset = TensorDataset(
                            other_features['features'], other_features['labels']
                        )
                        all_dataset = TensorDataset(
                            all_features['features'], all_features['labels']
                        )

                        # 4: zero-shot-classifier with {other} modality
                        test_acc = evaluate(clip_encoder, classifier_head, logit, other_dataset, test_dataset)
                        result_dict[task]['zero_shot'][split_index][shot_num][other][seed_idx] = test_acc
                        print(f"Zero-shot-{other}-classifier for {task} classification: {test_acc} ({dataset}-{split_index})")

                        hyperparams = HYPER_DICT[hyperparams_audio]
                        # Caveat: Not filtering out invalid batch sizes
                        VALID_BATCH_SIZES = hyperparams['batch_size']

                        def get_experiment_count(hyperparams):
                            count = 1
                            count *= len(hyperparams['lr'])
                            count *= len(hyperparams['weight_decay'])
                            count *= len(VALID_BATCH_SIZES)
                            count *= len(hyperparams['max_iter'])
                            return count
                        experiment_count = get_experiment_count(hyperparams)
                        cur_count = 0

                        # sweep through hyperparameters
                        for lr in hyperparams['lr']:
                            for wd in hyperparams['weight_decay']:
                                for batch_size in VALID_BATCH_SIZES:
                                    for iters in hyperparams['max_iter']:
                                        cur_count += 1

                                        hyperparams_str = get_hyperparams_str(
                                            hyperparams['optim'], lr, wd, batch_size, iters)
                                        
                                        # check if experiment has been done
                                        print(f"[{cur_count}/{experiment_count}]: {hyperparams_str}. Running")

                                        train_mode_dict = {
                                            'linear_prob': ('none', 0., 0., 0.),
                                            'audiovisual_linear_prob': ('zeroshot', 0.5, 0., 0.),
                                            'plustext_linear_prob': ('zeroshot', 0., 0., 0.5),
                                            'allmodal_linear_prob': ('zeroshot', 0., 0.5, 0.),
                                        }
                                        for train_mode in train_mode_dict.keys():
                                            head_type, other_batch_ratio, all_batch_ratio, text_batch_ratio = train_mode_dict[train_mode]
                                            
                                            if all_batch_ratio > 0:
                                                zeroshot_dataset = all_dataset
                                            elif text_batch_ratio > 0:
                                                zeroshot_dataset = text_dataset
                                            elif other_batch_ratio > 0:
                                                zeroshot_dataset = other_dataset
                                            else:
                                                zeroshot_dataset = other_dataset # won't be used for init because head_type is 'none'
                                            head, _, _ = make_classifier_head(
                                                                classifier_head,
                                                                clip_encoder,
                                                                head_type,
                                                                zeroshot_dataset,
                                                         )
                                            logit_head = LogitHead(
                                                head,
                                                logit_scale=logit,
                                            ).train().cuda()

                                            # Create the optimizer
                                            params_groups = [
                                                {'params': logit_head.parameters()},
                                            ]
                                            optimizer = build_optimizer(params_groups, hyperparams['optim'], lr, wd)
                                            scheduler = build_lr_scheduler(
                                                optimizer,
                                                hyperparams['lr_scheduler'],
                                                hyperparams['warmup_iter'],
                                                iters,
                                                warmup_type=hyperparams['warmup_type'],
                                                warmup_lr=hyperparams['warmup_min_lr']
                                            )
                                            criterion = torch.nn.CrossEntropyLoss()
                                            all_batch_size = int(batch_size * all_batch_ratio)
                                            other_batch_size = int(batch_size * other_batch_ratio)
                                            text_batch_size = int(batch_size * text_batch_ratio)

                                            if all_batch_ratio > 0:
                                                assert other_batch_ratio == 0
                                                assert text_batch_ratio == 0
                                                train_batch_size = batch_size - all_batch_size
                                            elif text_batch_ratio > 0:
                                                assert other_batch_ratio == 0
                                                assert all_batch_ratio == 0
                                                train_batch_size = batch_size - text_batch_size
                                            else:
                                                assert text_batch_ratio == 0
                                                assert all_batch_ratio == 0
                                                train_batch_size = batch_size - other_batch_size

                                            other_loader = None
                                            if other_batch_ratio > 0:
                                                other_loader = DataLoader(
                                                    other_dataset,
                                                    batch_size=other_batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    drop_last=True,
                                                )
                                            elif all_batch_ratio > 0:
                                                other_loader = DataLoader(
                                                    all_dataset,
                                                    batch_size=all_batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    drop_last=True,
                                                )
                                            elif text_batch_ratio > 0:
                                                other_loader = DataLoader(
                                                    text_dataset,
                                                    batch_size=text_batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    drop_last=True,
                                                )
                                        
                                            train_loader = None
                                            if train_batch_size > 0:
                                                train_loader = DataLoader(
                                                    train_dataset,
                                                    batch_size=train_batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    drop_last=True,
                                                )
                                            
                                            val_loader = DataLoader(
                                                val_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                            )

                                            test_loader = DataLoader(
                                                test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                            )

                                            best_val_dict = train(
                                                logit_head,
                                                train_loader, val_loader, other_loader, test_loader,
                                                optimizer, scheduler, criterion, iters,
                                                eval_freq=100
                                            )
                                            
                                            if not split_index in result_dict[task][train_mode]:
                                                result_dict[task][train_mode][split_index] = {}
                                            if not shot_num in result_dict[task][train_mode][split_index]:
                                                result_dict[task][train_mode][split_index][shot_num] = {}
                                            if not seed_idx in result_dict[task][train_mode][split_index][shot_num]:
                                                result_dict[task][train_mode][split_index][shot_num][seed_idx] = {
                                                    'all': {},
                                                    'best': {},
                                                }
                                            result_dict[task][train_mode][split_index][shot_num][seed_idx]['all'][hyperparams_str] = {
                                                "val_acc": best_val_dict["val_acc"],
                                                "test_acc": best_val_dict["test_acc"],
                                            }
                                            print(f"Test acc {dataset} {train_mode} {split_index} {shot_num} {seed_idx} {hyperparams_str}: {best_val_dict['test_acc']}")
                                        
                        for train_mode in train_mode_dict.keys():
                            best_val_acc = 0.
                            for hyperparams_str in result_dict[task][train_mode][split_index][shot_num][seed_idx]['all'].keys():
                                val_acc = result_dict[task][train_mode][split_index][shot_num][seed_idx]['all'][hyperparams_str]["val_acc"]
                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    best_test_acc = result_dict[task][train_mode][split_index][shot_num][seed_idx]['all'][hyperparams_str]["test_acc"]
                                    best_hyperparams_str = hyperparams_str
                            result_dict[task][train_mode][split_index][shot_num][seed_idx]['best'] = {
                                "val_acc": best_val_acc,
                                "test_acc": best_test_acc,
                                "hyperparams_str": best_hyperparams_str,
                            }
        torch.save(result_dict, result_dict_path)

    METHODS = ['linear_prob',
               'audiovisual_linear_prob',
               'plustext_linear_prob',
               'allmodal_linear_prob',
               'zero_shot_text',
               'zero_shot_image',
               'zero_shot_audio',
    ]
    # Take average
    for dataset in CLASS_MAP:
        print(f"Dataset: {dataset}")
        result_dict_path = os.path.join(result_dir, f"{dataset}_result_all.pt")

        assert os.path.exists(result_dict_path), f"Result dict not found: {result_dict_path}"
        result_dict = torch.load(result_dict_path)

        avg_dict = {
            task: {}
            for task in result_dict.keys()
        }

        for task in result_dict.keys():
            if task == 'audio':
                other = 'image'
            else:
                other = 'audio'
            
            for shot_num in SHOTS:
                avg_dict[task][shot_num] = {
                    method: {'mean': None, 'std': None}
                    for method in METHODS
                }

                zero_shot_tasks = []
                zero_shot_others = []
                linear_probs = []
                audiovisual_linear_probs = []
                allmodal_linear_probs = []
                plustext_linear_probs = []
                win = 0
                for split_index in SPLITS:
                    zero_shot_text = result_dict[task]['zero_shot'][split_index]['text']
                    zero_shot_task = result_dict[task]['zero_shot'][split_index][shot_num][task]
                    zero_shot_tasks.append(zero_shot_task)
                    for seed_idx in SPLITS:
                        zero_shot_other = result_dict[task]['zero_shot'][split_index][shot_num][other][seed_idx]
                        zero_shot_others.append(zero_shot_other)
                        linear_prob = result_dict[task]['linear_prob'][split_index][shot_num][seed_idx]['best']['test_acc']
                        linear_probs.append(linear_prob)
                        audiovisual_linear_prob = result_dict[task]['audiovisual_linear_prob'][split_index][shot_num][seed_idx]['best']['test_acc']
                        audiovisual_linear_probs.append(audiovisual_linear_prob)
                        plustext_linear_prob = result_dict[task]['plustext_linear_prob'][split_index][shot_num][seed_idx]['best']['test_acc']
                        plustext_linear_probs.append(plustext_linear_prob)
                        allmodal_linear_prob = result_dict[task]['allmodal_linear_prob'][split_index][shot_num][seed_idx]['best']['test_acc']
                        allmodal_linear_probs.append(allmodal_linear_prob)
                        methods = {
                            'linear_prob': linear_prob,
                            'audiovisual_linear_prob': audiovisual_linear_prob,
                            'plustext_linear_prob': plustext_linear_prob,
                            'allmodal_linear_prob': allmodal_linear_prob,
                            'zero_shot_text': zero_shot_text,
                            'zero_shot_image': zero_shot_task if task == 'image' else zero_shot_other,
                            'zero_shot_audio': zero_shot_task if task == 'audio' else zero_shot_other,
                        }
                        if methods['audiovisual_linear_prob'] > methods['linear_prob']:
                            win += 1
                print(f"Audiovisual linear prob outperforms unimodal linear prob: {win}/{len(SPLITS)*len(SPLITS)} for {task} {shot_num}-shot")
                avg_dict[task][shot_num]['zero_shot_text'] = {'mean': zero_shot_text, 'std': 0.0}
                if task == 'audio':
                    zero_shot_images = zero_shot_others
                    zero_shot_audios = zero_shot_tasks
                else:
                    zero_shot_images = zero_shot_tasks
                    zero_shot_audios = zero_shot_others
                avg_dict[task][shot_num]['zero_shot_image'] = {'mean': np.mean(zero_shot_images), 'std': np.std(zero_shot_images)}
                avg_dict[task][shot_num]['zero_shot_audio'] = {'mean': np.mean(zero_shot_audios), 'std': np.std(zero_shot_audios)}
                avg_dict[task][shot_num]['linear_prob'] = {'mean': np.mean(linear_probs), 'std': np.std(linear_probs)}
                avg_dict[task][shot_num]['plustext_linear_prob'] = {'mean': np.mean(plustext_linear_probs), 'std': np.std(plustext_linear_probs)}
                avg_dict[task][shot_num]['audiovisual_linear_prob'] = {'mean': np.mean(audiovisual_linear_probs), 'std': np.std(audiovisual_linear_probs)}
                avg_dict[task][shot_num]['allmodal_linear_prob'] = {'mean': np.mean(allmodal_linear_probs), 'std': np.std(allmodal_linear_probs)}

    import json
    print(json.dumps(avg_dict, indent=2))


if __name__ == "__main__":
    main()

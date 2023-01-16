import os, argparse
import torch
torch.set_num_threads(4)
import numpy as np
import csv

from engine.config import default
from engine.datasets.utils import TensorDataset, TextTensorDataset
from engine.optimizer.default import HYPER_DICT
from features import get_image_features_path, get_text_features_path
from domain_shift import get_hyperparams_str, get_save_dir, get_valid_batch_sizes
from domain_shift import IMAGENET_TESTSETS
EVAL_DIR = "./results/" # Default save to ./results/ directory


DATASETS = [
    "imagenet",
]

SEEDS = [
    1,
    2,
    3,
]

SHOTS = [
    16
]

def take_average(all_seed_dict,
                 ALL_EVAL_TYPES=["head"],
                 TESTSETS=IMAGENET_TESTSETS,
                 ):
    header = ['hyperparameter', 'iter_mean', 'iter_std', 'eval_type', 'val_acc_mean', 'val_acc_std', 'test_acc_mean', 'test_acc_std']
    for test_dataset in TESTSETS:
        header += [f'{test_dataset}_acc_mean', f'{test_dataset}_acc_std']
    columns = []
    ALL_SEEDS = list(all_seed_dict.keys())
    result_dict = {}
    avg_dict = {}
    std_dict = {}
    for eval_type in ALL_EVAL_TYPES:
        result_dict[eval_type] = {
            'val_acc': [],
            'test_acc': [],
            'iter': [],
            'hyperparameter': [],
        }
        for test_dataset in TESTSETS:
            result_dict[eval_type][test_dataset] = []
        avg_dict[eval_type] = {}
        std_dict[eval_type] = {}
        for seed in ALL_SEEDS:
            best_hyper = None
            for hyper in all_seed_dict[seed]:
                if best_hyper is None or all_seed_dict[seed][hyper]['val_acc'] > all_seed_dict[seed][best_hyper]['val_acc']:
                    best_hyper = hyper
            result_dict[eval_type]['val_acc'].append(all_seed_dict[seed][best_hyper]['val_acc'])
            result_dict[eval_type]['test_acc'].append(all_seed_dict[seed][best_hyper]['test_accs'][eval_type])
            result_dict[eval_type]['iter'].append(all_seed_dict[seed][best_hyper]['iter'])
            result_dict[eval_type]['hyperparameter'].append(best_hyper)
            for test_dataset in TESTSETS:
                result_dict[eval_type][test_dataset].append(all_seed_dict[seed][best_hyper]['domain_shift_accs'][test_dataset][eval_type])
        avg_dict[eval_type]['val_acc'] = np.mean(result_dict[eval_type]['val_acc'])
        avg_dict[eval_type]['test_acc'] = np.mean(result_dict[eval_type]['test_acc'])
        avg_dict[eval_type]['iter'] = np.mean(result_dict[eval_type]['iter'])
        std_dict[eval_type]['val_acc'] = np.std(result_dict[eval_type]['val_acc'])
        std_dict[eval_type]['test_acc'] = np.std(result_dict[eval_type]['test_acc'])
        std_dict[eval_type]['iter'] = np.std(result_dict[eval_type]['iter'])
        for test_dataset in TESTSETS:
            avg_dict[eval_type][test_dataset] = np.mean(result_dict[eval_type][test_dataset])
            std_dict[eval_type][test_dataset] = np.std(result_dict[eval_type][test_dataset])
        column = [str(result_dict[eval_type]['hyperparameter']),
                  avg_dict[eval_type]['iter'], std_dict[eval_type]['iter'],
                  eval_type,
                  avg_dict[eval_type]['val_acc'], std_dict[eval_type]['val_acc'],
                  avg_dict[eval_type]['test_acc'], std_dict[eval_type]['test_acc']]
        for test_dataset in TESTSETS:
            column += [avg_dict[eval_type][test_dataset], std_dict[eval_type][test_dataset]]
        columns.append(column)
        
    return header, columns

def get_eval_dir(shots,
                 clip_encoder,
                 image_layer_idx,
                 text_layer_idx,
                 text_augmentation,
                 image_augmentation,
                 image_views,
                 modality,
                 classifier_init,
                 hyperparams_str,
                 logit,
                 wise_ft,
                 eval_dir=EVAL_DIR):
    return os.path.join(
        eval_dir,
        "domain_shift",
        f"shots_{shots}",
        f"{clip_encoder}_im{image_layer_idx}_tx{text_layer_idx}",
        f"imaug_{image_augmentation}_imview{image_views}_txaug_{text_augmentation}",
        f"modality_{modality}_init_{classifier_init}_logit_{logit}_wise_ft_{wise_ft}_hyper_{hyperparams_str}",
    )


def save_csv(header,
             columns,
             result_path,
             dataset,
             shots,
             clip_encoder,
             image_layer_idx,
             text_layer_idx,
             text_augmentation,
             image_augmentation,
             image_views,
             modality,
             classifier_init,
             hyperparams,
             logit):
    all_headers = ['dataset', 'shots', 'clip_encoder', 'image_layer', 'text_layer', 'text_aug', 'image_aug', 'image_views','modality', 'init', 'logit', 'hyper'] + header
    all_columns = [[dataset, shots, clip_encoder, image_layer_idx, text_layer_idx, text_augmentation, image_augmentation, image_views, modality, classifier_init, logit, hyperparams] + column for column in columns]
    save_all_csv(all_headers, all_columns, result_path)
    return all_headers, all_columns

def save_all_csv(all_headers, all_columns, result_path):
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(result_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(all_columns)

def main(args):
    NAME = "_".join((
        "imagenet_domain_shift",
        args.clip_encoder,
        args.mode,
        args.text_augmentation,
        args.image_augmentation,
        str(args.image_views),
        args.modality,
        args.classifier_init,
        f"wiseft_{args.wise_ft}"
    ))

    args.text_layer_idx = 0
    if args.mode == 'linear':
        args.classifier_head = 'linear'
        args.image_layer_idx = 0
        args.hyperparams = 'linear'
        args.logit = 4.60517
    elif args.mode == 'adapter':
        args.classifier_head = 'adapter'
        args.image_layer_idx = 0
        args.hyperparams = 'adapter'
        args.logit = 4.60517
    elif args.mode == 'partial':
        args.classifier_head = 'linear'
        args.image_layer_idx = 1
        args.hyperparams = 'partial'
        args.logit = 4.0
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    all_columns = []
    all_headers = None
    for shots_idx, shots in enumerate(SHOTS):
        args.train_shot = shots
        print(f"Shots: {shots} | {shots_idx + 1}/{len(SHOTS)}")
        eval_dir = get_eval_dir(
            shots,
            args.clip_encoder,
            args.image_layer_idx,
            args.text_layer_idx,
            args.text_augmentation,
            args.image_augmentation,
            args.image_views,
            args.modality,
            args.classifier_init,
            args.hyperparams,
            args.logit,
            args.wise_ft,
            eval_dir=EVAL_DIR
        )
        all_dataset_dict = {}
        for dataset_idx, dataset in enumerate(DATASETS):
            print(f"Dataset: {dataset} | {dataset_idx + 1}/{len(DATASETS)}")
            args.dataset = dataset
            all_seed_finished = True
            all_seed_dict = {}
            for seed in SEEDS:
                args.seed = seed
                text_features_path = get_text_features_path(
                    dataset,
                    args.feature_dir,
                    args.clip_encoder,
                    args.text_layer_idx,
                    args.text_augmentation
                )
                text_features = torch.load(text_features_path)
                text_dataset = TextTensorDataset(
                    text_features['features'], text_features['labels'], text_features['eot_indices'])

                ccrop_features_path = get_image_features_path(
                    args.dataset,
                    args.train_shot,
                    args.seed,
                    args.feature_dir,
                    args.clip_encoder,
                    args.image_layer_idx,
                    "none",
                )
                ccrop_features = torch.load(ccrop_features_path)
                if args.image_augmentation == "none":
                    train_features = ccrop_features['train']['features']
                    train_labels = ccrop_features['train']['labels']
                else:
                    # Add extra views
                    image_features_path = get_image_features_path(
                        args.dataset,
                        args.train_shot,
                        args.seed,
                        args.feature_dir,
                        args.clip_encoder,
                        args.image_layer_idx,
                        args.image_augmentation,
                        image_views=args.image_views,
                    )
                    image_features = torch.load(image_features_path)
                    train_features = torch.cat([ccrop_features['train']['features'], image_features['train']['features']], dim=0)
                    train_labels = torch.cat([ccrop_features['train']['labels'], image_features['train']['labels']], dim=0)
                
                image_train_dataset = TensorDataset(
                    train_features,
                    train_labels
                )
                save_dir = get_save_dir(args)
                hyperparams = HYPER_DICT[args.hyperparams]
                VALID_BATCH_SIZES = get_valid_batch_sizes(
                    hyperparams, text_dataset, image_train_dataset, modality=args.modality)
                def get_experiment_count(hyperparams):
                    count = 1
                    count *= len(hyperparams['lr'])
                    count *= len(hyperparams['weight_decay'])
                    count *= len(VALID_BATCH_SIZES)
                    count *= len(hyperparams['max_iter'])
                    return count
                experiment_count = get_experiment_count(hyperparams)

                cur_count = 0
                all_hyper_finished = True
                all_hyper_dict = {}
                # sweep through hyperparameters
                for lr in hyperparams['lr']:
                    for wd in hyperparams['weight_decay']:
                        for batch_size in VALID_BATCH_SIZES:
                            for iters in hyperparams['max_iter']:
                                cur_count += 1
                                hyperparams_str = get_hyperparams_str(
                                    hyperparams['optim'], lr, wd, batch_size, iters)
                                
                                # all_hyper_dict[hyperparams_str] = {}
                                # check if experiment has been done
                                checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                                test_result_path = os.path.join(checkpoint_dir, "domain_shift_result.pth")
                                if not os.path.exists(test_result_path):
                                    print(f"Experiment {cur_count}/{experiment_count} not finished")
                                    import pdb; pdb.set_trace()
                                    continue
                                else:
                                    try:
                                        test_result_dict = torch.load(test_result_path)
                                    except:
                                        import pdb; pdb.set_trace()
                                    print(test_result_dict)
                                    print(f"Finished testing {hyperparams_str} {cur_count}/{experiment_count}")
                                    all_hyper_dict[hyperparams_str] = test_result_dict
                                
                if not all_hyper_finished:
                    print(f"Seed {seed} not finished!")
                    # break
                else:
                    all_seed_dict[seed] = all_hyper_dict
            
            if all_seed_finished:
                print(f"Dataset {dataset} finished! Taking average...")
                if args.wise_ft:
                    eval_types = ['head_wiseft_0.5']
                else:
                    eval_types = ['head']
                all_dataset_dict[dataset] = take_average(all_seed_dict, ALL_EVAL_TYPES=eval_types)
                this_headers, this_columns = save_csv(all_dataset_dict[dataset][0], all_dataset_dict[dataset][1],
                    os.path.join(eval_dir, dataset, "all_results.csv"),
                    dataset,
                    shots,
                    args.clip_encoder,
                    args.image_layer_idx,
                    args.text_layer_idx,
                    args.text_augmentation,
                    args.image_augmentation,
                    args.image_views,
                    args.modality,
                    args.classifier_init,
                    args.hyperparams,
                    args.logit
                )
                if all_headers == None:
                    all_headers = this_headers
                all_columns = all_columns + this_columns
            else:
                print(f"Dataset {dataset} not finished!")
    
    csv_path = os.path.join(EVAL_DIR, f"{NAME}.csv")
    print(f"Saving to {csv_path}")
    save_all_csv(all_headers, all_columns, csv_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip-encoder",
        type=str,
        default="RN50",
        choices=["ViT-B/16", "RN50"],
        help="image encoder of clip",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="linear",
        choices=[
            "linear",
            "partial",
            "adapter",
        ],
        help="finetuning mode",
    )
    parser.add_argument(
        "--text-augmentation",
        type=str,
        default='hand_crafted',
        choices=['hand_crafted', # tip_adapter selected
                'classname', # plain class name
                'vanilla', # a photo of a {cls}.
                'template_mining' # examples of best zero-shot templates for few-shot val set
                ],
        help="specify the text augmentation to use.",
    )
    parser.add_argument(
        "--image-augmentation",
        type=str,
        default='none',
        choices=['none', # only a single center crop
                'flip', # add random flip view
                'randomcrop', # add random crop view
                ],
        help="specify the image augmentation to use.",
    )
    parser.add_argument(
        "--image-views",
        type=int,
        default=1,
        help="if image-augmentation is not None, then specify the number of extra views.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="cross_modal",
        choices=["cross_modal", # half batch image, half batch text
                "uni_modal", # whole batch image
        ],
        help="whether or not to perform cross-modal training (ie. half batch is image, half batch is text)",
    )
    parser.add_argument(
        "--classifier_init",
        type=str,
        default="zeroshot",
        choices=["zeroshot", # zero-shot/one-shot-text-based initialization
                "random", # random initialization
        ],
        help="classifier head initialization",
    )
    parser.add_argument(
        "--wise_ft",
        type=bool,
        default=False,
        help="wise_ft with 0.5 ratio",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default.DATA_DIR,
        help="where the dataset is saved",
    )
    parser.add_argument(
        "--indices_dir",
        type=str,
        default=default.FEW_SHOT_DIR,
        help="where the (few-shot) indices is saved",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=default.FEATURE_DIR,
        help="where to save pre-extracted features",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=default.RESULT_DIR,
        help="where to save experiment results",
    )
    args = parser.parse_args()
    with torch.no_grad():
        main(args)
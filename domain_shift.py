from copy import deepcopy
import os

import torch
from torch.utils.data import DataLoader

from engine.config import parser

from engine.transforms.default import build_transform
from engine.tools.utils import makedirs, set_random_seed
from engine import clip
from engine.datasets.utils import TensorDataset, TextTensorDataset, get_label_map, get_testset
from engine.model.head import make_classifier_head, get_zero_shot_weights
from engine.model.logit import LogitHead
from engine.optimizer.default import HYPER_DICT
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from train import validate, \
                  get_save_dir, \
                  get_hyperparams_str, \
                  get_eval_heads, \
                  train, \
                  get_valid_batch_sizes
from features import get_backbone_name, \
                     extract_features, \
                     get_image_encoder, \
                     get_image_features_path, \
                     get_text_features_path, \
                     get_image_encoder_dir, \
                     get_text_encoder_dir, \
                     get_test_features_path
            

torch.set_num_threads(4) # To maximize efficiency, please tune the number of threads for your machine

CROSS_MODAL_BATCH_RATIO = 0.5 # Half of the batch is image, the other half is text
EVAL_FREQ = 100 # Evaluate on val set per 100 iterations (for early stopping)

IMAGENET_TESTSETS = [
    'imagenetv2',
    'imagenet_sketch',
    'imagenet_a',
    'imagenet_r',
]


def prepare_domain_shift_testset_features(args, TESTSETS=IMAGENET_TESTSETS):
    ########################################
    #   Setup Network
    ########################################
    clip_model, _ = clip.load(args.clip_encoder, jit=False)
    clip_model.float()
    clip_model.eval()

    image_encoder = get_image_encoder(clip_model, args)
    for testset in TESTSETS:
        # Check if features are saved already
        test_features_path = get_test_features_path(
            testset,
            args.feature_dir,
            args.clip_encoder,
            args.image_layer_idx
        )

        makedirs(os.path.dirname(test_features_path))
        if os.path.exists(test_features_path):
            print(f"Test features already saved at {test_features_path}")
        else:
            print(f"Saving features to {test_features_path}")
            test_transform = build_transform('none')
            benchmark_test = get_testset(testset, args.data_dir)
            print(f"Extracting features for test split ...")
            test_features = extract_features(
                image_encoder, 
                benchmark_test, test_transform,
                num_views=1, test_batch_size=args.test_batch_size, num_workers=args.num_workers)
            torch.save(test_features, test_features_path)


def main(args):
    if args.seed >= 0:
        print("Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    ########################################
    #   Prepare for domain shift testset features
    ########################################
    prepare_domain_shift_testset_features(args)

    ### Before scripts are mostly taken from train.py main() except for 
    ### evaluating on the domain shifted test set

    image_encoder_dir = get_image_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")

    text_encoder_dir = get_text_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx
    )
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")

    text_features_path = get_text_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx,
        args.text_augmentation
    )
    text_features = torch.load(text_features_path)
    # text_features['features'] = torch.nn.functional.normalize(text_features['features'], dim=1)
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
    image_val_dataset = TensorDataset(
        ccrop_features['val']['features'],
        ccrop_features['val']['labels']
    )

    test_features_path = get_test_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    test_features = torch.load(test_features_path)
    test_dataset = TensorDataset(
        test_features['features'],
        test_features['labels']
    )
    
    save_dir = get_save_dir(args)

    hyperparams = HYPER_DICT[args.hyperparams]
    # filter out invalid batch sizes
    VALID_BATCH_SIZES = get_valid_batch_sizes(hyperparams, text_dataset, image_train_dataset, modality=args.modality)

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
                    checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                    makedirs(checkpoint_dir)
                    test_result_dict = {}

                    domain_shift_result_path = os.path.join(checkpoint_dir, "domain_shift_result.pth")
                    if os.path.exists(domain_shift_result_path):
                        print(f"Already exists: {hyperparams_str} {cur_count}/{experiment_count}")
                        test_result_dict = torch.load(domain_shift_result_path)
                        continue
                    else:
                        print(f"Starting: {hyperparams_str} {cur_count}/{experiment_count}")
                    
                    # train logreg

                    image_encoder = torch.load(
                        image_encoder_path).partial_model.train().cuda()
                    text_encoder = torch.load(
                        text_encoder_path).partial_model.train().cuda()
                    head, num_classes, in_features = make_classifier_head(
                        args.classifier_head,
                        args.clip_encoder,
                        args.classifier_init,
                        text_dataset,
                        text_encoder
                    )
                    logit_head = LogitHead(
                        head,
                        logit_scale=args.logit,
                    ).train().cuda()

                    # Create the optimizer
                    params_groups = [
                        {'params': logit_head.parameters()},
                        {'params': image_encoder.parameters()},
                        {'params': text_encoder.parameters()},
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

                    text_batch_size = int(batch_size * CROSS_MODAL_BATCH_RATIO)
                    image_batch_size = batch_size - text_batch_size

                    text_loader = None
                    if text_batch_size > 0:
                        text_loader = DataLoader(
                            text_dataset,
                            batch_size=text_batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                        )
                    
                    image_loader = None
                    if image_batch_size > 0:
                        image_loader = DataLoader(
                            image_train_dataset,
                            batch_size=image_batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                        )
                    
                    val_loader = DataLoader(
                        image_val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )

                    result_dict = train(
                        logit_head, image_encoder, text_encoder, 
                        image_loader, val_loader, text_loader, 
                        optimizer, scheduler, criterion, iters,
                        eval_freq=EVAL_FREQ
                    )
                    
                    test_result_dict = {}
                    test_result_dict['val_acc'] = result_dict['val_acc']
                    test_result_dict['iter'] = result_dict['iter']
                    test_result_dict['test_accs'] = {}
                    test_result_dict['domain_shift_accs'] = {}

                    # Create the logreg model and load the weights
                    head, num_classes, in_features = make_classifier_head(
                        args.classifier_head,
                        args.clip_encoder,
                        args.classifier_init,
                        text_dataset,
                        text_encoder,
                        bias=False
                    )
                    old_logit_head = LogitHead(
                        head,
                        logit_scale=args.logit,
                    )
                    old_logit_head.load_state_dict(result_dict['logit_head'])

                    zero_shot_weights = get_zero_shot_weights(text_dataset, num_classes, in_features)
                    eval_heads = get_eval_heads(
                        deepcopy(old_logit_head.head),
                        zero_shot_weights,
                        logit=args.logit,
                        ratio_list=[0.5]
                    )

                    image_encoder = torch.load(image_encoder_path).partial_model
                    image_encoder.load_state_dict(result_dict['image_encoder'])
                    image_encoder = image_encoder.cuda().eval()
                    text_encoder = torch.load(text_encoder_path).partial_model
                    text_encoder.load_state_dict(result_dict['text_encoder'])
                    text_encoder = text_encoder.cuda().eval()

                    for eval_type in eval_heads:
                        eval_head = eval_heads[eval_type]
                        eval_head.cuda().eval()
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                        )
                        test_acc = validate(eval_head, image_encoder, test_loader, device="cuda")
                        test_result_dict['test_accs'][eval_type] = test_acc
                        eval_head.cpu()

                    ### eval for separate testset
                    for test_dataset_name in IMAGENET_TESTSETS:
                        test_result_dict['domain_shift_accs'][test_dataset_name] = {}
                        extra_test_features_path = os.path.join(
                            args.feature_dir,
                            'image',
                            "_".join([get_backbone_name(args.clip_encoder), str(args.image_layer_idx)]),
                            test_dataset_name,
                            "test.pth"
                        )
                        extra_test_features = torch.load(extra_test_features_path)
                        extra_test_dataset = TensorDataset(
                            extra_test_features['features'], extra_test_features['labels'])
                        label_map = get_label_map(args.data_dir, test_dataset_name)
                        for eval_type in eval_heads:
                            eval_head = eval_heads[eval_type]
                            eval_head.cuda().eval()
                            test_loader = DataLoader(
                                extra_test_dataset,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True,
                            )
                            if label_map is None:
                                test_acc = validate(eval_head, image_encoder, test_loader, device="cuda")
                            else:
                                # change eval_head to use label_map
                                assert isinstance(eval_head.head, torch.nn.Linear)
                                new_head = deepcopy(eval_head)
                                new_linear_head = torch.nn.Linear(eval_head.head.in_features, len(label_map), bias=False).cuda()
                                new_linear_head.weight.data = eval_head.head.weight.data[label_map]
                                new_head.head = new_linear_head
                                test_acc = validate(new_head, image_encoder, test_loader, device="cuda")
                            test_result_dict['domain_shift_accs'][test_dataset_name][eval_type] = test_acc
                    torch.save(test_result_dict, domain_shift_result_path)
                    print(test_result_dict)
                    print(f"Finished testing {hyperparams_str} {cur_count}/{experiment_count}")


if __name__ == "__main__":
    # other arguments follow features.py
    # parser.add_argument(
    #     "--modality",
    #     type=str,
    #     default="cross_modal",
    #     choices=["cross_modal", # half batch image, half batch text
    #             "uni_modal", # whole batch image
    #     ],
    #     help="whether or not to perform cross-modal training (ie. half batch is image, half batch is text)",
    # )
    # parser.add_argument(
    #     "--classifier_head",
    #     type=str,
    #     default="linear",
    #     choices=["linear", # linear classifier
    #             "adapter", # 2-layer MLP with 0.2 residual ratio following CLIP-adapter + linear classifier
    #     ],
    #     help="classifier head architecture",
    # )
    # parser.add_argument(
    #     "--classifier_init",
    #     type=str,
    #     default="zeroshot",
    #     choices=["zeroshot", # zero-shot/one-shot-text-based initialization
    #             "random", # random initialization
    #     ],
    #     help="classifier head initialization",
    # )
    # parser.add_argument(
    #     "--logit",
    #     type=float,
    #     default=4.60517, # CLIP's default logit scaling
    #     choices=[4.60517, # CLIP's default logit scaling
    #             4.0, # for partial finetuning
    #     ],
    #     help="logit scale (exp(logit) is the inverse softmax temperature)",
    # )
    # parser.add_argument(
    #     "--hyperparams",
    #     type=str,
    #     default="linear",
    #     choices=["linear", # linear hyper
    #             "adapter", # adapter hyper
    #             "partial", # partial hyper
    #     ],
    #     help="hyperparams sweep",
    # )
    args = parser.parse_args()
    assert args.dataset == "imagenet"
    main(args)

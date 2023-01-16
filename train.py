from copy import deepcopy
import os

import torch
from torch.utils.data import DataLoader

from engine.config import parser

from engine.tools.utils import makedirs, set_random_seed
from engine.datasets.utils import TensorDataset, TextTensorDataset
from engine.model.head import make_classifier_head, get_zero_shot_weights
from engine.model.logit import LogitHead
from engine.optimizer.default import HYPER_DICT
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from features import get_backbone_name, \
                     get_few_shot_setup_name, \
                     get_view_name, \
                     get_image_features_path, \
                     get_text_features_path, \
                     get_image_encoder_dir, \
                     get_text_encoder_dir, \
                     get_test_features_path

torch.set_num_threads(4) # To maximize efficiency, please tune the number of threads for your machine

CROSS_MODAL_BATCH_RATIO = 0.5 # Half of the batch is image, the other half is text
EVAL_FREQ = 100 # Evaluate on val set per 100 iterations (for early stopping)


def get_benchmark_name(dataset, train_shot, seed):
    benchmark_name = "-".join([
        dataset,
        get_few_shot_setup_name(train_shot, seed)
    ])
    return benchmark_name


def get_modality_name(modality,
                      clip_encoder,
                      image_augmentation,
                      text_augmentation,
                      image_layer_idx,
                      text_layer_idx,
                      image_views=1):
    text_feature_name = f"text_{text_layer_idx}_{text_augmentation}"
    image_feature_name = f"image_{image_layer_idx}_{get_view_name(image_augmentation, image_views=image_views)}"
    if modality == "cross_modal":
        feature_name = f"{text_feature_name}-{image_feature_name}"
    elif modality == "uni_modal":
        feature_name = image_feature_name
    return os.path.join(
        get_backbone_name(clip_encoder),
        feature_name
    )


def get_architecture_name(classifier_head, classifier_init):
    return classifier_head + "_" + classifier_init


def get_logit_name(logit):
    name = f"logit_{logit}"
    return name


def get_save_dir(args):
    save_dir = os.path.join(
        args.result_dir,
        get_benchmark_name(
            args.dataset,
            args.train_shot,
            args.seed
        ),
        get_modality_name(
            args.modality,
            args.clip_encoder,
            args.image_augmentation,
            args.text_augmentation,
            args.image_layer_idx,
            args.text_layer_idx,
            image_views=args.image_views
        ),
        get_architecture_name(
            args.classifier_head,
            args.classifier_init
        ),
        get_logit_name(
            args.logit
        ),
    )
    return save_dir


def get_hyperparams_str(optim,
                        lr,
                        wd,
                        batch_size,
                        iters):
    hyperparams_str = f"optim_{optim}-lr_{lr}-wd_{wd}-bs_{batch_size}-iters_{iters}"
    return hyperparams_str


def get_wiseft(head, zero_shot_weights, wiseft_ratio=0.5):
    if type(head) == torch.nn.Linear:
        head.weight.data = (1 - wiseft_ratio) * head.weight.data + wiseft_ratio * torch.nn.functional.normalize(zero_shot_weights, dim=1)
    elif type(head) == torch.nn.Sequential:
        assert type(head[-1]) == torch.nn.Linear, f"Invalid head: {head}"
        head[-1].weight.data = (1 - wiseft_ratio) * head[-1].weight.data + wiseft_ratio * torch.nn.functional.normalize(zero_shot_weights, dim=1)
    return head


def get_eval_heads(head, zero_shot_weights, ratio_list=[0.5], logit=None):
    logit_head = LogitHead(
        deepcopy(head),
        logit_scale=logit,
    )

    eval_heads = {
        'head': logit_head.cuda().eval(),
    }
    for ratio in ratio_list:
        # TODO (Warning): This wise-ft does not consider partial finetuning of image encoder
        wiseft = get_wiseft(deepcopy(head), zero_shot_weights, ratio)
        wiseft_head = LogitHead(
            wiseft,
            logit_scale=logit,
        )
        eval_heads[f'head_wiseft_{ratio}'] = wiseft_head.cuda().eval()
    return eval_heads


def train(logit_head, image_encoder, text_encoder,
          image_loader, val_loader, text_loader,
          optimizer, scheduler, criterion, iters,
          eval_freq=EVAL_FREQ, device="cuda"):
    if image_loader is None and text_loader is None:
        raise ValueError("Both image_loader and text_loader are None")
    if image_loader is not None:
        image_loader_iter = iter(image_loader)
    else:
        image_loader_iter = None
    if text_loader is not None:
        text_loader_iter = iter(text_loader)
    else:
        text_loader_iter = None

    result_dict = {
        "iter": None,
        "val_acc": None,
        "image_encoder": None,
        "text_encoder": None,
        "logit_head": None,
    }

    for i in range(iters):
        logit_head.train()
        image_encoder.train()
        text_encoder.train()
        if image_loader_iter is not None:
            try:
                image, image_label = next(image_loader_iter)
            except StopIteration:
                image_loader_iter = iter(image_loader)
                image, image_label = next(image_loader_iter)
            image = image.to(device)
            image_label = image_label.to(device)
            image_feature = image_encoder(image)
        else:
            image_feature = None
        
        if text_loader_iter is not None:
            try:
                text, text_label, eot_indices = next(text_loader_iter)
            except StopIteration:
                text_loader_iter = iter(text_loader)
                text, text_label, eot_indices = next(text_loader_iter)
            text = text.to(device)
            text_label = text_label.to(device)
            eot_indices = eot_indices.to(device)
            text_feature = text_encoder(text, eot_indices)
        else:
            text_feature = None
        
        if image_feature is not None and text_feature is not None:
            feature = torch.cat([image_feature, text_feature], dim=0)
            label = torch.cat([image_label, text_label], dim=0)
        elif image_feature is not None:
            feature = image_feature
            label = image_label
        elif text_feature is not None:
            feature = text_feature
            label = text_label
        else:
            raise ValueError("Both image_feature and text_feature are None")

        logit = logit_head(feature)
        loss = criterion(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % eval_freq == 0:
            val_acc = validate(logit_head, image_encoder, val_loader, device=device)
            if result_dict["val_acc"] is None or val_acc > result_dict["val_acc"]:
                result_dict["iter"] = i
                result_dict["val_acc"] = val_acc
                result_dict["image_encoder"] = deepcopy(image_encoder.state_dict())
                result_dict["text_encoder"] = deepcopy(text_encoder.state_dict())
                result_dict["logit_head"] = deepcopy(logit_head.state_dict())
    
    # load best model
    image_encoder.load_state_dict(result_dict["image_encoder"])
    text_encoder.load_state_dict(result_dict["text_encoder"])
    logit_head.load_state_dict(result_dict["logit_head"])
    val_acc = validate(logit_head, image_encoder, val_loader, device=device)
    print(f"Best val acc: {result_dict['val_acc']:.4f} at iter {result_dict['iter']}")
    return result_dict
            

def validate(logit_head, image_encoder, val_loader, device="cuda"):
    with torch.no_grad():
        logit_head.eval()
        image_encoder.eval()
        val_acc = 0
        val_count = 0.
        for image, image_label in val_loader:
            image = image.to(device)
            image_label = image_label.to(device)
            image_feature = image_encoder(image)
            logit = logit_head(image_feature)
            pred = torch.argmax(logit, dim=1)
            val_acc += torch.sum(pred == image_label).item()
            val_count += image_label.size(0)
            image.cpu()
        val_acc /= val_count
    return val_acc

def get_valid_batch_sizes(hyperparams, text_dataset, image_train_dataset, batch_ratio=CROSS_MODAL_BATCH_RATIO, modality='cross_modal'):
    VALID_BATCH_SIZES = []
    if modality == 'uni_modal':
        batch_ratio = 0.
    for batch_size in hyperparams['batch_size']:
        text_batch_size = int(batch_size * batch_ratio)
        image_batch_size = batch_size - text_batch_size
        # check if text batch size is smaller than the size of text dataset
        if text_batch_size == 0 or text_batch_size < len(text_dataset):
            # check if image batch size is smaller than the size of image dataset
            if image_batch_size == 0 or image_batch_size < len(image_train_dataset):
                VALID_BATCH_SIZES.append(batch_size)
    if len(VALID_BATCH_SIZES) == 0:
        raise ValueError("No valid batch size found. You should consider reducing the batch size.")
    print("Valid batch sizes: {}/{}".format(len(VALID_BATCH_SIZES), len(hyperparams['batch_size'])))
    return VALID_BATCH_SIZES

def main(args):
    if args.seed >= 0:
        print("Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

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
                    test_result_path = os.path.join(checkpoint_dir, "test_result.pth")
                    if os.path.exists(test_result_path):
                        print(f"Already exists: {hyperparams_str} {cur_count}/{experiment_count}")
                        test_result_dict = torch.load(test_result_path)
                        continue
                    else:
                        print(f"Starting: {hyperparams_str} {cur_count}/{experiment_count}")
                    
                    # train logreg

                    # Create the logreg model
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

                    if args.modality == "cross_modal":
                        text_batch_size = int(batch_size * CROSS_MODAL_BATCH_RATIO)
                    elif args.modality == "uni_modal":
                        text_batch_size = 0
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

                    image_encoder = torch.load(image_encoder_path).partial_model
                    image_encoder.load_state_dict(result_dict['image_encoder'])
                    image_encoder = image_encoder.cuda().eval()
                    text_encoder = torch.load(text_encoder_path).partial_model
                    text_encoder.load_state_dict(result_dict['text_encoder'])
                    text_encoder = text_encoder.cuda().eval()
                    original_text_encoder = torch.load(text_encoder_path).partial_model
                    original_text_encoder = original_text_encoder.eval()

                    zero_shot_weights = get_zero_shot_weights(text_dataset, num_classes, in_features, deepcopy(original_text_encoder).cuda())
                    # zero_shot_weights = get_zero_shot_weights(text_dataset, num_classes, in_features)
                    eval_heads = get_eval_heads(
                        deepcopy(old_logit_head.head),
                        zero_shot_weights,
                        logit=args.logit,
                        ratio_list=[0.5]
                    )


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
                    torch.save(test_result_dict, test_result_path)
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
    main(args)
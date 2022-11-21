import os

from engine.config import parser

import torch

from engine.tools.utils import makedirs, set_random_seed
from engine.transforms.default import build_transform
from engine.datasets.utils import DatasetWrapper, get_few_shot_setup_name, get_few_shot_benchmark
from engine.templates import get_templates
from engine import clip
from engine.clip import partial_model


def get_backbone_name(clip_encoder):
    return clip_encoder.replace("/", "-")


def get_image_encoder_name(clip_encoder, image_layer_idx):
    return "_".join([get_backbone_name(clip_encoder), str(image_layer_idx)])


def get_text_encoder_name(clip_encoder, text_layer_idx):
    return "_".join([get_backbone_name(clip_encoder), str(text_layer_idx)])


def get_view_name(image_augmentation, image_views=1):
    name = f"{image_augmentation}"
    if image_augmentation != "none":
        assert image_views > 0
        name += f"_view_{image_views}"
    return name


def get_image_encoder_dir(feature_dir, clip_encoder, image_layer_idx):
    image_encoder_path = os.path.join(
        feature_dir,
        'image',
        get_image_encoder_name(clip_encoder, image_layer_idx)
    )
    return image_encoder_path


def get_image_features_path(dataset,
                            train_shot,
                            seed,
                            feature_dir,
                            clip_encoder,
                            image_layer_idx,
                            image_augmentation,
                            image_views=1):
    image_features_path = os.path.join(
        get_image_encoder_dir(feature_dir, clip_encoder, image_layer_idx),
        dataset,
        get_view_name(image_augmentation, image_views),
        f"{get_few_shot_setup_name(train_shot, seed)}.pth")
    return image_features_path


def get_test_features_path(dataset,
                           feature_dir,
                           clip_encoder,
                           image_layer_idx):
    test_features_path = os.path.join(
        get_image_encoder_dir(feature_dir, clip_encoder, image_layer_idx),
        dataset,
        "test.pth"
    )
    return test_features_path


def get_text_encoder_dir(feature_dir,
                         clip_encoder,
                         text_layer_idx):
    text_encoder_path = os.path.join(
        feature_dir,
        'text',
        get_text_encoder_name(clip_encoder, text_layer_idx)
    )
    return text_encoder_path


def get_text_features_path(dataset,
                           feature_dir,
                           clip_encoder,
                           text_layer_idx,
                           text_augmentation):
    text_features_path = os.path.join(
        get_text_encoder_dir(feature_dir, clip_encoder, text_layer_idx),
        dataset,
        f"{text_augmentation}.pth")
    return text_features_path



def extract_text_features(dataset, text_augmentation, text_encoder, lab2cname):
    # Extract text features from CLIP
    features_dict = {
        'features': None,
        'labels': None,
        'eot_indices': None,
        'prompts': {},
        'lab2cname': lab2cname,
    }
    templates = get_templates(dataset, text_augmentation)
    text_encoder.feature_extractor.eval()
    with torch.no_grad():
        for label, cname in lab2cname.items():
            str_prompts = [template.format(cname.replace("_", " ")) for template in templates]
            prompts = torch.cat([clip.tokenize(p) for p in str_prompts]).cuda()
            features, eot_indices = text_encoder.feature_extractor(prompts)
            features = features.cpu()
            eot_indices = eot_indices.cpu()
            labels = torch.Tensor([label for _ in templates]).long()
            if features_dict['features'] is None:
                features_dict['features'] = features
                features_dict['labels'] = labels
                features_dict['eot_indices'] = eot_indices
            else:
                features_dict['features'] = torch.cat((features_dict['features'], features), 0)
                features_dict['labels'] = torch.cat((features_dict['labels'], labels))
                features_dict['eot_indices'] = torch.cat((features_dict['eot_indices'], eot_indices))
            features_dict['prompts'][label] = str_prompts
    return features_dict



def extract_features(image_encoder, data_source, transform, num_views=1, test_batch_size=32, num_workers=4):
    features_dict = {
        'features': torch.Tensor(),
        'labels': torch.Tensor(),
        'paths': [],
    }
    ######################################
    #   Setup DataLoader
    ######################################
    loader = torch.utils.data.DataLoader(
        DatasetWrapper(data_source, transform=transform),
        batch_size=test_batch_size,
        sampler=None,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    ########################################
    # Start Feature Extractor
    ########################################
    image_encoder.feature_extractor.eval()

    with torch.no_grad():
        for _ in range(num_views):
            for batch_idx, batch in enumerate(loader):
                data = batch["img"].cuda()
                feature = image_encoder.feature_extractor(data) # This is not L2 normed
                feature = feature.cpu()
                if batch_idx == 0:
                    features_dict['features'] = feature
                    features_dict['labels'] = batch['label']
                    features_dict['paths'] = batch['impath']
                else:
                    features_dict['features'] = torch.cat((features_dict['features'], feature), 0)
                    features_dict['labels'] = torch.cat((features_dict['labels'], batch['label']))
                    features_dict['paths'] = features_dict['paths'] + list(batch['impath'])
    return features_dict


def prepare_text_features(clip_model, args, lab2cname):
    text_encoder_dir = get_text_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx
    )
    makedirs(text_encoder_dir)
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    # Check if text partial model exists already
    if os.path.exists(text_encoder_path):
        print(f"text encoder already saved at {text_encoder_path}")
        text_encoder = torch.load(text_encoder_path)
    else:
        print(f"Saving text encoder to {text_encoder_path}")
        text_encoder = partial_model.get_text_encoder(
            args.text_layer_idx,
            clip_model
        )
        torch.save(text_encoder, text_encoder_path)

    # Text features extraction
    text_features_path = get_text_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx,
        args.text_augmentation
    )

    makedirs(os.path.dirname(text_features_path))

    if os.path.exists(text_features_path):
        print(f"Text features already saved at {text_features_path}")
    else:
        print(f"Saving features to {text_features_path}")
        text_features = {
            'features': torch.Tensor(),
            'labels': torch.Tensor(),
            'prompts': [],
            'classnames': [],
        }
        print(f"Extracting features for texts ...")
        text_features = extract_text_features(
            args.dataset, args.text_augmentation, text_encoder, lab2cname)
        torch.save(text_features, text_features_path)


def get_image_encoder(clip_model, args):
    image_encoder_dir = get_image_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    makedirs(image_encoder_dir)
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")
    # Check if image partial model exists already
    if os.path.exists(image_encoder_path):
        print(f"Image encoder already saved at {image_encoder_path}")
        image_encoder = torch.load(image_encoder_path)
    else:
        print(f"Saving image encoder to {image_encoder_path}")
        image_encoder = partial_model.get_image_encoder(
            args.clip_encoder,
            args.image_layer_idx,
            clip_model
        )
        torch.save(image_encoder, image_encoder_path)
    return image_encoder


def prepare_few_shot_image_features(clip_model, args, benchmark_train, benchmark_val):
    image_encoder = get_image_encoder(clip_model, args)
    # Check if (image) features are saved already
    image_features_path = get_image_features_path(
        args.dataset,
        args.train_shot,
        args.seed,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx,
        args.image_augmentation,
        image_views=args.image_views
    )

    makedirs(os.path.dirname(image_features_path))
    
    # import pdb; pdb.set_trace()
    if os.path.exists(image_features_path):
        print(f"Features already saved at {image_features_path}")
    else:
        print(f"Saving features to {image_features_path}")
        image_features = {
            'train': {},
            'val': {},
        }
        train_transform = build_transform(args.image_augmentation)
        test_transform = build_transform('none')
        print(f"Extracting features for train split ...")
        if args.image_augmentation == 'none':
            num_views = 1
        else:
            num_views = args.image_views
        assert num_views > 0, "Number of views must be greater than 0"
        image_features['train'] = extract_features(
            image_encoder, benchmark_train, 
            train_transform, num_views=num_views, test_batch_size=args.test_batch_size, num_workers=args.num_workers)
        
        print(f"Extracting features for val split ...")
        image_features['val'] = extract_features(
            image_encoder, benchmark_val,
            test_transform, num_views=1, test_batch_size=args.test_batch_size, num_workers=args.num_workers)
    
        torch.save(image_features, image_features_path)


def prepare_test_image_features(clip_model, args, benchmark_test):
    image_encoder = get_image_encoder(clip_model, args)
    # Check if features are saved already
    test_features_path = get_test_features_path(
        args.dataset,
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
    #   Train/Val/Test Split
    ########################################
    few_shot_benchmark = get_few_shot_benchmark(
        args.data_dir,
        args.indices_dir,
        args.dataset,
        args.train_shot,
        args.seed
    )

    ########################################
    #   Setup Network
    ########################################
    clip_model, _ = clip.load(args.clip_encoder, jit=False)
    clip_model.float()
    clip_model.eval()


    ########################################
    #   Feature Extraction
    ########################################
    prepare_text_features(clip_model, args, few_shot_benchmark['lab2cname'])

    prepare_few_shot_image_features(clip_model, args, few_shot_benchmark['train'], few_shot_benchmark['val'])

    prepare_test_image_features(clip_model, args, few_shot_benchmark['test'])


if __name__ == "__main__":
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="",
    #     choices=dataset_classes.keys(),
    #     help="number of train shot",
    # )
    # parser.add_argument(
    #     "--train-shot",
    #     type=int,
    #     default=1,
    #     help="number of train shot",
    # )
    # parser.add_argument(
    #     "--max-val-shot",
    #     type=int,
    #     default=4,
    #     help="number of val shot is min(max_val_shot, train_shot)",
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=1,
    #     help="seed number",
    # )
    # parser.add_argument(
    #     "--clip-encoder",
    #     type=str,
    #     default="RN50",
    #     choices=["ViT-B/16", "ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16"],
    #     help="specify the clip encoder to use",
    # )
    # parser.add_argument(
    #     "--image-layer-idx",
    #     type=int,
    #     default=0,
    #     choices=[0, 1],
    #     help="specify how many image encoder layers to finetune. 0 means none. -1 means full finetuning.",
    # )
    # parser.add_argument(
    #     "--text-layer-idx",
    #     type=int,
    #     default=0,
    #     choices=[0, 1],
    #     help="specify how many text encoder layers to finetune. 0 means none. -1 means full finetuning.",
    # )
    # parser.add_argument(
    #     "--text-augmentation",
    #     type=str,
    #     default='hand_crafted',
    #     choices=['hand_crafted', # tip_adapter selected
    #             'classname', # plain class name
    #             'vanilla', # a photo of a {cls}.
    #             'template_mining' # examples of best zero-shot templates for few-shot val set
    #             ],
    #     help="specify the text augmentation to use.",
    # )
    # parser.add_argument(
    #     "--image-augmentation",
    #     type=str,
    #     default='none',
    #     choices=['none', # only a single center crop
    #             'flip', # add random flip view
    #             'randomcrop', # add random crop view
    #             ],
    #     help="specify the image augmentation to use.",
    # )
    # parser.add_argument(
    #     "--image-views",
    #     type=int,
    #     default=1,
    #     help="if image-augmentation is not None, then specify the number of extra views.",
    # )
    args = parser.parse_args()
    main(args)
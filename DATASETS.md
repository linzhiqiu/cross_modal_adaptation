# How to install datasets

*The dataset download instruction is modified from official [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md). We also include installation of the audio classification dataset ESC-50 in this tutorial.*

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:

- [ImageNet](#imagenet)
- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [StanfordCars](#stanfordcars)
- [Flowers102](#flowers102)
- [Food101](#food101)
- [FGVCAircraft](#fgvcaircraft)
- [SUN397](#sun397)
- [DTD](#dtd)
- [EuroSAT](#eurosat)
- [UCF101](#ucf101)
- [ImageNetV2](#imagenetv2)
- [ImageNet-Sketch](#imagenet-sketch)
- [ImageNet-A](#imagenet-a)
- [ImageNet-R](#imagenet-r)
- [ESC-50](#esc-50)

The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us.

### ImageNet

- Create a folder named `imagenet/` under `$DATA`.
- Download `split_ImageNet.json` [(google drive link))](https://drive.google.com/file/d/1SvPIN6iV6NP2Oulj19a869rBXrB5SNFo/view) to this folder. (Note that the original CoOp does not make a train/val split.)
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`. The directory structure should look like

```
imagenet/
|–– split_ImageNet.json
|–– images/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
```

- If you had downloaded the ImageNet dataset before, you can create symbolic links to map the training and validation sets to `$DATA/imagenet/images`.
- Download the `classnames.txt` to `$DATA/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

### Caltech101

- Create a folder named `caltech-101/` under `$DATA`.
- Download `caltech-101.zip` from <https://data.caltech.edu/records/20086> and extract `101_ObjectCategories.tar.gz`  under `$DATA/caltech-101`.
- Copy `split_zhou_Caltech101.json` from [splits/split_zhou_Caltech101.json](splits/split_zhou_Caltech101.json) to this folder.

The directory structure should look like

```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### OxfordPets

- Create a folder named `oxford_pets/` under `$DATA`.
- Download the images from <https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz>.
- Download the annotations from <https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz>.
- Copy `split_zhou_OxfordPets.json` from [splits/split_zhou_OxfordPets.json](splits/split_zhou_OxfordPets.json) to this folder.

The directory structure should look like

```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### StanfordCars

- Create a folder named `stanford_cars/` under `$DATA`.
- Download the train images <http://ai.stanford.edu/~jkrause/car196/cars_train.tgz>.
- Download the test images <http://ai.stanford.edu/~jkrause/car196/cars_test.tgz>.
- Download the train labels <https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz>.
- Download the test labels <http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat>.
- Copy `split_zhou_StanfordCars.json` from [splits/split_zhou_StanfordCars.json](splits/split_zhou_StanfordCars.json) to this folder.

The directory structure should look like

```
stanford_cars/
|–– cars_test\
|–– cars_test_annos_withlabels.mat
|–– cars_train\
|–– devkit\
|–– split_zhou_StanfordCars.json
```

### Flowers102

- Create a folder named `oxford_flowers/` under `$DATA`.
- Download and extract the images and labels from <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz> and <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat> respectively.
- Copy `split_zhou_OxfordFlowers.json` from [splits/split_zhou_OxfordFlowers.json](splits/split_zhou_OxfordFlowers.json) to this folder.
- Copy `cat_to_name.json` from [splits/cat_to_name.json](splits/cat_to_name.json) to this folder.

The directory structure should look like

```
oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### Food101

- Download the dataset from <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/> and extract the file `food-101.tar.gz` under `$DATA`, resulting in a folder named `$DATA/food-101/`.
- Copy `split_zhou_Food101.json` from [splits/split_zhou_Food101.json](splits/split_zhou_Food101.json) to this folder.

The directory structure should look like

```
food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### FGVCAircraft

- Download the data from <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz>.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like

```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### SUN397

- Create a folder named  `sun397/` under `$DATA`.
- Download the images <http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz>.
- Download the partitions <https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip>.
- Extract these files under `$DATA/sun397/`.
- Copy `split_zhou_SUN397.json` from [splits/split_zhou_SUN397.json](splits/split_zhou_SUN397.json) to this folder.

The directory structure should look like

```
sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### DTD

- Download the dataset from <https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz> and extract it to `$DATA`. This should lead to `$DATA/dtd/`.
- Copy `split_zhou_DescribableTextures.json` from [splits/split_zhou_DescribableTextures.json](splits/split_zhou_DescribableTextures.json) to this folder.

The directory structure should look like

```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT

- Create a folder named `eurosat/` under `$DATA`.
- Download the dataset from <http://madm.dfki.de/files/sentinel/EuroSAT.zip> and extract it to `$DATA/eurosat/`.
- Copy `split_zhou_EuroSAT.json` from [splits/split_zhou_EuroSAT.json](splits/split_zhou_EuroSAT.json) to this folder.

The directory structure should look like

```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### UCF101

- Create a folder named `ucf101/` under `$DATA`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `$DATA/ucf101/`. This zip file contains the extracted middle video frames.
- Copy `split_zhou_UCF101.json` from [splits/split_zhou_UCF101.json](splits/split_zhou_UCF101.json) to this folder.

The directory structure should look like

```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```

### ImageNetV2

- Create a folder named `imagenetv2/` under `$DATA`.
<!-- - Go to this github repo <https://github.com/modestyachts/ImageNetV2>. -->
- Download the matched-frequency dataset from <https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-matched-frequency.tar.gz> and extract it to `$DATA/imagenetv2/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenetv2/`.

The directory structure should look like

```
imagenetv2/
|–– imagenetv2-matched-frequency-format-val/
|–– classnames.txt
```

### ImageNet-Sketch

- Create a folder named `imagenet-sketch/` under `$DATA`.
- Download the dataset from <https://github.com/HaohanWang/ImageNet-Sketch> (if you have [gdown](https://pypi.org/project/gdown/) installed, you can run `gdown https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA`).
- Extract the dataset to `$DATA/imagenet-sketch`. Rename the folder `sketch/` to `images/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-sketch/`.

The directory structure should look like

```
imagenet-sketch/
|–– images/ # contains 1,000 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-A

- Create a folder named `imagenet-adversarial/` under `$DATA`.
- Download the dataset from <https://github.com/hendrycks/natural-adv-examples> (or run `wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar`)
- Extract it to `$DATA/imagenet-adversarial/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-adversarial/`.

The directory structure should look like

```
imagenet-adversarial/
|–– imagenet-a/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-R

- Create a folder named `imagenet-rendition/` under `$DATA`.
- Download the dataset from <https://github.com/hendrycks/imagenet-r> (or run `wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar`)
- Extract it to `$DATA/imagenet-rendition/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-rendition/`.

The directory structure should look like

```
imagenet-rendition/
|–– imagenet-r/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```

### ESC-50

- Create a folder named `esc-50/` under `$DATA`.
- Download the master.zip from <https://github.com/karolpiczak/ESC-50> (or run `wget https://github.com/karoldvl/ESC-50/archive/master.zip`)
- Extract it to `$DATA/esc-50/`.
- Download AudioCLIP checkpoint with Frozen CLIP to this folder via `wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Partial-Training.pt`
<!-- - Download word embedding to this folder via `wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz` -->

The directory structure should look like

```
esc-50/
|–– ESC-50-master/ # contains ESC-50 original dataset
|–– AudioCLIP-Partial-Training.pt
```

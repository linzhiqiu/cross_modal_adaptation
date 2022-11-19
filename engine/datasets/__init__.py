from engine.datasets.oxford_pets import OxfordPets
from engine.datasets.oxford_flowers import OxfordFlowers
from engine.datasets.fgvc_aircraft import FGVCAircraft
from engine.datasets.dtd import DescribableTextures
from engine.datasets.eurosat import EuroSAT
from engine.datasets.stanford_cars import StanfordCars
from engine.datasets.food101 import Food101
from engine.datasets.sun397 import SUN397
from engine.datasets.caltech101 import Caltech101
from engine.datasets.ucf101 import UCF101
from engine.datasets.imagenet import ImageNet
from engine.datasets.imagenetv2 import ImageNetV2
from engine.datasets.imagenet_sketch import ImageNetSketch
from engine.datasets.imagenet_a import ImageNetA
from engine.datasets.imagenet_r import ImageNetR


dataset_classes = {
    "oxford_pets": OxfordPets,
    "oxford_flowers": OxfordFlowers,
    "fgvc_aircraft": FGVCAircraft,
    "dtd": DescribableTextures,
    "eurosat": EuroSAT,
    "stanford_cars": StanfordCars,
    "food101": Food101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "ucf101": UCF101,
    "imagenet": ImageNet,
    "imagenetv2": ImageNetV2,
    "imagenet_sketch": ImageNetSketch,
    "imagenet_a": ImageNetA,
    "imagenet_r": ImageNetR,
}
import os

from engine.tools.utils import listdir_nohidden
from engine.datasets.benchmark import Benchmark
from engine.datasets.imagenet import read_classnames

TO_BE_IGNORED = ["README.txt"]


class ImageNetR(Benchmark):
    """ImageNet-R(endition).

    This dataset is used for testing only.
    """

    dataset_name = "imagenet-rendition"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.original_imagenet_dir = os.path.join(root, "imagenet")
        original_text_file = os.path.join(self.original_imagenet_dir, "classnames.txt")
        original_classnames = read_classnames(original_text_file)

        self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)

        data, label_map = self.read_data(classnames, original_classnames)
        self.label_map = label_map

        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames, original_classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]

        original_folders = [folder for folder in original_classnames]
        label_map = [original_folders.index(folder) for folder in folders]

        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = {"impath": impath, "label": label, "classname": classname}
                items.append(item)

        return items, label_map

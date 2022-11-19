import os

from engine.datasets.benchmark import Benchmark
from engine.tools.utils import listdir_nohidden
from engine.datasets.imagenet import read_classnames


class ImageNetV2(Benchmark):
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_name = "imagenetv2"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = {"impath": impath, "label": label, "classname": classname}
                items.append(item)

        return items

import os
from collections import OrderedDict

from engine.datasets.benchmark import read_split, save_split, split_trainval, Benchmark
from engine.tools.utils import listdir_nohidden


def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames

class ImageNet(Benchmark):

    dataset_name = "imagenet"
    split_google_url = "https://drive.google.com/file/d/1SvPIN6iV6NP2Oulj19a869rBXrB5SNFo/view"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_ImageNet.json")

        if not os.path.exists(self.split_path):
            print(
                f"Please download the split path from {self.split_google_url}"
                f" and put it to {self.split_path}")
            raise FileNotFoundError(self.split_path)
        
        train, val, test = read_split(self.split_path, self.image_dir)

        # # Uncomment the following lines to generate a new split
        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames = self.read_classnames(text_file)
        # train = self.read_data(classnames, "train")
        # # Follow standard practice to perform evaluation on the val set
        # # Also used as the val set (so evaluate the last-step model)
        # test = self.read_data(classnames, "val")
        # # If you want to generate new split, uncomment the following lines.
        # train, val = split_trainval(train)
        # save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = {'impath': impath,
                        'label': label,
                        'classname': classname}
                items.append(item)

        return items
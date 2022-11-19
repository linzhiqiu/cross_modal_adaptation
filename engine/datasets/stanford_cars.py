import os
from scipy.io import loadmat

from engine.datasets.benchmark import Benchmark, read_split, save_split, split_trainval

class StanfordCars(Benchmark):
    dataset_name = "stanford_cars"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")

        train, val, test = read_split(self.split_path, self.dataset_dir)

        # # Uncomment the following lines to generate a new split
        # trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
        # test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
        # meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
        # trainval = self.read_data("cars_train", trainval_file, meta_file)
        # test = self.read_data("cars_test", test_file, meta_file)
        # train, val = split_trainval(trainval)
        # save_split(train, val, test, self.split_path, self.dataset_dir)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = {"impath": impath, "label": label, "classname": classname}
            items.append(item)

        return items

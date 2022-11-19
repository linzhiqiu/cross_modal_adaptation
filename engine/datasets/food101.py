import os

from engine.datasets.benchmark import Benchmark, read_split, read_and_split_data, save_split


class Food101(Benchmark):

    dataset_name = "food-101"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")

        train, val, test = read_split(self.split_path, self.image_dir)

        # # Uncomment the following lines to generate a new split
        # train, val, test = read_and_split_data(self.image_dir)
        # save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)

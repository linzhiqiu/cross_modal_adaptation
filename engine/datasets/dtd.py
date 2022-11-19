import os

from engine.datasets.benchmark import read_and_split_data, read_split, Benchmark


class DescribableTextures(Benchmark):

    dataset_name = "dtd"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")
        
        assert os.path.exists(self.split_path)
        train, val, test = read_split(self.split_path, self.image_dir)
        # # Uncomment the following lines to generate a new split
        # train, val, test = read_and_split_data(self.image_dir)
        # save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)
import os

from engine.datasets.benchmark import read_split, split_trainval, Benchmark

class OxfordPets(Benchmark):

    dataset_name = "oxford_pets"

    def __init__(self, data_dir):
        root = data_dir

        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")

        assert os.path.exists(self.split_path)
        train, val, test = read_split(self.split_path, self.image_dir)
        
        # If you want to generate new split, uncomment the following lines.
        # trainval = self.read_data(split_file="trainval.txt")
        # test = self.read_data(split_file="test.txt")
        # train, val = split_trainval(trainval)
        # self.save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, _, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = {'impath': impath,
                        'label': label,
                        'classname': breed}
                items.append(item)

        return items
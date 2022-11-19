import os

from engine.datasets.benchmark import Benchmark, read_split, save_split, split_trainval


class SUN397(Benchmark):

    dataset_name = "sun397"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")

        train, val, test = read_split(self.split_path, self.image_dir)
        # classnames = []
        # with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip()[1:]  # remove /
        #         classnames.append(line)
        # cname2lab = {c: i for i, c in enumerate(classnames)}
        # trainval = self.read_data(cname2lab, "Training_01.txt")
        # test = self.read_data(cname2lab, "Testing_01.txt")
        # train, val = split_trainval(trainval)
        # save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)
                item = {'impath': impath, 'label': label, 'classname': classname}
                items.append(item)

        return items

import os

from engine.config import parser
from engine.datasets import dataset_classes

from engine.tools.utils import makedirs, save_as_json, set_random_seed
from engine.datasets.utils import get_few_shot_setup_name
from engine.datasets.benchmark import generate_fewshot_dataset


def main(args):
    if args.seed >= 0:
        print("Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    # Check if the dataset is supported
    assert args.dataset in dataset_classes
    few_shot_index_file = os.path.join(
        args.indices_dir,
        args.dataset,
        get_few_shot_setup_name(args.train_shot, args.seed) + ".json"
    )
    if os.path.exists(few_shot_index_file):
        # If the json file exists, then load it
        print(f"Few-shot data exists at {few_shot_index_file}.")
    else:
        # If the json file does not exist, then create it
        print(f"Few-shot data does not exist at {few_shot_index_file}. Sample a new split.")
        makedirs(os.path.dirname(few_shot_index_file))
        benchmark = dataset_classes[args.dataset](args.data_dir)
        few_shot_dataset = generate_fewshot_dataset(
            benchmark.train,
            benchmark.val,
            num_shots=args.train_shot,
            max_val_shots=args.max_val_shot,
        )
        save_as_json(few_shot_dataset, few_shot_index_file)


if __name__ == "__main__":
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="",
    #     help="Name of the dataset",
    # )
    # parser.add_argument(
    #     "--train-shot",
    #     type=int,
    #     default=1,
    #     help="number of train shot",
    # )
    # parser.add_argument(
    #     "--max-val-shot",
    #     type=int,
    #     default=4,
    #     help="number of val shot is min(max_val_shot, train_shot). default is 4 following CoOp's protocol",
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=1,
    #     help="seed number",
    # )
    args = parser.parse_args()
    main(args)
    

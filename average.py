import csv
import os
import numpy as np
import argparse
from eval import EVAL_DIR

AVG_DIR = "./average_results"

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

def write_csv(filename, data):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

DATASETS = [
    'caltech101',
    'imagenet',
    'dtd',
    'eurosat',
    'fgvc_aircraft',
    'food101',
    'oxford_flowers',
    'oxford_pets',
    'stanford_cars',
    'sun397',
    'ucf101',
]
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the csv file to be averaged",
    )
    args = parser.parse_args()
    
    os.makedirs(AVG_DIR, exist_ok=True)

    rows = []
    data = read_csv(f'{EVAL_DIR}/{args.name}.csv')
    header = data[0]
    rows += data[1:]
    DATASET_INDEX = header.index('dataset')
    SHOTS_INDEX = header.index('shots')
    VAL_ACC_INDEX = header.index('val_acc_mean')
    TEST_ACC_INDEX = header.index('test_acc_mean')
    TEST_STD_INDEX = header.index('test_acc_std')
    def find_rows(rows, shot_num):
        subset = []
        test_accs = []
        test_stds = []
        for dataset in DATASETS:
            dataset_candidates = []
            for row in rows:
                if row[DATASET_INDEX] == dataset \
                    and int(row[SHOTS_INDEX]) == shot_num:
                    dataset_candidates.append(row)
                    
            if len(dataset_candidates) == 0:
                import pdb; pdb.set_trace()
                raise ValueError('No candidates found for dataset: {}'.format(dataset))
            else:
                assert len(dataset_candidates) == 1, 'Multiple candidates found for dataset: {}'.format(dataset)
                max_val_acc = max([float(row[VAL_ACC_INDEX]) for row in dataset_candidates])
                max_val_acc_candidates = [row for row in dataset_candidates if float(row[VAL_ACC_INDEX]) >= max_val_acc]
                max_test_acc = max([float(row[TEST_ACC_INDEX]) for row in max_val_acc_candidates])
                test_accs.append(max_test_acc)
                if len(max_val_acc_candidates) > 1:
                    max_test_acc_candidates = [row for row in max_val_acc_candidates if float(row[TEST_ACC_INDEX]) == max_test_acc]
                    if len(max_test_acc_candidates) > 1:
                        print('Multiple candidates {} found for dataset: {}'.format(len(max_test_acc_candidates), dataset))
                    subset.append(max_test_acc_candidates[0])
                    test_stds.append(float(max_test_acc_candidates[0][TEST_STD_INDEX]))
                else:
                    subset.append(max_val_acc_candidates[0])
                    test_stds.append(float(max_val_acc_candidates[0][TEST_STD_INDEX]))
        return subset, np.mean(np.array(test_accs)), np.mean(np.array(test_stds))

    save_name = args.name
    save_dir = f"{AVG_DIR}/{save_name}"
    os.makedirs(save_dir, exist_ok=True)
    for shot_num in [1, 2, 4, 8, 16]:
        all_rows_to_write = []
        subset, mean_test_acc, mean_test_std = find_rows(rows, shot_num)
        all_rows_to_write.extend(subset)
        all_rows_to_write.append(['Mean Test Acc/Std', mean_test_acc, mean_test_std])
        all_rows_to_write.append([])
        all_rows_to_write.append([])

        write_csv(f'{save_dir}/shot_{shot_num}.csv', [header] + all_rows_to_write)
    print(f"Saved to {save_dir}")
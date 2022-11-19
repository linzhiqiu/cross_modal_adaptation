# This script is modifed from https://github.com/AndreyGuzhov/AudioCLIP/blob/master/demo/AudioCLIP.ipynb

# Modify this PATH variable if you install ESC-50 somewhere else
PATH = '../data/esc-50/'

import sys
print(sys.version)
import os

import librosa
import numpy as np
from tqdm import tqdm

import torch

# sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

import csv
from model import AudioCLIP
from utils.transforms import ToTensor1D

torch.set_grad_enabled(False)

MODEL_FILENAME = 'AudioCLIP-Partial-Training.pt'
# derived from ESResNeXt
SAMPLE_RATE = 44100

aclp = AudioCLIP(pretrained=f'{PATH}/{MODEL_FILENAME}')
audio_transforms = ToTensor1D()


ESC_path = f"{PATH}/ESC-50-master"


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

metadata = read_csv(f'{ESC_path}/meta/esc50.csv')

save_path = f'{PATH}/features.pt'
FILE_PATH_INDEX = 0
FOLD_INDEX = 1
CLASS_INDEX = 2
CLASSNAME_INDEX = 3
ID_INDEX = 5

if not os.path.exists(save_path):
    track_list = {
        i: list()
        for i in range(5)
    }
    meta = {
        i: list()
        for i in range(5)
    }
    for idx, line in tqdm(enumerate(metadata[1:])):
        path_to_audio = f'{ESC_path}/audio/{line[FILE_PATH_INDEX]}'
        fold_index = int(line[FOLD_INDEX]) - 1
        track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
        if track.shape[0] > 220500:
            track = track[:220500]
        else:
            track = np.pad(track, (0, 220500 - track.shape[0]), 'constant')
        track_list[fold_index].append(track)
       
        meta[fold_index].append(
            {
                'class': int(line[CLASS_INDEX]),
                'classname' : line[CLASSNAME_INDEX],
                'path': line[FILE_PATH_INDEX]
            }
        )

    esc50_dict = {
        i: dict()
        for i in range(5)
    }
    batch_size = 10
    for fold_index in range(5):
        # chunk track_list into batches
        tracks_allfold = [track_list[fold_index][i:i + batch_size] for i in range(0, len(track_list[fold_index]), batch_size)]
        
        final_audio_stacked = None
        for tracks in tqdm(tracks_allfold):
            audio_stacked = torch.stack([audio_transforms(track.reshape(1, -1)) for track in tracks])
            ((audio_features_stacked, _, _), _), _ = aclp(audio=audio_stacked)
            audio_features_stacked = audio_features_stacked / torch.linalg.norm(audio_features_stacked, dim=-1, keepdim=True)

            if final_audio_stacked is None:
                final_audio_stacked = audio_features_stacked
            else:
                final_audio_stacked = torch.cat((final_audio_stacked, audio_features_stacked), dim=0)
        assert len(meta[fold_index]) == final_audio_stacked.shape[0]
        esc50_dict[fold_index]['features'] = final_audio_stacked
        esc50_dict[fold_index]['labels'] = torch.Tensor([d['class'] for d in meta[fold_index]]).long()
        esc50_dict[fold_index]['labelnames'] = [d['classname'] for d in meta[fold_index]]
        esc50_dict[fold_index]['paths'] = [d['path'] for d in meta[fold_index]]
    torch.save(esc50_dict, save_path)
else:
    esc50_dict = torch.load(save_path)

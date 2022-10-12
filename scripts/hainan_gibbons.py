from collections import defaultdict
import json
from random import sample
import sys

import pandas as pd
from pathlib import Path
from plumbum import local, FG

from beans.utils import divide_waveform_to_chunks, divide_annotation_to_chunks, get_wav_length_in_secs

CHUNK_SIZE = 60     # in seconds
TARGET_SAMPLE_RATE = 9600

local['mkdir']['-p', 'data/hainan_gibbons/wav']()
local['mkdir']['-p', 'data/hainan_gibbons/Train']()
local['wget']['https://zenodo.org/record/3991714/files/Train.zip?download=1', '-O', 'data/hainan_gibbons/Train.zip'] & FG
local['unzip']['data/hainan_gibbons/Train.zip', '-d', 'data/hainan_gibbons/Train/'] & FG
local['wget']['https://zenodo.org/record/3991714/files/Train_Labels.zip?download=1', '-O', 'data/hainan_gibbons/Train_Labels.zip'] & FG
local['unzip']['data/hainan_gibbons/Train_Labels.zip', '-d', 'data/hainan_gibbons/'] & FG


def get_split(file_id):
    # 1:5:3:3
    if file_id == 0:
        return 'train-low'
    elif 1 <= file_id <= 5:
        return 'train'
    elif 6 <= file_id <= 8:
        return 'valid'
    else:
        return 'test'

datasets = defaultdict(list)

for file_id, wav_path in enumerate(sorted(Path('data/hainan_gibbons/').glob('Train/*.wav'))):
    print(f'Converting {wav_path} ...', file=sys.stderr)

    target_paths = divide_waveform_to_chunks(
        path=wav_path,
        target_dir='data/hainan_gibbons/wav',
        chunk_size=CHUNK_SIZE,
        target_sample_rate=TARGET_SAMPLE_RATE
    )

    df = pd.read_csv(str(wav_path.parent.parent / 'Train_Labels' / ('g_'+wav_path.stem)) + '.data')
    annotations = []
    for _, row in df.iterrows():
        st, ed, type = row['Start'], row['End'], row['Type']
        annotations.append({'st': st, 'ed': ed, 'label': type})

    chunks = divide_annotation_to_chunks(
        annotations=annotations,
        chunk_size=CHUNK_SIZE)

    split = get_split(file_id)

    for chunk, path in enumerate(target_paths):
        if chunk % 3 != 0:
            continue

        datasets[split].append({
            'path': path,
            'length': get_wav_length_in_secs(path),
            'annotations': chunks[chunk]
        })


for split in ['train', 'train-low', 'valid', 'test']:
    with open(f'data/hainan_gibbons/{split}.jsonl', mode='w') as f:
        if split == 'train':    # 'train' = 'train' + 'train-low'
            for data in datasets['train-low']:
                print(json.dumps(data), file=f)
        for data in datasets[split]:
            print(json.dumps(data), file=f)

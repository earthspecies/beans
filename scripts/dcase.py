from collections import defaultdict
import json
import sys
from pathlib import Path

import pandas as pd
from plumbum import local, FG

from beans.utils import divide_waveform_to_chunks, divide_annotation_to_chunks, get_wav_length_in_secs

CHUNK_SIZE = 60     # in seconds
TARGET_SAMPLE_RATE = 16_000

def get_split(chunk_id, total_num_chunks):
    if chunk_id / total_num_chunks < .12:
        return 'train-low'
    elif chunk_id / total_num_chunks < .6:
        return 'train'
    elif chunk_id / total_num_chunks < .8:
        return 'valid'
    else:
        return 'test'

local['mkdir']['-p', 'data/dcase/wav']()
local['wget']['https://zenodo.org/record/5412896/files/Development_Set.zip?download=1', '-O', 'data/dcase/Development_set.zip'] & FG
local['unzip']['data/dcase/Development_set.zip', '-d', 'data/dcase/'] & FG

datasets = defaultdict(list)

for wav_path in sorted(Path('data/dcase/Development_Set/').glob('**/*.wav')):
    csv_path = wav_path.parent / (wav_path.stem + '.csv')
    print(f'Converting {wav_path} and {csv_path} ...', file=sys.stderr)

    target_paths = divide_waveform_to_chunks(
        path=wav_path,
        target_dir='data/dcase/wav/',
        chunk_size=CHUNK_SIZE,
        target_sample_rate=TARGET_SAMPLE_RATE)
    print(f'num_chunks = {len(target_paths)}', file=sys.stderr)

    df = pd.read_csv(csv_path)
    annotations = []
    for _, row in df.iterrows():
        st, ed = row['Starttime'], row['Endtime']

        for species, label in row.iloc[3:].items():
            if label == 'POS':
                if species in {'AGGM', 'SOCM'}:
                    # these species have very few annotations and will result in zero samples in either train or test sets after split
                    continue
                annotations.append({'st': st, 'ed': ed, 'label': species})

    chunks = divide_annotation_to_chunks(
        annotations=annotations,
        chunk_size=CHUNK_SIZE)

    for chunk, path in enumerate(target_paths):
        split = get_split(chunk, len(target_paths))
        datasets[split].append({
            'path': path,
            'length': get_wav_length_in_secs(path),
            'annotations': chunks[chunk],
        })

for split in ['train', 'train-low', 'valid', 'test']:
    with open(f'data/dcase/{split}.jsonl', mode='w') as f:
        if split == 'train':    # 'train' = 'train' + 'train-low'
            for data in datasets['train-low']:
                print(json.dumps(data), file=f)
        for data in datasets[split]:
            print(json.dumps(data), file=f)


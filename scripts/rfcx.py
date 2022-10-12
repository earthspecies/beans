import json
import os
import sys

import pathlib
from plumbum import local, FG
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

from beans.utils import get_wav_length_in_secs

dataset = {}

local['mkdir']['-p', 'data/rfcx/wav']()
local['kaggle']['competitions', 'download', '-p', 'data/rfcx', 'rfcx-species-audio-detection'] & FG
local['unzip']['data/rfcx/rfcx-species-audio-detection.zip', '-d', 'data/rfcx/'] & FG

dest_dir = pathlib.Path('data/rfcx/wav')
for src_file in sorted(pathlib.Path('data/rfcx/train').glob('*.flac')):
    dest_file = dest_dir / (src_file.stem + '.wav')
    if not os.path.exists(dest_file):
        print(f"Converting {src_file} ...", file=sys.stderr)
        subprocess.run(
            ['sox', src_file, '-r 48000', '-R', dest_file]
        )
    dataset[src_file.stem] = {
        'path': str(dest_file),
        'length': get_wav_length_in_secs(dest_file),
        'annotations': []}


df = pd.read_csv('data/rfcx/train_tp.csv')
for _, row in df.iterrows():
    dataset[row['recording_id']]['annotations'].append(
        {'st': row['t_min'], 'ed': row['t_max'], 'label': row['species_id']})

# split to train:valid:test = 6:2:2
dataset = list(dataset.values())
df_train, df_valid_test = train_test_split(dataset, test_size=0.4, random_state=42, shuffle=True)
df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=42, shuffle=True)
df_train_low, _ = train_test_split(df_train, test_size=0.66, random_state=42, shuffle=True)

df_train.sort(key=lambda x: x['path'])
df_train_low.sort(key=lambda x: x['path'])
df_valid.sort(key=lambda x: x['path'])
df_test.sort(key=lambda x: x['path'])

with open('data/rfcx/train.jsonl', mode='w') as f:
    for data in df_train:
        print(json.dumps(data), file=f)

with open('data/rfcx/train-low.jsonl', mode='w') as f:
    for data in df_train_low:
        print(json.dumps(data), file=f)

with open('data/rfcx/valid.jsonl', mode='w') as f:
    for data in df_valid:
        print(json.dumps(data), file=f)

with open('data/rfcx/test.jsonl', mode='w') as f:
    for data in df_test:
        print(json.dumps(data), file=f)

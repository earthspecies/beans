from collections import defaultdict
import json
from pathlib import Path
import pandas as pd
from plumbum import local, FG
import sys

from beans.utils import divide_waveform_to_chunks, divide_annotation_to_chunks, get_wav_length_in_secs

CHUNK_SIZE = 60     # in seconds
TARGET_SAMPLE_RATE = 32_000

local['mkdir']['-p', 'data/enabirds/wav']()
local['wget']['https://datadryad.org/stash/downloads/file_stream/641808', '-O', 'data/enabirds/wav_Files.zip'] & FG
local['unzip']['data/enabirds/wav_Files.zip', '-d', 'data/enabirds/'] & FG

local['wget']['https://datadryad.org/stash/downloads/file_stream/641805', '-O', 'data/enabirds/annotation_Files.zip'] & FG
local['unzip']['data/enabirds/annotation_Files.zip', '-d', 'data/enabirds/'] & FG

def get_split(chunk_id, total_num_chunks):
    if chunk_id / total_num_chunks < .12:
        return 'train-low'
    elif chunk_id / total_num_chunks < .6:
        return 'train'
    elif chunk_id / total_num_chunks < .8:
        return 'valid'
    else:
        return 'test'

datasets = defaultdict(list)

for wav_path in sorted(Path('data/enabirds/').glob('Recording_?/*.wav')):
    print(f'Converting {wav_path} ...', file=sys.stderr)

    target_paths = divide_waveform_to_chunks(
        path=wav_path,
        target_dir='data/enabirds/wav',
        chunk_size=CHUNK_SIZE,
        target_sample_rate=TARGET_SAMPLE_RATE
    )

    df = pd.read_csv(str(wav_path.parent / wav_path.stem) + '.Table.1.selections.txt', sep='\t')

    annotations = []
    for _, row in df.iterrows():
        st, ed = row['Begin Time (s)'], row['End Time (s)']
        annotations.append({'st': st, 'ed': ed, 'label': row['Species']})

    chunks = divide_annotation_to_chunks(
        annotations=annotations,
        chunk_size=CHUNK_SIZE)

    for chunk, path in enumerate(target_paths):
        split = get_split(chunk, len(target_paths))
        datasets[split].append({
            'path': path,
            'length': get_wav_length_in_secs(path),
            'annotations': chunks[chunk]
        })

for split in ['train', 'train-low', 'valid', 'test']:
    with open(f'data/enabirds/{split}.jsonl', mode='w') as f:
        if split == 'train':    # 'train' = 'train' + 'train-low'
            for data in datasets['train-low']:
                print(json.dumps(data), file=f)
        for data in datasets[split]:
            print(json.dumps(data), file=f)

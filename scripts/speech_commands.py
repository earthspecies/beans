from pathlib import Path
import pandas as pd
from plumbum import local, FG

local['mkdir']['-p', 'data/speech_commands']()
local['wget']['http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', '-O', 'data/speech_commands/speech_commands_v0.02.tar.gz'] & FG
local['tar']['zxvf', 'data/speech_commands/speech_commands_v0.02.tar.gz', '-C', 'data/speech_commands/'] & FG

all_list = set()
for fn in Path('data/speech_commands').glob('*/*.wav'):
    if '_background_noise_' in str(fn):
        continue
    fn = '/'.join(str(fn).split('/')[-2:])
    all_list.add(fn)


testing_list = set()
with open('data/speech_commands/testing_list.txt') as f:
    for line in f:
        testing_list.add(line.strip())

validation_list = set()
with open('data/speech_commands/validation_list.txt') as f:
    for line in f:
        validation_list.add(line.strip())

training_list = all_list - validation_list - testing_list
training_list_low = [fn for i, fn in enumerate(sorted(training_list)) if i % 100 == 0]

def _to_df(lst):
    data = []
    for fn in lst:
        label = fn.split('/')[0]
        path = f"data/speech_commands/{fn}"
        data.append({'path': path, 'label': label})

    return pd.DataFrame.from_dict(data)

_to_df(sorted(training_list)).to_csv('data/speech_commands/train.csv')
_to_df(sorted(training_list_low)).to_csv('data/speech_commands/train-low.csv')
_to_df(sorted(validation_list)).to_csv('data/speech_commands/valid.csv')
_to_df(sorted(testing_list)).to_csv('data/speech_commands/test.csv')

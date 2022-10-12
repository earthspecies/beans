import pandas as pd
from plumbum import local

target_dir = 'data/esc50'

git = local['git']
git['clone', 'https://github.com/karolpiczak/ESC-50.git', target_dir]()

df = pd.read_csv(f'{target_dir}/meta/esc50.csv')

def convert(row):
    new_row = pd.Series({
        'path': f"data/esc50/audio/{row['filename']}",
        'label': row['target'],
        'fold': row['fold']
    })

    return new_row

df = df.apply(convert, axis=1)

def _get_fold(row):
    return int(row['fold'])
    # return int(row['filename'][0])

df_train = df[df.apply(lambda r: _get_fold(r) <= 3, axis=1)]
df_train_low = df[df.apply(lambda r: _get_fold(r) == 1, axis=1)]
df_valid = df[df.apply(lambda r: _get_fold(r) == 4, axis=1)]
df_test = df[df.apply(lambda r: _get_fold(r) == 5, axis=1)]

df_train.to_csv(f'{target_dir}/meta/esc50.train.csv')
df_train_low.to_csv(f'{target_dir}/meta/esc50.train-low.csv')
df_valid.to_csv(f'{target_dir}/meta/esc50.valid.csv')
df_test.to_csv(f'{target_dir}/meta/esc50.test.csv')

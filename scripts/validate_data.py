from beans.utils import check_md5


with open('data/file_hashes') as f:
    for line in f:
        path, md5 = line.strip().split('\t')
        print(f'Validating {path} ...')
        check_md5(path, md5)

print('Validation succeeded!')

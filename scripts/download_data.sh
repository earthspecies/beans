#! /bin/env bash

mkdir -p data
python scripts/watkins.py 
python scripts/bats.py
python scripts/cbi.py
python scripts/humbugdb.py
python scripts/dogs.py
python scripts/dcase.py
python scripts/enabirds.py
mkdir data/hiceas
wget https://storage.googleapis.com/ml-bioacoustics-datasets/hiceas_1-20_minke-detection.zip -O data/hiceas/hiceas.zip
unzip data/hiceas/hiceas.zip -d data/hiceas
python scripts/rfcx.py
python scripts/hainan_gibbons.py
python scripts/esc50.py
python scripts/speech_commands.py
python scripts/validate_data.py

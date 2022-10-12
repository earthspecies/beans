import torch
import torchaudio
import pytest
from beans.utils import divide_waveform_to_chunks, divide_annotation_to_chunks

def test_divide_waveform_to_chunks(tmp_path):
    sr = 16000
    # create a 5-second wave file that has a monotonically increasing waveform
    waveform = torch.arange(0, 1, 1/(5*sr)).unsqueeze(0)
    torchaudio.save(tmp_path / 'orig.wav', waveform, sr)

    # divide it into 1-second chunks
    divide_waveform_to_chunks(tmp_path / 'orig.wav', tmp_path, 1, sr)

    for i in [0, 1, 2, 3, 4]:
        chunk, sr_chunk = torchaudio.load(tmp_path / f'orig.00{i}.wav')
        assert sr_chunk == sr
        assert chunk.shape == (1, sr)
        assert chunk[0, 0] == pytest.approx(0.2 * i)
        assert chunk[0, -1] == pytest.approx(0.2 * i + 0.2 - 1/(5*sr))


def test_divide_annotation_to_chunks():
    annotations = [
        {'st': 0.1, 'ed': 0.2},
        {'st': 0.8, 'ed': 1.2},
        {'st': 2.8, 'ed': 4.2}
    ]

    chunks = divide_annotation_to_chunks(annotations, chunk_size=1)
    assert chunks[0] == [
        {'st': pytest.approx(0.1), 'ed': pytest.approx(0.2)},
        {'st': pytest.approx(0.8), 'ed': pytest.approx(1)}
    ]
    assert chunks[1] == [
        {'st': pytest.approx(0.0), 'ed': pytest.approx(0.2)}
    ]
    assert chunks[2] == [
        {'st': pytest.approx(0.8), 'ed': pytest.approx(1.0)}
    ]
    assert chunks[3] == [
        {'st': pytest.approx(0.0), 'ed': pytest.approx(1.0)}
    ]
    assert chunks[4] == [
        {'st': pytest.approx(0.0), 'ed': pytest.approx(0.2)}
    ]

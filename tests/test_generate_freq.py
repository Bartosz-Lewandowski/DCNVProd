import numpy as np
import pytest

from src.cnv_generator import CNVGenerator
from src.utils import BedFormat


@pytest.fixture
def cnv_generator(sample_fasta_file):
    cnv_gen = CNVGenerator(50, 75, 10, 0.7, sample_fasta_file)
    return cnv_gen


@pytest.fixture
def dup_dele_chr_info_sample():
    dup = [
        BedFormat(chr="1", start=401, end=405, cnv_type="normal", freq=1),
        BedFormat(chr="1", start=101, end=106, cnv_type="normal", freq=1),
        BedFormat(chr="1", start=451, end=457, cnv_type="normal", freq=1),
        BedFormat(chr="1", start=351, end=361, cnv_type="normal", freq=1),
    ]
    dele = [
        BedFormat(chr="1", start=550, end=554, cnv_type="normal", freq=1),
        BedFormat(chr="1", start=20, end=25, cnv_type="normal", freq=1),
        BedFormat(chr="1", start=200, end=206, cnv_type="normal", freq=1),
        BedFormat(chr="1", start=300, end=310, cnv_type="normal", freq=1),
    ]
    chr_info = [BedFormat(chr="1", start=1, end=600, cnv_type="normal", freq=1)]

    return dup, dele, chr_info


def test_create_dele_freqs(cnv_generator, dup_dele_chr_info_sample, mocker):
    mocker.patch.object(
        cnv_generator,
        "_create_bed_with_coords",
        return_value=dup_dele_chr_info_sample,
    )
    dup, dele, chr_info = cnv_generator._clean_bed_dup_del()
    result = cnv_generator._create_dele_freqs(dele)
    assert isinstance(result, np.ndarray)
    assert all(item.freq == 0 for item in result)
    assert all(item.cnv_type == "del" for item in result)
    assert all(result[i].start <= result[i + 1].start for i in range(len(result) - 1))
    assert all(result[i].end <= result[i + 1].end for i in range(len(result) - 1))
    assert len(result) == 4


def test_create_dup_freqs(cnv_generator, dup_dele_chr_info_sample, mocker):
    mocker.patch.object(
        cnv_generator,
        "_create_bed_with_coords",
        return_value=dup_dele_chr_info_sample,
    )
    dup, dele, chr_info = cnv_generator._clean_bed_dup_del()
    result = cnv_generator._create_dup_freqs(dup)
    assert isinstance(result, np.ndarray)
    assert all(item.freq > 1 and item.freq <= 10 for item in result)
    assert all(item.cnv_type == "dup" for item in result)
    assert all(result[i].start <= result[i + 1].start for i in range(len(result) - 1))
    assert all(result[i].end <= result[i + 1].end for i in range(len(result) - 1))
    assert len(result) == 4

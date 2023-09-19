import numpy as np
import pytest

from src.cnv_generator import CNVGenerator
from src.utils import BedFormat


@pytest.fixture
def cnv_generator(sample_fasta_file):
    cnv_gen = CNVGenerator(sample_fasta_file, 50, 75, 10, 0.7)
    return cnv_gen


@pytest.fixture
def chr_info():
    return [BedFormat(chr="1", start=1, end=600, cnv_type="normal", freq=1)]


# Test bedfile with normal sequence when no CNVs are present
def test_bedfile_with_normal_seq_no_CNV(cnv_generator, chr_info):
    dup = []
    dele = []
    result = cnv_generator._bedfile_with_normal_seq(dup, dele, chr_info)
    assert len(result) == 1
    assert isinstance(result, np.ndarray)
    assert result == [BedFormat(chr="1", start=1, end=600, cnv_type="normal", freq=1)]


# Test bedfile with normal sequence when CNVs are present
def test_bedfile_with_normal_seq_CNV(cnv_generator, chr_info):
    dup = [
        BedFormat(chr="1", start=101, end=201, cnv_type="dup", freq=2),
        BedFormat(chr="1", start=301, end=401, cnv_type="dup", freq=2),
    ]
    dele = [
        BedFormat(chr="1", start=0, end=100, cnv_type="del", freq=0),
        BedFormat(chr="1", start=501, end=551, cnv_type="del", freq=0),
    ]
    result = cnv_generator._bedfile_with_normal_seq(dup, dele, chr_info)
    assert len(result) == 3
    assert result[0] == BedFormat(
        chr="1", start=201, end=301, cnv_type="normal", freq=1
    )
    assert result[1] == BedFormat(
        chr="1", start=401, end=501, cnv_type="normal", freq=1
    )
    assert result[2] == BedFormat(
        chr="1", start=551, end=600, cnv_type="normal", freq=1
    )


# Test bedfile with normal sequence when CNVs are present and no space between them
def test_bedfile_with_normal_seq_CNV_no_space(cnv_generator, chr_info):
    dup = [
        BedFormat(chr="1", start=101, end=201, cnv_type="dup", freq=2),
        BedFormat(chr="1", start=201, end=301, cnv_type="dup", freq=2),
        BedFormat(chr="1", start=301, end=401, cnv_type="dup", freq=2),
    ]
    dele = [
        BedFormat(chr="1", start=0, end=100, cnv_type="del", freq=0),
        BedFormat(chr="1", start=401, end=551, cnv_type="del", freq=0),
        BedFormat(chr="1", start=551, end=600, cnv_type="del", freq=0),
    ]
    result = cnv_generator._bedfile_with_normal_seq(dup, dele, chr_info)
    assert len(result) == 0

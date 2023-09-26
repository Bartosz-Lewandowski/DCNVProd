import numpy as np
import pytest
from pybedtools import BedTool

from src.cnv_generator import CNVGenerator
from src.utils import BedFormat


@pytest.fixture
def cnv_generator(sample_fasta_file):
    cnv_gen = CNVGenerator(50, 75, 10, 0.7, sample_fasta_file)
    return cnv_gen


@pytest.fixture
def dup_del_normal():
    dup = [
        BedFormat(chr="1", start=101, end=201, cnv_type="dup", freq=2),
        BedFormat(chr="1", start=301, end=401, cnv_type="dup", freq=2),
    ]
    dele = [
        BedFormat(chr="1", start=0, end=100, cnv_type="del", freq=0),
        BedFormat(chr="1", start=501, end=551, cnv_type="del", freq=0),
    ]
    normal = [
        BedFormat(chr="1", start=201, end=301, cnv_type="normal", freq=1),
        BedFormat(chr="1", start=401, end=501, cnv_type="normal", freq=1),
        BedFormat(chr="1", start=551, end=600, cnv_type="normal", freq=1),
    ]

    return dup, dele, normal


@pytest.fixture
def total_files(cnv_generator, dup_del_normal):
    dup, dele, normal = dup_del_normal
    total_bedformat, total_bedtools = cnv_generator._create_total_bed(dup, dele, normal)
    return total_bedformat, total_bedtools


def test_total_bedformat_instance(total_files):
    total_bedformat, total_bedtools = total_files
    assert isinstance(total_bedformat, np.ndarray)
    assert all(isinstance(item, BedFormat) for item in total_bedformat)


def test_total_bedtools_instance(total_files):
    total_bedformat, total_bedtools = total_files
    assert isinstance(total_bedtools, BedTool)


def test_total_bedformat_sorted(total_files):
    total_bedformat, total_bedtools = total_files
    assert all(
        total_bedformat[i].start <= total_bedformat[i + 1].start
        for i in range(len(total_bedformat) - 1)
    )
    assert all(
        total_bedformat[i].end <= total_bedformat[i + 1].end
        for i in range(len(total_bedformat) - 1)
    )


def test_total_bedformat_right_concatenate(total_files):
    total_bedformat, total_bedtools = total_files
    assert len([item for item in total_bedformat if item.cnv_type == "dup"]) == 2
    assert len([item for item in total_bedformat if item.cnv_type == "del"]) == 2
    assert len([item for item in total_bedformat if item.cnv_type == "normal"]) == 3
    assert all(item.freq >= 0 for item in total_bedformat)


def test_bedtools_is_equal_bedformat(total_files):
    total_bedformat, total_bedtools = total_files
    assert len(total_bedformat) == len(total_bedtools)
    for x, y in zip(total_bedformat, total_bedtools):
        assert x.chr == y.chrom
        assert x.start == y.start
        assert x.end == y.end
        assert x.cnv_type == y.name
        assert str(x.freq) == y.score

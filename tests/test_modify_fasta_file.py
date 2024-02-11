import os
import tempfile

import pytest
from Bio import SeqIO

from src.cnv_generator import CNVGenerator
from src.paths import MODIFIED_FASTA_FILE_NAME
from src.utils import BedFormat


@pytest.fixture
def temp_dir():
    return tempfile.TemporaryDirectory()


@pytest.fixture
def cnv_generator(sample_fasta_file, temp_dir):
    cnv_gen = CNVGenerator(50, 75, 10, 0.7, sample_fasta_file, temp_dir.name)
    return cnv_gen


@pytest.fixture
def dup_del_normal():
    dup = [
        BedFormat(chr="1", start=101, end=201, cnv_type="dup", freq=2),
        BedFormat(chr="1", start=301, end=401, cnv_type="dup", freq=4),
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


@pytest.fixture
def seqs_and_freqs(sample_fasta_file, dup_del_normal):
    fasta = SeqIO.parse(open(sample_fasta_file), "fasta")
    seq = str(next(fasta).seq)
    seq_and_freq = []
    for x in dup_del_normal:
        for z in x:
            seq_and_freq.append((seq[z.start : z.end], z.freq))
    return seq_and_freq


# Test if file exits
def test_modify_fasta_file_exists(total_files, cnv_generator, temp_dir):
    cnv_generator.modify_fasta_file(total_files[0])
    assert os.path.exists(os.path.join(temp_dir.name, MODIFIED_FASTA_FILE_NAME))


# Test if file is modified correctly
def test_modify_fasta_file_correct(
    total_files, cnv_generator, temp_dir, seqs_and_freqs
):
    cnv_generator.modify_fasta_file(total_files[0])
    with open(os.path.join(temp_dir.name, MODIFIED_FASTA_FILE_NAME)) as f:
        modified = f.read()
        for x in seqs_and_freqs:
            assert modified.count(x[0]) == x[1]

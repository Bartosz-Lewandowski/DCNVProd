import os
import tempfile

import pytest

from src.sim_reads import SimReads


@pytest.mark.parametrize(
    "chr_len, cov, model, expected_result",
    [
        (1000, 30, "novaseq", 200),
        (2000, 20, "hiseq", 320),
        (500, 10, "miseq", 17),
    ],
)
def test_calc_N_reads(chr_len, cov, model, expected_result):
    simreads = SimReads(cov, 10, model)
    # Test with various chromosome lengths and expected results
    assert simreads._SimReads__calc_N_reads(chr_len) == expected_result


@pytest.mark.parametrize(
    "model, expected_result",
    [
        ("novaseq", 150),
        ("hiseq", 125),
        ("miseq", 300),
    ],
)
def test_get_read_length(model, expected_result):
    simreads = SimReads(10, 10, model)
    # Test with various models and expected results
    assert simreads._SimReads__get_read_length() == expected_result


def test_get_read_length_ValueError():
    # Test with wrong model
    with pytest.raises(ValueError):
        SimReads(10, 10, "test")


def test_sim_reads(sample_fasta_file):
    # Test if sim_reads_genome() function creates two fastq files
    with tempfile.TemporaryDirectory() as temp_dir:
        simreads = SimReads(10, 10, "novaseq", sample_fasta_file, temp_dir)
        simreads.sim_reads_genome()
        assert len(os.listdir(temp_dir)) == 3
        assert os.path.exists(os.path.join(temp_dir, "10_R1.fastq"))
        assert os.path.exists(os.path.join(temp_dir, "10_R2.fastq"))
        assert os.path.exists(os.path.join(temp_dir, "10_abundance.txt"))

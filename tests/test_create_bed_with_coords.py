import random

import numpy as np
import pytest
from Bio import SeqIO

from src.cnv_generator import (
    CNVGenerator,  # Replace 'your_module' and 'YourClass' with actual module and class names
)
from src.utils import BedFormat  # Import the BedFormat class


@pytest.fixture
def chr_data(sample_fasta_file):
    fasta = SeqIO.parse(open(sample_fasta_file), "fasta")
    chr = next(fasta)
    return chr


@pytest.fixture
def cnv_generator(sample_fasta_file):
    cnv_gen = CNVGenerator(sample_fasta_file, 50)
    return cnv_gen


def test_create_bed_with_coords(cnv_generator, mocker):
    mocker.patch.object(
        cnv_generator,
        "_CNVGenerator__generate_coords",
        return_value=[
            BedFormat(chr="1", start=401, end=405, cnv_type="normal", freq=1),
            BedFormat(chr="1", start=101, end=106, cnv_type="normal", freq=1),
            BedFormat(chr="1", start=451, end=457, cnv_type="normal", freq=1),
            BedFormat(chr="1", start=351, end=361, cnv_type="normal", freq=1),
        ],
    )

    mocker.patch.object(
        cnv_generator,
        "_CNVGenerator__generate_coords",
        return_value=[
            BedFormat(chr="1", start=550, end=554, cnv_type="normal", freq=1),
            BedFormat(chr="1", start=20, end=25, cnv_type="normal", freq=1),
            BedFormat(chr="1", start=200, end=206, cnv_type="normal", freq=1),
            BedFormat(chr="1", start=300, end=310, cnv_type="normal", freq=1),
        ],
    )
    mocker.patch.object(
        cnv_generator,
        "_CNVGenerator__chromosome_info",
        return_value=[BedFormat(chr="1", start=1, end=600, cnv_type="normal", freq=1)],
    )

    result = cnv_generator._create_bed_with_coords()

    assert all(isinstance(item, np.ndarray) for item in result)
    assert len(result) == 3
    assert len(result[0]) == len(result[1]) == 4


def test_generate_coords(cnv_generator, mocker, chr_data):
    mocker.patch.object(
        cnv_generator,
        "_CNVGenerator__find_CNV_windows",
        return_value=([401, 101, 451, 351], [405, 106, 457, 361]),
    )
    mocker.patch.object(
        cnv_generator, "_CNVGenerator__max_cnv_lenght_too_large", return_value=False
    )
    mocker.patch.object(
        cnv_generator, "_CNVGenerator__too_large_cnvs_number", return_value=False
    )
    mocker.patch.object(random, "randrange", side_effect=[4, 5, 6, 10])

    result = cnv_generator._CNVGenerator__generate_coords(chr_data, 75)
    assert isinstance(result, np.ndarray)
    assert len(result) == 4
    assert all(isinstance(item, BedFormat) for item in result)


def test_find_CNV_windows(cnv_generator, mocker, chr_data):
    # Mock the random.randrange method to return a fixed start value
    mocker.patch.object(random, "randrange", side_effect=[1, 6, 12])
    lengths = [4, 5, 6]

    starts, ends = cnv_generator._CNVGenerator__find_CNV_windows(
        chr_data, lengths, min_cnv_gap=1
    )
    assert len(starts) == len(ends)
    assert starts == [1, 6, 12]  # Mocked start value is used for all CNVs
    assert ends == [5, 11, 18]  # Stop coordinates are calculated based on lengths


def test_find_CNV_windows_with_overlap(cnv_generator, mocker, chr_data):
    # Mock the __check_overlapping method to return True (indicating overlap)
    mocker.patch.object(random, "randrange", return_value=1)
    mocker.patch.object(
        cnv_generator, "_CNVGenerator__check_overlaping", return_value=False
    )

    lengths = [4]
    with pytest.raises(RuntimeError):
        starts, ends = cnv_generator._CNVGenerator__find_CNV_windows(chr_data, lengths)


def test_find_CNV_windows_with_no_occur(cnv_generator, mocker, chr_data):
    # Mock the __check_occur method to return False (indicating no occurrence)
    mocker.patch.object(random, "randrange", return_value=1)
    mocker.patch.object(cnv_generator, "_CNVGenerator__check_occur", return_value=False)

    lengths = [4]
    with pytest.raises(RuntimeError):
        starts, ends = cnv_generator._CNVGenerator__find_CNV_windows(chr_data, lengths)


@pytest.mark.parametrize(
    "starts, ends, start, stop, min_cnv_gap, expected",
    [
        ([100, 200, 300], [150, 250, 350], 399, 450, 50, False),
        ([100, 200, 300], [150, 250, 350], 400, 450, 50, True),
        ([100, 200, 400], [150, 250, 450], 0, 51, 50, False),
        ([100, 200], [150, 250], 0, 50, 50, True),
        ([], [], 100, 200, 50, True),
        ([100], [200], 150, 250, 50, False),
        ([100, 400], [300, 600], 200, 300, 50, False),
        ([200, 300, 400], [250, 350, 450], 100, 150, 50, True),
        ([100], [200], 50, 150, 50, False),
        ([100], [200], 50, 120, 50, False),
        ([100], [200], 50, 90, 10, True),
    ],
)
def test_check_overlapping(
    cnv_generator, starts, ends, start, stop, min_cnv_gap, expected
):
    result = cnv_generator._CNVGenerator__check_overlaping(
        start, stop, starts, ends, min_cnv_gap
    )
    assert result is expected


def test_get_N_percentage(cnv_generator):
    seq = "ACTGNNNNACTGACTG"
    percentage = cnv_generator._CNVGenerator__get_N_percentage(seq)
    assert percentage == 0.25


def test_max_cnv_length_too_large(cnv_generator):
    # Test when len_fasta is greater than max_cnv_length
    result = cnv_generator._CNVGenerator__max_cnv_lenght_too_large(
        len_fasta=1000, max_cnv_length=500
    )
    assert result is False

    # Test when len_fasta is equal to max_cnv_length
    result = cnv_generator._CNVGenerator__max_cnv_lenght_too_large(
        len_fasta=500, max_cnv_length=500
    )
    assert result is True

    # Test when len_fasta is less than max_cnv_length
    result = cnv_generator._CNVGenerator__max_cnv_lenght_too_large(
        len_fasta=200, max_cnv_length=500
    )
    assert result is True

    # Test with negative values
    result = cnv_generator._CNVGenerator__max_cnv_lenght_too_large(
        len_fasta=-100, max_cnv_length=500
    )
    assert result is True

    # Test with zero values
    result = cnv_generator._CNVGenerator__max_cnv_lenght_too_large(
        len_fasta=0, max_cnv_length=500
    )
    assert result is True  # Assuming that a CNV of length 0 is allowed


def test_too_large_cnvs_number(cnv_generator):
    # Test when total CNV length plus gaps can fit in data
    lengths = [100, 200, 150]
    cnv_count = len(lengths)
    min_cnv_gap = 50
    len_fasta = sum(lengths) + (cnv_count - 1) * min_cnv_gap + 100
    result = cnv_generator._CNVGenerator__too_large_cnvs_number(
        len_fasta=len_fasta,
        lenghts=lengths,
        cnv_count=cnv_count,
        min_cnv_gap=min_cnv_gap,
    )
    assert result is False

    # Test when total CNV length plus gaps exceed data length
    lengths = [100, 200, 150]
    cnv_count = len(lengths)
    min_cnv_gap = 50
    len_fasta = sum(lengths) + (cnv_count - 1) * min_cnv_gap - 100
    result = cnv_generator._CNVGenerator__too_large_cnvs_number(
        len_fasta=len_fasta,
        lenghts=lengths,
        cnv_count=cnv_count,
        min_cnv_gap=min_cnv_gap,
    )
    assert result is True

    # Test with minimum gap equal to 0
    lengths = [100, 200, 150]
    cnv_count = len(lengths)
    min_cnv_gap = 0
    len_fasta = sum(lengths) + (cnv_count - 1) * min_cnv_gap + 100
    result = cnv_generator._CNVGenerator__too_large_cnvs_number(
        len_fasta=len_fasta,
        lenghts=lengths,
        cnv_count=cnv_count,
        min_cnv_gap=min_cnv_gap,
    )
    assert result is False

    # Test when total CNV lenght plus gaps is equal to data length
    lengths = [100, 200, 150]
    cnv_count = len(lengths)
    min_cnv_gap = 50
    len_fasta = sum(lengths) + (cnv_count - 1) * min_cnv_gap
    result = cnv_generator._CNVGenerator__too_large_cnvs_number(
        len_fasta=len_fasta,
        lenghts=lengths,
        cnv_count=cnv_count,
        min_cnv_gap=min_cnv_gap,
    )
    assert result is True


def test_chromosome_info(cnv_generator, chr_data):
    result = cnv_generator._CNVGenerator__chromosome_info(chr_data)
    assert len(result) == 1
    assert result[0].chr == "1"
    assert result[0].start == 1
    assert result[0].end == 600
    assert result[0].cnv_type == "normal"
    assert result[0].freq == 1

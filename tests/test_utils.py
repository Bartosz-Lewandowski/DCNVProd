import gzip
import os
import tempfile

import numpy as np
import pytest
from pybedtools import BedTool

from src.utils import (  # Replace with your actual module name
    BedFormat,
    BedFormat_to_BedTool,
    BedTool_to_BedFormat,
    combine_and_cleanup_reference_genome,
    download_reference_genome,
)


@pytest.fixture
def temp_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_bedtool():
    # Create a sample BedTool for testing.
    bed_string = "chr1 100 200 cnv1 1\nchr2 300 400 cnv2 2\n"
    return BedTool(bed_string, from_string=True)


def test_BedTool_to_BedFormat(sample_bedtool):
    # Test BedTool_to_BedFormat function
    bed_formats = BedTool_to_BedFormat(sample_bedtool)
    assert len(bed_formats) == 2
    assert isinstance(bed_formats[0], BedFormat)
    assert bed_formats[0].chr == "chr1"
    assert bed_formats[0].start == 100
    assert bed_formats[0].end == 200
    assert bed_formats[0].cnv_type == "cnv1"
    assert bed_formats[0].freq == 1


def test_BedTool_to_BedFormat_short_v(sample_bedtool):
    # Test BedTool_to_BedFormat with short_v=True
    bed_formats = BedTool_to_BedFormat(sample_bedtool, short_v=True)
    assert len(bed_formats) == 2
    assert isinstance(bed_formats[0], BedFormat)
    assert bed_formats[0].chr == "chr1"
    assert bed_formats[0].start == 100
    assert bed_formats[0].end == 200
    assert bed_formats[0].cnv_type == "normal"
    assert bed_formats[0].freq == 1


def test_BedFormat_to_BedTool():
    # Test BedFormat_to_BedTool function
    bed_formats = [
        BedFormat("chr1", 100, 200, "cnv1", 1),
        BedFormat("chr2", 300, 400, "cnv2", 2),
    ]
    bed_tool = BedFormat_to_BedTool(np.array(bed_formats))
    assert isinstance(bed_tool, BedTool)
    assert str(bed_tool) == "chr1\t100\t200\tcnv1\t1\nchr2\t300\t400\tcnv2\t2\n"


def test_download_first_and_last_chromosomes(temp_directory):
    chrs_to_download = ["18"]
    download_reference_genome(chrs_to_download, temp_directory)

    assert os.path.exists(temp_directory)

    # Check if the downloaded files exist
    for chr_num in chrs_to_download:
        filename = f"Sus_scrofa.Sscrofa11.1.dna.chromosome.{chr_num}.fa.gz"
        file_path = os.path.join(temp_directory, filename)
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0

    with pytest.raises(IndexError):
        download_reference_genome(["19"], temp_directory)
        download_reference_genome(["0"], temp_directory)


def test_combine_and_cleanup_reference_genome():
    # Create temporary directory and files for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and write a gzipped file
        gz_file_path_1 = os.path.join(temp_dir, "test_file1.fa.gz")
        gz_file_path_2 = os.path.join(temp_dir, "test_file2.fa.gz")
        for gz_file_path in [gz_file_path_1, gz_file_path_2]:
            with gzip.open(gz_file_path, "wb") as gz_file:
                gz_file.write(b">header\nACGT\n")

        # Call the function being tested
        output_file = os.path.join(temp_dir, "output.fa")
        combine_and_cleanup_reference_genome(temp_dir, output_file)

        # Assert the output file exists and has content
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

        # Assert content of two files were combined
        with open(output_file, "r") as output_file:
            output_file_content = output_file.read()
            assert ">header\nACGT\n>header\nACGT\n" in output_file_content

        # Assert the gzipped file was cleaned up
        assert not os.path.exists(gz_file_path_1)
        assert not os.path.exists(gz_file_path_2)

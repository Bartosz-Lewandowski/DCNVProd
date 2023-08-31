import gzip
import os
import tempfile

import pytest

from src.utils import (  # Replace with your actual module name
    combine_and_cleanup_reference_genome,
    download_reference_genome,
)


@pytest.fixture
def temp_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_download_first_and_last_chromosomes(temp_directory):
    chrs_to_download = ["1", "18"]
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

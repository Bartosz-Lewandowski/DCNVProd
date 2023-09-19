import os
import tempfile

import pandas as pd
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


def test_create_total_windows(cnv_generator, total_files, chr_info):
    total_bedformat, total_bedtools = total_files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set TARGET_DATA_PATH to the temporary directory.
        TARGET_DATA_PATH = temp_dir
        TARGET_DATA_FILE_NAME = "test_output.csv"
        cnv_generator.target_data_path = TARGET_DATA_PATH
        cnv_generator.target_data_file_name = TARGET_DATA_FILE_NAME
        # Call the function to create total windows.
        cnv_generator._create_total_windows(total_bedtools, chr_info)

        # Check if the output file exists.
        output_file_path = os.path.join(TARGET_DATA_PATH, TARGET_DATA_FILE_NAME)
        assert os.path.exists(output_file_path)


def test_create_total_windows_output(cnv_generator, total_files, chr_info):
    total_bedformat, total_bedtools = total_files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set TARGET_DATA_PATH to the temporary directory.
        TARGET_DATA_PATH = temp_dir
        TARGET_DATA_FILE_NAME = "test_output.csv"
        cnv_generator.target_data_path = TARGET_DATA_PATH
        cnv_generator.target_data_file_name = TARGET_DATA_FILE_NAME
        # Call the function to create total windows.
        cnv_generator._create_total_windows(total_bedtools, chr_info)
        data = pd.read_csv(
            os.path.join(TARGET_DATA_PATH, TARGET_DATA_FILE_NAME),
            sep="\t",
            dtype={"chr": object},
            header=None,
            names=["chr", "start", "end", "cnv_type"],
        )
        assert len(data) == chr_info[0].end / 50
        assert len(data[data.cnv_type == "dup"]) == 4
        assert len(data[data.cnv_type == "del"]) == 3
        assert len(data[data.cnv_type == "normal"]) == 5
        assert all(data[(data.start >= 101) & (data.end <= 200)].cnv_type == "dup")
        assert all(data[(data.start >= 401) & (data.end <= 500)].cnv_type == "normal")
        assert all(data[(data.start >= 501) & (data.end <= 550)].cnv_type == "del")

import os
import tempfile

import great_expectations as ge
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES_COMBINED_FILE, SIM_BAM_FILE_NAME
from src.generate_features import Stats, fastest_sum, numba_calc


@pytest.fixture
def tempdir():
    return tempfile.TemporaryDirectory()


@pytest.fixture
def gen_stats(tempdir):
    bam_file = "/".join(["./tests/test_generate_features", SIM_BAM_FILE_NAME])
    # Create the Stats object
    stats = Stats(
        10,
        bam_file,
        output_folder=tempdir.name,
        sim_data_path="./tests/test_generate_features",
    )
    return stats


@pytest.fixture
def values(gen_stats):
    # Generate the stats
    values = gen_stats.generate_stats(chrs="1", window_size=50)
    return values


@pytest.fixture()
def df(values, gen_stats, tempdir):
    gen_stats.combine_into_one_big_file(values)
    df = ge.dataset.PandasDataset(
        pd.read_csv(os.path.join(tempdir.name, FEATURES_COMBINED_FILE))
    )
    return df


# Test if columns in df are as expected
def test_expect_table_columns_to_match_ordered_list(df):
    df.expect_table_columns_to_match_ordered_list(
        [
            "chr",
            "start",
            "end",
            "overlap",
            "intq" "means" "std",
            "BAM_CMATCH",
            "BAM_CINS",
            "BAM_CDEL",
            "BAM_CSOFT_CLIP",
            "NM tag",
            "BAM_CROSS",
            "STAT_CROSS",
            "STAT_CROSS2",
            "cnv_type",
        ]
    )


def test_expect_column_values_to_be_of_type(df):
    df.expect_column_values_to_be_of_type("chr", "object")
    df.expect_column_values_to_be_of_type("start", "int64")
    df.expect_column_values_to_be_of_type("end", "int64")
    df.expect_column_values_to_be_of_type("overlap", "int64")
    df.expect_column_values_to_be_of_type("intq", "int64")
    df.expect_column_values_to_be_of_type("means", "float64")
    df.expect_column_values_to_be_of_type("std", "float64")
    df.expect_column_values_to_be_of_type("BAM_CMATCH", "int64")
    df.expect_column_values_to_be_of_type("BAM_CINS", "int64")
    df.expect_column_values_to_be_of_type("BAM_CDEL", "int64")
    df.expect_column_values_to_be_of_type("BAM_CSOFT_CLIP", "int64")
    df.expect_column_values_to_be_of_type("NM tag", "int64")
    df.expect_column_values_to_be_of_type("BAM_CROSS", "int64")
    df.expect_column_values_to_be_of_type("STAT_CROSS", "float64")
    df.expect_column_values_to_be_of_type("STAT_CROSS2", "float64")
    df.expect_column_values_to_be_of_type("cnv_type", "object")


def test_expect_column_values_to_be_in_set(df):
    df.expect_column_values_to_be_in_set("chr", ["1"])
    df.expect_column_values_to_be_in_set("cnv_type", ["dup", "del", "normal"])


# Unique combinations of features (detect data leaks!)
def test_expect_compound_columns_to_be_unique(df):
    df.expect_compound_columns_to_be_unique(["chr", "start", "end"])


# Test if dataset is not empty
def test_expect_table_row_count_to_be_between(df):
    df.expect_table_row_count_to_be_between(1, None)


def test_column_values_to_not_be_null(df):
    df.expect_column_values_to_not_be_null("chr")
    df.expect_column_values_to_not_be_null("start")
    df.expect_column_values_to_not_be_null("end")
    df.expect_column_values_to_not_be_null("overlap")
    df.expect_column_values_to_not_be_null("intq")
    df.expect_column_values_to_not_be_null("means")
    df.expect_column_values_to_not_be_null("std")
    df.expect_column_values_to_not_be_null("BAM_CMATCH")
    df.expect_column_values_to_not_be_null("BAM_CINS")
    df.expect_column_values_to_not_be_null("BAM_CDEL")
    df.expect_column_values_to_not_be_null("BAM_CSOFT_CLIP")
    df.expect_column_values_to_not_be_null("NM tag")
    df.expect_column_values_to_not_be_null("BAM_CROSS")
    df.expect_column_values_to_not_be_null("STAT_CROSS")
    df.expect_column_values_to_not_be_null("STAT_CROSS2")
    df.expect_column_values_to_not_be_null("cnv_type")


# Test if the output file is created
def test_existing_output_file(gen_stats, values, tempdir):
    gen_stats.combine_into_one_big_file(values)
    assert os.path.exists(os.path.join(tempdir.name, FEATURES_COMBINED_FILE))


# Test the numba_calc function
def test_numba_calc(cov_summed):
    result = numba_calc(cov_summed)

    # Check if the result is a list with two elements
    assert isinstance(result, list)
    assert len(result) == 2

    # Check if the mean and std values are close to the expected values
    expected_mean = sum(cov_summed) / len(cov_summed)
    variance = sum((x - expected_mean) ** 2 for x in cov_summed) / len(cov_summed)
    expected_std = variance**0.5

    assert pytest.approx(result[0], abs=1e-6) == expected_mean
    assert pytest.approx(result[1], abs=1e-6) == expected_std


def test_fastest_sum(cov):
    cov_numba = np.array([np.array(x) for x in cov])
    result = fastest_sum(cov_numba)
    # Check if the result is a tuple with four elements
    assert isinstance(result, np.ndarray)
    assert all([len(x) == len(result) for x in cov])

    # Calculate the expected result by summing each list in the input
    expected_result = np.array(np.sum(cov, axis=0))
    # Check if the calculated result matches the expected result
    assert np.array_equal(result, expected_result)

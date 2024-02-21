import os
import tempfile

import great_expectations as ge
import numpy as np
import pandas as pd
import pytest

from src.generate_features import Stats, fastest_sum, numba_calc
from src.paths import FEATURES_COMBINED_FILE, SIM_BAM_FILE_NAME


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
        pd.read_csv(
            os.path.join(tempdir.name, FEATURES_COMBINED_FILE),
            sep=",",
            dtype={
                "chr": "int8",
                "start": "int32",
                "end": "int32",
                "overlap": "float32",
                "intq": "float16",
                "means": "float16",
                "std": "float16",
                "BAM_CMATCH": "int32",
                "BAM_CINS": "int16",
                "BAM_CDEL": "int16",
                "BAM_CSOFT_CLIP": "int16",
                "NM tag": "int16",
                "STAT_CROSS": "float16",
                "STAT_CROSS2": "float16",
                "BAM_CROSS": "int64",
                "cnv_type": "object",
                "NXT_5": "float32",
                "PR_5": "float32",
                "NXT_10": "float64",
                "PR_10": "float64",
                "NXT_20": "float64",
                "PR_20": "float64",
            },
        )
    )
    return df


# Test if columns in df are as expected
def test_expect_table_columns_to_match_ordered_list(df):
    result = df.expect_table_columns_to_match_ordered_list(
        [
            "chr",
            "start",
            "end",
            "overlap",
            "intq",
            "means",
            "std",
            "BAM_CMATCH",
            "BAM_CINS",
            "BAM_CDEL",
            "BAM_CSOFT_CLIP",
            "NM tag",
            "BAM_CROSS",
            "STAT_CROSS",
            "STAT_CROSS2",
            "cnv_type",
            "PR_5",
            "PR_10",
            "PR_20",
            "NXT_5",
            "NXT_10",
            "NXT_20",
        ]
    )
    assert result["success"]


def test_expect_column_values_to_be_of_type(df):
    assert df.expect_column_values_to_be_of_type("chr", "int8")["success"]
    assert df.expect_column_values_to_be_of_type("start", "int32")["success"]
    assert df.expect_column_values_to_be_of_type("end", "int32")["success"]
    assert df.expect_column_values_to_be_of_type("overlap", "float32")["success"]
    assert df.expect_column_values_to_be_of_type("intq", "float16")["success"]
    assert df.expect_column_values_to_be_of_type("means", "float16")["success"]
    assert df.expect_column_values_to_be_of_type("std", "float16")["success"]
    assert df.expect_column_values_to_be_of_type("BAM_CMATCH", "int32")["success"]
    assert df.expect_column_values_to_be_of_type("BAM_CINS", "int16")["success"]
    assert df.expect_column_values_to_be_of_type("BAM_CDEL", "int16")["success"]
    assert df.expect_column_values_to_be_of_type("BAM_CSOFT_CLIP", "int16")["success"]
    assert df.expect_column_values_to_be_of_type("NM tag", "int16")["success"]
    assert df.expect_column_values_to_be_of_type("BAM_CROSS", "int64")["success"]
    assert df.expect_column_values_to_be_of_type("STAT_CROSS", "float16")["success"]
    assert df.expect_column_values_to_be_of_type("STAT_CROSS2", "float16")["success"]
    assert df.expect_column_values_to_be_of_type("PR_5", "float32")["success"]
    assert df.expect_column_values_to_be_of_type("NXT_5", "float32")["success"]
    assert df.expect_column_values_to_be_of_type("PR_10", "float64")["success"]
    assert df.expect_column_values_to_be_of_type("NXT_10", "float64")["success"]
    assert df.expect_column_values_to_be_of_type("PR_20", "float64")["success"]
    assert df.expect_column_values_to_be_of_type("NXT_20", "float64")["success"]
    assert df.expect_column_values_to_be_of_type("cnv_type", "object")["success"]


def test_expect_chr_values_to_be_in_set(df):
    result = df.expect_column_values_to_be_in_set("chr", [1])
    assert result["success"]


def test_expected_target_values_to_be_in_set(df):
    result = df.expect_column_values_to_be_in_set("cnv_type", ["dup", "del", "normal"])
    assert result["success"]


# Unique combinations of features (detect data leaks!)
def test_expect_compound_columns_to_be_unique(df):
    assert df.expect_compound_columns_to_be_unique(["chr", "start", "end"])["success"]


# Test if dataset is not empty
def test_expect_table_row_count_to_be_between(df):
    assert df.expect_table_row_count_to_be_between(1, None)["success"]


def test_features_nxt(df):
    assert (
        df.groupby("chr")
        .tail(5)
        .expect_column_values_to_be_in_set("NXT_5", [0.0])["success"]
    )
    assert (
        df.groupby("chr")
        .tail(10)
        .expect_column_values_to_be_in_set("NXT_10", [0.0])["success"]
    )
    assert (
        df.groupby("chr")
        .tail(20)
        .expect_column_values_to_be_in_set("NXT_20", [0.0])["success"]
    )


def test_features_pr(df):
    print(df)
    assert (
        df.groupby("chr")
        .head(5)
        .expect_column_values_to_be_in_set("PR_5", [0.0])["success"]
    )
    assert (
        df.groupby("chr")
        .head(10)
        .expect_column_values_to_be_in_set("PR_10", [0.0])["success"]
    )
    assert (
        df.groupby("chr")
        .head(20)
        .expect_column_values_to_be_in_set("PR_20", [0.0])["success"]
    )


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


def test_numba_calc_results_are_as_expected(cov_summed):
    result = numba_calc(cov_summed)
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

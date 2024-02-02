from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.metrics import CNV, CNVMetric


@pytest.fixture
def sample_data():
    df = pd.read_csv("./tests/test_metrics/preds.csv")
    return df


@pytest.fixture
def metrics(sample_data):
    metrics = CNVMetric(sample_data)
    return metrics


@pytest.fixture
def cnvs(metrics):
    real_cnvs = metrics._extract_real_cnvs()
    predicted_cnvs = metrics._extract_predicted_cnvs()
    cnvs = metrics._get_childs_intersecting_and_incorrect(real_cnvs, predicted_cnvs)
    return cnvs


@pytest.mark.parametrize("w", np.linspace(-10, 100, 100))
def test_base_metric_in_range(metrics, w):
    base_metric = metrics.base_metric(w)
    assert 0 <= base_metric <= 1


@pytest.fixture
def mock_get_metrics_best():
    return {
        "predicted_incorrectly": 0,
        "prediction_cov": 1.0,
        "all_true_cnvs": 20,
        "all_predicted_cnvs": 25,
    }


def test_base_metric_best(mock_get_metrics_best):
    with patch.object(CNVMetric, "get_metrics", return_value=mock_get_metrics_best):
        cnv_metric = CNVMetric(None)  # Pass whatever dataframe you want for testing
        w = 2  # or any other value you want to test
        result = cnv_metric.base_metric(w)
        assert result == 1.0


@pytest.fixture
def mock_get_metrics_worst():
    return {
        "predicted_incorrectly": 25,
        "prediction_cov": 0.0,
        "all_true_cnvs": 20,
        "all_predicted_cnvs": 25,
    }


def test_base_metric_worst(mock_get_metrics_worst):
    with patch.object(CNVMetric, "get_metrics", return_value=mock_get_metrics_worst):
        cnv_metric = CNVMetric(None)  # Pass whatever dataframe you want for testing
        w = 2  # or any other value you want to test
        result = cnv_metric.base_metric(w)
        assert result == 0.0


def test_extract_real_cnvs(metrics):
    cnvs = metrics._extract_real_cnvs()
    assert len(cnvs) == 20
    assert cnvs[-1] == CNV(
        cnv_type="normal",
        chr=2,
        start=1751,
        end=1800,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )
    assert cnvs[8] == CNV(
        cnv_type="normal",
        chr=1,
        start=1101,
        end=1150,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )
    assert cnvs[9] == CNV(
        cnv_type="normal",
        chr=2,
        start=1,
        end=250,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )


def test_extract_predicted_cnvs(metrics):
    cnvs = metrics._extract_predicted_cnvs()
    assert len(cnvs) == 28
    assert cnvs[-1] == CNV(
        cnv_type="del",
        chr=2,
        start=1651,
        end=1800,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )
    assert cnvs[11] == CNV(
        cnv_type="normal",
        chr=1,
        start=1101,
        end=1150,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )

    assert cnvs[12] == CNV(
        cnv_type="normal",
        chr=2,
        start=1,
        end=150,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )


def test_get_childs_intersecting_and_incorrect(cnvs):
    assert len(cnvs) == 20


def test_child_correct(cnvs):
    assert len(cnvs[0].childs) == 2
    assert len(cnvs[3].childs) == 1
    assert len(cnvs[3].intersected) == 0
    assert len(cnvs[3].incorrect_childs) == 0
    assert cnvs[3].childs[0] == CNV(
        cnv_type="dup",
        chr=1,
        start=401,
        end=550,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )


def test_intersected(cnvs):
    assert len(cnvs[5].intersected) == 1
    assert len(cnvs[5].childs) == 0
    assert len(cnvs[5].incorrect_childs) == 0
    assert cnvs[5].intersected[0] == CNV(
        cnv_type="dup",
        chr=1,
        start=701,
        end=800,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )


def test_child_incorrect(cnvs):
    assert len(cnvs[0].incorrect_childs) == 2
    incorrect_childs = cnvs[0].incorrect_childs
    assert incorrect_childs[0] == CNV(
        cnv_type="del",
        chr=1,
        start=1,
        end=50,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )
    assert incorrect_childs[1] == CNV(
        cnv_type="dup",
        chr=1,
        start=151,
        end=200,
        childs=[],
        intersected=[],
        incorrect_childs=[],
    )


def test_predicted_correctly(metrics, cnvs):
    predicted_correctly = metrics._CNVMetric__predicted_correctly(cnvs)
    assert predicted_correctly == {"dup": 2, "del": 1}


def test_predicted_half_correctly(metrics, cnvs):
    predicted_half_correctly = metrics._CNVMetric__predicted_half_correctly(cnvs)
    assert predicted_half_correctly == {"dup": 1, "del": 2}


def test_predicted_incorrectly(metrics, cnvs):
    intersected_half_correctly = metrics._CNVMetric__predicted_half_correctly(cnvs)
    predicted_incorrectly = metrics._CNVMetric__predicted_incorrectly(
        cnvs, intersected_half_correctly
    )
    assert predicted_incorrectly == 3


def test_intersected_half_correctly(metrics, cnvs):
    intersected_half_correctly = metrics._CNVMetric__intersected_half_correctly(cnvs)
    assert intersected_half_correctly == {"dup": 1, "del": 0}


def test_correct_is_no_higher_than_real_cnvs(metrics):
    metrics = metrics.get_metrics()
    correct_cnvs = {cnv_type: 0 for cnv_type in metrics["predicted_correctly"]}
    for category in [
        "predicted_correctly",
        "predicted_half_correctly",
        "intersected_half_correctly",
    ]:
        for cnv_type in metrics[category]:
            correct_cnvs[cnv_type] += metrics[category][cnv_type]

    total_correct = sum(correct_cnvs.values())
    assert total_correct <= metrics["all_true_cnvs"]


def test_incorrect_is_no_higher_than_predicted(metrics):
    metrics = metrics.get_metrics()
    incorrect_cnvs = metrics["predicted_incorrectly"]
    assert incorrect_cnvs <= metrics["all_predicted_cnvs"]


def test_cov_lower_than_100(metrics):
    metrics = metrics.get_metrics()
    assert metrics["prediction_cov"] <= 1

import numpy as np
import pytest
from pybedtools import BedTool

from src.cnv_generator import CNVGenerator


@pytest.fixture
def cnv_generator(sample_fasta_file):
    cnv_gen = CNVGenerator(50, 75, 10, 0.7, sample_fasta_file)
    return cnv_gen


def test_clean_bed_dup_del(cnv_generator):
    result = cnv_generator._clean_bed_dup_del()
    assert len(result) == 3
    assert isinstance(result[0], BedTool)
    assert isinstance(result[1], BedTool)
    assert isinstance(result[2], np.ndarray)

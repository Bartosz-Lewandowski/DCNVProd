import gzip
import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from pybedtools import BedTool
from tqdm.std import tqdm

from .paths import (
    FEATURES_COMBINED_FILE,
    REF_GEN_PATH,
    STATS_FOLDER,
    TEST_FOLDER,
    TEST_PATH,
    TRAIN_FOLDER,
    TRAIN_PATH,
    VAL_FOLDER,
    VAL_PATH,
)


@dataclass
class BedFormat:
    chr: str
    start: int
    end: int
    cnv_type: str = "normal"
    freq: int = 1


def BedTool_to_BedFormat(bedfile: BedTool, short_v: bool = False) -> np.ndarray:
    out = []
    if short_v:
        return [
            BedFormat(line.chrom, int(line.start), int(line.end)) for line in bedfile
        ]
    for line in bedfile:
        out.append(
            BedFormat(
                line.chrom,
                int(line.start),
                int(line.end),
                line.name,
                int(line.score),
            )
        )
    return np.array(out)


def BedFormat_to_BedTool(seq: np.array) -> BedTool:
    out_str = ""
    for line in seq:
        out_str += f"{line.chr} {line.start} {line.end} {line.cnv_type} {line.freq}\n"
    bedfile = BedTool(out_str, from_string=True)
    return bedfile


def download_reference_genome(chrs: list, output_folder: str = REF_GEN_PATH) -> None:
    """Download reference genome from Ensembl.
    Save it in reference_genome folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    URL = "http://ftp.ensembl.org/pub/release-107/fasta/sus_scrofa/dna/"
    page = requests.get(URL)
    cont = page.content

    texts_search = [
        f"Sus_scrofa.Sscrofa11.1.dna.chromosome.{chr}.fa.gz" for chr in chrs
    ]  # find all files with chromosomes from 1 to 18 without X and Y
    try:
        chrs = [re.findall(text_search, str(cont))[0] for text_search in texts_search]
    except IndexError:
        raise IndexError(
            "Chromosome not found. Check if the chromosome number is correct."
        )
    for chr in chrs:
        with requests.get(URL + chr, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))
            with tqdm.wrapattr(
                r.raw, "read", total=total_length, desc="Downloading reference genome"
            ) as raw:
                with open(output_folder + "/" + chr, "wb") as output:
                    shutil.copyfileobj(raw, output)


def combine_and_cleanup_reference_genome(input_folder, output_file):
    with open(output_file, "wb") as out_f:
        for filename in os.listdir(input_folder):
            if filename.endswith(".gz"):
                with gzip.open(os.path.join(input_folder, filename), "rb") as in_f:
                    out_f.write(in_f.read())
    # Clean up the .gz files
    for filename in os.listdir(input_folder):
        if filename.endswith(".fa.gz"):
            os.remove(os.path.join(input_folder, filename))


def prepare_data() -> None:
    os.makedirs(TRAIN_FOLDER, exist_ok=True)
    os.makedirs(TEST_FOLDER, exist_ok=True)
    os.makedirs(VAL_FOLDER, exist_ok=True)
    data_file = "/".join([STATS_FOLDER, FEATURES_COMBINED_FILE])
    sim_data = pd.read_csv(
        data_file,
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
        },
    )
    test = sim_data[sim_data["chr"].isin([3, 13, 18])].reset_index(drop=True)
    val = sim_data[sim_data["chr"].isin([4, 9])].reset_index(drop=True)
    train = sim_data[~sim_data["chr"].isin([3, 4, 9, 13, 18])].reset_index(drop=True)

    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

import glob
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from numba import jit
from tqdm import tqdm

from src.config import (
    FEATURES_COMBINED_FILE,
    SIM_BAM_FILE_NAME,
    SIM_DATA_PATH,
    STATS_FOLDER,
    TARGET_DATA_FILE_NAME,
)


@jit(nopython=True)
def numba_calc(cov):
    means = np.mean(cov)
    std = np.std(cov)
    return [means, std]


@jit(nopython=True)
def fastest_sum(list_of_lists):
    return (
        list_of_lists[0][:]
        + list_of_lists[1][:]
        + list_of_lists[2][:]
        + list_of_lists[3][:]
    )


class Stats:
    def __init__(self, cpus: int = 2) -> None:
        self.bam_file = "/".join([SIM_DATA_PATH, SIM_BAM_FILE_NAME])
        self.output_folder = STATS_FOLDER
        self.cpus = cpus
        pysam.index(self.bam_file)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def combine_into_one_big_file(self) -> pd.DataFrame:
        logging.info("Combining all stats into one big file")
        isExist: bool = os.path.exists(
            "/".join([self.output_folder, FEATURES_COMBINED_FILE])
        )
        if isExist:
            os.system(f"rm -r {self.output_folder}/{FEATURES_COMBINED_FILE}")
        files_for_this_idv = glob.glob(os.path.join(self.output_folder, "*.csv"))
        df_y = pd.read_csv(
            "/".join([SIM_DATA_PATH, TARGET_DATA_FILE_NAME]),
            sep="\t",
            dtype={"chr": object},
            header=None,
            names=["chr", "start", "end", "cnv_type"],
        )
        df_X = pd.concat(
            (
                pd.read_csv(f, sep="\t", dtype={"chr": object})
                for f in files_for_this_idv
            )
        )
        df = pd.merge(df_X, df_y, on=["chr", "start", "end"])
        df.to_csv("/".join([self.output_folder, FEATURES_COMBINED_FILE]), index=False)

    def generate_stats(self, chrs: list, window_size: int) -> pd.DataFrame:
        with pysam.AlignmentFile(self.bam_file, "rb", threads=self.cpus) as bam:
            refname = bam.references
            seqlen = bam.lengths
        all_chrs = {rname: slen for rname, slen in zip(refname, seqlen)}
        chrs_to_calc = {your_key: all_chrs[your_key] for your_key in chrs}

        logging.info(
            f"Calculating features for chromosomes: {chrs_to_calc.keys()} with lengths {chrs_to_calc.values()}"
        )
        out_columns = [
            [
                "chr",
                "start",
                "end",
                "means",
                "std",
                "BAM_CMATCH",
                "BAM_CINS",
                "BAM_CDEL",
                "BAM_CSOFT_CLIP",
            ]
        ]
        with Pool(processes=self.cpus) as p:
            start_time = time.time()
            arg = [
                (ref, seq, window_size, out_columns)
                for ref, seq in chrs_to_calc.items()
            ]
            p.map(self.get_features, arg)
            logging.info(
                f"Time elapsed to calculate features for all chromosomes {time.time() - start_time}"
            )

    def get_features(
        self,
        args: tuple[str, int, int, list],
    ) -> None:
        refname = args[0]
        seqlen = args[1]
        window_size = args[2]
        out_columns = args[3]
        starts = range(1, seqlen - window_size + 2, window_size)
        total = len(list(starts))
        ends = range(window_size, seqlen + 1, window_size)
        output = []
        with pysam.AlignmentFile(self.bam_file, "rb", threads=self.cpus) as bam:
            for start, end in tqdm(zip(starts, ends), total=total):
                cov = bam.count_coverage(refname, start, end)
                cov = np.array([list(x) for x in cov])
                cov_all = fastest_sum(cov)
                stats = numba_calc(cov_all)
                cigar = self._get_cigar_stats(refname, start, end, bam)
                out = [refname, start, end, *stats, *cigar]
                output.append(out)
        df = pd.DataFrame(output, columns=out_columns)
        df.to_csv(f"{self.output_folder}/{refname}.csv", sep="\t", index=False)

    def _get_cigar_stats(self, chr: str, start: int, end: int, bam):
        cigar = np.sum(
            (
                [
                    x.get_cigar_stats()[0]
                    for x in bam.fetch(chr, start, end, until_eof=True)
                ]
            ),
            axis=0,
        )
        if isinstance(cigar, float):
            cigar = [0 for _ in range(11)]
        return [cigar[0], cigar[1], cigar[2], cigar[4]]

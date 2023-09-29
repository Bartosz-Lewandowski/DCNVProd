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
    SIM_BAM_FILE_PATH,
    SIM_DATA_PATH,
    STATS_FOLDER,
    TARGET_DATA_FILE_NAME,
)


@jit(nopython=True)
def numba_calc(cov: list) -> list:
    """
    Calculates mean and std of coverage.
    """
    means = np.mean(cov)
    std = np.std(cov)
    return [means, std]


@jit(nopython=True)
def fastest_sum(list_of_lists: list) -> tuple:
    """
    Sums all lists in list_of_lists.
    Faster than np.sum(list_of_lists, axis=0)

    Args:
        list_of_lists (list): list of lists to sum

    Returns:
        tuple: tuple of summed lists

    """
    return (
        list_of_lists[0][:]
        + list_of_lists[1][:]
        + list_of_lists[2][:]
        + list_of_lists[3][:]
    )


class Stats:
    """
    Class for calculating stats from bam file.
    Calculates mean and std of coverage and cigar stats.
    Next those stats are combined with target file and passed to the model as training data.

    Args:
        cpus (int): number of cpus to use
    """

    def __init__(self, cpus: int, bam_file: str = SIM_BAM_FILE_PATH) -> None:
        self.bam_file = bam_file
        self.output_folder = STATS_FOLDER
        self.cpus = cpus
        pysam.index(self.bam_file)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def combine_into_one_big_file(self) -> pd.DataFrame:
        """
        Combines all stats into one big file with target file.

        Returns:
            pd.DataFrame: dataframe with all stats and target column (dup, del, normal)
        """
        logging.info("Combining all stats into one big file")
        training_file = "/".join([self.output_folder, FEATURES_COMBINED_FILE])
        isExist: bool = os.path.exists(training_file)
        if isExist:
            os.system(f"rm -r {training_file}")
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
        df.to_csv(training_file, index=False)

    def generate_stats(self, chrs: list, window_size: int) -> pd.DataFrame:
        """
        Generates stats for each chromosome in genome.
        It uses multiprocessing to speed up the process.
        Stats are calculated for each window_size of the chromosome.

        Args:
            chrs (list): list of chromosomes to calculate stats for
            window_size (int): window size for calculating stats

        Returns:
            pd.DataFrame: dataframe with stats for each window_size
        """
        with pysam.AlignmentFile(self.bam_file, "rb", threads=self.cpus) as bam:
            refname = bam.references
            seqlen = bam.lengths
        all_chrs = {rname: slen for rname, slen in zip(refname, seqlen)}
        chrs_to_calc = {
            chr_key: all_chrs[chr_key] for chr_key in chrs
        }  # take only chrs that were passed as argument

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
        """
        Calculates stats for a given chromosome.
        Calculates mean and std of coverage and cigar stats for each window.
        Saves stats to a csv file.

        Args:
            args (tuple): tuple with arguments for the function
        """
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

    def _get_cigar_stats(self, chr: str, start: int, end: int, bam) -> list:
        """
        Get cigar stats for a given region.

        Args:
            chr (str): chromosome
            start (int): start of the region
            end (int): end of the region
            bam ([type]): bam file

        Returns:
            list: list of cigar stats
        """
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

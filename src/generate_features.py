import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from numba import jit
from scipy import stats as st
from tqdm import tqdm

from src.paths import (
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
def fastest_sum(list_of_lists: np.ndarray) -> np.ndarray:
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

    def __init__(
        self,
        cpus: int,
        bam_file: str = SIM_BAM_FILE_PATH,
        output_folder: str = STATS_FOLDER,
        sim_data_path: str = SIM_DATA_PATH,
    ) -> None:
        self.bam_file = bam_file
        self.output_folder = output_folder
        self.sim_data_path = sim_data_path
        self.cpus = cpus
        pysam.index(self.bam_file)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def combine_into_one_big_file(self, values: list) -> pd.DataFrame:
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
        df_X = pd.DataFrame(
            values,
            columns=[
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
            ],
        )
        df_y = pd.read_csv(
            "/".join([self.sim_data_path, TARGET_DATA_FILE_NAME]),
            sep="\t",
            dtype={"chr": object},
            header=None,
            names=["chr", "start", "end", "cnv_type"],
        )

        df = pd.merge(df_X, df_y, on=["chr", "start", "end"])
        df_prev_next = self.__get_next_and_prev(df)
        df_prev_next.to_csv(training_file, index=False)

    def generate_stats(self, chrs: list, window_size: int) -> list:
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
        with Pool(processes=self.cpus) as p:
            start_time = time.time()
            args = []
            total = 0

            for ref, seq in chrs_to_calc.items():
                starts, ends = self.__calc_starts_ends(seq, window_size)
                total += len(starts)

                for s, e in zip(starts, ends):
                    args.append((ref, s, e))

            results = list(
                tqdm(
                    p.imap(self.get_features, args),
                    total=total,
                    desc="Calculating features",
                )
            )
            logging.info(
                f"Time elapsed to calculate features for all chromosomes {time.time() - start_time}"
            )
            return results

    def __calc_starts_ends(self, seqlen: int, window_size: int) -> tuple:
        """
        Calculates starts and ends for a given chromosome.

        Args:
            seqlen (int): length of the chromosome
            window_size (int): window size for calculating stats

        Returns:
            tuple: tuple of starts and ends
        """
        starts = range(1, seqlen - window_size + 2, window_size)
        ends = range(window_size, seqlen + 1, window_size)
        return starts, ends

    def get_features(
        self,
        args: tuple[str, int, int],
    ) -> list:
        """
        Calculates stats for a given chromosome.
        Calculates mean and std of coverage and cigar stats for each window.
        Saves stats to a csv file.

        Args:
            args (tuple): list with arguments for the function
        """
        refname = args[0]
        start = args[1]
        end = args[2]
        with pysam.AlignmentFile(self.bam_file, "rb", threads=1) as bam:
            cov = bam.count_coverage(refname, start, end)
            overlap_list = [
                x.get_overlap(start, end) for x in bam.fetch(refname, start, end)
            ]
            overlap = np.sum([x if x is not None else 0 for x in overlap_list])
            cov = np.array([list(x) for x in cov])
            intq = st.iqr(cov)
            cov_all = fastest_sum(cov)
            stats = numba_calc(cov_all)
            cigar = self._get_cigar_stats(refname, start, end, bam)
            feature_crosses = self._feature_crossing(cigar, stats)
            out = [refname, start, end, overlap, intq, *stats, *cigar, *feature_crosses]
        return out

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
        return [cigar[0], cigar[1], cigar[2], cigar[4], cigar[-1]]

    def _feature_crossing(self, cigar_stats: list, cov_stats: list) -> list:
        """
        Crosses cigar stats and coverage stats.

        Args:
            cigar_stats (list): list of cigar stats
            cov_stats (list): list of coverage stats

        Returns:
            list: list of crossed stats
        """
        bam_cross = abs(np.prod(cigar_stats))
        mean, std = cov_stats
        cov1_cross = mean / (std + 0.0001)
        cov2_cross = mean * std
        return [bam_cross, cov1_cross, cov2_cross]

    def __get_next_and_prev(self, df: pd.DataFrame) -> pd.DataFrame:
        df["PR_5"] = (
            df["means"].rolling(5, min_periods=5).sum().fillna(0).round(decimals=5)
        )
        df["PR_10"] = (
            df["means"].rolling(10, min_periods=10).sum().fillna(0).round(decimals=5)
        )
        df["PR_20"] = (
            df["means"].rolling(20, min_periods=20).sum().fillna(0).round(decimals=5)
        )
        df["NXT_5"] = (
            df["means"][::-1]
            .rolling(5, min_periods=5)
            .sum()[::-1]
            .fillna(0)
            .round(decimals=5)
        )
        df["NXT_10"] = (
            df["means"][::-1]
            .rolling(10, min_periods=10)
            .sum()[::-1]
            .fillna(0)
            .round(decimals=5)
        )
        df["NXT_20"] = (
            df["means"][::-1]
            .rolling(20, min_periods=20)
            .sum()[::-1]
            .fillna(0)
            .round(decimals=5)
        )
        df = self.__correct_next_and_prev(df)
        return df

    def __correct_next_and_prev(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df.groupby("chr").head(5).index, ["PR_5"]] = 0
        df.loc[df.groupby("chr").head(10).index, ["PR_10"]] = 0
        df.loc[df.groupby("chr").head(20).index, ["PR_20"]] = 0
        df.loc[df.groupby("chr").tail(5).index, ["NXT_5"]] = 0
        df.loc[df.groupby("chr").tail(10).index, ["NXT_10"]] = 0
        df.loc[df.groupby("chr").tail(20).index, ["NXT_20"]] = 0
        return df

import glob
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from numba import jit
from tqdm import tqdm


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
    def __init__(self, bam_file: str, output_folder: str, cpus: int = 2) -> None:
        self.bam_file = bam_file
        self.output_folder = f"stats/{output_folder}"
        self.cpus = cpus
        pysam.index(self.bam_file)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def combine_into_one_big_file(self, target_file: str) -> pd.DataFrame:
        print("comining into one big file...")
        isExist: bool = os.path.exists(f"{self.output_folder}/combined.csv")
        if isExist:
            os.system(f"rm -r {self.output_folder}/combined.csv")
        files_for_this_idv = glob.glob(os.path.join(self.output_folder, "*.csv"))
        df_y = pd.read_csv(
            target_file,
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
        df.to_csv(f"{self.output_folder}/combined.csv", index=False)

    def generate_stats(self, chrs: list, window_size: int, step: int) -> pd.DataFrame:
        with pysam.AlignmentFile(self.bam_file, "rb", threads=self.cpus) as bam:
            refname = bam.references
            seqlen = bam.lengths
        all_chrs = {rname: slen for rname, slen in zip(refname, seqlen)}
        chrs_to_calc = {your_key: all_chrs[your_key] for your_key in chrs}

        print(
            f"Calculating features for chromosoms: {chrs_to_calc.keys()} with lengths {chrs_to_calc.values()}"
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
                (ref, seq, window_size, step, out_columns)
                for ref, seq in chrs_to_calc.items()
            ]
            p.map(self.get_features, arg)
            print(
                f"Time elapsed to calculate features for individual {self.output_folder}, with time {time.time() - start_time}"
            )

    def get_features(
        self,
        args: tuple[str, int, int, int, list],
    ) -> None:
        refname = args[0]
        seqlen = args[1]
        window_size = args[2]
        step = args[3]
        out_columns = args[4]
        starts = range(1, seqlen - window_size, step)
        total = len(list(starts))
        ends = range(step, seqlen + 1, step)
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

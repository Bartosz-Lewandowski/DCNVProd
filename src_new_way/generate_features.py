import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import scipy.stats


class Stats:
    def __init__(self, bam_file: str, output_folder: str) -> None:
        self.bam_file = bam_file
        self.output_folder = f"stats/{output_folder}"
        pysam.index(self.bam_file)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def generate_stats(self, window_size: int = 50, step: int = 50) -> pd.DataFrame:
        with pysam.AlignmentFile(self.bam_file, "rb", threads=8) as bam:
            refname = bam.references
            seqlen = bam.lengths

        with Pool(processes=10) as p:
            arg = [(ref, seq, window_size, step) for ref, seq in zip(refname, seqlen)]
            p.map(self.get_features, arg)

    def combine_into_one_big_file(self, target_file: str) -> pd.DataFrame:
        isExist: bool = os.path.exists(f"{self.output_folder}/combined.csv")
        if isExist:
            os.system(f"rm {self.output_folder}/combined.csv")
        files_for_this_idv = os.listdir(self.output_folder)
        df_target = pd.read_csv(target_file, sep="\t", header=None)
        df = pd.DataFrame()
        for file in sorted(files_for_this_idv):
            tmp = pd.read_csv(f"{self.output_folder}/{file}", sep="\t")
            tmp["cnv_type"] = df_target[3]
            df = pd.concat([df, tmp])
        df.to_csv(f"{self.output_folder}/combined.csv", index=False)

    def get_features(
        self,
        args: tuple[str, int, int, int],
    ) -> pd.DataFrame:
        print(args)
        refname = args[0]
        seqlen = args[1]
        window_size = args[2]
        step = args[3]

        out_np = [
            [
                "chr",
                "start",
                "end",
                "count",
                "means",
                "med",
                "intq",
                "std",
                "num_aligned",
                "nseg",
                "overlap",
                "BAM_CMATCH",
                "BAM_CINS",
                "BAM_CDEL",
                "BAM_CREF_SKIP",
                "BAM_CSOFT_CLIP",
                "BAM_CHARD_CLIP",
                "BAM_CPAD",
                "BAM_CEQUAL",
                "BAM_CDIFF",
                "BAM_CBACK",
                "NM tag",
            ]
        ]
        starts = range(1, seqlen - window_size, step)
        ends = range(step, seqlen, step)
        start_time = time.time()
        for start, end in zip(starts, ends):
            stats = self._calc_stats(refname, start, end)
            out_np.append(stats)
        print(time.time() - start_time)
        df = pd.DataFrame(out_np[1:], columns=out_np[0])
        df.to_csv(f"{self.output_folder}/{refname}.csv", sep="\t", index=False)

    def _get_cigar_stats(self, chr: str, start: int, end: int):
        with pysam.AlignmentFile(self.bam_file, "rb") as bam:
            cigar = np.sum(
                np.sum(
                    [x.get_cigar_stats() for x in bam.fetch(chr, start, end)], axis=0
                ),
                axis=0,
            )
        if isinstance(cigar, float):
            cigar = [0 for _ in range(11)]
        # return [cigar[1], cigar[2], cigar[4], cigar[5], cigar[10]]
        return cigar

    def _calc_stats(self, chr: str, start: int, end: int) -> pd.DataFrame:
        with pysam.AlignmentFile(self.bam_file, "rb", threads=8) as bam:
            count = bam.count(chr, start, end)
            cov = np.sum(bam.count_coverage(chr, start, end), axis=0)
            means = np.mean(cov)
            med = np.median(cov)
            intq = scipy.stats.iqr(cov)
            std = np.std(cov)
            num_aligned = np.sum(
                [x.get_num_aligned() for x in bam.pileup(chr, start, end)]
            )
            nseg = np.sum([x.nsegments for x in bam.pileup(chr, start, end)])
            overlap_list = [
                x.get_overlap(start, end) for x in bam.fetch(chr, start, end)
            ]
            overlap = np.sum([x if x is not None else 0 for x in overlap_list])

        cigar = self._get_cigar_stats(chr, start, end)
        out_df = [
            chr,
            start,
            end,
            count,
            means,
            med,
            intq,
            std,
            num_aligned,
            nseg,
            overlap,
        ]
        out_df.extend(cigar)
        return out_df

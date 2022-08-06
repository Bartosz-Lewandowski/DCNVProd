import time
from multiprocessing import Process

import numpy as np
import pandas as pd
import pysam
import scipy.stats


class Stats:
    def __init__(self, bam_file: str) -> None:
        self.bam_file = bam_file
        self.out_df = pd.DataFrame()
        pysam.index(self.bam_file)

    def generate_stats(
        self, output_file: str = "stats_windows", window_size: int = 50, step: int = 50
    ) -> pd.DataFrame:
        print("START")
        with pysam.AlignmentFile(self.bam_file, "rb", threads=8) as bam:
            refname = bam.references
            seqlen = bam.lengths

        processes = []
        for i in range(len(refname)):
            p = Process(
                target=self.target, args=(refname[i], seqlen[i], window_size, step, output_file)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print(processes)

    def target(self, refname, seqlen, window_size, step, output_file):
        out_np = np.array(
            [
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
        )
        start = range(1, seqlen - window_size, step)
        end = range(step, seqlen, step)
        print("START LOOP")
        start_time = time.time()
        for start, end in zip(start, end):
            stats = self._calc_stats(refname, start, end)
            out_np = np.append(out_np, stats, axis=0)
        print(time.time() - start_time)
        df = pd.DataFrame(out_np[1:], columns=out_np[0])
        df.to_csv(f"{output_file}_{refname}.csv", sep="\t", index=False)

    def get_cigar_stats(self, chr, start, end):
        with pysam.AlignmentFile(self.bam_file, "rb") as bam:
            cigar = np.sum(
                np.sum([x.get_cigar_stats() for x in bam.fetch(chr, start, end)], axis=0), axis=0
            )
        if isinstance(cigar, float):
            cigar = [0 for x in range(11)]
        return cigar

    def _calc_stats(self, chr: str, start: int, end: int) -> pd.DataFrame:
        with pysam.AlignmentFile(self.bam_file, "rb", threads=8) as bam:
            count = bam.count(chr, start, end)
            cov = np.sum(bam.count_coverage(chr, start, end), axis=0)
            means = np.mean(cov)
            med = np.median(cov)
            intq = scipy.stats.iqr(cov)
            std = np.std(cov)
            num_aligned = np.sum([x.get_num_aligned() for x in bam.pileup(chr, start, end)])
            nseg = np.sum([x.nsegments for x in bam.pileup(chr, start, end)])
            overlap_list = [x.get_overlap(start, end) for x in bam.fetch(chr, start, end)]
            overlap = np.sum([x if x is not None else 0 for x in overlap_list])

        cigar = self.get_cigar_stats(chr, start, end)
        out_df = np.array(
            [chr, start, end, count, means, med, intq, std, num_aligned, nseg, overlap]
        )
        out_df = np.append(out_df, cigar)
        return np.array([out_df])

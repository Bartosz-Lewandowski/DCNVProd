import numpy as np
import pandas as pd
import scipy.stats
from pybedtools import BedTool


class Stats:
    def __init__(self) -> None:
        pass

    def create_genomecov(self, bam_file: str) -> pd.DataFrame:
        bam = BedTool(bam_file)
        bam_cov = bam.genome_coverage(d=True)
        df = bam_cov.to_dataframe(names=["chrom", "depth", "n"], dtype={"chrom": object})
        return df

    def generate_stats(
        self,
        df: pd.DataFrame,
        window_size: int = 50,
        step_size: int = 50,
        output_file: str = "stats_windows.csv",
    ) -> pd.DataFrame:
        chrs = df["chrom"].unique()
        out_df = pd.DataFrame(columns=["chr", "start", "end", "med", "iqr", "mean", "std"])
        for i in chrs:
            values_for_chrom = df[df["chrom"] == i]["n"].to_numpy()
            wins_step = self._calc_stats(i, values_for_chrom, window_size - 1, step_size)
            print("Chromosome " + str(i) + " processed")
            out_df = pd.concat([out_df, wins_step])
        out_df.to_csv(output_file, sep="\t", index=False)
        return out_df

    def _calc_stats(self, chr: str, values: np.array, window: int, step: int) -> pd.DataFrame:
        vert_idx_list = np.arange(1, len(values) - window, step)
        windows_values = [values[i : i + window] for i in vert_idx_list]
        med = [np.around(np.median(x), 4) for x in windows_values]
        intq = [np.around(scipy.stats.iqr(x), 4) for x in windows_values]
        means = [np.around(np.mean(x), 4) for x in windows_values]
        std = [np.around(np.std(x), 4) for x in windows_values]
        return pd.DataFrame(
            {
                "chr": chr,
                "start": vert_idx_list,
                "end": vert_idx_list + window,
                "med": med,
                "iqr": intq,
                "mean": means,
                "std": std,
            }
        )

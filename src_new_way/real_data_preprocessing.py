import os
import re

import numpy as np
import pandas as pd
import pysam
from numba import jit
from pybedtools import BedTool
from utils import BedFormat_to_BedTool, BedTool_to_BedFormat

pd.options.mode.chained_assignment = None


def cnvnator_file_name(number):
    return (
        f"real_data_target/cnvnator/cnv-data_cnvnator_wgs_S4389Nr{number}.cnvnator.200"
    )


def pindel_del_file_name(number):
    return f"real_data_target/pindel/cnv-data_pindel_pindel.del.nr.{number}"


def pindel_dup_file_name(number):
    return f"real_data_target/pindel/cnv-data_pindel_pindel.dup.nr.{number}"


def bam_file_name(number):
    return f"real_data/cnv-data_wgs_S4389Nr{number}.fixmate.srt.markdup.recal.bam"


@jit(nopython=True)
def numba_e(lines):
    out = []
    for line in lines:
        if line[0].isdigit():
            x = line.replace("\t", " ")
            y = x.split(" ")
            out.append([y[7], y[9], y[10], y[1]])
    return out


class CleanRealTarget:
    def __init__(self, cpus: int = 8) -> None:
        self.cpus = cpus

    def combine(self, ind_n: str) -> None:
        pindel = pd.read_csv(
            f"real_data_target/pindel/{ind_n}_pindel.csv",
            sep="\t",
            names=["chr", "start", "stop", "type", "freq"],
        )
        cnvnator = pd.read_csv(
            f"real_data_target/cnvnator/{ind_n}_cnvnator.csv",
            sep="\t",
            names=["chr", "start", "stop", "type", "freq"],
        )
        total = pd.concat(
            [
                self._get_valid_cnv(pindel, cnvnator, cnv_type)
                for cnv_type in ["dup", "del"]
            ]
        )
        total = total[total["chr"] == "1"]
        total.drop(["valid"], axis=1, inplace=True)
        os.makedirs("real_data_target/valid_target/", exist_ok=True)
        total.to_csv(
            f"real_data_target/valid_target/{ind_n}.csv",
            sep="\t",
            index=False,
            header=False,
        )
        self._make_windows(ind_n)

    def clean_cnvnator(self, ind_n: str) -> None:
        data = pd.read_csv(cnvnator_file_name(ind_n), sep="\t", header=None)
        types = [self._get_type(value) for value in data[0].values]
        chrom, start, end = self._get_cnv_info(data[1].values)
        data_new = pd.DataFrame(
            {"chr": chrom, "start": start, "end": end, "type": types}
        )
        data_new["freq"] = "0"
        data_new.to_csv(
            f"real_data_target/cnvnator/{ind_n}_cnvnator.csv",
            index=False,
            header=False,
            sep="\t",
        )

    def clean_pindel(self, ind_n: str) -> None:
        pindel_dup = pd.read_csv(
            pindel_dup_file_name(ind_n),
            sep=" ",
            names=["prefix", "chr", "start", "end", "cnv_len", "support"],
            dtype={"chr": object},
        )
        pindel_del = pd.read_csv(
            pindel_del_file_name(ind_n),
            sep=" ",
            names=["prefix", "chr", "start", "end", "cnv_len", "support"],
            dtype={"chr": object},
        )
        pindel_dup_clean = self._filter_data(pindel_dup)
        pindel_del_clean = self._filter_data(pindel_del)

        pindel_dup_clean["type"] = "dup"
        pindel_del_clean["type"] = "del"
        pindel_all = pd.concat([pindel_dup_clean, pindel_del_clean])
        pindel_all.drop(["prefix", "cnv_len", "support"], axis=1, inplace=True)
        pindel_all["freq"] = 0
        pindel_all.to_csv(
            f"real_data_target/pindel/{ind_n}_pindel.csv",
            index=False,
            header=False,
            sep="\t",
        )

    def _get_valid_cnv(self, pindel, cnvnator, cnv_type):
        pindel_tmp = pindel[pindel["type"] == cnv_type].sort_values(by="start")
        cnvnator_tmp = cnvnator[cnvnator["type"] == cnv_type].sort_values(by="start")
        valid = []
        print("find overlaping")
        for r in zip(*cnvnator_tmp.to_dict("list").values()):
            overlap = sum(
                [
                    self._get_overlap([r[1], r[2]], [r2[1], r2[2]])
                    for r2 in zip(*pindel_tmp.to_dict("list").values())
                ]
            )
            window_len = r[2] - r[1]
            cov_ratio = overlap / window_len
            if cov_ratio > 0.7:
                valid.append(True)
            else:
                valid.append(False)
        cnvnator_tmp["valid"] = valid
        return cnvnator_tmp[cnvnator_tmp["valid"]]

    def _get_overlap(self, a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    def _get_type(self, value: str) -> str:
        match value:
            case "deletion":
                out = "del"
            case "duplication":
                out = "dup"
        return out

    def _get_cnv_info(self, values: list) -> tuple[np.array, np.array, np.array]:
        chroms, start, end = ([], [], [])
        for x in values:
            pattern = re.split(":|-", x)
            chroms.append(pattern[0])
            start.append(pattern[1])
            end.append(pattern[2])
        return np.array(chroms), np.array(start), np.array(end)

    def _filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[(data["cnv_len"] < 5000000) & (data["cnv_len"] > 50)]
        data = data[data["support"] > 3]
        return data

    def _make_windows(self, ind_n: str) -> None:
        print("START WINDOWS")
        bamfile_path = bam_file_name(ind_n)
        if not os.path.exists(bamfile_path + ".bai"):
            pysam.index(bamfile_path, threads=self.cpus)
        print("AFTER ALIGMENT")
        with pysam.AlignmentFile(bamfile_path, "rb", threads=self.cpus) as bam:
            refname = bam.references
            seqlen = bam.lengths
        refname = [sorted(refname)[0]]
        seqlen = [sorted(seqlen)[0]]
        str_out = ""
        for x, y in zip(refname, seqlen):
            str_out += f"{x} 1 {y}\n"
        data_bedtool = BedTool(f"real_data_target/valid_target/{ind_n}.csv")
        genome = BedTool(str_out, from_string=True)
        print("first windowmaker")
        normal = BedTool().window_maker(genome, w=50)
        print("INTERSECT")
        normal2 = normal.intersect(data_bedtool, v=True, wa=True).sort().merge()
        print("FIRST SWITCH")
        normal2_bedformat = BedTool_to_BedFormat(normal2, short_v=True)
        print("SWITCH TO BEDFORMAT")
        data_bedtool_bedformat = BedTool_to_BedFormat(data_bedtool)
        print("CONCAT")
        total = np.concatenate((normal2_bedformat, data_bedtool_bedformat), axis=0)
        print("ANOTHER SWITCH")
        total_sorted = BedFormat_to_BedTool(total).sort()
        print("LAST WINDOWMAKER")
        BedTool.window_maker(genome, b=total_sorted, w=50, s=50, i="src").saveas(
            f"real_data_target/valid_target/{ind_n}.csv"
        )
import os
import random
from dataclasses import dataclass

import numpy as np
from Bio import SeqIO
from pybedtools import BedTool
from tqdm import tqdm


@dataclass
class BedFormat:
    chr: str
    start: int
    end: int
    cnv_type: str = ""
    cov: float = 0.0
    freq: int = 0

    def get_str(self):
        return "\t".join(
            (
                str(self.chr)
                + str(self.start)
                + str(self.end)
                + self.cnv_type
                + str(self.cov)
                + str(self.freq)
            )
        )


class CNVGenerator:
    def __init__(self, fasta_file: str, pathout: str = "train_bed/") -> None:
        self.fasta_file = fasta_file
        self.pathout = pathout
        isExist = os.path.exists(self.pathout)
        if not isExist:
            os.makedirs(self.pathout)
        fasta = SeqIO.parse(open(self.fasta_file), "fasta")
        self.num_of_chroms = len(tuple(fasta))

    def generate_cnv(self):
        dup, dele, chr_info = self.clean_bed_dup_del()
        dup_freq, dele_freq = self.generate_freq(dup, dele)
        normal = self.bedfile_with_normal_seq(dup_freq, dele_freq, chr_info)
        total = self.create_total_bed(dup_freq, dele_freq, normal)
        return total

    def clean_bed_dup_del(self) -> tuple[BedTool, BedTool, BedTool]:
        dup, dele, chr_info = self._create_bed_with_coords()
        dele_merged = self._cnv_sort_and_merge(dele)
        dup_merged = self._cnv_sort_and_merge(dup)
        dup_intersect, dele_intersect = self._cnv_intersect(dup_merged, dele_merged)
        return dup_intersect, dele_intersect, chr_info

    def generate_freq(self, dup, dele):
        chr_freq = {i: round(i / self.num_of_chroms, 3) for i in range(1, self.num_of_chroms + 1)}
        dup_with_freq = self._create_dup_freqs(dup, chr_freq)
        dele_with_freq = self._create_dele_freqs(dele, chr_freq)
        return dup_with_freq, dele_with_freq

    def bedfile_with_normal_seq(self, dup, dele, chr_info):
        chr_info_bed = CNVGenerator.__BedFormat_to_BedTool(chr_info)
        normal = BedTool().window_maker(chr_info_bed, w=5)
        dup_bedtools = CNVGenerator.__BedFormat_to_BedTool(dup).saveas(f"{self.pathout}dup.bed")
        dele_bedtools = CNVGenerator.__BedFormat_to_BedTool(dele).saveas(f"{self.pathout}del.bed")
        normal2 = (
            normal.intersect(dup_bedtools, v=True, wa=True)
            .intersect(dele_bedtools, v=True, wa=True)
            .sort()
            .merge()
        )
        out_normal = np.array([])
        for line in normal2:
            out_normal = np.append(
                out_normal, BedFormat(line.chrom, line.start, line.end, "normal", 1, 1)
            )
        return out_normal

    def create_total_bed(self, dup, dele, normal):
        total = np.concatenate((dup, dele, normal), axis=0)
        total_sorted = (
            CNVGenerator.__BedFormat_to_BedTool(total).sort().saveas(f"{self.pathout}total.bed")
        )
        total_BedFormat = CNVGenerator.__BedTool_to_BedFormat(total_sorted)
        return total_BedFormat

    def _create_dele_freqs(self, dele: np.array, chr_freq) -> np.array:
        out = np.array([])
        for line in dele:
            num = random.randint(1, self.num_of_chroms)
            out = np.append(
                out,
                np.array(BedFormat(line.chrom, line.start, line.end, "del", chr_freq[num], 0)),
            )
        return out

    def _create_dup_freqs(self, dup: np.array, chr_freq) -> np.array:
        out = np.array([])
        for line in dup:
            num = random.randint(1, self.num_of_chroms)
            count = np.random.choice(
                list(range(2, 10)), 1, p=[0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
            )[0]
            out = np.append(
                out,
                np.array(BedFormat(line.chrom, line.start, line.end, "dup", chr_freq[num], count)),
            )
        return out

    def _cnv_sort_and_merge(self, cnv) -> BedTool:
        cnv_bed = CNVGenerator.__BedFormat_to_BedTool(cnv)
        cnv_merged = cnv_bed.sort().merge()
        return cnv_merged

    def _cnv_intersect(self, dup: BedTool, dele: BedTool) -> tuple[BedTool, BedTool]:
        dup_intersect = dup.intersect(dele, wa=True, v=True)
        dele_intersect = dele.intersect(dup, wa=True, v=True)
        return dup_intersect, dele_intersect

    def _create_bed_with_coords(
        self,
    ):
        dup = np.array([])
        dele = np.array([])
        chr_info = np.array([])
        for chr in tqdm(SeqIO.parse(open(self.fasta_file), "fasta"), total=self.num_of_chroms):
            dup = np.append(dup, np.array([self.__generate_coords(chr)]))
            dele = np.append(dele, np.array([self.__generate_coords(chr)]))
            chr_info = np.append(chr_info, np.array([self.__chromosome_info(chr)]))

        return dup, dele, chr_info

    def __generate_coords(self, fasta_file):
        len_fasta = len(fasta_file.seq)
        lenghts = []
        max_cnv_length = 1e2
        cnv_count = int(
            (len_fasta / max_cnv_length) / 2
        )  # number of cnv that can fit in data devided by two because there are two types of cnv (duplications and deletions)
        while len(lenghts) < cnv_count:
            cnv_range = random.randint(50, max_cnv_length)
            lenghts.append(cnv_range)
        # dodać tutaj test -> assert, że suma tych dwóch list będzie mniejsza od długości całego genomu
        # he = (np.sum(dup_lengths) + np.sum(del_lengths))/len_fasta
        # print(f"Duplikacje i delecje stanowią {he}% całego genomu")
        start = np.random.randint(1, len_fasta, size=(cnv_count))
        end = [st + lgt for st, lgt in zip(start, lenghts)]
        ids = [fasta_file.id] * cnv_count
        return np.array([BedFormat(id, st, en) for id, st, en in zip(ids, start, end)])

    def __chromosome_info(self, fasta_file) -> BedFormat:
        len_fasta = len(fasta_file.seq)
        return np.array([BedFormat(fasta_file.id, 1, len_fasta)])

    @staticmethod
    def __BedTool_to_BedFormat(bedfile: BedTool) -> np.array:
        out = np.array([])
        for line in bedfile:
            out = np.append(
                out,
                BedFormat(
                    line.chrom,
                    int(line.start),
                    int(line.end),
                    line.name,
                    float(line.score),
                    int(line.strand),
                ),
            )
        return out

    @staticmethod
    def __BedFormat_to_BedTool(seq: np.array) -> BedTool:
        out_str = ""
        for line in seq:
            out_str += (
                f"{line.chr} {line.start} {line.end} {line.cnv_type} {line.cov} {line.freq}\n"
            )
        bedfile = BedTool(out_str, from_string=True)
        return bedfile

    def modify_fasta_file(self, total_file: np.array):
        fasta_original = {
            fasta.id: fasta.seq for fasta in SeqIO.parse(open(self.fasta_file), "fasta")
        }
        fasta_modified = {fasta.id: "" for fasta in SeqIO.parse(open(self.fasta_file), "fasta")}
        for line in tqdm(total_file, total=len(total_file)):
            if line.cnv_type == "del":
                continue
            elif line.cnv_type == "normal":
                fasta_modified[line.chr] += fasta_original[line.chr][line.start : line.end]
            elif line.cnv_type == "dup":
                seq_to_copy = self._find_seq_to_copy(line, fasta_original)
                str_modified = seq_to_copy * line.freq
                fasta_modified[line.chr] += str_modified

        return self.save_fasta_CNV(fasta_modified)

    def save_fasta_CNV(self, fasta_modified: dict) -> str:
        fasta_cnv_name = f"{self.pathout}fasta_CNV.fa"

        if os.path.exists(fasta_cnv_name):
            os.remove(fasta_cnv_name)

        for id, fasta_str in fasta_modified.items():
            with open(fasta_cnv_name, "a") as fasta_cnv:
                fasta_cnv.write(f">{id}\n{fasta_str}\n")
        return fasta_cnv_name

    def _find_seq_to_copy(self, line: BedFormat, fasta_original: dict[str, str]) -> str:
        dup_len = line.end - line.start
        seq_len = int(dup_len / line.freq)
        seq_to_copy = fasta_original[line.chr][line.start : line.start + seq_len]
        return seq_to_copy

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
    freq: int = 0


class CNVGenerator:
    """
    CNV simulator.
    With given reference genome this function
    generate random CNV's across the chronosomes.

    Args:
        fasta_file: reference genome in fasta format
        pathout: folder where to store output files
    """

    def __init__(self, fasta_file: str, pathout: str = "train_bed/") -> None:
        self.fasta_file = fasta_file
        self.pathout = pathout
        isExist: bool = os.path.exists(self.pathout)
        if not isExist:
            os.makedirs(self.pathout)
        fasta = SeqIO.parse(open(self.fasta_file), "fasta")
        self.num_of_chroms: int = len(tuple(fasta))

    def generate_cnv(self) -> BedFormat:
        """_summary_

        Returns:
            BedFormat: _description_
        """
        dup, dele, chr_info = self.clean_bed_dup_del()
        dup_freq, dele_freq = self.generate_freq(dup, dele)
        normal = self.bedfile_with_normal_seq(dup_freq, dele_freq, chr_info)
        total, total_bedtool = self.create_total_bed(dup_freq, dele_freq, normal)
        self.create_total_windows(total_bedtool, chr_info)
        return total

    def create_total_windows(self, total_bedtool, chr_info):
        chr_info_bed = CNVGenerator.__BedFormat_to_BedTool(chr_info)
        BedTool.window_maker(chr_info_bed, b=total_bedtool, w=50, s=50, i="src").saveas(
            f"{self.pathout}total.bed"
        )

    def clean_bed_dup_del(self) -> tuple[BedTool, BedTool, np.array]:
        dup, dele, chr_info = self._create_bed_with_coords()
        print("Sortking and merging dele")
        dele_merged = self._cnv_sort_and_merge(dele)
        print("sorting and merging dup")
        dup_merged = self._cnv_sort_and_merge(dup)
        print("INTERSECTING")
        dup_intersect, dele_intersect = self._cnv_intersect(dup_merged, dele_merged)
        return dup_intersect, dele_intersect, chr_info

    def generate_freq(self, dup: np.array, dele: np.array) -> tuple[np.array, np.array]:
        """Method to generate frequencies.

        Args:
            dup (np.array): array with BedFromat as values from intersect.
            dele (np.array): array with BedFromat as values from intersect.

        Returns:
            tuple[np.array, np.array]: Returns same dup and dele object, but with frequencies
        """
        print("generating dup freq")
        dup_with_freq: np.array = self._create_dup_freqs(dup)
        print("generating dele freq")
        dele_with_freq: np.array = self._create_dele_freqs(dele)
        return dup_with_freq, dele_with_freq

    def bedfile_with_normal_seq(
        self, dup: np.array, dele: np.array, chr_info: np.array
    ) -> np.array:
        print("creating normal seq")
        chr_info_bed = CNVGenerator.__BedFormat_to_BedTool(chr_info)
        print("window maker")
        normal = BedTool().window_maker(chr_info_bed, w=50)
        print("dup dele")
        dup_bedtools = CNVGenerator.__BedFormat_to_BedTool(dup).saveas(f"{self.pathout}dup.bed")
        dele_bedtools = CNVGenerator.__BedFormat_to_BedTool(dele).saveas(f"{self.pathout}del.bed")
        print("normal 2")
        normal2 = (
            normal.intersect(dup_bedtools, v=True, wa=True)
            .intersect(dele_bedtools, v=True, wa=True)
            .sort()
            .merge()
        )
        out_normal = np.array([])
        for line in normal2:
            out_normal = np.append(
                out_normal, BedFormat(line.chrom, line.start, line.end, "normal", 1)
            )
        return out_normal

    def create_total_bed(self, dup: np.array, dele: np.array, normal: np.array) -> np.array:
        print("creating total file")
        total = np.concatenate((dup, dele, normal), axis=0)
        total_sorted = CNVGenerator.__BedFormat_to_BedTool(total).sort()
        total_BedFormat = CNVGenerator.__BedTool_to_BedFormat(total_sorted)
        return total_BedFormat, total_sorted

    def _create_dele_freqs(self, dele: np.array) -> np.array:
        out = np.array([])
        for line in dele:
            out = np.append(
                out,
                np.array(BedFormat(line.chrom, line.start, line.end, "del", 0)),
            )
        return out

    def _create_dup_freqs(self, dup: np.array) -> np.array:
        out = np.array([])
        for line in dup:
            count = np.random.choice(
                list(range(2, 10)), 1, p=[0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
            )[0]
            out = np.append(
                out,
                np.array(BedFormat(line.chrom, line.start, line.end, "dup", count)),
            )
        return out

    def _cnv_sort_and_merge(self, cnv: np.array) -> BedTool:
        cnv_bed = CNVGenerator.__BedFormat_to_BedTool(cnv)
        cnv_merged = cnv_bed.sort().merge()
        return cnv_merged

    def _cnv_intersect(self, dup: BedTool, dele: BedTool) -> tuple[BedTool, BedTool]:
        dup_intersect = dup.intersect(dele, wa=True, v=True)
        dele_intersect = dele.intersect(dup, wa=True, v=True)
        return dup_intersect, dele_intersect

    def _create_bed_with_coords(
        self,
    ) -> tuple[np.array, np.array, np.array]:
        dup = np.array([])
        dele = np.array([])
        chr_info = np.array([])
        print("Creating bed objects with coordinates of cnv")
        for chr in tqdm(SeqIO.parse(open(self.fasta_file), "fasta"), total=self.num_of_chroms):
            dup = np.append(dup, np.array([self.__generate_coords(chr)]))
            dele = np.append(dele, np.array([self.__generate_coords(chr)]))
            chr_info = np.append(chr_info, np.array([self.__chromosome_info(chr)]))

        return dup, dele, chr_info

    def __generate_coords(self, fasta_file) -> np.array:
        len_fasta: int = len(fasta_file.seq)
        lenghts: list = []
        max_cnv_length = 1000
        cnv_count = int(
            (len_fasta / max_cnv_length) / 2
        )  # number of cnv that can fit in data devided by two because there are two types of cnv (duplications and deletions)
        while len(lenghts) < cnv_count:
            cnv_range = random.randrange(50, max_cnv_length, step=50)
            lenghts.append(cnv_range)
        # dodać tutaj test -> assert, że suma tych dwóch list będzie mniejsza od długości całego genomu
        # he = (np.sum(dup_lengths) + np.sum(del_lengths))/len_fasta
        # print(f"Duplikacje i delecje stanowią {he}% całego genomu")
        start = [random.randrange(1, len_fasta, step=50) for _ in range(cnv_count)]
        end = [st + lgt for st, lgt in zip(start, lenghts)]
        ids = [fasta_file.id] * cnv_count
        return np.array([BedFormat(id, st, en) for id, st, en in zip(ids, start, end)])

    def __chromosome_info(self, fasta_file) -> np.array:
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
                    int(line.score),
                ),
            )
        return out

    @staticmethod
    def __BedFormat_to_BedTool(seq: np.array) -> BedTool:
        out_str = ""
        for line in seq:
            out_str += f"{line.chr} {line.start} {line.end} {line.cnv_type} {line.freq}\n"
        bedfile = BedTool(out_str, from_string=True)
        return bedfile

    def modify_fasta_file(self, total_file: np.array) -> str:
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
                seq_to_copy = fasta_original[line.chr][line.start : line.end]
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

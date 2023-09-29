import logging
import os
import random
from collections import Counter

import numpy as np
from Bio import SeqIO, SeqRecord
from pybedtools import BedTool
from tqdm import tqdm

from .config import (
    MODIFIED_FASTA_FILE_PATH,
    REF_FASTA_FILE,
    SIM_DATA_PATH,
    TARGET_DATA_FILE_NAME,
)
from .utils import BedFormat, BedFormat_to_BedTool, BedTool_to_BedFormat


class CNVGenerator:
    """
    CNV simulator.
    With given reference genome this function
    generate random CNV's across the chronosomes.

    Args:
        fasta_file: reference genome in fasta format
        window_size: size of genomic window for CNV's
        max_cnv_length: maximum length of CNV's
        min_cnv_gap: minimum gap between CNV's
        N_percentage: percentage of N's in sequence
        pathout: folder where to store output files
    """

    def __init__(
        self,
        window_size: int,
        max_cnv_length: int,
        min_cnv_gap: int,
        N_percentage: float,
        fasta_file: str = REF_FASTA_FILE,
        pathout: str = SIM_DATA_PATH,
    ) -> None:
        self.fasta_file = fasta_file
        self.pathout = pathout
        self.target_data_file_name = TARGET_DATA_FILE_NAME
        self.window_size = window_size
        self.max_cnv_length = max_cnv_length
        self.min_cnv_gap = min_cnv_gap
        self.N_percentage = N_percentage
        if not os.path.exists(self.pathout):
            os.makedirs(self.pathout)
        fasta = SeqIO.parse(open(self.fasta_file), "fasta")
        self.num_of_chroms: int = len(tuple(fasta))

    def generate_cnv(self) -> BedFormat:
        """
        Main method to generate CNV's.
        First it creates BedFormat objects with coordinates of CNV's.
        Then it intersect duplications and deletions to avoid overlapping.
        Then it generates frequencies for duplications and deletions.
        Then it creates normal sequence without CNV's.
        At the end it creates total file with duplications, deletions and normal sequence.

        Returns:
            BedFormat: BedFormat object with all CNV's and normal sequence
        """
        dup, dele, chr_info = self._clean_bed_dup_del()
        dup_freq, dele_freq = self._generate_freq(dup, dele)
        normal = self._bedfile_with_normal_seq(dup_freq, dele_freq, chr_info)
        total_BedFormat, total_Bedtool = self._create_total_bed(
            dup_freq, dele_freq, normal
        )
        self._create_total_windows(total_Bedtool, chr_info)
        return total_BedFormat

    def modify_fasta_file(self, total_file: np.array) -> None:
        """
        Method to modify fasta file.
        It generates fasta file with CNV's.
        Args:
            total_file: BedFormat object with all CNV's

        Returns:
            str: path to fasta file with CNV's
        """
        fasta_original = {
            fasta.id: fasta.seq for fasta in SeqIO.parse(open(self.fasta_file), "fasta")
        }
        fasta_modified = {
            fasta.id: "" for fasta in SeqIO.parse(open(self.fasta_file), "fasta")
        }
        for line in tqdm(total_file, total=len(total_file)):
            if line.cnv_type == "del":
                continue
            elif line.cnv_type == "normal":
                fasta_modified[line.chr] += fasta_original[line.chr][
                    line.start : line.end
                ]
            elif line.cnv_type == "dup":
                seq_to_copy = fasta_original[line.chr][line.start : line.end]
                str_modified = seq_to_copy * line.freq
                fasta_modified[line.chr] += str_modified
        self._save_fasta_CNV(fasta_modified)

    def _clean_bed_dup_del(self) -> tuple[BedTool, BedTool, np.array]:
        """
        Method to clean bed files.
        It sorts and merges duplications and deletions.
        Then it intersects duplications and deletions to avoid overlapping.


        Returns:
            tuple[BedTool, BedTool, np.array]: Returns duplications, deletions and chromosome info
        """
        dup, dele, chr_info = self._create_bed_with_coords()
        logging.info("Sorting and merging deletions")
        dele_merged = self._cnv_sort_and_merge(dele)
        logging.info("Sorting and merging duplications")
        dup_merged = self._cnv_sort_and_merge(dup)
        logging.info("Intersecting duplications and deletions")
        dup_intersect, dele_intersect = self._cnv_intersect(dup_merged, dele_merged)
        return dup_intersect, dele_intersect, chr_info

    def _create_bed_with_coords(
        self,
    ) -> tuple[np.array, np.array, np.array]:
        """
        Method to create bed files with coordinates of CNV's using SeqIO.

        Returns:
            tuple[np.array, np.array, np.array]: Returns duplications, deletions and chromosome info
        """
        dup = np.array([])
        dele = np.array([])
        chr_info = np.array([])
        logging.info("Creating bed files with coordinates of CNV's")
        for chr in tqdm(
            SeqIO.parse(open(self.fasta_file), "fasta"),
            total=self.num_of_chroms,
            desc="Creating bed files",
        ):
            dup = np.append(dup, np.array([self.__generate_coords(chr)]))
            dele = np.append(dele, np.array([self.__generate_coords(chr)]))
            chr_info = np.append(chr_info, np.array([self.__chromosome_info(chr)]))

        return dup, dele, chr_info

    def _create_total_windows(self, total_bedtool, chr_info):
        chr_info_bed = BedFormat_to_BedTool(chr_info)
        BedTool.window_maker(
            chr_info_bed,
            b=total_bedtool,
            w=self.window_size - 1,
            s=self.window_size,
            i="src",
        ).saveas("/".join([self.pathout, self.target_data_file_name]))

    def _generate_freq(self, dup: BedTool, dele: BedTool) -> tuple[np.array, np.array]:
        """
        Method to generate frequencies for duplications and deletions.
        It generates random frequencies for duplications and deletions.
        Frequencies are for duplications from 2 to 10 and for deletions 0.
        They are responsible for how many times CNV is repeated.

        Args:
            dup: BedFormat object with duplications
            dele: BedFormat object with deletions

        Returns:
            tuple[np.array, np.array]: tuple of BedFormat objects with duplications and deletions
        """
        logging.info("Generating duplications frequencies")
        dup_with_freq: np.array = self._create_dup_freqs(dup)
        logging.info("Generating deletions frequencies")
        dele_with_freq: np.array = self._create_dele_freqs(dele)
        return dup_with_freq, dele_with_freq

    def _bedfile_with_normal_seq(
        self, dup: np.array, dele: np.array, chr_info: np.array
    ) -> np.array:
        """
        Method to create normal sequence.
        It creates normal sequence without CNV's.
        To do this it uses BedTools to create windows and then intersect duplications and deletions to avoid overlapping.
        Next it sorts and merges normal sequence.
        Finally it creates BedFormat object with normal sequence.

        Args:
            dup: BedFormat object with duplications
            dele: BedFormat object with deletions
            chr_info: BedFormat object with chromosome info

        Returns:
            np.array: BedFormat object with normal sequence
        """
        logging.info("Creating normal sequence")
        chr_info_bed = BedFormat_to_BedTool(chr_info)
        normal = BedTool().window_maker(chr_info_bed, w=self.window_size)
        dup_bedtools = BedFormat_to_BedTool(dup)  # convert to bedtools for intersection
        dele_bedtools = BedFormat_to_BedTool(dele)
        normal2 = (
            normal.intersect(dup_bedtools, v=True, wa=True)
            .intersect(dele_bedtools, v=True, wa=True)
            .sort()
            .merge()
        )  # after this step we have only normal genomic windows without CNV's
        out_normal = np.array([])
        for line in normal2:
            out_normal = np.append(
                out_normal, BedFormat(line.chrom, line.start, line.end, "normal", 1)
            )
        return out_normal

    def _create_total_bed(
        self, dup: np.array, dele: np.array, normal: np.array
    ) -> tuple[np.array, BedTool]:
        """
        Method to create total bed file.
        It concatenates duplications, deletions and normal sequence.
        Then it sorts and merges total bed file.
        Finally it creates BedFormat object with total bed file.

        Args:
            dup: BedFormat object with duplications
            dele: BedFormat object with deletions
            normal: BedFormat object with normal sequence

        Returns:
            tuple[np.array[BedFormat], BedTool]: tuple of BedFormat and BedTool object with sorted and merged total bed file
                                                BedFormat is neccessary for next steps like modify fasta file
                                                BedTool is neccessary for creating windows and save file as target file for training
        """
        logging.info("Creating total bed file")
        total = np.concatenate((dup, dele, normal), axis=0)
        total_Bedtool = BedFormat_to_BedTool(total).sort()
        total_BedFormat = BedTool_to_BedFormat(total_Bedtool)
        return total_BedFormat, total_Bedtool

    def _create_dele_freqs(self, dele: np.array) -> np.array:
        """
        Method to create deletions with frequencies.
        It generates frequencies for deletions which are always 0.

        Args:
            dele: BedFormat object with deletions

        Returns:
            np.array: BedFormat object with deletions and frequencies
        """
        out = np.array([])
        for line in dele:
            out = np.append(
                out,
                np.array(BedFormat(line.chrom, line.start, line.end, "del", 0)),
            )
        return out

    def _create_dup_freqs(self, dup: np.array) -> np.array:
        """
        Method to create duplications with frequencies.
        It generates frequencies for duplications from 2 to 10 with probabilities:
        2 - 50%
        3 to 5 - 10%
        6 to 10 - 5%
        Frequencies are responsible for how many times CNV is repeated.

        Args:
            dup: BedFormat object with duplications

        Returns:
            np.array: BedFormat object with duplications and frequencies
        """
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
        """
        Method to sort and merge CNV's using BedTools.

        Args:
            cnv: BedFormat object with CNV's

        Returns:
            BedTool: BedTool object with sorted and merged CNV's
        """
        cnv_bed = BedFormat_to_BedTool(cnv)
        cnv_merged = cnv_bed.sort().merge()
        return cnv_merged

    def _cnv_intersect(self, dup: BedTool, dele: BedTool) -> tuple[BedTool, BedTool]:
        """
        Method to intersect duplications and deletions using BedTools.

        Args:
            dup: BedTool object with duplications
            dele: BedTool object with deletions

        Returns:
            tuple[BedTool, BedTool]: tuple of BedTool objects with duplications and deletions
        """
        dup_intersect = dup.intersect(dele, wa=True, v=True)
        dele_intersect = dele.intersect(dup, wa=True, v=True)
        return dup_intersect, dele_intersect

    def __generate_coords(self, chr: SeqRecord.SeqRecord) -> np.array:
        """
        Method to generate coordinates of CNV's.
        It generates random coordinates of CNV's and checks if they are overlapping.
        If they are overlapping it generates new coordinates.

        Args:
            chr: reference chromosome in fasta format

        Returns:
            np.array: array with BedFormat as values
        """
        len_fasta: int = len(chr.seq)
        if self.__max_cnv_lenght_too_large(len_fasta):
            raise Exception("CNV's are longer than maximum length")
        cnv_count = int(
            (len_fasta / self.max_cnv_length) / 2
        )  # number of cnv that can fit in data devided by two
        # because there are two types of cnv (duplications and deletions)
        lenghts: list = sorted(
            [
                random.randrange(
                    self.window_size, self.max_cnv_length, step=self.window_size
                )
                for _ in range(cnv_count)
            ],
            reverse=True,
        )
        if self.__too_large_cnvs_number(len_fasta, lenghts, cnv_count):
            raise Exception(
                "CNV's are too long, try smaller max_cnv_length or smaller min_cnv_gap"
            )

        start, end = self.__find_CNV_windows(chr, lenghts)
        ids = [chr.id] * cnv_count
        return np.array([BedFormat(id, st, en) for id, st, en in zip(ids, start, end)])

    def __max_cnv_lenght_too_large(self, len_fasta: int) -> bool:
        """
        Method to check if CNV's are not longer than maximum length.

        Args:
            chr: reference chromosome in fasta format

        Returns:
            bool: True if CNV's are not longer than maximum length, False otherwise
        """
        if len_fasta <= self.max_cnv_length:
            return True
        else:
            return False

    def __too_large_cnvs_number(
        self, len_fasta: int, lenghts: list, cnv_count: int
    ) -> bool:
        """
        Check if all cnv's length can fit in data with min_cnv_gap between them.

        Args:
            lenghts: list of lengths of cnv's
            cnv_count: number of cnv's

        Returns:
            bool: True if all cnv's can fit in data, False otherwise
        """
        total_cnv_length = sum(lenghts)
        if total_cnv_length + (cnv_count * self.min_cnv_gap) >= len_fasta:
            return True
        else:
            return False

    def __find_CNV_windows(
        self, chr: SeqRecord.SeqRecord, lenghts: list
    ) -> tuple[list[int], list[int]]:
        """
        Method to find CNV windows.
        It generates random coordinates of CNV's and checks if they are overlapping.
        If they are overlapping it generates new coordinates.

        Args:
            chr: reference chromosome in fasta format
            lenghts: list of lengths of CNV's

        Returns:
            tuple[list[int], list[int]]: tuple of lists with start and end coordinates of CNV's
        """
        starts: list = []
        ends: list = []
        for i in lenghts:
            wrong_cnv_coords = True
            n = 0
            while wrong_cnv_coords:
                start = random.randrange(1, len(chr.seq) - i, step=self.window_size)
                stop = start + i
                if self.__check_overlaping(
                    start, stop, starts, ends
                ) and self.__check_occur(chr, start, stop):
                    wrong_cnv_coords = False

                n += 1
                if n > 200:
                    raise RuntimeError("Could not find CNV coordinates")
            starts.append(start)
            ends.append(stop)
        return starts, ends

    def __check_occur(self, chr: SeqRecord.SeqRecord, start: int, stop: int) -> bool:
        """
        Method to check if there are more than 70% of N's in sequence.

        Args:
            chr: reference chromosome in fasta format
            start: start coordinate of CNV
            stop: stop coordinate of CNV

        Returns:
            bool: True if there are less than 70% of N's in sequence, False otherwise
        """
        if CNVGenerator.__get_N_percentage(chr.seq[start:stop]) < self.N_percentage:
            return True
        else:
            return False

    def __check_overlaping(
        self, start: int, stop: int, starts: list, ends: list
    ) -> bool:
        """
        Method to check if CNV's are overlapping.

        Args:
            start: start coordinate of CNV
            stop: stop coordinate of CNV
            starts: list of start coordinates of CNV's
            ends: list of stop coordinates of CNV's

        Returns:
            bool: True if CNV's are not overlapping, False otherwise
        """
        for start_taken, stop_taken in zip(starts, ends):
            if (start - stop_taken >= self.min_cnv_gap) or (
                start_taken - stop >= self.min_cnv_gap
            ):
                continue  # CNVs are non-overlapping or have a sufficient gap
            else:
                return False  # CNVs are overlapping
        return True  # All CNVs are non-overlapping

    @staticmethod
    def __get_N_percentage(seq) -> float:
        """
        Method to get percentage of N's in sequence.

        Args:
            seq: sequence to check

        Returns:
            float: percentage of N's in sequence
        """
        total = len(seq)
        N_num = Counter(seq).get("N", 0)
        return N_num / total

    def __chromosome_info(self, fasta_file) -> np.array:
        len_fasta = len(fasta_file.seq)
        return np.array([BedFormat(fasta_file.id, 1, len_fasta)])

    def _save_fasta_CNV(self, fasta_modified: dict) -> None:
        """
        Method to save fasta file with CNV's.

        Args:
            fasta_modified: dictionary with fasta id as key and fasta sequence as value

        Returns:
            str: path to fasta file with CNV's
        """

        if os.path.exists(MODIFIED_FASTA_FILE_PATH):
            os.remove(MODIFIED_FASTA_FILE_PATH)

        for id, fasta_str in fasta_modified.items():
            with open(MODIFIED_FASTA_FILE_PATH, "a") as fasta_cnv:
                fasta_cnv.write(f">{id}\n{fasta_str}\n")

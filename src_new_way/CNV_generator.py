from dataclasses import dataclass
from multiprocessing.spawn import prepare
import pandas as pd
import numpy as np
from Bio import SeqIO
import pybedtools
import random
import os
from tqdm import tqdm
import timeit

@dataclass
class BedFormat:
    chr: str
    start: int
    end: int
    cnv_type: str = None
    cov: float = None
    freq: float = None

class CNVGenerator:
    def __init__(self, fasta_file: str, pathout: str = "train/") -> None:
        self.fasta_file = fasta_file
        self.pathout = pathout
        isExist = os.path.exists(self.pathout)
        if not isExist:
            os.makedirs(self.pathout)
        fasta = SeqIO.parse(open(self.fasta_file), "fasta")
        self.num_of_chroms = len(tuple(fasta))

    def generate_cnv(self):
        chr_info = self.bedfile_chr_info()
        cnvs = self.bedfile_dup_del()

    def bedfile_chr_info(self):
        chr_info = np.array([])
        for chr in tqdm(SeqIO.parse(open(self.fasta_file), "fasta"), total=self.num_of_chroms):
            chr_info = np.append(chr_info, self.chromosome_info(chr))
        return chr_info


    def bedfile_dup_del(self,) -> None:
        for cnv_type in "del", "dup":
            print(f"Creating {cnv_type}...")
            out = np.array([])
            for chr in tqdm(SeqIO.parse(open(self.fasta_file), "fasta"), total=self.num_of_chroms):
                out = np.append(np.array([out,self.generate_coords(chr)]))
        return out

    def cnv_sort_and_merge(self, cnv_type):
        cnv = pybedtools.BedTool(self.pathout + cnv_type +'.bed')
        cnv_merged = cnv.sort().merge()
        return cnv_merged
    
    def cnv_intersect(self, cnv_merged, cnv_other):
        cnv_clean = cnv_merged.intersect(cnv_other, wa = True, v = True)
        return cnv_clean
    
    def bedtools_dup_del(self, cnv_type: str) -> tuple[pybedtools.bedtool.BedTool,pybedtools.bedtool.BedTool]:
        if cnv_type not in ["dup", "dele"]:
            raise ValueError("Wrong CNV type. Try 'dup' or 'dele'")

        dele = self.cnv_sort_and_merge("del")
        dup = self.cnv_sort_and_merge("dup")
        if cnv_type == "dup":
            return self.cnv_intersect(dup, dele)
        elif cnv_type == "dele":
            return self.cnv_intersect(dele, dup)

    def generate_coords(self, fasta_file):
        str_fasta = str(fasta_file.seq)
        len_fasta = len(str_fasta)
        lenghts = []
        max_cnv_length = 1e6
        cnv_count = int((len_fasta/max_cnv_length)/2)  #number of cnv that can fit in data devided by two because there are two types of cnv (duplications and deletions)
        while len(lenghts) < cnv_count:
            cnv_range = random.randint(50,max_cnv_length)
            lenghts.append(cnv_range)
        # dodać tutaj test -> assert, że suma tych dwóch list będzie mniejsza od długości całego genomu
        # he = (np.sum(dup_lengths) + np.sum(del_lengths))/len_fasta
        # print(f"Duplikacje i delecje stanowią {he}% całego genomu")
        start = np.random.randint(1,len_fasta, size=(cnv_count))
        end = [st + lgt for st, lgt in zip(start, lenghts)]
        ids = list(fasta_file.id)*cnv_count
        return np.array([BedFormat(id, st, en) for id,st,en in zip(ids, start, end)])

    def chromosome_info(self, fasta_file) -> BedFormat:
        str_fasta = str(fasta_file.seq)
        len_fasta = len(str_fasta)
        return np.array([BedFormat(fasta_file.id,"1",len_fasta)])

    # def write_chromosome_info(self, chr_info):
    #     chr_info_str = "\n".join([x for x in chr_info])
    #     with open(self.pathout + "chrs.bed", "w") as chr_file:
    #         chr_file.write(chr_info_str)

    def generate_freq(self):
        chr_freq = {i: round(i/self.num_of_chroms,3) for i in range(1,self.num_of_chroms+1)}
        output = np.array([])
        for cnv_type in ["del","dup"]:
            out = np.array([])
            for line in self.bedtools_dup_del(cnv_type):
                if cnv_type == "del":
                    num = random.randint(1,self.num_of_chroms)
                    out = np.append(out, np.array(BedFormat(line.chrom, line.start, line.end, "del", chr_freq[num], 0.0)))
                elif cnv_type == "dup":
                    num = random.randint(1,self.num_of_chroms)
                    count = np.random.choice(list(range(2,11)), 1, p=[0.5, 0.1, 0.1, 0.05, 0.05,0.05,0.05,0.05,0.05])[0]
                    out = np.append(out, np.array(BedFormat(line.chrom, line.start, line.end, "dup", chr_freq[num], count)))
            output = np.append(output, out)
        return output

    def bedfile_with_normal_seq(self, dup, dele):
        for i in self.num_of_chroms:
            print("Creating bedfiles for sample " + str(i))
            normal = pybedtools.BedTool().window_maker(b=self.pathOut + "chrs.bed", w = 5)
            normal2 = normal.intersect(dup, v=True, wa=True).intersect(dele, v=True, wa=True).sort().merge()
            out = np.array([])
            for line in normal2:
                out = np.append(out, BedFormat(line.chrom, line.start, line.end, "normal", 1, 1))
            return out

    def create_total_bed(self, dup, dele, normal):
        total = np.append(dup, dele, normal)
        total_sorted = pybedtools.BedTool(self.prepare_data_for_BedTool(total)).sort()
        return total_sorted

    def prepare_data_for_BedTool(self, seq: np.array) -> tuple:
        for line in seq:
            yield (line.chr, line.start, line.end, line.cnv_type, line.cov, line.freq)

    # def modify_fasta_file(self):
    #     os.system("cp " + "fasta.fa" + " " + pathOut + "train" + "_noCNV.fa")
    #     chrs = []
    #     chr = {}
    #     chr2 = {}
    #     for r in SeqIO.parse(open(pathOut + "train" + "_noCNV.fa"),"fasta"):
    #         chrs.append(r.id)
    #         chr[r.id] = str(r.seq)
    #         chr2[r.id] = ""
    #     for line in open("train/total.1.bed"):
    #         if line.split()[3].rstrip() == "normal":
    #             chr2[line.split()[0]] += chr[line.split()[0]][int(line.split()[1]):int(line.split()[2])]
    #         elif line.split()[3].rstrip() == "del":
    #             pass
    #         elif line.split()[3].rstrip() == "dup":
    #             if float(line.split()[-1].rstrip()) > 1.5:
    #                 for v in range(0,int(line.split()[-1].rstrip())):
    #                     chr2[line.split()[0]] += chr[line.split()[0]][int(line.split()[1]):int(line.split()[2])]
    #             else:
    #                 chr2[line.split()[0]] += chr[line.split()[0]][int(line.split()[1]):int(line.split()[2])]
    #                 chr2[line.split()[0]] += chr[line.split()[0]][int(line.split()[1]):int(line.split()[2])]
    #     for i in chrs:
    #         out = open(pathOut + i + "_" + "train" + "_CNV.fa","w")
    #         out.write(">" + i + "\n" + chr2[i] + "\n")
    #     out.close()

if __name__ == "__main__":
    cnv = CNVGenerator("reference_genome/ref_genome_short.fa")
    dele = cnv.cnv_sort_and_merge("del")
    dup = cnv.cnv_sort_and_merge("dup")
    dup_intersect = cnv.cnv_intersect(dup, dele)


"""
PO WSZYSTKIM ZOSTAJE PLIK DEL.1.bed, dup.1.bed, total.1.bed
gdzie tak naprawdę tylko total1.bed jest używany w simChr.py - więc on jest tylko wykorzystywany
mogę tę strukturę total1.bed wykorzystać jako dataframe i wtedy go zapisać w pamięci, zamiast zapisywać w pliku
Tyczy się tekstu na górze:
Zamiast zachowywać to w padnasie zachowuj to w arrayu i wtedy będę mógł to przyspieszyć z numbą.
Generowanie tej frekwencji nic kompletnie nie daje. Można to zrobić od razu z modyfikacją plikuuuu fasta
"""
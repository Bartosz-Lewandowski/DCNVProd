from dataclasses import dataclass
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
    cnv_type: str
    cov: float
    freq: float

class CNVGenerator:
    def __init__(self, fasta_file: str, pathout: str = "train/") -> None:
        self.fasta_file = fasta_file
        self.pathout = pathout
        isExist = os.path.exists(self.pathout)
        if not isExist:
            os.makedirs(self.pathout)
        fasta = SeqIO.parse(open(self.fasta_file), "fasta")
        self.num_of_chroms = len(tuple(fasta))
        
    def generate_cnv(self,) -> None:
        chr_info = []
        for cnv_type in "del", "dup":
            print(f"Creating {cnv_type}...")
            df = pd.DataFrame()
            for chr in tqdm(SeqIO.parse(open(self.fasta_file), "fasta"), total=self.num_of_chroms):
                chr_info.append(self.chromosome_info(chr))
                df = pd.concat([df,self.generate_coords(chr)])
            df.to_csv(self.pathout + cnv_type + ".bed", header=False, index=False, sep = "\t")
        chr_info_wo_duplicates = list(set(chr_info))
        self.write_chromosome_info(sorted(chr_info_wo_duplicates))

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
        cnv = pd.DataFrame({1:[fasta_file.id]*cnv_count,2:start,3:end,4:lenghts})
        return cnv 

    def chromosome_info(self, fasta_file):
        str_fasta = str(fasta_file.seq)
        len_fasta = len(str_fasta)
        return "\t".join([fasta_file.id,"1",str(len_fasta)])

    def write_chromosome_info(self, chr_info):
        chr_info_str = "\n".join([x for x in chr_info])
        with open(self.pathout + "chrs.bed", "w") as chr_file:
            chr_file.write(chr_info_str)

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

    def remove_overlaps(self):
        for i in self.num_of_chroms:
            dele, dup = self.generate_freq()
            print("Creating bedfiles for sample " + str(i))
            normal = pybedtools.BedTool().window_maker(b=self.pathOut + "chrs.bed", w = 5)

            os.system(bedtools_path + " intersect -v -wa -a " + pathOut + "normal." + str(i) + ".bed -b " + pathOut + "dup." + str(i) + ".bed | " + 
                        bedtools_path + " intersect -v -wa -a stdin -b " + pathOut + "del." + str(i) + ".bed | " + 
                        bedtools_path + " sort -i stdin | " + bedtools_path + " merge -i stdin > " + pathOut + "normal2." + str(i) + ".bed")

            normal2 = normal.intersect(dup, v=True, wa=True).intersect(dele, )

            out = open(pathOut + "normal3." + str(i) + ".bed","w")
            for line in open(pathOut + "normal2." + str(i) + ".bed"):
                out.write(line.rstrip() + "\tnormal\t1\t1\n")
            out.close()

    # def create_total_bed(self):
    #     os.system("cat " + pathOut + "normal3." + str(i) + ".bed " + pathOut + "dup." + str(i) + ".bed " + pathOut + "del." + str(i) + ".bed | " + bedtools_path + " sort -i stdin > " + pathOut + "total." + str(i) + ".bed")

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
    for line in dup_intersect:
        print(line)
        break


"""
PO WSZYSTKIM ZOSTAJE PLIK DEL.1.bed, dup.1.bed, total.1.bed
gdzie tak naprawdę tylko total1.bed jest używany w simChr.py - więc on jest tylko wykorzystywany
mogę tę strukturę total1.bed wykorzystać jako dataframe i wtedy go zapisać w pamięci, zamiast zapisywać w pliku
Tyczy się tekstu na górze:
Zamiast zachowywać to w padnasie zachowuj to w arrayu i wtedy będę mógł to przyspieszyć z numbą.
Generowanie tej frekwencji nic kompletnie nie daje. Można to zrobić od razu z modyfikacją plikuuuu fasta
"""
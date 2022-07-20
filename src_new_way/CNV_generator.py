import pandas as pd
import numpy as np
from Bio import SeqIO
import pybedtools
import random
import os

class CNVGenerator:
    def __init__(self, fasta_file: str, pathout: str = "train/") -> None:
        self.fasta_file = fasta_file
        self.pathout = pathout
        isExist = os.path.exists(self.pathout)
        if not isExist:
            os.makedirs(self.pathout)

        self.num_of_chroms = 1
        
    def generate_cnv(self,) -> None:
        for cnv_type in "del", "dup":
            df = pd.DataFrame()
            for chr in SeqIO.parse(open(self.fasta_file), "fasta"):
                self.write_chromosome_info(chr)
                df = pd.concat([df,self.generate_coords(chr)])
            df.to_csv(self.pathout + cnv_type + ".bed", header=False, index=False, sep = "\t")

    def cnv_sort_merge_intersect(self, cnv_type):
        cnv = pybedtools.BedTool(self.pathout + cnv_type +'.bed')
        cnv_merged = cnv.sort().merge()
        cnv_clean = cnv_merged.intersect(cnv, wa = True, v = True)
        return cnv_clean
        
    def generate_coords(self, fasta_file):
        str_fasta = str(fasta_file.seq)
        len_fasta = len(str_fasta)
        lenghts = []
        max_cnv_length = 1e3
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

    def write_chromosome_info(self, fasta_file):
        str_fasta = str(fasta_file.seq)
        len_fasta = len(str_fasta)
        with open(self.pathout + "chrs.bed", "w") as chr_file:
            chr_info = "\t".join([fasta_file.id,"1",str(len_fasta)]) + "\n"
            chr_file.write(chr_info)

if __name__ == "__main__":
    cnv = CNVGenerator("fasta.fa")
    cnv.generate_cnv()
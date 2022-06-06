bedtools_path = "/usr/bin/bedtools"
FASTA="Sus_scrofa.Sscrofa11.1.dna.chromosome.1.fa"

import pandas as pd
import numpy as np
from Bio import SeqIO
import random
import os
df_del = pd.DataFrame(columns = [1,2,3,4])
df_dup = pd.DataFrame(columns = [1,2,3,4])
train_path = "train/"
isExist = os.path.exists(train_path)

if not isExist:
  os.makedirs(train_path)
pathOut = "train/"
if pathOut != "" and pathOut.endswith("/") == False:
    pathOut += "/"
out = open(pathOut + "chrs.bed","w")
print("Generating duplication and deletion coordinates")
for fasta_file in SeqIO.parse(open(FASTA),"fasta"):
    #first line is fasta ID, 1(?), length of fasta file
    first_line = "\t".join([fasta_file.id,"1",str(len(str(fasta_file.seq)))]) + "\n"
    out.write(first_line)
    dup_lengths = []
    del_lengths = []
    #why 1000000 in cnv count? -> because we calculate it for megabase 
    max_cnv_length = 1e5
    cnv_count = int((len(str(fasta_file.seq))/max_cnv_length)/2)  #number of cnv that can fit in data devided by two because there are two types of cnv (duplications and deletions)
    while len(dup_lengths) < cnv_count:
        cnv_range = random.randint(50,100000)
        dup_lengths.append(cnv_range)
    while len(del_lengths) < cnv_count:
        cnv_range = random.randint(50,100000)
        del_lengths.append(cnv_range)
    # dodać tutaj test -> assert, że suma tych dwóch list będzie mniejsza od długości całego genomu
    dup_start = np.random.randint(1,len(str(fasta_file.seq)), size=(1, cnv_count))[0]
    del_start = np.random.randint(1,len(str(fasta_file.seq)), size=(1, cnv_count))[0]
    dup_ends = [int(a + b) for a, b in zip(dup_start, dup_lengths)]
    del_ends = [int(a + b) for a, b in zip(del_start, del_lengths)]
    dups = pd.DataFrame({1:[fasta_file.id]*cnv_count,2:dup_start,3:dup_ends,4:dup_lengths})
    dels = pd.DataFrame({1:[fasta_file.id]*cnv_count,2:del_start,3:del_ends,4:del_lengths})
    #tutaj zmienić ten append
    df_dup = df_dup.append(dups)
    df_del = df_del.append(dels)
out.close()
df_dup.to_csv(pathOut + "dup.bed",header=False,index=False,sep="\t")
df_del.to_csv(pathOut + "del.bed",header=False,index=False,sep="\t")
os.system(bedtools_path + " sort -i " + pathOut + "dup.bed | " + bedtools_path + " merge -i stdin > " + pathOut + "dup2.bed")
os.system(bedtools_path + " sort -i " + pathOut + "del.bed | " + bedtools_path + " merge -i stdin > " + pathOut + "del2.bed")
os.system("cp "+ pathOut + "del2.bed "+ pathOut + "del3.bed")
os.system("cp "+ pathOut + "dup2.bed "+ pathOut + "dup3.bed")
os.system(bedtools_path + " intersect -wa -v -a " + pathOut + "dup3.bed -b " + pathOut + "del3.bed > " + pathOut + "dup4.bed")
os.system(bedtools_path + " intersect -wa -v -a " + pathOut + "del3.bed -b " + pathOut + "dup3.bed > " + pathOut + "del4.bed")
no_chrs = list(range(1, int(1)+1))
chr_freq = {}
for i in no_chrs:
    chr_freq[i] = i/1
no_chrs = list(range(1, int(1)+1))
chr_freq = {}
print("Generating duplication and deletion frequencies")
print(no_chrs)
for i in no_chrs:
    chr_freq[i] = round(i/1,3)
for i in ["del","dup"]:
    out = open(pathOut + str(i) + "5.bed","w")
    for line in open(pathOut + i + "4.bed"):
        if i == "del":
            num = random.randint(1,1)
            out.write(line.rstrip() + "\tdel\t" + str(chr_freq[num]) + "\t0\n")
        elif i == "dup":
            num = random.randint(1,1)
            count = np.random.choice([2,3,4,5,6,7,8,9,10], 1, p=[0.5, 0.1, 0.1, 0.05, 0.05,0.05,0.05,0.05,0.05])[0]
            freqs = num/1
            cp = (count*freqs) + ((1-freqs) * 1)
            while cp == 1.0:
                num = random.randint(1,1)
                count = np.random.choice([2,3,4,5,6,7,8,9,10], 1, p=[0.5, 0.1, 0.1, 0.05, 0.05,0.05,0.05,0.05,0.05])[0]
            out.write(line.rstrip() + "\tdup\t" + str(chr_freq[num]) + "\t" + str(count) + "\n")
    out.close()
    for j in chr_freq:
        out = open(pathOut + i + "." + str(j) + ".bed","w")
        for line in open(pathOut + i + "5.bed"):
            if float(line.split()[4]) >= chr_freq[j]:
                out.write(line)
        out.close()
print("Removing overlaps, generating total file")
for i in no_chrs:
    print("Creating bedfiles for sample " + str(i))
    os.system("bedtools makewindows -b " + pathOut + "chrs.bed -w 5 > " + pathOut + "normal." + str(i) + ".bed")
    os.system(bedtools_path + " intersect -v -wa -a " + pathOut + "normal." + str(i) + ".bed -b " + pathOut + "dup." + str(i) + ".bed | " + bedtools_path + " intersect -v -wa -a stdin -b " + pathOut + "del." + str(i) + ".bed | " + bedtools_path + " sort -i stdin | " + bedtools_path + " merge -i stdin > " + pathOut + "normal2." + str(i) + ".bed")
    out = open(pathOut + "normal3." + str(i) + ".bed","w")
    for line in open(pathOut + "normal2." + str(i) + ".bed"):
        out.write(line.rstrip() + "\tnormal\t1\t1\n")
    out.close()
    os.system("cat " + pathOut + "normal3." + str(i) + ".bed " + pathOut + "dup." + str(i) + ".bed " + pathOut + "del." + str(i) + ".bed | " + bedtools_path + " sort -i stdin > " + pathOut + "total." + str(i) + ".bed")
    os.remove(pathOut + "normal3." + str(i) + ".bed")
    os.remove(pathOut + "normal2." + str(i) + ".bed")
    os.remove(pathOut + "normal." + str(i) + ".bed")
os.remove(pathOut + "del.bed")
os.remove(pathOut + "del2.bed")
os.remove(pathOut + "del3.bed")
os.remove(pathOut + "del4.bed")
os.remove(pathOut + "del5.bed")
os.remove(pathOut + "dup.bed")
os.remove(pathOut + "dup2.bed")
os.remove(pathOut + "dup3.bed")
os.remove(pathOut + "dup4.bed")
os.remove(pathOut + "dup5.bed")
os.remove(pathOut + "chrs.bed")
import pandas as pd
import numpy as np
from Bio import SeqIO
import pybedtools
import random
import os

bedtools_path = "/usr/bin/bedtools"
FASTA="fasta.fa"

pathOut = "train/"
isExist = os.path.exists(pathOut)
if not isExist:
  os.makedirs(pathOut)

df_del = pd.DataFrame(columns = [1,2,3,4])  #? 
df_dup = pd.DataFrame(columns = [1,2,3,4])  #?

out = open(pathOut + "chrs.bed","w")
print("Generating duplication and deletion coordinates")
for fasta_file in SeqIO.parse(open(FASTA),"fasta"):
    print(type(fasta_file))
    #first line is fasta ID, 1(?), length of fasta file
    first_line = "\t".join([fasta_file.id,"1",str(len(str(fasta_file.seq)))]) + "\n"
    out.write(first_line)
    dup_lengths = []
    del_lengths = []
    max_cnv_length = 1e3
    cnv_count = int((len(str(fasta_file.seq))/max_cnv_length)/2)  #number of cnv that can fit in data devided by two because there are two types of cnv (duplications and deletions)
    while len(dup_lengths) < cnv_count:
        cnv_range = random.randint(50,max_cnv_length)
        dup_lengths.append(cnv_range)
    while len(del_lengths) < cnv_count:
        cnv_range = random.randint(50,max_cnv_length)
        del_lengths.append(cnv_range)
    # dodać tutaj test -> assert, że suma tych dwóch list będzie mniejsza od długości całego genomu
    he = (np.sum(dup_lengths) + np.sum(del_lengths))/len(str(fasta_file.seq))
    print(f"Duplikacje i delecje stanowią {he}% całego genomu")
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
dup = pybedtools.BedTool(pathOut + 'dup.bed')
dup2 = dup.sort().merge()
dele = pybedtools.BedTool(pathOut + 'del.bed')
dele2 = dele.sort().merge()
dup2.intersect(dele, wa = True, v = True).saveas(pathOut + "dup2.bed")
dele2.intersect(dup, wa = True, v = True).saveas(pathOut + "del2.bed")
##########################################################################################################################

print("Generating duplication and deletion frequencies")
NUMBER_OF_CHROMOSOMES = 1
no_chrs = range(1, NUMBER_OF_CHROMOSOMES+1)
chr_freq = {}
for i in no_chrs:
    chr_freq[i] = round(i/1,3)

for i in ["del","dup"]:
    out = open(pathOut + i + "3.bed","w")
    for line in open(pathOut + i + "2.bed"):
        if i == "del":
            #1 bo dla jednego (i pierwszego) chromosomu chcemy zrobić modyfikację
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
        for line in open(pathOut + i + "3.bed"):
            if float(line.split()[4]) >= chr_freq[j]:
                out.write(line)
        out.close()


print("Removing overlaps, generating total file")
for i in no_chrs:
    print("Creating bedfiles for sample " + str(i))
    normal = pybedtools.BedTool().window_maker(b=pathOut + "chrs.bed", w = 5).saveas(pathOut + "normal." + str(i) + ".bed")

    os.system(bedtools_path + " intersect -v -wa -a " + pathOut + "normal." + str(i) + ".bed -b " + pathOut + "dup." + str(i) + ".bed | " + 
                bedtools_path + " intersect -v -wa -a stdin -b " + pathOut + "del." + str(i) + ".bed | " + 
                bedtools_path + " sort -i stdin | " + bedtools_path + " merge -i stdin > " + pathOut + "normal2." + str(i) + ".bed")

    #normal2 = pybedtools.BedTool(pathOut + "normal." + str(i) + ".bed").intersect(dup, v=True, wa=True).intersect(dele, )

    out = open(pathOut + "normal3." + str(i) + ".bed","w")
    for line in open(pathOut + "normal2." + str(i) + ".bed"):
        out.write(line.rstrip() + "\tnormal\t1\t1\n")
    out.close()
    os.system("cat " + pathOut + "normal3." + str(i) + ".bed " + pathOut + "dup." + str(i) + ".bed " + pathOut + "del." + str(i) + ".bed | " + bedtools_path + " sort -i stdin > " + pathOut + "total." + str(i) + ".bed")
    os.remove(pathOut + "normal3." + str(i) + ".bed")
    os.remove(pathOut + "normal2." + str(i) + ".bed")
    os.remove(pathOut + "normal." + str(i) + ".bed")
# os.remove(pathOut + "del.bed")
# os.remove(pathOut + "del2.bed")
# os.remove(pathOut + "del3.bed")
# os.remove(pathOut + "dup.bed")
# os.remove(pathOut + "dup2.bed")
# os.remove(pathOut + "dup3.bed")
# os.remove(pathOut + "chrs.bed")
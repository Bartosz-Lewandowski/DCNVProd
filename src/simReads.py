from Bio import SeqIO
import os
cov = 20
pathOut = "sim_train/"
isExist = os.path.exists(pathOut)
wgsim_path = "/home/bartek/anaconda3/bin/wgsim"
if not isExist:
    os.makedirs(pathOut)
chr_lens = {}
FASTA="fasta.fa"
for r in SeqIO.parse(open(FASTA),"fasta"):
    chr_lens[r.id] = len(str(r.seq))
for chr in chr_lens:
    READ_LENGTH = 100
    reads = round(chr_lens[chr]/(2*int(READ_LENGTH)))*int(cov)
    command = wgsim_path + " -N " + str(reads) + " -1 " + str(READ_LENGTH) + " -2 " + str(READ_LENGTH) + " " + pathOut + chr + "_" + "train" + "_CNV.fa " + pathOut + chr + "_1.fq " + pathOut + chr + "_2.fq > stdout"
    print(command)
    os.system(command)
for chr in chr_lens:
    os.system("cat " + pathOut + chr + "_1.fq >> " + pathOut + "train" + "_" + str(cov) + "_1.fq")
    os.system("cat " + pathOut + chr + "_2.fq >> " + pathOut + "train" + "_" + str(cov) + "_2.fq")
    os.remove(pathOut + chr + "_1.fq")
    os.remove(pathOut + chr + "_2.fq")
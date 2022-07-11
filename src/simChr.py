import os
pathOut = "sim_train/"
isExist = os.path.exists(pathOut)
if not isExist:
    os.makedirs(pathOut)
from Bio import SeqIO
import os
os.system("cp " + "fasta.fa" + " " + pathOut + "train" + "_noCNV.fa")
chrs = []
chr = {}
chr2 = {}
for r in SeqIO.parse(open(pathOut + "train" + "_noCNV.fa"),"fasta"):
    chrs.append(r.id)
    chr[r.id] = str(r.seq)
    chr2[r.id] = ""
for line in open("train/total.1.bed"):
    if line.split()[3].rstrip() == "normal":
        chr2[line.split()[0]] += chr[line.split()[0]][int(line.split()[1]):int(line.split()[2])]
    elif line.split()[3].rstrip() == "del":
        pass
    elif line.split()[3].rstrip() == "dup":
        if float(line.split()[-1].rstrip()) > 1.5:
            for v in range(0,int(line.split()[-1].rstrip())):
                chr2[line.split()[0]] += chr[line.split()[0]][int(line.split()[1]):int(line.split()[2])]
        else:
            chr2[line.split()[0]] += chr[line.split()[0]][int(line.split()[1]):int(line.split()[2])]
            chr2[line.split()[0]] += chr[line.split()[0]][int(line.split()[1]):int(line.split()[2])]
for i in chrs:
    out = open(pathOut + i + "_" + "train" + "_CNV.fa","w")
    out.write(">" + i + "\n" + chr2[i] + "\n")
out.close()
os.remove(pathOut + "train" + "_noCNV.fa")
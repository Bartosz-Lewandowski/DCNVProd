import os
import pandas as pd
import numpy as np
import math
from shutil import copyfile
pathOut = "train/"
WINDOW_SIZE = 50
WINDOW = 5 
INPUT = "output.bed"
OUTPUT = "results.bed"
DUPLICATION = "train/dup.1.bed"
DELETION = "train/del.1.bed"
STEP_SIZE = 50

bedtools_path = "/usr/bin/bedtools"
def roundup(x):
    return int(math.ceil(x / WINDOW_SIZE)) * WINDOW_SIZE
def rounddown(x):
    return int(math.floor(x / WINDOW_SIZE)) * WINDOW_SIZE
"""If ignoring TEs is required, due to their inherit weirdness with split reads/coverage, this removes windows with TE sequences."""
os.system("cp " + INPUT + " " + "train/dudeml_temp.bed")
del_cp = {}
dup_cp = {}
dup_temp_1 = open("dup_temp_1.bed","w")
del_temp_1 = open("del_temp_1.bed","w")
"""Reformat deletion and duplication windows to find overlapping windows with"""
for line in open(DUPLICATION):
    line = line.rstrip()
    cp = str((float(line.split()[5])*float(line.split()[4])) + ((1-float(line.split()[4])) * 1))
    dup_temp_1.write("\t".join([line.split()[0],str(rounddown(int(line.split()[1]))),str(roundup(int(line.split()[2]))),cp]) + "\n")
for line in open(DELETION):
    line = line.rstrip()
    cp = str((float(line.split()[5])*float(line.split()[4])) + ((1-float(line.split()[4])) * 1))
    del_temp_1.write("\t".join([line.split()[0],str(rounddown(int(line.split()[1]))),str(roundup(int(line.split()[2]))),cp]) + "\n")
dup_temp_1.close()
del_temp_1.close()
os.system(bedtools_path + " makewindows -b dup_temp_1.bed -w " + str(WINDOW_SIZE) + " -s " + str(STEP_SIZE) + " -i src > dup_temp_2.bed")
os.system(bedtools_path + " makewindows -b del_temp_1.bed -w " + str(WINDOW_SIZE) + " -s " + str(STEP_SIZE) + " -i src > del_temp_2.bed")
for line in open("dup_temp_2.bed"):
    dup_cp[line.split()[0] + "\t" + str(int(line.split()[1]) + 1) + "\t" + line.split()[2]] = line.split()[3]
for line in open("del_temp_2.bed"):
    del_cp[line.split()[0] + "\t" + str(int(line.split()[1]) + 1) + "\t" + line.split()[2]] = line.split()[3]
out = open(pathOut + "dudeml_temp2.bed","w")
for line in open(pathOut + "dudeml_temp.bed"):
    copy = "N"
    line = line.rstrip()
    liner = line.split()
    if line.split()[0] + "\t" + line.split()[1] + "\t" + str(int(line.split()[2])) in dup_cp:
        out.write("\t".join([liner[0],liner[1],liner[2],"dup",dup_cp[line.split()[0] + "\t" + line.split()[1] + "\t" + str(int(line.split()[2]))], "\t".join(line.split()[3:])]) + "\n")
    elif line.split()[0] + "\t" + line.split()[1] + "\t" + str(int(line.split()[2])) in del_cp:
        out.write("\t".join([liner[0],liner[1],liner[2],"del",del_cp[line.split()[0] + "\t" + line.split()[1] + "\t" + str(int(line.split()[2]))], "\t".join(line.split()[3:])]) + "\n")
    else:
        if len(liner) == 5 or len(liner) == 7 or len(liner) == 8:
            out.write("\t".join([liner[0],liner[1],liner[2],"N","1.0", "\t".join(line.split()[3:])]) + "\n")
out.close()
v=WINDOW_SIZE
if STEP_SIZE is not None:
    v=int(STEP_SIZE)
elif STEP_SIZE is None:
    v=int(WINDOW_SIZE)
window_pos = [[0,1,2,3,4,5]] * ((2*WINDOW) + 1)
output = open(OUTPUT,"w")
count = 0
for line in open(pathOut + "dudeml_temp2.bed"):
    count += 1
    if count % 100000 == 0:
        print(int(count),"windows processed")
    window_pos += [window_pos.pop(0)]
    window_pos[(2*WINDOW)] = line.rstrip().split()
    class_ud = "N"
    if len(list(set([item[0] for item in window_pos]))) == 1:
        if window_pos[WINDOW][3] == "dup" or window_pos[WINDOW][3] == "Dup":
            class_ud = "Dup"
        elif window_pos[WINDOW][3] == "del" or window_pos[WINDOW][3] == "Del":
            class_ud = "Del"
        cc = 0
        cv = 0
        for k in window_pos:
            if int(k[1]) == int(window_pos[WINDOW][1]) - (v*(WINDOW - cc)):
                cv += 1
            cc += 1
        if cv == len(window_pos):
            cq = [str(window_pos[WINDOW][0]),str(window_pos[WINDOW][1]), str(window_pos[WINDOW][2]), class_ud,str(window_pos[WINDOW][4])]
            for k in window_pos:
                cq.append(str(k[5]))
                cq.append(str(k[6]))
                cq.append(str(k[7]))
                cq.append(str(k[8]))
            output.write("\t".join(cq) + "\n")
output.close()
os.remove(pathOut + "dudeml_temp.bed")
os.remove(pathOut + "dudeml_temp2.bed")
os.remove("dup_temp_1.bed")
os.remove("del_temp_1.bed")
os.remove("dup_temp_2.bed")
os.remove("del_temp_2.bed")
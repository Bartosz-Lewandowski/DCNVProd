from Bio import SeqIO
import random
import argparse
from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, dest='file', help='FASTA file to process')
parser.add_argument('-n', dest='number', type=int, help='number of cnv to create')

def find_CNV_window(taken_places: list, file_len: int, cnv_range: int) -> Tuple[int, int]:
    wrong_cnv_coords = True
    while wrong_cnv_coords: 
        start = random.randint(1, file_len - cnv_range)
        stop = start + cnv_range + 1
        if taken_places:
            for start_taken, stop_taken in taken_places:
                if not (start >= start_taken and start <= stop_taken) or not \
                    (stop <= stop_taken and stop >= start_taken):
                    wrong_cnv_coords = False
        else:
            wrong_cnv_coords = False
    return start, stop

def generate_CNV(cnv_num: int, file: str) -> None:
    with open(file) as handle:
        CNV_num = cnv_num
        taken_places = []
        cnv_types = []
        for record in SeqIO.parse(handle, "fasta"):
            record_str = str(record.seq)
            file_len = len(record.seq)
            for _ in range(CNV_num):
                cnv_range = random.randint(50,100000)
                cnv_type = random.choice(['del','dup'])
                if cnv_type == 'dup':
                    print("Creating duplication")
                    start, stop = find_CNV_window(taken_places, file_len, cnv_range)
                    input_nuc = random.choice(["A","C","G","T"]) 
                    record_str = record_str[:start] + input_nuc * cnv_range + record_str[stop:]    
                    taken_places.append((start, stop))
                    cnv_types.append(cnv_type + "_" + input_nuc)
                elif cnv_type == 'del':
                    print("Creating deletion")
                    start, stop = find_CNV_window(taken_places, file_len, cnv_range)
                    record_str = record_str[:start] + record_str[stop:]    
                    taken_places.append((start, stop))
                    cnv_types.append(cnv_type)

    return taken_places, cnv_types

if __name__ == "__main__":
    args = parser.parse_args()
    taken_places, cnv_type = generate_CNV(args.number, args.file)
    for coords, type in zip(taken_places, cnv_type):
        print(f"{type}: {coords}")
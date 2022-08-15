import time
from subprocess import call

from CNV_generator import CNVGenerator
from generate_features import Stats
from sim_reads import SimReads
from train import train_data


def create_sim_bam(ref_genome_fasta: str) -> None:
    CNV = CNVGenerator(ref_genome_fasta)
    total = CNV.generate_cnv()
    fasta_modified = CNV.modify_fasta_file(total)
    sim_reads = SimReads(fasta_modified, 20)
    r1, r2 = sim_reads.sim_reads_genome()
    call(f"bwa index {ref_genome_fasta}", shell=True)
    bwa_command = f"bwa mem -t 8 {ref_genome_fasta} {r1} {r2} | samtools view -Shb - | samtools sort - > train_sim/total.bam"
    call(bwa_command, shell=True)


if __name__ == "__main__":
    start_time = time.time()
    ref_genome_fasta = "reference_genome/ref_genome.fa"
    create_sim_bam(ref_genome_fasta)
    stats = Stats("train_sim/total.bam", output_folder="sim")
    df_with_stats = stats.generate_stats()
    stats.combine_into_one_big_file("train_bed/total.bed")
    train_data("stats/sim/combined.csv")
    print(f"Time elapsed:  {time.time() - start_time}")

import os

from CNV_generator import CNVGenerator
from generate_stats import Stats
from sim_reads import SimReads


def create_sim_bam(ref_genome_fasta: str) -> None:
    CNV = CNVGenerator(ref_genome_fasta)
    total = CNV.generate_cnv()
    fasta_modified = CNV.modify_fasta_file(total)
    sim_reads = SimReads(fasta_modified, 20)
    r1, r2 = sim_reads.sim_reads_genome()
    os.system(f"bwa index {ref_genome_fasta}")
    bwa_command = f"bwa mem -t 8 {ref_genome_fasta} {r1} {r2} | samtools view -Shb - | samtools sort - > train_sim/total.bam"
    os.system(bwa_command)


if __name__ == "__main__":
    ref_genome_fasta = "reference_genome/ref_genome_short.fa"
    create_sim_bam(ref_genome_fasta)
    stats = Stats()
    df = stats.create_genomecov("train_sim/total.bam")
    df_with_stats = stats.generate_stats(df)

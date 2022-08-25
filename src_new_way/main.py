import glob
import os
from subprocess import call

from argparser import CHRS, arg_parser
from CNV_generator import CNVGenerator
from generate_features import Stats
from google.api_core.exceptions import NotFound
from real_data_preprocessing import CleanRealTarget
from sim_reads import SimReads
from train import train_data
from utils import GCP, download_reference_genom, get_number_of_individuals


def create_sim_bam(cpus: int, window_size: int) -> None:
    download_reference_genom()
    call("./src_new_way/combine_ref_gens.sh", shell=True)
    ref_genome_fasta = "reference_genome/ref_genome.fa"
    CNV = CNVGenerator(ref_genome_fasta, window_size)
    total = CNV.generate_cnv()
    fasta_modified = CNV.modify_fasta_file(total)
    sim_reads = SimReads(fasta_modified, 10, cpu=cpus)
    r1, r2 = sim_reads.sim_reads_genome()
    r1 = "train_sim/10_R1.fastq"
    r2 = "train_sim/10_R2.fastq"
    call(f"bwa index {ref_genome_fasta}", shell=True)
    os.makedirs("sim_data/", exist_ok=True)
    bwa_command = f"bwa mem -t {cpus} {ref_genome_fasta} {r1} {r2} | samtools view -Shb - | samtools sort - > sim_data/sim_data.bam"
    call(bwa_command, shell=True)


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    if args.chrs == ["all"]:
        args.chrs = CHRS[:-1]
    gcp = GCP("cnv-detection-9531ae84cdd2.json")
    if args.command == "sim":
        if args.new_data and not args.new_features:
            raise Exception(
                "Creating new data make sure to calculate new features as well!"
            )
        if args.new_data:
            create_sim_bam(args.cpus, args.window_size)
            gcp.upload_blob(
                "cnv-data-ml", "sim_data/sim_data.bam", "cnv-data/sim_data/sim_data.bam"
            )
            gcp.upload_blob(
                "cnv-data-ml",
                "sim_target/sim_target.csv",
                "cnv-data/sim_target/sim_target.csv",
            )
        else:
            os.makedirs("sim_data/", exist_ok=True)
            gcp.download_blob(
                "cnv-data-ml", "cnv-data/sim_data/sim_data.bam", "sim_data/sim_data.bam"
            )
            os.makedirs("sim_target/", exist_ok=True)
            gcp.download_blob(
                "cnv-data-ml",
                "cnv-data/sim_target/sim_target.csv",
                "sim_target/sim_target.csv",
            )
        if args.new_features:
            print("##Creating features for simulated data...")
            stats = Stats("sim_data/sim_data.bam", output_folder="sim", cpus=args.cpus)
            df_with_stats = stats.generate_stats(
                chrs=args.chrs, window_size=args.window_size, step=args.window_size
            )
            stats.combine_into_one_big_file("sim_target/sim_target.csv")
            gcp.upload_blob(
                "cnv-data-ml",
                "stats/sim/combined.csv",
                "cnv-data/stats/sim/combined.csv",
            )
        else:
            gcp.download_blob(
                "cnv-data", "cnv-data/stats/sim/combined.csv", "stats/sim/combined.csv"
            )

    if args.command == "real":
        if not os.path.exists("real_data/"):
            os.makedirs("real_data/", exist_ok=True)
            gcp.download_folder_blob("cnv-data-ml", "cnv-data/real_data/", "real_data/")
        if not os.path.exists("real_data_target"):
            os.makedirs("real_data_target/pindel", exist_ok=True)
            os.makedirs("real_data_target/cnvnator", exist_ok=True)
            gcp.download_folder_blob(
                "cnv-data-ml", "cnv-data/pindel/", "real_data_target/pindel/"
            )
            gcp.download_folder_blob(
                "cnv-data-ml", "cnv-data/cnvnator/", "real_data_target/cnvnator/"
            )

        if args.new_features:
            bam_real_files = sorted(glob.glob(os.path.join("real_data/", "*.bam")))
            indv = get_number_of_individuals(bam_real_files)
            target = CleanRealTarget(window_size=args.window_size, cpus=args.cpus)
            for ind in indv:
                target.clean_cnvnator(ind)
                target.clean_pindel(ind)
                target.combine(ind)
            print("##Creating features for real data..")
            for bam_file, ind in zip(bam_real_files, indv):
                stats = Stats(bam_file, output_folder=ind, cpus=args.cpus)
                df_with_stats = stats.generate_stats(
                    chrs=args.chrs,
                    window_size=args.window_size,
                    step=args.window_size,
                )
                stats.combine_into_one_big_file(
                    f"real_data_target/valid_target/{ind}.csv"
                )
                gcp.upload_blob(
                    "cnv-data-ml",
                    f"stats/{ind}/combined.csv",
                    f"cnv-data/stats/{ind}/combined.csv",
                )
        else:
            for ind in [2, 3, 4, 5, 11, 12]:
                try:
                    gcp.download_folder_blob(
                        "cnv-data-ml", f"cnv-data/stats/{ind}/", f"stats/{ind}"
                    )
                except NotFound:
                    print(f"Stats for {ind} not found!")

    if args.command == "train":
        train_data("stats/sim/combined.csv")

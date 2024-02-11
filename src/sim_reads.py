import gc
import os
from subprocess import call

import numpy as np
from Bio import SeqIO

from src.paths import MODIFIED_FASTA_FILE_NAME, SIM_DATA_PATH, SIM_READS_FOLDER


class SimReads:
    """
    Function to simulate reads from a fasta file using InSilicoSeq.
    Simulate behaviour of NovaSeq, HiSeq or MiSeq sequencers.
    Different machines have different read lengths and error rates.
    NovaSeq: 150 bp,
    HiSeq: 125 bp,
    MiSeq: 300 bp,
    The bigger coverage, the data is more accurate.
    Coverage can't be too big, because it will be too expensive to simulate.

    Parameters
    ----------
    cov : int
        Coverage of the data.
    cpu : int
        Number of CPUs to use.
    model : str
        Type of sequencer to simulate. One of: "novaseq", "hiseq", "miseq".
    """

    def __init__(
        self,
        cov: int,
        cpu: int,
        model: str = "novaseq",
        fasta_file: str = "/".join([SIM_DATA_PATH, MODIFIED_FASTA_FILE_NAME]),
        pathout: str = SIM_READS_FOLDER,
    ) -> None:
        gc.collect()
        models = ["novaseq", "hiseq", "miseq"]
        if model in models:
            self.model = model
        else:
            raise ValueError(f"Wrong type of model. Try one of{models}")
        self.model = model
        self.fasta_file = fasta_file
        self.read_len = self.__get_read_length()
        self.cov = cov
        self.cpu = cpu
        self.pathout = pathout
        os.makedirs(pathout, exist_ok=True)

    def sim_reads_genome(self) -> None:
        """
        Main function to simulate reads from a fasta file using InSilicoSeq..
        """
        chrs = np.array(
            [len(fasta.seq) for fasta in SeqIO.parse(open(self.fasta_file), "fasta")]
        )
        chrs_len = np.sum(chrs)
        N = self.__calc_N_reads(chrs_len)
        self._sim_reads_with_InSilicoSeq(N)

    def _sim_reads_with_InSilicoSeq(self, N: int) -> None:
        """
        Simulate reads from a fasta file using InSilicoSeq.
        """
        command = f"iss generate --model {self.model} --genomes {self.fasta_file} --n_reads {N} --cpus {self.cpu} --output {self.pathout}/{self.cov}"
        call(command, shell=True)

    def __get_read_length(self) -> int:
        """
        Get read length of the sequencer.
        """
        match self.model:
            case "novaseq":
                return 150
            case "hiseq":
                return 125
            case "miseq":
                return 300
        return 150

    def __calc_N_reads(self, chr_len: int) -> int:
        """
        Calculate number of reads to simulate for a given coverage.

        Parameters
        ----------
        chr_len : int
            Length of the all chromosomes.

        Returns
        -------
        int
            Number of reads to simulate.
        """
        return round(chr_len / self.read_len * self.cov)

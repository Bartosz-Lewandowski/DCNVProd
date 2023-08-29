import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import requests
from pybedtools import BedTool
from tqdm.std import tqdm


@dataclass
class BedFormat:
    chr: str
    start: int
    end: int
    cnv_type: str = "normal"
    freq: int = 1


def BedTool_to_BedFormat(bedfile: BedTool, short_v: bool = False) -> list:
    out = []
    if short_v:
        return [
            BedFormat(line.chrom, int(line.start), int(line.end)) for line in bedfile
        ]
    for line in bedfile:
        out.append(
            BedFormat(
                line.chrom,
                int(line.start),
                int(line.end),
                line.name,
                int(line.score),
            )
        )
    return out


def BedFormat_to_BedTool(seq: np.array) -> BedTool:
    out_str = ""
    for line in seq:
        out_str += f"{line.chr} {line.start} {line.end} {line.cnv_type} {line.freq}\n"
    bedfile = BedTool(out_str, from_string=True)
    return bedfile


def get_number_of_individuals(file_names: list[str]) -> list:
    output = []
    for file in file_names:
        file_pattern = re.search(r"Nr\d+", file)
        assert file_pattern is not None
        output.append(file_pattern.group().replace("Nr", ""))
    return output


def download_reference_genom() -> None:
    URL = "http://ftp.ensembl.org/pub/release-107/fasta/sus_scrofa/dna/"
    page = requests.get(URL)
    cont = page.content
    output_folder = "reference_genome/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    text_search = r"Sus_scrofa.Sscrofa11.1.dna.chromosome.[\d]?[\d]?.fa.gz"
    chrs = set(re.findall(text_search, str(cont)))
    for chr in chrs:
        with requests.get(URL + chr, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))
            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                with open(output_folder + chr, "wb") as output:
                    shutil.copyfileobj(raw, output)

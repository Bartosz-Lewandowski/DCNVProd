import requests
import re
import os
from tqdm.auto import tqdm
import shutil

URL = "http://ftp.ensembl.org/pub/release-107/fasta/sus_scrofa/dna/"
page = requests.get(URL)
cont = page.content
output_folder = "reference_genome/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

text_search = "Sus_scrofa.Sscrofa11.1.dna.chromosome.[\dXY]?[\d]?.fa.gz"
chrs = set(re.findall(text_search, str(cont)))
for chr in chrs:
    with requests.get(URL + chr, stream=True) as r:
        total_length = int(r.headers.get("Content-Length"))
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
            with open(output_folder + chr, 'wb') as output:
                shutil.copyfileobj(raw, output)
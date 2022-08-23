import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import requests
from google.cloud import storage
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
                line.chrom, int(line.start), int(line.end), line.name, int(line.score)
            )
        )
    return out


def BedFormat_to_BedTool(seq: np.array) -> BedTool:
    out_str = ""
    for line in seq:
        out_str += f"{line.chr} {line.start} {line.end} {line.cnv_type} {line.freq}\n"
    bedfile = BedTool(out_str, from_string=True)
    return bedfile


def get_number_of_individuals(file_names: list) -> list:
    return [re.search(r"Nr\d+", file).group().replace("Nr", "") for file in file_names]


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


class GCP:
    def __init__(self, credential_json) -> None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_json

    def upload_blob(
        self, bucket_name: str, source_file_name: str, destination_blob_name: str
    ) -> None:
        """Uploads a file to the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"
        # The path to your file to upload
        # source_file_name = "local/path/to/file"
        # The ID of your GCS object
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(f"File {source_file_name} uploaded to {destination_blob_name}.")

    def download_blob(
        self, bucket_name: str, source_blob_name: str, destination_file_name: str
    ) -> None:
        """Downloads a blob from the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"

        # The ID of your GCS object
        # source_blob_name = "storage-object-name"

        # The path to which the file should be downloaded
        # destination_file_name = "local/path/to/file"

        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = bucket.get_blob(source_blob_name)
        with open(destination_file_name, "wb") as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                storage_client.download_blob_to_file(blob, file_obj)

        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
                source_blob_name, bucket_name, destination_file_name
            )
        )

    def download_folder_blob(
        self, bucket_name: str, source_blob_name: str, destination_folder_name: str
    ) -> None:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=source_blob_name)  # Get list of files
        for blob in blobs:
            if blob.size > 0:
                print(
                    "Downloading storage object {} from bucket {} to local folder {}.".format(
                        source_blob_name, bucket_name, destination_folder_name
                    )
                )
                filename = blob.name.replace("/", "_")
                blob.download_to_filename(
                    destination_folder_name + filename
                )  # Download

    def upload_folder_blob(
        self, bucket_name: str, source_folder_name: str, destination_blob_name: str
    ) -> None:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        files_in_folder = os.listdir(source_folder_name)
        for file in files_in_folder:
            print(f"File {file} uploaded to {destination_blob_name}.")
            blob.upload_from_filename(file)

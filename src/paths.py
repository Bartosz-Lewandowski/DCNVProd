SIM_DATA_PATH = "sim_data"
SIM_BAM_FILE_NAME = "sim_data.bam"
SIM_BAM_FILE_PATH = "/".join([SIM_DATA_PATH, SIM_BAM_FILE_NAME])

TARGET_DATA_FILE_NAME = "target_file.csv"
MODIFIED_FASTA_FILE_NAME = "modified_fasta_CNV.fa"

REF_GEN_PATH = "reference_genome"
REF_GEN_FILE_NAME = "ref_genome.fa"
REF_FASTA_FILE = "/".join([REF_GEN_PATH, REF_GEN_FILE_NAME])

STATS_FOLDER = "stats"

SIM_READS_FOLDER = "/".join([SIM_DATA_PATH, "sim_reads"])

FEATURES_COMBINED_FILE = "combined.csv"

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
VAL_FOLDER = "val"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
VAL_FILE = "val.csv"

TRAIN_PATH = "/".join([TRAIN_FOLDER, TRAIN_FILE])
TEST_PATH = "/".join([TEST_FOLDER, TEST_FILE])
VAL_PATH = "/".join([VAL_FOLDER, VAL_FILE])

MODELS_FOLDER = "models"
BASIC_MODEL_FILE = "ML_model.pkl"
BASIC_MODEL_PATH = "/".join([MODELS_FOLDER, BASIC_MODEL_FILE])


OVR_MODEL_FILE = "ML_ovr_model.pkl"
OVR_BASIC_MODEL_PATH = "/".join([MODELS_FOLDER, OVR_MODEL_FILE])

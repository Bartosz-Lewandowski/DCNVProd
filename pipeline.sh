python3.10 src/CNV_generator_new.py
python3.10 src/simChr.py
python3.10 src/simReads.py
bwa index fasta.fa
bwa mem -t 4 fasta.fa sim_train/train_20_1.fq sim_train/train_20_2.fq | samtools view -Shb - | samtools sort - > sim_train/total.bam
python3.10 src/Stats.py
python3.10 src/fvecTrain.py
python3.10 src/train.py
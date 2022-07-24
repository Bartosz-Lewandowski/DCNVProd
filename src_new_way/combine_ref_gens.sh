for f in reference_genome/*.gz; do
  zcat "$f" | head -n 5000
done | gzip -c > reference_genome/ref_genome_short.fa.gz

rm -r reference_genome/Sus_scrofa.Sscrofa11.1.dna*.fa.gz
gunzip reference_genome/ref_genome_short.fa.gz
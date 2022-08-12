#!/bin/bash
rm -r reference_genome/ref_genome.fa
for f in reference_genome/*.gz
do
  echo "Combining $f file..."
  zcat < $f | head -n 10000 >> reference_genome/ref_genome.fa
done 
rm -r reference_genome/Sus_scrofa.Sscrofa11.1.dna*.fa.gz
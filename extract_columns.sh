#!/bin/bash
for name in dev devtest test; do
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < ${name}.txt > ${name}.ja.txt
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < ${name}.txt > ${name}.en.txt
done
for name in train-1; do
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < ${name}.txt > ${name}.ja.txt
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[4], "\n";' < ${name}.txt > ${name}.en.txt
done
for file in train-1 dev devtest test; do
  mv ${file}.ja.txt ${file}.ja.txt.org
  cat ${file}.ja.txt.org | perl -CSD -Mutf8 -pe 's/(.)［[０-９．]+］$/${1}/;' > ${file}.ja.txt
done
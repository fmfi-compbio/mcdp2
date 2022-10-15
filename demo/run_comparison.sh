#!/bin/bash

mcdp2 single tcga_intervals.txt hirt_intervals.txt chr_sizes.txt -o output_dir -t 4
mcdp2 single tcga_intervals.txt hirt_intervals.txt chr_sizes.txt -o output_dir_fast -t 4 -a 'fast'
mcdp2 single tcga_intervals.txt hirt_intervals.txt chr_sizes.txt -o output_dir_bases -t 4 -s "bases"

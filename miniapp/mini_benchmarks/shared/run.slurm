#!/bin/bash
#SBATCH -o deps_comp_benchmarks.%j.out
#SBATCH -n 1
#SBATCH -c 32

python3 deps_comp_benchmarks.py

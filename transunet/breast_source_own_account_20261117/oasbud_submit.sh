#!/bin/bash
for seed in 1337 2027 2028; do
  sbatch_out=$(/opt/slurm/default/bin/sbatch -J oasbud_s${seed} /scratch/xl5874/Eve/birp_breast_source/train.sbatch oasbud ${seed})
  echo "$sbatch_out"
done

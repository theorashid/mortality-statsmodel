#!/bin/sh
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=2:mem=96gb

module load anaconda3/personal
source activate Rmortstat

Rscript $HOME/mortality-statsmodel/Models/parametric/run_model.R LSOA BYM 1 25000 20000 --num_chains=2 --test

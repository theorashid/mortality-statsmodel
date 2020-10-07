#!/bin/sh
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=4:mem=96gb

module load anaconda3/personal
source activate Rmortstat

Rscript $HOME/mortality-statsmodel/Models/parametric/inits.R

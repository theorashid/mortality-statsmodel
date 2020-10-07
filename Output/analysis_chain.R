# Theo AO Rashid -- June 2020

# ----- Samples analysis -----
library(dplyr)
library(ggplot2)
library(rstan)
library(mcmcplots)

data_path    <- "/rds/general/user/tar15/home/mortality-statsmodel/Data/"
output_path  <- "/rds/general/user/tar15/home/mortality-statsmodel/Output/"

# ----- IMPORT DATA -----
name_run  <- "BYM"

sex_choice <- 1 # 1 male, 2 female
london     <- TRUE # if false, loads Hammersmith and Fulham

samples   <- FALSE # if false, loads samples2
n_samples <- 3 # number of rows of samples to subset (mortality rates)

# rows - iteration, columns - node
if (london) {
  if (samples) {
    samples <- readRDS(paste0(output_path, name_run, sex_choice, "_ldn", "_samples.rds"))
    samples <- samples[,sample(ncol(samples), n_samples)] # subset
  } else {
    samples <- readRDS(paste0(output_path, name_run, sex_choice, "_ldn", "_samples2.rds"))
  }
} else {
  if (samples) {
    samples <- readRDS(paste0(output_path, name_run, sex_choice, "_hf", "_samples.rds"))
    samples <- samples[,sample(ncol(samples), n_samples)] # subset
  } else {
    samples <- readRDS(paste0(output_path, name_run, sex_choice, "_hf", "_samples2.rds"))
  }
}

# ----- CHAIN CONVERGENCE -----
summary(samples)

# Convert to a format for stan analysis
# list of matrices, one for each parameter
# matrix 
samples_stan

# Rhat compares between- and within- chain estimates for parameters. Need a value
# of less than 1.05 from at least 4 chains
# https://mc-stan.org/rstan/reference/Rhat.html
Rhat(samples_stan)

# ess_bulk measures sampling efficiency in bulk of distribution
# ess_tail measures in tail of distribution
# need both above 100
ess_bulk(samples_stan)
ess_tail(samples_stan)

# ----- CHAIN PLOTS -----
mcmcplot(samples)

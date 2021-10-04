# Theo AO Rashid -- October 2020

# ----- Samples analysis -----
library(tidyverse)
library(reshape2)
library(rstan)
library(foreach)

print(paste0("AVAILABLE CORES = ", detectCores()))
doParallel::registerDoParallel(cores = 8)

# ----- IMPORT DATA -----
region <- "LSOA"
model  <- "nested"
sex    <- 1 # 1 male, 2 female
test   <- TRUE

print("----- LOADING DATA... -----")
if (!test) {
  chain_output <- readRDS(
    here::here(
      "Output", "mcmc_output", paste0(region, model, sex, "_mcmc_out.rds")
    )
  )
} else {
  chain_output <- readRDS(
    here::here(
      "Output", "mcmc_output", paste0(region, model, sex, "_T", "_mcmc_out.rds")
    )
  )
}

n_chains   <- length(chain_output)
n_samples  <- nrow(chain_output[[1]]$samples)
n_samples2 <- nrow(chain_output[[1]]$samples2)
n_params   <- ncol(chain_output[[1]]$samples)
n_hyps     <- ncol(chain_output[[1]]$samples2)

names_params <- colnames(chain_output[[1]]$samples)
names_hyps   <- colnames(chain_output[[1]]$samples2)

# nimble MCMC output is of the following form:
# chain_output is a list of n_chains, one for each MCMC chain
# chain_output[[i]]$samples are the mortality rates
# chain_output[[i]]$samples2 are the hyperparameters
# For samples, rows are each iteration of MCMC, columns are parameter name

# ----- MORTALITY RATES -----
# information in samples

# convert to rstan sims format
# list, one for each parameter
# rows are each iteration of MCMC, column for each chain
print("----- REORGANISE MORTALITY -----")

#' regorganises from nimble output to stan-like output for analysis
reorganise_mortality <- function(
  chain_output, params_vec, n_samples, n_chains
) {
  mr <- list()
  for (i in params_vec) {
    par_mat <- matrix(, nrow = n_samples, ncol = n_chains)
    for (c in 1:n_chains) {
      par_mat[, c] <- chain_output[[c]]$samples[, i]
    }
    name <- names_params[i]
    mr[[name]] <- par_mat
  }
  return(mr)
}

# split the matrix process over many nodes
chunk_size <- 100
tmp <- split(1:n_params, ceiling(seq_along(1:n_params) / chunk_size))
print(
  system.time({
    mr <- foreach(params_vec = tmp) %dopar% {
      reorganise_mortality(
        chain_output = chain_output,
        params_vec   = params_vec,
        n_samples    = n_samples,
        n_chains     = n_chains
      )
    }
    mr <- unlist(mr, recursive = FALSE)
  })
)

# 1. R-hat (mc-stan.org/rstan/reference/Rhat.html)
print("----- R-HAT -----")
print(
  system.time(
    rhats <- foreach(i = mr, .combine = "c") %dopar% {
      Rhat(i)
    }
  )
)

print("R-hat range: ")
print(range(rhats))

# 2. Bulk effective sample size
print("----- BULK-ESS -----")
print(
  system.time(
    ess_bulks <- foreach(i = mr, .combine = "c") %dopar% {
      ess_bulk(i)
    }
  )
)

print("ess-bulk range: ")
print(range(ess_bulks))

# 3. Tail effective sample size
print("----- TAIL-ESS -----")
print(
  system.time(
    ess_tails <- foreach(i = mr, .combine = "c") %dopar% {
      ess_tail(i)
    }
  )
)

print("ess-tail range: ")
print(range(ess_tails))

# ----- HYPERPARAMETERS -----
hyp <- list()
for (i in 1:n_hyps) {
  par_mat <- matrix(, nrow = n_samples2, ncol = n_chains)
  for (c in 1:n_chains) {
    par_mat[, c] <- chain_output[[c]]$samples2[, i]
  }
  name <- names_hyps[i]
  hyp[[name]] <- par_mat
}

if (!test) {
  saveRDS(
    hyp,
    here::here("Output", "e0_samples",
    paste0(region, model, sex, "_e0_samples.csv"))
  )
} else {
  saveRDS(
    hyp,
    here::here("Output", "e0_samples",
    paste0(region, model, sex, "_T", "_e0_samples.csv"))
  )
}
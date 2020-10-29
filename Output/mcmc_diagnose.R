# Theo AO Rashid -- October 2020

# ----- Samples analysis -----
library(dplyr)
library(reshape2)
library(rstan)
library(ggplot2)
library(foreach)

print(paste0("AVAILABLE CORES = ", detectCores()))
# numCores <- detectCores()
numCores <- 8
doParallel::registerDoParallel(cores = numCores)

# ----- IMPORT DATA -----
region <- "LSOA"
model  <- "nested"
sex    <- 1 # 1 male, 2 female
test   <- TRUE

print("----- LOADING DATA... -----")
if (!test) {
  chain_output <- readRDS(
    here::here("Output", "mcmc_output", 
    paste0(region, model, sex, "_mcmc_out.rds"))
  )
} else {
  chain_output <- readRDS(
    here::here("Output", "mcmc_output", 
    paste0(region, model, sex, "_T", "_mcmc_out.rds"))
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
# list, one for each parameter, rows are each iteration of MCMC, column for each chain
print("----- REORGANISE MORTALITY -----")

#' regorganises from nimble output to stan-like output for analysis
reorganise_mortality <- function(chain_output, params_vec, n_samples, n_chains) {
  mr <- list()
  for (i in params_vec) {
    par_mat <- matrix(, nrow = n_samples, ncol = n_chains)
    for(c in 1:n_chains) {
      par_mat[, c] <- chain_output[[c]]$samples[,i]
    }
    name <- names_params[i]
    mr[[name]] <- par_mat
  }
  return(mr)
}

# split the matrix process over many nodes
chunk_size <- 100
tmp <- split(1:n_params, ceiling(seq_along(1:n_params)/chunk_size))
print(
  system.time({
    mr <- foreach(params_vec = tmp) %dopar% {
      reorganise_mortality(
        chain_output = chain_output,
        params_vec = params_vec,
        n_samples = n_samples,
        n_chains = n_chains
      )
    }
    mr <- unlist(mr, recursive = FALSE)
  })
)

# Convergence and efficiency diagnostics (https://mc-stan.org/rstan/reference/Rhat.html)
# 1. R-hat
# The Rhat function compares the between- and within-chain estimates for model parameters.
# If chains have not mixed well (ie, the between- and within-chain estimates don't agree),
# R-hat is larger than 1. Need R-hat < 1.05.
print("----- R-HAT -----")
print(
  system.time(
    rhats <- foreach(i = mr, .combine = "c") %dopar% {
      Rhat(i)
    }
  )
)

if (!test) {
  png(
    file = here::here("Output", "convergence", 
    paste0(region, model, sex, "_rhat.png")),
    width = 600, height = 350
    )
} else {
  png(
    file = here::here("Output", "convergence", 
    paste0(region, model, sex, "_T", "_rhat.png")),
    width = 600, height = 350
  )
}
hist(rhats, col = rgb(0, 0.545, 0.545, 0.7), xlab = "R-hat statistic", main = "")
dev.off()

print("R-hat range: ")
print(range(rhats))

# 2. Bulk effective sample size
# Bulk-ESS is useful measure for sampling efficiency in the bulk of the distribution
# (related e.g. to efficiency of mean and median estimates)
print("----- BULK-ESS -----")
print(
  system.time(
    ess_bulks <- foreach(i = mr, .combine = "c") %dopar% {
      ess_bulk(i)
    }
  )
)

if (!test) {
  png(
    file = here::here("Output", "convergence", 
    paste0(region, model, sex, "_ess_bulk.png")),
    width = 600, height = 350
    )
} else {
  png(
    file = here::here("Output", "convergence", 
    paste0(region, model, sex, "_T", "_ess_bulk.png")),
    width = 600, height = 350
  )
}
hist(ess_bulks, col = rgb(0.69, 0.769, 0.871, 0.7), xlab = "ess-bulk", main = "")
dev.off()

print("ess-bulk range: ")
print(range(ess_bulks))

# The ess_tail function produces computes the minimum of effective sample sizes for 5%
# and 95% quantiles (related e.g. to efficiency of variance and tail quantile estimates).
print("----- TAIL-ESS -----")
print(
  system.time(
    ess_tails <- foreach(i = mr, .combine = "c") %dopar% {
      ess_tail(i)
    }
  )
)

if (!test) {
  png(
    file = here::here("Output", "convergence", 
    paste0(region, model, sex, "_ess_tail.png")),
    width = 600, height = 350
    )
} else {
  png(
    file = here::here("Output", "convergence", 
    paste0(region, model, sex, "_T", "_ess_tail.png")),
    width = 600, height = 350
  )
}
hist(ess_tails, col = rgb(0.439, 0.502, 0.565, 0.7), xlab = "ess-tail statistic", main = "")
dev.off()

print("ess-tail range: ")
print(range(ess_tails))

# 2. Traceplots
print("----- PLOTTING LOGRATE CONVERGENCE -----")
n_sub <- 100 # work on 100 random subsamples

set.seed(1)
sub <- sample.int(n_params, n_sub) 

traces <- list()
for (i in sub) {
  tmp <- as.data.frame(mr[[i]])
  name <- names_params[i]
  tmp$iteration <- 1:n_samples
  tmp <- melt(tmp,  id.vars = "iteration", variable.name = "chain")
  tmp$name <- name
  traces[[name]] <- tmp
}
traces <- do.call(rbind, traces)

p <- traces %>% 
  ggplot(aes(x = iteration, y = value, colour = chain)) +
  geom_line(alpha = 0.7, size = 0.1) +
  labs(x = "", y = "") +
  theme(legend.position = "none", text = element_text(size = 6)) + 
  facet_wrap( ~ name, ncol = 10)
ggsave(
  here::here("Output", "convergence", 
  paste0(region, model, sex, "_T", "_mr_trace.png")),
  p, scale = 4
)

# ----- HYPERPARAMETERS -----
hyp <- list()
for (i in 1:n_hyps) {
  par_mat <- matrix(, nrow = n_samples2, ncol = n_chains)
  for(c in 1:n_chains) {
    par_mat[, c] <- chain_output[[c]]$samples2[,i]
  }
  name <- names_hyps[i]
  hyp[[name]] <- par_mat
}

print("----- HYPERPARAMETER SUMMARY -----")
for (i in 1:length(hyp)) {
  print(names_hyps[i])
  print(summary(hyp[[i]]))
}

print("----- PLOTTING HYPERPARAMETER CONVERGENCE -----")
traces <- list()
for (i in 1:length(hyp)) {
  tmp <- as.data.frame(hyp[[i]])
  name <- names_hyps[i]
  tmp$iteration <- 1:n_samples2
  tmp <- melt(tmp,  id.vars = "iteration", variable.name = "chain")
  tmp$name <- name
  traces[[name]] <- tmp
}
traces <- do.call(rbind, traces)

p <- traces %>% 
  ggplot(aes(x = iteration, y = value, colour = chain)) +
  geom_line(alpha = 0.7, size = 0.1) +
  labs(x = "", y = "") +
  theme(legend.position = "none", text = element_text(size = 8)) + 
  facet_wrap( ~ name, ncol = 8)
ggsave(
  here::here("Output", "convergence", 
  paste0(region, model, sex, "_T", "_hyp_trace.png")),
  p, scale = 4
)

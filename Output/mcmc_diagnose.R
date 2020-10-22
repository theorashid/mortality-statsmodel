# Theo AO Rashid -- October 2020

# ----- Samples analysis -----
library(dplyr)
library(reshape2)
library(rstan)
library(ggplot2)

source(paste0("/rds/general/user/tar15/home/mortality-statsmodel/Models/parametric/",
              "path_info.R"))

# ----- IMPORT DATA -----
region <- "LSOA"
model  <- "nested"
sex    <- 1 # 1 male, 2 female
test   <- TRUE

print("----- LOADING DATA... -----")
if (!test) {
  chain_output <- readRDS(paste0(output_path, "mcmc_output/",
                                 region, model, sex,
                                 "_mcmc_out.rds"))
} else {
  chain_output <- readRDS(paste0(output_path, "mcmc_output/",
                                 region, model, sex,
                                 "_T", "_mcmc_out.rds"))
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
mr <- list()
for (i in 1:n_params) {
  par_mat <- matrix(, nrow = n_samples, ncol = n_chains)
  for(c in 1:n_chains) {
    par_mat[, c] <- chain_output[[c]]$samples[,i]
  }
  name <- names_params[i]
  mr[[name]] <- par_mat
}

# Convergence and efficiency diagnostics (https://mc-stan.org/rstan/reference/Rhat.html)
# 1. R-hat
# The Rhat function compares the between- and within-chain estimates for model parameters.
# If chains have not mixed well (ie, the between- and within-chain estimates don't agree),
# R-hat is larger than 1. Need R-hat < 1.05.
print("----- R-HAT -----")
rhats <- lapply(mr, Rhat)

if (!test) {
  png(file = paste0(output_path, "convergence/", region, model, sex, "_rhat.png"),
      width = 600, height = 350)
} else {
  png(file = paste0(output_path, "convergence/", region, model, sex, "_T", "_rhat.png"),
      width = 600, height = 350)
}
hist(as.numeric(unlist(rhats)), col = rgb(0, 0.545, 0.545, 0.7), xlab = "R-hat statistic", main = "")
dev.off()

print("R-hat range: ")
print(range(rhats))

# 2. Bulk effective sample size
# Bulk-ESS is useful measure for sampling efficiency in the bulk of the distribution
# (related e.g. to efficiency of mean and median estimates)
print("----- BULK-ESS -----")
ess_bulks <- lapply(mr, ess_bulk)
if (!test) {
  png(file = paste0(output_path, "convergence/", region, model, sex, "_ess_bulk.png"),
      width = 600, height = 350)
} else {
  png(file = paste0(output_path, "convergence/", region, model, sex, "_T", "_ess_bulk.png"),
      width = 600, height = 350)
}
hist(as.numeric(unlist(ess_bulks)), col = rgb(0.69, 0.769, 0.871, 0.7), xlab = "ess-bulk", main = "")
dev.off()

print("ess-bulk range: ")
print(range(ess_bulks))

# The ess_tail function produces computes the minimum of effective sample sizes for 5%
# and 95% quantiles (related e.g. to efficiency of variance and tail quantile estimates).
print("----- TAIL-ESS -----")
ess_tails <- lapply(mr, ess_tail)
if (!test) {
  png(file = paste0(output_path, "convergence/", region, model, sex, "_ess_tail.png"),
      width = 600, height = 350)
} else {
  png(file = paste0(output_path, "convergence/", region, model, sex, "_T", "_ess_tail.png"),
      width = 600, height = 350)
}
hist(as.numeric(unlist(ess_tails)), col = rgb(0.439, 0.502, 0.565, 0.7), xlab = "ess-tail statistic", main = "")
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
ggsave(paste0(output_path, "convergence/", region, model, sex, "_mr_trace.png"), p, scale = 4)

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
ggsave(paste0(output_path, "convergence/", region, model, sex, "_hyp_trace.png"), p, scale = 3)


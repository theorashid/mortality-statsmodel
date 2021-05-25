"Run mortality model.

Usage:
    run_model.R <region> <model> <sex> <num_iter> <num_burn> [--num_chains=<num_chains>] [--thin_mort=<thin_mort>] [--thin_param=<thin_param>] [--test]
    run_model.R (-h | --help)
    
Options:
    -h --help                       Show this help message and exit.
    --test                          Run the smaller test dataset (London for MSOA, Hammersmith and Fulham for LSOA).
    --num_chains=<num_chains>       Number of parallel chains [default: 1].
    --thin_mort=<thin_mort>         Thinning interval for mortality rates [default: 10].
    --thin_param=<thin_param>       Thinning interval for model parameters (must be greater than thin_mort) [default: 100].

Arguments:
    <region>        Spatial unit of interest (MSOA | LSOA).
    <model>         Name of the model (BYM | nested ).
    <sex>           Sex of the run (1 for male, 2 for female).
    <num_iter>      Number of iterations for the MCMC.
    <num_burn>      Number of burn in iterations for the MCMC.
" -> doc

args <- docopt::docopt(doc)

suppressPackageStartupMessages({
  library(tidyverse)
  library(nimble)
})

source(here::here("Models", "parametric", "prepare_model.R"))
source(here::here("Models", "parametric", "nimble_model.R"))

# test parameters for running interactively
args <- list("LSOA", "BYM", "1", TRUE, "100", "10", "1", "2", "10")
names(args) <- list(
  "region", "model", "sex", "test",
  "num_iter", "num_burn", "num_chains",
  "thin_mort", "thin_param"
)

# ----- IMPORT MORTALITY DATA -----
mortality <- load_data(
  data_path = here::here("Data"),
  region    = args$region,
  sex       = as.numeric(args$sex),
  test      = args$test
)

# add ID columns, hier3 as lowest level of hierarchy
if (args$region == "MSOA") {
  mortality <- mortality %>%
    mutate(
      hier1.id = GOR.id,
      hier2.id = LAD.id,
      hier3.id = MSOA.id
    )
} else if (args$region == "LSOA") {
  mortality <- mortality %>%
    mutate(
      hier1.id = LAD.id,
      hier2.id = MSOA.id,
      hier3.id = LSOA.id
    )
}

# ----- IMPORT INITIAL VALUES -----
if (!args$test) {
  initial <- readRDS(
    file = here::here(
      "Data", "Inits", paste0(args$region, args$sex, "_inits.rds")
    )
  )
} else {
  initial <- readRDS(
    file = here::here(
      "Data", "Inits", paste0(args$region, args$sex, "_T", "_inits.rds")
    )
  )
}

inital$r                <- 10.0
initial$sigma_alpha_age <- 0.6
initial$sigma_beta_age  <- 0.01
initial$sigma_nu        <- 0.05
initial$sigma_xi        <- 0.15
initial$sigma_gamma     <- 0.1

# ----- RUN MODEL PREPROCESSING ------
# reduced adjacency matrix information for BYM
model_inputs <- prep_model(
  data_path = here::here("Data"),
  mortality = mortality, 
  region    = args$region, 
  model     = args$model
)

# ----- SET UP CLUSTER AND RUN -----
if (as.numeric(args$num_chains) == 1) {
  print("----- RUNNING CHAIN -----")
  system.time(
    chain_output <- run_MCMC_allcode(
      seed       = 1,
      model_name = args$model,
      mortdata   = mortality,
      init_vals  = initial,
      n_iter     = as.numeric(args$num_iter),
      n_burn     = as.numeric(args$num_burn),
      thin_1     = as.numeric(args$thin_mort),
      thin_2     = as.numeric(args$thin_param),
      inputs     = model_inputs
    )
  )
  print("----- ONE CHAIN RUN -----")
} else {
  library(parallel)
  this_cluster <- makeCluster(as.numeric(args$num_chains))
  print(this_cluster)
  print("----- RUNNING CHAINS -----")
  system.time(
    chain_output <- parLapply(
      cl         = this_cluster,
      X          = 1:as.numeric(args$num_chains),
      fun        = run_MCMC_allcode,
      model_name = args$model,
      mortdata   = mortality,
      init_val   = initial,
      n_iter     = as.numeric(args$num_iter),
      n_burn     = as.numeric(args$num_burn),
      thin_1     = as.numeric(args$thin_mort),
      thin_2     = as.numeric(args$thin_param),
      inputs     = model_inputs
    )
  )
  print("----- CHAINS RUN -----")
  stopCluster(this_cluster)
}

print("----- SAVING DATA... -----")
if (!args$test) {
  saveRDS(
    chain_output,
    file = here::here(
      "Output", "mcmc_output",
      paste0(args$region, args$model, args$sex, "_mcmc_out.rds")
    )
  )
} else {
  saveRDS(
    chain_output,
    file = here::here(
      "Output", "mcmc_output",
      paste0(args$region, args$model, args$sex, "_T", "_mcmc_out.rds")
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(nimble)
})

source(here::here("Models", "parametric", "prepare_model.R"))
source(here::here("Models", "parametric", "nimble_model_debug.R"))

# test parameters for running interactively
args <- list("LSOA", "nested", "1", TRUE, "5", "0", "1", "1", "10")
names(args) <- list("region", "model", "sex", "test",
                    "num_iter", "num_burn", "num_chains",
                    "thin_mort", "thin_param")

# ----- IMPORT MORTALITY DATA -----
mortality <- load_data(
  data_path = here::here("Data"),
  region = args$region,
  sex = as.numeric(args$sex),
  test = args$test
)

# add ID columns, hier3 as lowest level of hierarchy
if (args$region == "MSOA") {
  mortality$hier1.id <- mortality$GOR.id
  mortality$hier2.id <- mortality$LAD.id
  mortality$hier3.id <- mortality$MSOA.id
} else if (args$region == "LSOA") {
  mortality$hier1.id <- mortality$LAD.id
  mortality$hier2.id <- mortality$MSOA.id
  mortality$hier3.id <- mortality$LSOA.id
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

# ----- RUN MODEL PREPROCESSING ------
# reduced adjacency matrix information for BYM
model_inputs <- prep_model(
  data_path = here::here("Data"),
  mortality = mortality, 
  region = args$region, 
  model = args$model
)

# ----- SET UP CLUSTER AND RUN -----
if (as.numeric(args$num_chains) == 1) {
  print("----- RUNNING CHAIN -----")
  system.time(chain_output <- run_MCMC_allcode(seed = 1,
                                               model_name = args$model,
                                               mortdata = mortality,
                                               init_vals = initial,
                                               n_iter = as.numeric(args$num_iter),
                                               n_burn = as.numeric(args$num_burn),
                                               thin_1 = as.numeric(args$thin_mort),
                                               thin_2 = as.numeric(args$thin_param),
                                               inputs = model_inputs))
  print("----- ONE CHAIN RUN -----")
} else stop("1chainplease")
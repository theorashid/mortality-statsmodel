# Theo AO Rashid -- October 2020

# ----- mr2le -----
# Convert mortality into life expectancy
# Uses n_pd samples without replacement

library(tidverse)
library(foreach)

source(here::here("Output", "analysis_utils.R"))
source(here::here("Output", "period_life_table.R"))
source(here::here("Models", "parametric", "prepare_model.R"))

set.seed(1)
print(paste0("AVAILABLE CORES = ", detectCores()))
doParallel::registerDoParallel(cores = 8)

# ----- IMPORT MORTALITY -----
region <- "LSOA"
model  <- "nested"
sex    <- 1 # 1 male, 2 female
test   <- TRUE

n_pd <- 1000 # number of posterior draws

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

# ----- SAMPLE MORTALITY -----
mr <- stack_chains(chain_output)
rm(chain_output)

mr <- exp(mr) # samples of death rate per person in each stratum

print(paste0("NUMBER OF SAMPLES = ", dim(mr)[1]))

# get n_pd subsamples
print(paste0("NUMBER OF POSTERIOR DRAWS = ", n_pd))

if (n_pd > dim(mr)[1]) {
  print("NUMBER OF POSTERIOR DRAWS GREATER THAN NUMBER OF AVAILABLE SAMPLES")
  print("USING ALL AVAILABLE SAMPLES")
  n_pd <- dim(mr)[1]
} else {
  print(paste0("NUMBER OF POSTERIOR DRAWS = ", n_pd))
  mr <- mr[sample(nrow(mr), n_pd, replace = FALSE), ]
}

# ----- ARRANGE BY AGE GROUP -----
mr <- as.data.frame(t(mr))
mr <- unlograte(mr) %>%
  arrange(YEAR.id, hier3.id, age_group.id) # confirm age groups are in order

strata <- mr %>%
  select(age_group.id, hier3.id, YEAR.id)
print(paste0("NUMBER OF STRATA = ", nrow(strata)))

# ---- SPLIT DATA INTO MANAGEABLE CHUNKS -----
print("----- CHUNKING DATA -----")
ages <- c(c(0, 1), seq(5, 85, 5))
n_ages  <- max(strata$age_group.id)
n_locs  <- max(strata$hier3.id)
n_years <- max(strata$YEAR.id)

# the num_groups_per_core divides each group for parallel into
# num_groups_per_core*num_ages ~ 1500 (for London) which is near-optimal
num_groups_per_core <- n_years * 5
chunk_size <- num_groups_per_core * n_ages
print(paste0("CHUNK SIZE = ", chunk_size))

mr <- mr %>%
  select(-c(age_group.id, hier3.id, YEAR.id)) # samples only
mr <- as.vector(t(stack(mr)[1])) # stack all on top of each other

# this sets mrsample into list of length (4835/5)*n_pd with each element a
# vector of size ~ 1500
mr <- split(mr, ceiling(seq_along(mr) / chunk_size))
print(paste0("NUMBER OF CHUNKS = ", length(mr)))

# ---- CALCULATE LIFE EXPECTANCIES -----
paste0("----- CALCULATING LIFE EXPECTANCIES -----")
# calculate life expectancies for each element in mrsample list
print(
  system.time({
    result <- foreach(mx = mr) %dopar% {
      PeriodLifeTable(
        age = rep(ages, num_groups_per_core),
        mx  = mx,
        ax  = rep(NA, n_ages * num_groups_per_core),
        sex = sex
      )$ex
    }
  })
)

result <- unlist(result) # remove the chunking by flattening
le <- result[seq(1, length(result), 19)] # le at birth is every 19th

le <- split(le, ceiling(seq_along(le) / (n_locs * n_years)))

# data frame of size n_strata, n_pd
le <- data.frame(matrix(unlist(le), ncol = n_pd))

paste0("----- MERGING WITH STRATA -----")
LE_df <- strata %>%
  select(c(YEAR.id, hier3.id)) %>%
  distinct()

LE_df <- bind_cols(LE_df, le)

strata <- load_data(
  data_path = data_path, region = region, sex = sex, test = test
)

if (region == "MSOA") {
  strata <- strata %>%
    select(
      MSOA.id, YEAR.id,
      MSOA2011, LAD2020, GOR2011, YEAR
    ) %>%
    distinct() %>%
    mutate(hier3.id = MSOA.id)
  LE_df <- left_join(strata, LE_df) %>%
    select(-c(MSOA.id, hier3.id, YEAR.id))
} else if (region == "LSOA") {
  strata <- strata %>%
    select(
      LSOA.id, YEAR.id,
      LSOA2011, MSOA2011, LAD2020, GOR2011, YEAR
    ) %>%
    distinct() %>%
    mutate(hier3.id = LSOA.id)
  LE_df <- left_join(strata, LE_df) %>%
    select(-c(LSOA.id, hier3.id, YEAR.id))
}

paste0("----- SAVING LIFE EXPECTANCY SAMPLES -----")
if (!test) {
  write_csv(
    LE_df,
    here::here("Output", "e0_samples",
    paste0(region, model, sex, "_e0_samples.csv"))
  )
} else {
  write_csv(
    LE_df,
    here::here("Output", "e0_samples",
    paste0(region, model, sex, "_T", "_e0_samples.csv"))
  )
}

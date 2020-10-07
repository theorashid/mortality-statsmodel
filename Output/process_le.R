# Theo AO Rashid -- June 2020

# ----- Process life expectancy -----
# Convert mortality into life expectancy

library(dplyr)
library(foreach)
library(doParallel)

data_path    <- "/rds/general/user/tar15/home/mortality-statsmodel/Data/"
output_path  <- "/rds/general/user/tar15/home/mortality-statsmodel/Output/"

source(paste0(output_path, "unlograte.R"))
source(paste0(output_path, "period_life_table.R"))

set.seed(1)
# numCores <- detectCores()
numCores <- 8
doParallel::registerDoParallel(cores = numCores)

# ----- IMPORT MORTALITY -----
name_run <- "hier"

sex_choice <- 1 # 1 male, 2 female
london     <- TRUE # if false, loads Hammersmith and Fulham

n_pd <- 1000 # number of posterior draws

paste0("----- OPENING MORTALITY -----")
if (london) {
  mr <- readRDS(paste0(output_path, name_run, sex_choice, "_ldn", "_samples.rds"))
} else {
  mr <- readRDS(paste0(output_path, name_run, sex_choice, "_hf", "_samples.rds"))
}

paste0("MORTALITY dim = ", dim(mr))
paste0("mr size = ", (object.size(mr)))

# ----- LIFE EXPECTANCIES -----
# Calculate LE from n_pd posterior draws
paste0("ARRANGING MORTALITY")
mrdf <- as.data.frame(t(mr))
paste0("mrdf size = ", (object.size(mrdf)))

# Arrange by year, then LSOA, then age group (so age groups are 1-19 for LE calculation)
mrdf <- unlograte(mrdf) %>%
  arrange(YEAR.id, LSOA.id, age_group.id)

strata <- mrdf %>%
  select(YEAR.id, LSOA.id, age_group.id)
paste0("strata size = ", (object.size(strata)))

paste0("number of strata = ", nrow(strata))

# Columns are each iteration, rows are strata
# Pick random sample of columns, apply LE calc

# first N columns are iterations, last 3 are year, LSOA, age group
N <- dim(mr)[1] # number of iterations (max of sample)
paste0("Number of posterior draws = ", n_pd)

mrsample <- select(mrdf, sample(N, n_pd)) # excludes id cols

rm(mrdf) # remove some data for memory reasons
rm(mr)

ages <- c(c(0, 1), seq(5, 85, 5))
num_locs  <- max(strata$LSOA.id)
num_years <- max(strata$YEAR.id)
num_ages  <- max(strata$age_group.id)
# this num_groups_per_core divides each group for parallel into 
# num_groups_per_core*num_ages ~ 1500 which is near-optimal
num_groups_per_core <- num_years*5
chunk_size <- num_groups_per_core*num_ages

mrsample <- as.vector(t(stack(mrsample)[1])) # stack all on top of each other

# this sets mrsample into list of length (4835/5)*n_pd with each element a 
# vector of size ~ 1500
mrsample <- split(mrsample, ceiling(seq_along(mrsample)/chunk_size))

paste0("CALCULATING LIFE EXPECTANCIES")
# calculate life expectancies for each element in mrsample list
system.time({
  result <- foreach(mx = mrsample) %dopar% {
    PeriodLifeTable(
      age = rep(ages, num_groups_per_core), 
      mx = mx, 
      ax = rep(NA, num_ages * num_groups_per_core), 
      sex = sex_choice
    )$ex
  }
})

result <- do.call(c, result) # remove the chunking
le <- result[seq(1, length(result), 19)] # le at birth is every 19th

le <- split(le, ceiling(seq_along(le)/(num_locs*num_years))) # split into n_pd elements

# data frame of size n_strata, n_pd
le <- data.frame(matrix(unlist(le), ncol = n_pd))

paste0("----- MERGING WITH STRATA -----")
LE_df <- strata %>%
  select(c(YEAR.id, LSOA.id)) %>%
  distinct()

LE_df <- bind_cols(LE_df, le)

# # Calculate mean, sd, ci for LEs
# LE <- cbind(
#   mean      = apply(le, 1, mean),
#   sd        = apply(le, 1, sd),
#   ci95low   = apply(le, 1, function(x) quantile(x, 0.025)),
#   ci95upp   = apply(le, 1, function(x) quantile(x, 0.975))
# )

paste0("----- SAVING LIFE EXPECTANCY SAMPLES -----")
if (london) {
  saveRDS(LE_df, paste0(output_path, name_run, sex_choice, "_ldn", "_le_samples.rds"))
} else {
  saveRDS(LE_df, paste0(output_path, name_run, sex_choice, "_hf", "_le_samples.rds"))
}

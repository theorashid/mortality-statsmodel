# Theo AO Rashid -- June 2020

# ----- Initial values -----
# Get inits for global intercept and slope,
# spatial intercepts and slopes,
# and age intercepts
# --------------------

library(dplyr)
library(lme4)

source(here::here("Models", "parametric", "prepare_model.R"))

region <- "LSOA"
sex <- 1

mortality <- load_data(data_path, region = region, sex = sex, test = FALSE)

# model with only age intercepts, MSOA/LSOA intercepts and MSOA/LSOA slopes
if (region == "MSOA") {
  system.time(
    mod <- glmer(deaths ~ offset(log(population)) + YEAR.id + (1 | age_group.id) + (1 + YEAR.id | MSOA.id), family = "poisson", data = subset(mortality, population > 0))
  )
  bin <- ranef(mod)$MSOA.id
} else if (region == "LSOA") {
  system.time(
    mod <- glmer(deaths ~ offset(log(population)) + YEAR.id + (1 | age_group.id) + (1 + YEAR.id | LSOA.id), family = "poisson", data = subset(mortality, population > 0))
  )
  bin <- ranef(mod)$LSOA.id
} else {
  stop("invalid region: MSOA or LSOA only")
}

fixed <- coef(summary(mod))[, "Estimate"] # fixed effects
intercept <- fixed[1]
slope <- fixed[2]

space_int_inits <- bin$"(Intercept)"
# plot(space_int_inits)
space_slope_inits <- bin$YEAR.id
# plot(space_slope_inits)

bin <- ranef(mod)$age_group.id
age_inits <- bin$"(Intercept)"
# plot(age_inits) # familiar swoosh

summary(mod)

inits <- list(intercept, slope, space_int_inits, space_slope_inits, age_inits)
names(inits) <- c(
  "global.intercept", "global.slope", "space.intercepts",
  "space.slopes", "age.intercepts"
)

saveRDS(
  inits,
  file = here::here(
    "Data", "Inits", paste0(region, sex, "_inits.rds")
  )
)

if (region == "MSOA") {
  sub <- mortality %>% filter(GOR2011 == "E12000007")
  inits_sub <- list(
    intercept,
    slope,
    space_int_inits[unique(sub$MSOA.id)],
    space_slope_inits[unique(sub$MSOA.id)],
    age_inits
  )
} else if (region == "LSOA") {
  sub <- mortality %>% filter(LAD2011 == "E09000013")
  inits_sub <- list(
    intercept,
    slope,
    space_int_inits[unique(sub$LSOA.id)],
    space_slope_inits[unique(sub$LSOA.id)],
    age_inits
  )
}

names(inits_sub) <- c(
  "global.intercept", "global.slope", "space.intercepts",
  "space.slopes", "age.intercepts"
)

saveRDS(
  inits_sub,
  file = here::here(
    "Data", "Inits", paste0(region, sex, "_T", "_inits.rds")
  )
)

suppressPackageStartupMessages({
  library(dplyr)
  library(nimble)
})

#' function required to run the mcmc chains in parallel
#' this will build a separate model on each core
#' the function body contains only the nested model, but see the BUGS files for different models
run_MCMC_allcode <- function(seed,
                             model_name,
                             mortdata,
                             init_vals,
                             n_iter,
                             n_burn,
                             thin_1,
                             thin_2,
                             inputs) {
  library(nimble)
  library(dplyr)

  # ----- BUILD THE MODEL -----
  # Indices:
  # - a -- age
  # - s -- space
  # - t -- year (time)

  code <- nimbleCode({
    # Theo AO Rashid -- October 2020

    # ----- Nested model -----
    # Three tier nested hierarchy
    # Negative binomial likelihood
    #
    # Global terms (normal prior) +
    # Random effects for tier 1 (normal prior) +
    # Random effects for tier 2 (normal prior, hierarchy) +
    # Random effects for tier 3 (smallest) (normal prior, hierarchy) +
    # Age effects (random walk prior) +
    # Age-space interaction (normal prior) +
    # Age random walk +
    # Space random walk
    #
    # Designed for the ONS hierarchy LSOA -> MSOA -> LAD / MSOA -> LAD -> Region (3 -> 2 -> 1)
    # --------------------

    # PRIORS
    alpha0 ~ dnorm(0, 0.00001)
    beta0 ~ dnorm(0, 0.00001)
    sigma_alpha_s1 ~ dunif(0, 2)
    sigma_beta_s1 ~ dunif(0, 2)
    sigma_alpha_s2 ~ dunif(0, 2)
    sigma_beta_s2 ~ dunif(0, 2)
    sigma_alpha_s3 ~ dunif(0, 2)
    sigma_beta_s3 ~ dunif(0, 2)
    sigma_alpha_age ~ dunif(0, 2)
    sigma_beta_age ~ dunif(0, 2)
    sigma_xi ~ dunif(0, 2)
    sigma_nu ~ dunif(0, 2)
    sigma_gamma ~ dunif(0, 2)
    r ~ dunif(0, 50)

    # AREA TERMS -- random effects for intercepts and slopes
    # Lower level term centred on higher level
    # tier 1 terms
    for (s1 in 1:N_s1) {
      alpha_s1[s1] ~ dnorm(0, sd = sigma_alpha_s1)
      beta_s1[s1] ~ dnorm(0, sd = sigma_beta_s1)
    }

    # tier 2 terms
    for (s2 in 1:N_s2) {
      alpha_s2[s2] ~ dnorm(alpha_s1[grid.lookup.s2[s2, 2]], sd = sigma_alpha_s2) # centred on s1 terms
      beta_s2[s2] ~ dnorm(beta_s1[grid.lookup.s2[s2, 2]], sd = sigma_beta_s2)
    }

    # tier 3 terms
    for (s in 1:N_space) {
      # s = s3, N_space = N_s3
      alpha_s3[s] ~ dnorm(alpha_s2[grid.lookup[s, 2]], sd = sigma_alpha_s3) # centred on s2 terms
      beta_s3[s] ~ dnorm(beta_s2[grid.lookup[s, 2]], sd = sigma_beta_s3)
    }

    # AGE TERMS
    alpha_age[1] <- alpha0 # initialise first terms for RW
    beta_age[1] <- beta0
    for (a in 2:N_age_groups) {
      alpha_age[a] ~ dnorm(alpha_age[a - 1], sd = sigma_alpha_age) # RW based on previous age group
      beta_age[a] ~ dnorm(beta_age[a - 1], sd = sigma_beta_age)
    }

    # INTERACTIONS
    # age-space interactions
    for (a in 1:N_age_groups) {
      for (s in 1:N_space) {
        xi[a, s] ~ dnorm(alpha_age[a] + alpha_s3[s], sd = sigma_xi) # centred on age + space term
      }
    }

    # space-time random walk
    for (s in 1:N_space) {
      nu[s, 1] <- 0
      for (t in 2:N_year) {
        nu[s, t] ~ dnorm(nu[s, t - 1] + beta_s3[s], sd = sigma_nu)
      }
    }

    # age-time random walk
    for (a in 1:N_age_groups) {
      gamma[a, 1] <- 0
      for (t in 2:N_year) {
        gamma[a, t] ~ dnorm(gamma[a, t - 1] + beta_age[a], sd = sigma_gamma)
      }
    }

    # Put all parameters together into indexed lograte term
    for (a in 1:N_age_groups) {
      for (s in 1:N_space) {
        for (t in 1:N_year) {
          lograte[a, s, t] <- xi[a, s] + nu[s, t] + gamma[a, t]
        }
      }
    }

    # LIKELIHOOD
    # N total number of cells, i.e. ages*years*areas(*sex)
    for (i in 1:N) {
      # y is number of deaths in that cell
      # mu is predicted number of deaths in that cell
      y[i] ~ dnegbin(p[i], r)
      p[i] <- r / (r + mu[i])
      log(mu[i]) <- log(n[i]) + lograte[age[i], space[i], yr[i]]
    }
  })

  constants <- list(
    N = nrow(mortdata),
    N_year = max(mortdata$YEAR.id),
    N_age_groups = max(mortdata$age_group.id),
    N_space = max(mortdata$hier3.id),
    age = mortdata$age_group.id,
    space = mortdata$hier3.id,
    yr = mortdata$YEAR.id,
    N_s1 = max(mortdata$hier1.id),
    N_s2 = max(mortdata$hier2.id),
    grid.lookup = inputs[[1]],
    grid.lookup.s2 = inputs[[2]]
  )

  inits <- list(
    alpha0 = init_vals$global.intercept,
    beta0 = init_vals$global.slope,
    alpha_age = init_vals$global.intercept + init_vals$age.intercepts,
    sigma_alpha_age = init_vals$sigma_alpha_age,
    sigma_beta_age = init_vals$sigma_beta_age,
    sigma_nu = init_vals$sigma_nu,
    sigma_xi = init_vals$sigma_xi,
    sigma_gamma = init_vals$sigma_gamma,
    r = init_vals$r,
    alpha_s3 = init_vals$space.intercepts,
    beta_s3 = init_vals$space.slopes,
    sigma_alpha_s1 = 0.1,
    sigma_beta_s1 = 0.01,
    sigma_alpha_s2 = 0.1,
    sigma_beta_s2 = 0.01,
    sigma_alpha_s3 = 0.1,
    sigma_beta_s3 = 0.01
  )

  data <- list(
    y = mortdata$deaths,
    n = mortdata$population
  )

  # ----- CREATE THE MODEL IN R -----
  model <- nimbleModel(
    code      = code,
    constants = constants,
    inits     = inits,
    data      = data,
    calculate = FALSE
  )
  print("----- MODEL BUILT -----")

  # ----- COMPILE THE MODEL IN C -----
  Cmodel <- compileNimble(model)
  print("----- MODEL COMPILED -----")

  # ----- MCMC INTEGRATION -----
  # Monitor the death rate per person to avoid 0 population issues
  monitors <- c("lograte")

  # Hyperparameter monitors to check covergence, with some thinning
  sigmas <- c(
    "sigma_alpha_age", "sigma_beta_age",
    "sigma_xi", "sigma_gamma",
    "sigma_alpha_s1", "sigma_beta_s1",
    "sigma_alpha_s2", "sigma_beta_s2",
    "sigma_alpha_s3", "sigma_beta_s3"
  )
  monitors2 <- c(
    "r", "alpha0", "beta0", "alpha_age", "beta_age", "xi",
    "alpha_s1", "beta_s1",
    "alpha_s2", "beta_s2",
    "alpha_s3", "beta_s3",
    sigmas
  )

  # CUSTOMISABLE MCMC -- configureMCMC, buildMCMC, compileNimble, runMCMC
  # 1. MCMC Configuration -- can be customised with different samplers
  mcmcConf <- configureMCMC(
    model     = Cmodel,
    monitors  = monitors,
    monitors2 = monitors2,
    thin      = thin_1,
    thin2     = thin_2,
    print     = TRUE
  )

  # sample standard deviations on log scale
  mcmcConf$removeSamplers(sigmas)
  for (s in sigmas) {
    mcmcConf$addSampler(target = s, type = "RW", control = list(log = TRUE))
  }
  print("----- MCMC CONFIGURED -----")

  # 2. Build and compile the MCMC
  Rmcmc <- buildMCMC(mcmcConf) # Set enableWAIC = TRUE if we need to calculate WAIC
  print("----- MCMC BUILT -----")
  Cmcmc <- compileNimble(Rmcmc)
  print("----- MCMC COMPILED -----")

  # 3. Run MCMC
  # Return samples only
  mcmc.out <- runMCMC(
    Cmcmc,
    niter       = n_iter,
    nburnin     = n_burn,
    # nchains = 1, summary = TRUE,
    progressBar = TRUE,
    setSeed     = seed
  )

  return(mcmc.out)
}

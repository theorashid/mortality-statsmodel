suppressPackageStartupMessages({
  library(dplyr)
  library(nimble)
})

print(sessionInfo())

#' Matern kernel function defined as a nimbleFunction
dbetabin <- nimbleFunction(
  run = function(
      x = double(0),
      alpha = double(0),
      beta = double(0),
      size = double(0),
      log = integer(0, default = 0)
    ) {

    returnType(double(0))
    logProb <- lgamma(size+1) - lgamma(x+1) - lgamma(size - x + 1) +
      lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta) +
      lgamma(x + alpha) + lgamma(size - x + beta) - lgamma(size + alpha + beta)
    if(log) return(logProb)
    else return(exp(logProb))
  })

rbetabin <- nimbleFunction(
  run = function(
      n = integer(0),
      alpha = double(0),
      beta = double(0),
      size = double(0)
    ) {
    returnType(double(0))
    if(n != 1) print("rbetabin only allows n = 1; using n = 1.")
    p <- rbeta(1, alpha, beta)
    return(rbinom(1, size = size, prob = p))
  })

#' function required to run the mcmc chains in parallel
#' this will build a separate model on each core
#' the function body contains the BYM, nested and GP models
run_MCMC_allcode <- function(
    seed,
    mortdata,
    lookup,
    n_iter,
    n_burn,
    thin_1,
    thin_2
) {
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
    # Beta-binomial likelihood
    #
    # Global terms (normal prior) +
    # Random effects for tier 1 (normal prior) +
    # Random effects for tier 2 (normal prior, hierarchy) +
    # Random effects for tier 3 (smallest) (normal prior, hierarchy) +
    # Age effects (random walk prior) +
    # Age-space interaction (normal prior) +
    # Global random walk
    #
    # Designed for the hierarchy Region (3 -> 2 -> 1)
    # --------------------

    # PRIORS

    # COMMON TERMS
    alpha0 ~ dnorm(0, sd = 100)
    beta0  ~ dnorm(0, sd = 100)

    # AREA TERMS -- random effects for intercepts and slopes
    # Lower level term centred on higher level

    # tier 1 terms
    for(s1 in 1:N_s1){
      alpha_s1[s1] ~ dnorm(0, sd = sigma_alpha_s1)
      beta_s1[s1] ~ dnorm(0, sd = sigma_beta_s1)
    }
    sigma_alpha_s1 ~ dunif(0, 2)
    sigma_beta_s1  ~ dunif(0, 2)

    # tier 2 terms
    for(s in 1:N_s2){
      alpha_s2[s] ~ dnorm(alpha_s1[grid.lookup.s2[s, 2]], sd = sigma_alpha_s2)
      beta_s2[s] ~ dnorm(beta_s1[grid.lookup.s2[s, 2]], sd = sigma_beta_s2)
    }
    sigma_alpha_s2 ~ dunif(0, 2)
    sigma_beta_s2  ~ dunif(0, 2)

    # AGE TERMS
    alpha_age[1] <- alpha0 # initialise first terms for RW
    beta_age[1]  <- beta0
    for(a in 2:N_age_groups){
      alpha_age[a] ~ dnorm(alpha_age[a - 1], sd = sigma_alpha_age)
      beta_age[a]  ~ dnorm(beta_age[a - 1], sd = sigma_beta_age)
    }
    sigma_alpha_age ~ dunif(0, 2)
    sigma_beta_age ~ dunif(0, 2)

    # INTERACTIONS
    # age-space interactions
    for(a in 1:N_age_groups) {
      for(s in 1:N_s2) {
        xi[a, s] ~ dnorm(alpha_age[a] + alpha_s2[s], sd = sigma_xi)
      }
    }
    sigma_xi ~ dunif(0, 2)

    # space-time random walk
    for(s in 1:N_s2){
        nu[s, 1] <- 0
        for(t in 2:N_year) {
            nu[s, t] ~ dnorm(nu[s, t - 1] + beta_s2[s], sd = sigma_nu)
        }
    }
    sigma_nu ~ dunif(0,2)

    # age-time random walk
    for(a in 1:N_age_groups){
        gamma[a, 1] <- 0
        for(t in 2:N_year) {
            gamma[a, t] ~ dnorm(gamma[a, t - 1] + beta_age[a], sd = sigma_gamma)
        }
    }
    sigma_gamma ~ dunif(0, 2)

    # Put all parameters together into indexed lograte term
    for(a in 1:N_age_groups) {
      for(s in 1:N_s2) {
        for(t in 1:N_year) {
          rate[a, s, t] <- xi[a, s] + nu[s, t] + gamma[a, t]
        }
      }
    }

    # LIKELIHOOD
    # N total number of cells, i.e. ages*years*areas(*sex)
    for (i in 1:N) {
      # y is number of deaths in that cell
      # mu is predicted number of deaths in that cell
      y[i] ~ dbetabin(alpha[i], beta[i], n[i])
      alpha[i] <- mu[i] * theta
      beta[i] <- (1 - mu[i]) * theta
      logit(mu[i]) <- rate[age[i], space[i], yr[i]]
    }
    theta ~ dexp(rate = 0.1)
  })

  constants <- list(
    N = nrow(mortdata),
    N_year = max(mortdata$YEAR.id),
    N_age_groups = max(mortdata$age_group.id),
    N_s1 = max(mortdata$hier1.id),
    N_s2 = max(mortdata$hier2.id),
    age = mortdata$age_group.id,
    space = mortdata$hier2.id,
    yr = mortdata$YEAR.id,
    grid.lookup.s2 = lookup
  )
  
  inits <- list(
    alpha0 = -5.1,
    beta0 = -0.04,
    sigma_alpha_age = 1.0,
    sigma_beta_age = 0.003,
    sigma_alpha_s1 = 0.15,
    sigma_beta_s1 = 0.006,
    sigma_alpha_s2 = 0.15,
    sigma_beta_s2 = 0.003,
    sigma_xi = 0.26,
    sigma_gamma = 0.028,
    sigma_nu = 0.063,
    theta = 1400
  )

  data <- list(y = mortdata$deaths, n = mortdata$population)

  # ----- CREATE THE MODEL -----
  print("MODEL BUILD TIME...")
  print(
    system.time(
      model <- nimbleModel(
          code = code,
          constants = constants,
          inits = inits,
          data = data,
          calculate = FALSE)
    )
  )
  print("----- MODEL BUILT -----")

  # ----- COMPILE THE MODEL IN C-CODE -----
  print("MODEL COMPILE TIME...")
  print(system.time(Cmodel <- compileNimble(model)))
  print("----- MODEL COMPILED -----")
  
  # ----- MCMC INTEGRATION -----
  # Monitor the death rate per person to avoid 0 population issues
  monitors <- c("mu")

  # Hyperparameter monitors to check covergence, with some thinning
  sigmas <- c(
      "sigma_alpha_age", "sigma_beta_age",
      "sigma_xi", "sigma_gamma", "sigma_nu",
      "sigma_alpha_s1", "sigma_beta_s1",
      "sigma_alpha_s2", "sigma_beta_s2"
    )
  monitors2 <- c("theta", "alpha_age", "beta_age", "alpha0", "beta0", sigmas)

  # CUSTOMISABLE MCMC -- configureMCMC, buildMCMC, compileNimble, runMCMC
  # 1. MCMC Configuration -- can be customised with different samplers
  mcmcConf <- configureMCMC(
      model = Cmodel,
      monitors = monitors,
      monitors2 = monitors2,
      thin = thin_1,
      thin2 = thin_2,
      print = TRUE
    ) # input the R model

  # sample standard deviations on log scale
  mcmcConf$removeSamplers(sigmas)
  for (s in sigmas) {
    mcmcConf$addSampler(target = s, type = "RW", control = list(log = TRUE))
  }
  print("----- MCMC CONFIGURED -----")

  # 2. Build and compile the MCMC
  Rmcmc <- buildMCMC(mcmcConf)
  print("----- MCMC BUILT -----")
  Cmcmc <- compileNimble(Rmcmc)
  print("----- MCMC COMPILED -----")

  # 3. Run MCMC
  # Return samples only
  mcmc.out <- runMCMC(
      Cmcmc,
      niter = n_iter,
      nburnin = n_burn,#nchains = 1,summary = TRUE,
      progressBar = TRUE,
      setSeed = seed
    )
  return(mcmc.out)
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(nimble)
})

# ----- IMPORT MORTALITY DATA -----
mortality <- read_csv("Models/parametric/simulated_mortality.csv") %>%
    mutate(
        hier1.id = s1 + 1,
        hier2.id = s2 + 1,
        YEAR.id = t + 1,
        age_group.id = a + 1
    )

grid.lookup <- mortality %>%
    select(hier2.id, hier1.id) %>%
    distinct() %>%
    arrange(hier2.id)

system.time(
    chain_output <- run_MCMC_allcode(
        seed = 1,
        mortdata = mortality,
        lookup = grid.lookup,
        n_iter = 1000,
        n_burn = 100,
        thin_1 = 1,
        thin_2 = 1
    )
)

library(parallel)
this_cluster <- makeCluster(4)
print(this_cluster)
print("----- EXPORTING FUNCTIONS -----")
clusterExport(this_cluster, c("dbetabin"))
clusterExport(this_cluster, c("rbetabin"))
print("----- RUNNING CHAINS -----")
print(
  system.time(chain_output <- parLapply(cl = this_cluster,
                                        X = 1:as.numeric(args$num_chains),
                                        fun = run_MCMC_allcode,
                                        model_name = args$model,
                                        mortdata = mortality,
                                        init_val = initial,
                                        n_iter = as.numeric(args$num_iter),
                                        n_burn = as.numeric(args$num_burn),
                                        thin_1 = as.numeric(args$thin_mort),
                                        thin_2 = as.numeric(args$thin_param),
                                        inputs = model_inputs
  ))
)
  
print("----- CHAINS RUN -----")
stopCluster(this_cluster)


print("----- SAVING DATA... -----")
saveRDS(chain_output, file = paste0(output_path, "mcmc_output/",
                                    "betabintest_mcmc_out.rds"))

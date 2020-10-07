suppressPackageStartupMessages({
  library(dplyr)
  library(nimble)
})

#' Matern kernel function defined as a nimbleFunction
maternKernel <- nimbleFunction(     
  run = function(dists = double(2), l = double(0), sigma = double(0)) {
    returnType(double(2))
    n <- dim(dists)[1]
    result <- matrix(nrow = n, ncol = n, init = FALSE)
    sigma2 <- sigma*sigma   # calculate once
    l2 <- l*l
    for(i in 1:n)
      for(j in 1:n)
        # if-then-else statement remove for speed
        # comment out the required function (nu = 1/2, 3/2, 5/2)
        # result[i, j] <- sigma2 * exp(-dists[i,j]/l)
        # result[i, j] <- sigma2 * (1 + sqrt(3)*dists[i,j]/l) * exp(-sqrt(3)*dists[i,j]/l)
        result[i, j] <- sigma2 * (1 + sqrt(5)*dists[i,j]/l + 5*dists[i,j]^2 / (3*l2)) * exp(-sqrt(5)*dists[i,j]/l)
    return(result)
  })

#' function required to run the mcmc chains in parallel
#' this will build a separate model on each core
#' the function body contains the BYM, nested and GP models
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
  if (model_name == "BYM") {
    code <- nimbleCode({
      # Theo AO Rashid -- October 2020
      
      # ----- BYM model -----
      # Negative binomial likelihood
      #
      # Global terms (normal prior) +
      # BYM for each space +
      # Age effects (random walk prior) +
      # Age-space interaction (normal prior) +
      # Global random walk
      # --------------------
      
      # PRIORS
      
      # COMMON TERMS
      alpha0 ~ dnorm(0, 0.00001)
      beta0  ~ dnorm(0, 0.00001)
      
      # AREA TERMS -- BYM priors
      # No hierarchy
      # Structured intercept and slope with a CAR prior
      alpha_u[1:N_space] ~ dcar_normal(adj[1:L], weights[1:L], num[1:N_space], tau_alpha_u, zero_mean = 0)
      beta_u[1:N_space]  ~ dcar_normal(adj[1:L], weights[1:L], num[1:N_space], tau_beta_u, zero_mean = 0)
      # Unstructured IID intercept and slope
      for(s in 1:N_space) {
        alpha_v[s] ~ dnorm(alpha0 + alpha_u[s], sd = sigma_alpha_v) # centred on common + CAR term
        beta_v[s]  ~ dnorm(beta0 + beta_u[s], sd = sigma_beta_v)
      }
      sigma_alpha_u ~ dunif(0,2)
      tau_alpha_u <- pow(sigma_alpha_u,-2)
      sigma_beta_u ~ dunif(0,2)
      tau_beta_u <- pow(sigma_beta_u,-2)
      sigma_alpha_v ~ dunif(0,2)
      sigma_beta_v ~ dunif(0,2)
      
      # AGE TERMS
      alpha_age[1] <- 0 # initialise first terms for RW
      beta_age[1]  <- 0
      for(a in 2:N_age_groups) {
        alpha_age[a] ~ dnorm(alpha_age[a-1], sd = sigma_alpha_age) # RW based on previous age group
        beta_age[a]  ~ dnorm(beta_age[a-1], sd = sigma_beta_age)
      }
      sigma_alpha_age ~ dunif(0,2)
      sigma_beta_age ~ dunif(0,2)
      
      # INTERACTIONS
      # age-space interactions
      for(a in 1:N_age_groups) {
        for(s in 1:N_space) {
          xi[a, s] ~ dnorm(alpha_age[a] + alpha_v[s], sd = sigma_xi) # centred on age + space term
        }
      }
      sigma_xi ~ dunif(0,2)
      
      # space slope (as no space-time random walk)
      for(s in 1:N_space){
        space_slope[s, 1] <- 0
        for(t in 2:N_year) {
          space_slope[s, t] <- space_slope[s, t-1] + beta_v[s]
        }
      }
      
      # age slope (as no age-time random walk)
      for(a in 1:N_age_groups){
        age_slope[a, 1] <- 0
        for(t in 2:N_year) {
          age_slope[a, t] <- age_slope[a, t-1] + beta_age[a]
        }
      }
      
      # global random walk for non-linear time
      pi[1] <- 0
      for (t in 2:N_year) {
        pi[t] ~ dnorm(pi[t-1], sd = sigma_pi)
      }
      sigma_pi ~ dunif(0, 2)
      
      # Put all parameters together into indexed lograte term
      for(a in 1:N_age_groups) {
        for(s in 1:N_space) {
          for(t in 1:N_year) {
            lograte[a, s, t] <- xi[a, s] + space_slope[s, t] + age_slope[a, t] + pi[t]
          }
        }
      }
      
      # LIKELIHOOD
      # N total number of cells, i.e. ages*years*areas(*sex)
      for (i in 1:N) {
        # y is number of deaths in that cell
        # mu is predicted number of deaths in that cell
        y[i] ~ dnegbin(p[i], r)
        p[i] <- r/(r + mu[i])
        log(mu[i]) <- log(n[i]) + lograte[age[i], space[i], yr[i]]
      }
      r ~ dunif(0,50)
    })
    
    specific_constants <- list(L = length(inputs$adj), # inputs = nbInfo
                               adj = inputs$adj,
                               weights = inputs$weights,
                               num = inputs$num)
    specific_inits <- list(alpha0 = init_vals$global.intercept,
                           beta0 = init_vals$global.slope,
                           alpha_u = init_vals$space.intercepts,
                           beta_u = init_vals$space.slopes,
                           sigma_alpha_u = 0.1, sigma_beta_u = 0.1,
                           sigma_alpha_v = 0.1, sigma_beta_v = 0.1)
    
    specific_sigmas <- c("sigma_alpha_u", "sigma_beta_u",
                         "sigma_alpha_v", "sigma_beta_v")
    specific_monitors2 <- c("alpha0", "beta0")
    
  } else if (model_name == "nested") {
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
      # Global random walk
      #
      # Designed for the ONS hierarchy LSOA -> MSOA -> LAD / MSOA -> LAD -> Region (3 -> 2 -> 1)
      # --------------------
      
      # PRIORS
      
      # COMMON TERMS
      alpha0 ~ dnorm(0, 0.00001)
      beta0  ~ dnorm(0, 0.00001)
      
      # AREA TERMS -- random effects for intercepts and slopes
      # Lower level term centred on higher level
      
      # tier 1 terms
      for(s1 in 1:N_s1){
        alpha_s1[s1] ~ dnorm(alpha0, sd = sigma_alpha_s1) # centred on common terms
        beta_s1[s1] ~ dnorm(beta0, sd = sigma_beta_s1)
      }
      sigma_alpha_s1 ~ dunif(0,2)
      sigma_beta_s1  ~ dunif(0,2)
      
      # tier 2 terms
      for(s2 in 1:N_s2){
        alpha_s2[s2] ~ dnorm(alpha_s1[grid.lookup.s2[s2, 2]], sd = sigma_alpha_s2) # centred on s1 terms
        beta_s2[s2] ~ dnorm(beta_s1[grid.lookup.s2[s2, 2]], sd = sigma_beta_s2)
      }
      sigma_alpha_s2 ~ dunif(0,2)
      sigma_beta_s2  ~ dunif(0,2)
      
      # tier 3 terms
      for(s in 1:N_space){ # s = s3, N_space = N_s3
        alpha_s3[s] ~ dnorm(alpha_s2[grid.lookup[s, 2]], sd = sigma_alpha_s3) # centred on s2 terms
        beta_s3[s]  ~ dnorm(beta_s2[grid.lookup[s, 2]], sd = sigma_beta_s3)
      }
      sigma_alpha_s3 ~ dunif(0,2)
      sigma_beta_s3  ~ dunif(0,2)
      
      # AGE TERMS
      alpha_age[1] <- 0 # initialise first terms for RW
      beta_age[1]  <- 0
      for(a in 2:N_age_groups){
        alpha_age[a] ~ dnorm(alpha_age[a-1], sd = sigma_alpha_age) # RW based on previous age group
        beta_age[a]  ~ dnorm(beta_age[a-1], sd = sigma_beta_age)
      }
      sigma_alpha_age ~ dunif(0,2)
      sigma_beta_age ~ dunif(0,2)
      
      # INTERACTIONS
      # age-space interactions
      for(a in 1:N_age_groups) {
        for(s in 1:N_space) {
          xi[a, s] ~ dnorm(alpha_age[a] + alpha_s3[s], sd = sigma_xi) # centred on age + space term
        }
      }
      sigma_xi ~ dunif(0,2)
      
      # space slope (as no space-time random walk)
      for(s in 1:N_space){
        space_slope[s, 1] <- 0
        for(t in 2:N_year) {
          space_slope[s, t] <- space_slope[s, t-1] + beta_s3[s]
        }
      }
      
      # age slope (as no age-time random walk)
      for(a in 1:N_age_groups){
        age_slope[a, 1] <- 0
        for(t in 2:N_year) {
          age_slope[a, t] <- age_slope[a, t-1] + beta_age[a]
        }
      }
      
      # global random walk for non-linear time
      pi[1] <- 0
      for (t in 2:N_year) {
        pi[t] ~ dnorm(pi[t-1], sd = sigma_pi)
      }
      sigma_pi ~ dunif(0, 2)
      
      # Put all parameters together into indexed lograte term
      for(a in 1:N_age_groups) {
        for(s in 1:N_space) {
          for(t in 1:N_year) {
            lograte[a, s, t] <- xi[a, s] + space_slope[s, t] + age_slope[a, t] + pi[t]
          }
        }
      }
      
      # LIKELIHOOD
      # N total number of cells, i.e. ages*years*areas(*sex)
      for (i in 1:N) {
        # y is number of deaths in that cell
        # mu is predicted number of deaths in that cell
        y[i] ~ dnegbin(p[i], r)
        p[i] <- r/(r + mu[i])
        log(mu[i]) <- log(n[i]) + lograte[age[i], space[i], yr[i]]
      }
      r ~ dunif(0,50)
    })
    
    specific_constants <- list(N_s1 = max(mortdata$hier1.id),
                               N_s2 = max(mortdata$hier2.id),
                               grid.lookup = inputs[[1]],
                               grid.lookup.s2 = inputs[[2]])
    specific_inits <- list(alpha0 = init_vals$global.intercept,
                           beta0 = init_vals$global.slope,
                           alpha_s3 = init_vals$space.intercepts,
                           beta_s3 = init_vals$space.slopes,
                           sigma_alpha_s1 = 0.1, sigma_beta_s1 = 0.1,
                           sigma_alpha_s2 = 0.1, sigma_beta_s2 = 0.1,
                           sigma_alpha_s3 = 0.1, sigma_beta_s3 = 0.1)
    
    specific_sigmas <- c("sigma_alpha_s1", "sigma_beta_s1",
                         "sigma_alpha_s2", "sigma_beta_s2",
                         "sigma_alpha_s3", "sigma_beta_s3")
    specific_monitors2 <- c("alpha0", "beta0")
    
  } else if (model_name == "GP") {
    code <- nimbleCode({
      # Theo AO Rashid -- October 2020
      
      # ----- GP model -----
      # Negative binomial likelihood
      #
      # Gaussian process over space +
      # Age effects (random walk prior) +
      # Age-space interaction (normal prior) +
      # Global random walk
      #
      # Requires MaternKernel as a nimbleFunction
      
      # PRIORS
      
      # AREA TERMS
      # GP for intercepts
      mu_alpha[1:N_space] <- mu0_alpha*ones[1:N_space]
      cov_alpha[1:N_space, 1:N_space] <- maternKernel(dists[1:N_space, 1:N_space], l_alpha, sigma_alpha)
      alpha_space[1:N_space] ~ dmnorm(mu_alpha[1:N_space], cov = cov_alpha[1:N_space, 1:N_space])
      mu0_alpha   ~ dnorm(0, sd = 100)
      sigma_alpha ~ dunif(0, 2)
      l_alpha     ~ dunif(0, 5)
      
      # GP for slopes
      mu_beta[1:N_space] <- mu0_beta*ones[1:N_space]
      cov_beta[1:N_space, 1:N_space] <- maternKernel(dists[1:N_space, 1:N_space], l_beta, sigma_beta)
      beta_space[1:N_space] ~ dmnorm(mu_beta[1:N_space], cov = cov_beta[1:N_space, 1:N_space])
      mu0_beta   ~ dnorm(0, sd = 100)
      sigma_beta ~ dunif(0, 2)
      l_beta     ~ dunif(0, 5)
      # AGE TERMS
      alpha_age[1] <- 0 # initialise first terms for RW
      beta_age[1]  <- 0
      for(a in 2:N_age_groups){
        alpha_age[a] ~ dnorm(alpha_age[a-1], sd = sigma_alpha_age) # RW based on previous age group
        beta_age[a]  ~ dnorm(beta_age[a-1], sd = sigma_beta_age)
      }
      sigma_alpha_age ~ dunif(0,2)
      sigma_beta_age ~ dunif(0,2)
      
      # INTERACTIONS
      # age-space interactions
      for(a in 1:N_age_groups) {
        for(s in 1:N_space) {
          xi[a, s] ~ dnorm(alpha_age[a] + alpha_space[s], sd = sigma_xi) # centred on age + space term
        }
      }
      sigma_xi ~ dunif(0,2)
      
      # space slope (as no space-time random walk)
      for(s in 1:N_space){
        space_slope[s, 1] <- 0
        for(t in 2:N_year) {
          space_slope[s, t] <- space_slope[s, t-1] + beta_space[s]
        }
      }
      
      # age slope (as no age-time random walk)
      for(a in 1:N_age_groups){
        age_slope[a, 1] <- 0
        for(t in 2:N_year) {
          age_slope[a, t] <- age_slope[a, t-1] + beta_age[a]
        }
      }
      
      # global random walk for non-linear time
      pi[1] <- 0
      for (t in 2:N_year) {
        pi[t] ~ dnorm(pi[t-1], sd = sigma_pi)
      }
      sigma_pi ~ dunif(0, 2)
      
      # Put all parameters together into indexed lograte
      for(a in 1:N_age_groups) {
        for(s in 1:N_space) {
          for(t in 1:N_year) {
            lograte[a, s, t] <- xi[a, s] + space_slope[s, t] + age_slope[a, t] + pi[t]
          }
        }
      }
      
      # LIKELIHOOD
      # N total number of cells, i.e. ages*years*areas(*sex)
      for (i in 1:N) {
        # y is number of deaths in that cell
        # mu is predicted number of deaths in that cell
        y[i] ~ dnegbin(p[i], r)
        p[i] <- r/(r + mu[i])
        log(mu[i]) <- log(n[i]) + lograte[age[i], space[i], yr[i]]
      }
      r ~ dunif(0,50)
    })
    
    specific_constants <- list(dists = inputs,
                               ones = rep(1, max(mortdata$hier3.id)))
    specific_inits <- list(mu0_alpha = init_vals$global.intercept,
                           mu0_beta = init_vals$global.slope,
                           sigma_alpha = 0.1, sigma_beta = 0.1,
                           alpha_space = init_vals$space.intercepts + init_vals$global.intercept,
                           beta_space = init_vals$space.slopes + init_vals$global.slope)
      
    specific_sigmas <- c("sigma_alpha", "sigma_beta")
    specific_monitors2 <- c("mu0_alpha", "mu0_beta", "l_alpha", "l_beta")
    
  } else stop("BYM, nested or GP model names only")
  
  constants <- c(list(
    N = nrow(mortdata),
    N_year = max(mortdata$YEAR.id),
    N_age_groups = max(mortdata$age_group.id),
    N_space = max(mortdata$hier3.id),
    age = mortdata$age_group.id,
    space = mortdata$hier3.id, 
    yr = mortdata$YEAR.id
  ), specific_constants)
  
  inits <- c(list(
    alpha_age = init_vals$age.intercepts,
    sigma_alpha_age = 0.75, sigma_beta_age = 0.015,
    sigma_xi = 0.08, sigma_pi = 0.1, r = 20.0
  ), specific_inits)
  
  data <- list(y = mortdata$deaths,
               n = mortdata$population)
  
  # ----- CREATE THE MODEL -----
  model <- nimbleModel(code = code, constants = constants,
                       inits = inits, data = data,
                       calculate = FALSE) # model in R
  print("----- MODEL BUILT -----")
  
  # ----- COMPILE THE MODEL IN C-CODE -----
  Cmodel <- compileNimble(model)
  print("----- MODEL COMPILED -----")
  
  # ----- MCMC INTEGRATION -----
  # Monitor the death rate per person to avoid 0 population issues
  monitors <- c("lograte")
  
  # Hyperparameter monitors to check covergence, with some thinning
  sigmas <- c("sigma_alpha_age", "sigma_beta_age",
              "sigma_xi", "sigma_pi",
              specific_sigmas)
  monitors2 <- c("r", "alpha_age", "beta_age", 
                 specific_monitors2, sigmas)
  
  # CUSTOMISABLE MCMC -- configureMCMC, buildMCMC, compileNimble, runMCMC
  # 1. MCMC Configuration -- can be customised with different samplers
  mcmcConf <- configureMCMC(model = Cmodel,
                            monitors = monitors, monitors2 = monitors2,
                            thin = thin_1, thin2 = thin_2, print = TRUE) # input the R model
  
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
  mcmc.out <- runMCMC(Cmcmc,
                      niter = n_iter, nburnin = n_burn, #nchains = 1, summary = TRUE,
                      progressBar = TRUE, setSeed = seed)
  
  return(mcmc.out)
}

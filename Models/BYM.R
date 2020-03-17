# Theo AO Rashid -- March 2020

# ----- BYM model -----
# Common terms (normal prior) +
# BYM for each LSOA +
# Age effects (random walk prior)
# --------------------

library(rgdal)
library(spdep)
library(dplyr)
library(nimble)
library(igraph)

set.seed(1)

# ----- IMPORT DATA -----
# Mortality data

# Shape data
test <- readOGR(dsn = "Data/shapefiles/ldn_LSOA11",
                layer = "LSOA11_London")
# Merge with mortality

# test1 = subset(test, code=="HF") # load for hammersmith and fulham

# Extract adjacency matrix
W.nb <- poly2nb(test, row.names =  rownames(test@data))
nbInfo <- nb2WB(W.nb)
# nbInfo$adj
# adj = nbInfo$adj, weights = nbInfo$weights, num = nbInfo$num
# s[1:N] ~ dcar_normal(adj[1:L], weights[1:L], num[1:N], tau, zero_mean = 0)

# BYM - u structured CAR effect, v unstructured
# N is number of spatial units
# L is the length of the adjacency matrix
u[1:N] ~ dcar_normal(adj[1:L], weights[1:L], num[1:N], tau_u, zero_mean = 0)
for (j in 1:N) {
    tmp[j] <- alpha + u[j]
    v[j] ~ dnorm(tmp[j], tau_v)
}
sigma_u ~ dunif(0,2)
tau_u <- pow(sigma_u,-2)
sigma_v ~ dunif(0,2)
tau_v <- pow(sigma_v,-2)

# BYM2 pseudocode
# unstructured effect needs to be scaled so the mean of the marginal variances
# of the precision matrix is equal to one
u[1:N] ~ dcar_normal(adj[1:L], weights[1:L], num[1:N], tau_u, zero_mean = 0) # THIS BUT SCALED
for (j in 1:N) {
    # v[j] is N(0, 1)
    b[j] ~ (1 / sqrt(tau_b)) * ((sqrt(phi) * dnorm(0, 1)) + sqrt(1 - phi) * u[1:N])
}
phi ~ dnorm(0.0, 1.0) # half normal prior
tau_b ~ dbeta(0.5, 0.5)
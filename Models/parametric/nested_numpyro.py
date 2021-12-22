# %%
import pandas as pd
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import jax.numpy as jnp
from jax import random
from jax.scipy.special import expit
import arviz as az

numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

# %%
__author__ = "Theo Rashid"
__email__ = "tar15@ic.ac.uk"

data = pd.read_csv("/Users/tar15/Dropbox/SPH/Data/Mortality (simulated)/mortality_hf_ac_LSOA.csv")
# %%
data = data[data["sex"] == 1]
data = data.assign(
    year_id  = data["YEAR"].astype("category").cat.codes,
    age_id   = data["age_group"].astype("category").cat.codes,
    hier1_id = data["LAD2020"].astype("category").cat.codes,
    hier2_id = data["MSOA2011"].astype("category").cat.codes,
    hier3_id = data["LSOA2011"].astype("category").cat.codes,
)


# %%
def model(
    age_id,
    hier1_id,
    hier2_id,
    hier3_id,
    year_id,
    population,
    deaths=None
):

    N_s1 = len(np.unique(hier1_id))
    N_s2 = len(np.unique(hier2_id))
    N_s3 = len(np.unique(hier3_id))
    N_age = len(np.unique(age_id))
    N_t = len(np.unique(year_id))
    
    # plates
    space_plate = numpyro.plate("space", N_s3, dim=-1)
    age_plate = numpyro.plate("age_groups", N_age, dim=-2)

    # hyperparameters
    alpha0 = numpyro.sample("alpha0", dist.Normal(0., 100.))
    beta0 = numpyro.sample("beta0", dist.Normal(0., 100.))
    sigma_alpha_s1 = numpyro.sample("sigma_alpha_s1", dist.Uniform(0., 2.))
    sigma_beta_s1 = numpyro.sample("sigma_beta_s1", dist.Uniform(0., 2.))
    sigma_alpha_s2 = numpyro.sample("sigma_alpha_s2", dist.Uniform(0., 2.))
    sigma_beta_s2 = numpyro.sample("sigma_beta_s2", dist.Uniform(0., 2.))
    sigma_alpha_s3 = numpyro.sample("sigma_alpha_s3", dist.Uniform(0., 2.))
    sigma_beta_s3 = numpyro.sample("sigma_beta_s3", dist.Uniform(0., 2.))
    sigma_alpha_age = numpyro.sample("sigma_alpha_age", dist.Uniform(0., 2.))
    sigma_beta_age = numpyro.sample("sigma_beta_age", dist.Uniform(0., 2.))
    sigma_xi = numpyro.sample("sigma_xi", dist.Uniform(0., 2.))
    sigma_nu = numpyro.sample("sigma_nu", dist.Uniform(0., 2.))
    sigma_gamma = numpyro.sample("sigma_gamma", dist.Uniform(0.0, 2.0))
    theta = numpyro.sample("theta", dist.Exponential(0.1))

    # spatial hierarchy
    with numpyro.plate("plate_s1", N_s1):
        z_s1 = numpyro.sample("z_s1", dist.Normal(0, 1).expand([2]).to_event(1))
    
    with numpyro.plate("plate_s2", N_s2):
        z_s2 = numpyro.sample("z_s2", dist.Normal(0., 1.).expand([2]).to_event(1))
    
    with space_plate:
        z_s3 = numpyro.sample("z_s3", dist.Normal(0., 1.).expand([2]).to_event(1))
        # space-time random walk
        rw_nu = numpyro.sample(
            "rw_nu",
            dist.GaussianRandomWalk(scale=1., num_steps=(N_t-1))
        )
        rw_nu = jnp.pad(rw_nu, ((0, 0), (1, 0)))

    nu = rw_nu * sigma_nu

    alpha_s = z_s1[hier1_id, 0] * sigma_alpha_s1 \
        + z_s2[hier2_id, 0] * sigma_alpha_s2 \
        + z_s3[hier3_id, 0] * sigma_alpha_s3
    beta_s = z_s1[hier1_id, 1] * sigma_beta_s1 + \
        z_s2[hier2_id, 1] * sigma_beta_s2 + \
        z_s3[hier3_id, 1] * sigma_beta_s3

    # age
    rw_alpha_age = numpyro.sample(
        "rw_alpha_age", dist.GaussianRandomWalk(scale=1., num_steps=(N_age-1))
    )
    rw_beta_age = numpyro.sample(
        "rw_beta_age", dist.GaussianRandomWalk(scale=1., num_steps=(N_age-1))
    )

    alpha_age = jnp.pad(rw_alpha_age * sigma_alpha_age, (1,0))
    beta_age = jnp.pad(rw_beta_age * sigma_beta_age, (1,0))
    
    with age_plate:
        # age-time random walk
        rw_gamma = numpyro.sample(
            "rw_gamma",
            dist.GaussianRandomWalk(scale=1., num_steps=(N_t-1))
        )
        rw_gamma = jnp.squeeze(rw_gamma, axis=1)
        rw_gamma = jnp.pad(rw_gamma, ((0, 0), (1, 0)))
        # age-space interaction
        with space_plate:
            z_xi = numpyro.sample("z_xi", dist.Normal(0., 1.))
    xi = z_xi * sigma_xi
    
    gamma = rw_gamma * sigma_gamma
    
    # beta-binomial likelihood
    latent_rate = alpha0 + alpha_age[age_id] + alpha_s[hier3_id] \
        + (beta0 + beta_age[age_id] + beta_s[hier3_id]) * year_id \
        + xi[age_id, year_id] + nu[hier3_id, year_id] + gamma[age_id, year_id]
    mu = numpyro.deterministic("mu", expit(latent_rate))
    numpyro.sample("deaths", dist.BetaBinomial(mu * theta, (1 - mu) * theta, population), obs=deaths)

# %%
nuts_kernel = NUTS(model)
rng_key = random.PRNGKey(0)

# %%
prior = Predictive(nuts_kernel.model, num_samples=100)(
    rng_key,
    population=data["population"].values,
)

# check prior predictive death rates
p = prior["pbar"]
az.plot_kde(p)


# %%
mcmc = MCMC(nuts_kernel, num_samples=200, num_warmup=20, num_chains=1)
rng_key, rng_key_ = random.split(rng_key)
mcmc.run(
    rng_key,
    age_id=data["age_id"].values,
    hier1_id=data["hier1_id"].values,
    hier2_id=data["hier2_id"].values,
    hier3_id=data["hier3_id"].values,
    year_id=data["year_id"].values,
    population=data["population"].values,
    deaths=data["deaths"].values,
)
mcmc.print_summary()
# %%
fit = az.from_numpyro(mcmc)
# %%
az.plot_forest(
    fit,
    var_names=("alpha", "theta"),
)

# %%
az.plot_trace(
    fit,
    var_names=("alpha", "theta"),
)
# %%

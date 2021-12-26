# %%
from numpyro.primitives import sample
import pandas as pd
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam
import jax.numpy as jnp
from jax import random
from jax.scipy.special import expit
import arviz as az

numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

# %%
__author__ = "Theo Rashid"
__email__ = "tar15@ic.ac.uk"

data = pd.read_csv("mortality_hf_ac_LSOA.csv")
# %%
data = data[data["sex"] == 1]
data = data.assign(
    year_id  = data["YEAR"].astype("category").cat.codes,
    age_id   = data["age_group"].astype("category").cat.codes,
    hier1_id = data["LAD2020"].astype("category").cat.codes,
    hier2_id = data["MSOA2011"].astype("category").cat.codes,
    hier3_id = data["LSOA2011"].astype("category").cat.codes,
)

# grid to match hier3 to hier2
grid_lookup2 = data[["hier1_id", "hier2_id", "hier3_id"]].drop_duplicates()
grid_lookup2 = grid_lookup2.sort_values(by="hier3_id")

# grid to match hier2 to hier1
grid_lookup1 = grid_lookup2[["hier1_id", "hier2_id"]].drop_duplicates().sort_values(by="hier2_id")
grid_lookup1 = grid_lookup1.sort_values(by="hier2_id")


# %%
reparam_config = {
    k: LocScaleReparam(0) for k in [
        "alpha_s1",
        "beta_s1",
        "alpha_s2",
        "beta_s2",
        "alpha_s3",
        "beta_s3",
        "alpha_age_drift",
        "xi",
        "nu_drift",
        "gamma_drift"
    ]
}


@numpyro.handlers.reparam(config=reparam_config)
def model(
    space,
    age,
    time,
    lookup1,
    lookup2,
    population,
    deaths=None
):

    N_s1 = len(np.unique(lookup1))
    N_s2 = len(np.unique(lookup2))
    N_s3 = len(np.unique(space))
    N_age = len(np.unique(age))
    N_t = len(np.unique(time))
    N = len(population)
    
    # plates
    space_plate = numpyro.plate("space", N_s3, dim=-3)
    age_plate = numpyro.plate("age_groups", N_age, dim=-2)
    year_plate = numpyro.plate("year", N_t - 1, dim=-1)

    # hyperparameters
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
    with numpyro.plate("s1", N_s1, dim=-3):
        alpha_s1 = numpyro.sample("alpha_s1", dist.Normal(0, sigma_alpha_s1))
        beta_s1 = numpyro.sample("beta_s1", dist.Normal(0, sigma_beta_s1))
    with numpyro.plate("s2", N_s2, dim=-3):
        alpha_s2 = numpyro.sample("alpha_s2", dist.Normal(alpha_s1[lookup1], sigma_alpha_s2))
        beta_s2 = numpyro.sample("beta_s2", dist.Normal(beta_s1[lookup1], sigma_beta_s2))
    with space_plate:
        alpha_s3 = numpyro.sample("alpha_s3", dist.Normal(alpha_s2[lookup2], sigma_alpha_s3))
        beta_s3 = numpyro.sample("beta_s3", dist.Normal(beta_s2[lookup2], sigma_beta_s3))

    # age
    with age_plate:
        alpha_age_drift_scale = jnp.pad(
            jnp.broadcast_to(
                sigma_alpha_age,
                N_age - 1
            ),
            (1, 0),
            constant_values=100. # pad so first term is alpha0, the global intercept with prior N(0, 100)
        )[:, jnp.newaxis]
        alpha_age_drift = numpyro.sample("alpha_age_drift", dist.Normal(0, alpha_age_drift_scale))
        alpha_age = jnp.cumsum(alpha_age_drift, -2)

        beta_age_drift_scale = jnp.pad(
            jnp.broadcast_to(
                sigma_beta_age,
                N_age - 1
            ),
            (1, 0),
            constant_values=100.
        )[:, jnp.newaxis]
        beta_age_drift = numpyro.sample("beta_age_drift", dist.Normal(0, beta_age_drift_scale))
        beta_age = jnp.cumsum(beta_age_drift, -2)
    
    # age-space interactions
    with age_plate, space_plate:
        xi = numpyro.sample("xi", dist.Normal(alpha_age + alpha_s3, sigma_xi))
    
    # space-time random walk
    with space_plate, year_plate:
        nu_drift = numpyro.sample("nu_drift", dist.Normal(beta_s3, sigma_nu))
        nu = jnp.pad(jnp.cumsum(nu_drift, -1), [(0, 0), (0, 0), (1, 0)])

    # age-time random walk
    with age_plate, year_plate:
        gamma_drift = numpyro.sample("gamma_drift", dist.Normal(beta_age, sigma_gamma))
        gamma = jnp.pad(jnp.cumsum(gamma_drift, -1), [(0, 0), (1, 0)])
    
    # likelihood
    latent_rate = xi + nu + gamma
    with numpyro.plate("N", N):
        mu_logit = latent_rate[space, age, time]
        mu = numpyro.deterministic("mu", expit(mu_logit))
        numpyro.sample("deaths", dist.BetaBinomial(mu * theta, (1 - mu) * theta, population), obs=deaths)
    print("xi")
    print(xi)
    print("nu")
    print(nu)
    print("gamma")
    print(gamma)


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
mcmc = MCMC(nuts_kernel, num_samples=20, num_warmup=10, num_chains=1)
rng_key, rng_key_ = random.split(rng_key)
mcmc.run(
    rng_key,
    age=data["age_id"].values,
    space=data["hier3_id"].values,
    time=data["year_id"].values,
    lookup1=grid_lookup1["hier1_id"].values,
    lookup2=grid_lookup2["hier2_id"].values,
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

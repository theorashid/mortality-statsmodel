# %%
from numpyro.infer.initialization import init_to_feasible, init_to_median, init_to_uniform, init_to_value
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
# numpyro.enable_x64()

# %%
__author__ = "Theo Rashid"
__email__ = "tar15@ic.ac.uk"

# %%
def get_data(
    rng_key,
    alpha0=-5.1,
    beta0=-0.04,
    sigma_alpha_age=1.0,
    sigma_alpha_s1=0.15,
    sigma_alpha_s2=0.15,
    sigma_beta_age=0.003,
    sigma_beta_s1=0.006,
    sigma_beta_s2=0.003,
    sigma_gamma=0.028,
    sigma_nu=0.063,
    sigma_xi=0.26,
    theta=1400
):
    data = pd.read_csv("mortality_hf_ac_LSOA.csv")
    data = data[data["sex"] == 1]
    data = data.assign(
        year_id  = data["YEAR"].astype("category").cat.codes,
        age_id   = data["age_group"].astype("category").cat.codes,
        hier1_id = data["MSOA2011"].astype("category").cat.codes,
        hier2_id = data["LSOA2011"].astype("category").cat.codes,
    )

    N_s1 = len(np.unique(data["MSOA2011"]))
    N_s2 = len(np.unique(data["LSOA2011"]))
    N_age = len(np.unique(data["age_group"]))
    N_t = len(np.unique(data["YEAR"]))

    alpha_s1 = dist.Normal(0., sigma_alpha_s1).sample(rng_key, (N_s1,))
    alpha_s2 = dist.Normal(0., sigma_alpha_s2).sample(rng_key, (N_s2,))
    beta_s1 = dist.Normal(0., sigma_beta_s1).sample(rng_key, (N_s1,))
    beta_s2 = dist.Normal(0., sigma_beta_s2).sample(rng_key, (N_s2,))
    
    alpha_age = dist.GaussianRandomWalk(scale=sigma_alpha_age, num_steps=(N_age-1)).sample(rng_key)
    alpha_age = jnp.pad(alpha_age, (1,0))
    beta_age = dist.GaussianRandomWalk(scale=sigma_beta_age, num_steps=(N_age-1)).sample(rng_key)
    beta_age = jnp.pad(beta_age, (1,0))

    xi = dist.Normal(0., sigma_xi).sample(random.PRNGKey(0), (N_age, N_s2))
    nu = dist.GaussianRandomWalk(
        scale=sigma_nu,
        num_steps=(N_t-1)
    ).sample(rng_key, (N_s2,))
    nu = jnp.pad(nu, ((0, 0), (1, 0)))
    gamma = dist.GaussianRandomWalk(
        scale=sigma_gamma,
        num_steps=(N_t-1)
    ).sample(rng_key, (N_age,))
    gamma = jnp.pad(gamma, ((0, 0), (1, 0)))

    alpha = alpha0 + alpha_age[data["age_id"].values] + \
        alpha_s1[data["hier1_id"].values] + \
        alpha_s2[data["hier2_id"].values]
    beta = beta0 + beta_age[data["age_id"].values] + \
        beta_s1[data["hier1_id"].values] + \
        beta_s2[data["hier2_id"].values]

    mu_logit = alpha + beta * data["year_id"].values
    mu_logit += xi[data["age_id"].values, data["hier2_id"].values]
    mu_logit += nu[data["hier2_id"].values, data["year_id"].values]
    mu_logit += gamma[data["age_id"].values, data["year_id"].values]
    mu = expit(mu_logit)

    data["deaths"] = dist.BetaBinomial(
        mu * theta,
        (1 - mu) * theta,
        data["population"].values
    ).sample(rng_key)

    return data

# %%
rng_key = random.PRNGKey(0)
data = get_data(rng_key)

# grid to match hier2 to hier1
grid_lookup = data[["hier1_id", "hier2_id"]].drop_duplicates().sort_values(by="hier2_id")

# %%
reparam_config = {
    k: LocScaleReparam(0) for k in [
        "alpha_s1",
        "beta_s1",
        "alpha_s2",
        "beta_s2",
        "alpha_age_drift",
        "beta_age_drift",
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
    lookup,
    population,
    deaths=None
):

    N_s1 = len(np.unique(lookup))
    N_s2 = len(np.unique(space))
    N_age = len(np.unique(age))
    N_t = len(np.unique(time))
    N = len(population)
    
    # plates
    space_plate = numpyro.plate("space", N_s2, dim=-3)
    age_plate = numpyro.plate("age_groups", N_age, dim=-2)
    year_plate = numpyro.plate("year", N_t - 1, dim=-1)

    # hyperparameters
    sigma_alpha_s1 = numpyro.sample("sigma_alpha_s1", dist.Uniform(0., 2.))
    sigma_beta_s1 = numpyro.sample("sigma_beta_s1", dist.Uniform(0., 2.))
    sigma_alpha_s2 = numpyro.sample("sigma_alpha_s2", dist.Uniform(0., 2.))
    sigma_beta_s2 = numpyro.sample("sigma_beta_s2", dist.Uniform(0., 2.))
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
    with space_plate:
        alpha_s3 = numpyro.sample("alpha_s2", dist.Normal(alpha_s1[lookup], sigma_alpha_s2))
        beta_s3 = numpyro.sample("beta_s2", dist.Normal(beta_s1[lookup], sigma_beta_s2))

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


# %%
# nuts_kernel = NUTS(model, init_strategy=init_to_feasible)
nuts_kernel = NUTS(
    model,
    init_strategy=init_to_value(
        values={
            "alpha_age_drift_decentered[0,0]": 5.1,
            "beta_age_drift_decentered[0,0]": -0.04,
            "sigma_alpha_age": 1.0,
            "sigma_alpha_s1": 0.15,
            "sigma_alpha_s2": 0.15,
            "sigma_beta_age": 0.003,
            "sigma_beta_s1": 0.006,
            "sigma_beta_s2": 0.003,
            "sigma_gamma": 0.028,
            "sigma_nu": 0.063,
            "sigma_xi": 0.26,
            "theta": 1400
        }
    )
)

# %%
with numpyro.handlers.seed(rng_seed=1):
    trace = numpyro.handlers.trace(model).get_trace(
        age=data["age_id"].values,
        space=data["hier2_id"].values,
        time=data["year_id"].values,
        lookup=grid_lookup["hier1_id"].values,
        population=data["population"].values,
        deaths=data["deaths"].values,
    )
print(numpyro.util.format_shapes(trace))

# %%
numpyro.render_model(
    model,
    model_args=(
        data["hier2_id"].values,
        data["age_id"].values,
        data["year_id"].values,
        grid_lookup["hier1_id"].values,
        data["population"].values,
        data["deaths"].values,
    )
)

# %%
prior = Predictive(nuts_kernel.model, num_samples=100)(
    rng_key,
    age=data["age_id"].values,
    space=data["hier3_id"].values,
    time=data["year_id"].values,
    lookup=grid_lookup["hier1_id"].values,
    population=data["population"].values,
)

# check prior predictive death rates
p = prior["mu"]
az.plot_kde(p)


# %%
mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=500, num_chains=2)
rng_key, rng_key_ = random.split(rng_key)
mcmc.run(
    rng_key,
    age=data["age_id"].values,
    space=data["hier2_id"].values,
    time=data["year_id"].values,
    lookup=grid_lookup["hier1_id"].values,
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
    var_names=("sigma_xi", "theta"),
)
# %%

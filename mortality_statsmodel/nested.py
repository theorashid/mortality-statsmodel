"""Modelling mortality over space and time

Three-tier mortality model with beta-binomial likelihood.
"""

import argparse
import os
import pickle

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam


def load_data(data_dir="", region="LSOA", sex="male"):
    a = np.load(data_dir + region + "_" + "a.npy")
    s1 = np.load(data_dir + region + "_" + "s1.npy")
    s2 = np.load(data_dir + region + "_" + "s2.npy")
    s3 = np.load(data_dir + region + "_" + "s3.npy")
    t = np.load(data_dir + region + "_" + "t.npy")
    deaths = np.load(data_dir + region + "_" + sex + "_" + "deaths.npy")
    population = np.load(data_dir + region + "_" + sex + "_" + "population.npy")
    return a, s1, s2, s3, t, deaths, population


def create_lookup(s1, s2, s3):
    """
    Create map between:
            - s1 indices and unique s2 indices
            - s2 indices and unique s3 indices
    """
    lookup12 = np.column_stack([s1, s2])
    lookup12 = np.unique(lookup12, axis=0)
    lookup12 = lookup12[lookup12[:, 1].argsort()]

    lookup23 = np.column_stack([s2, s3])
    lookup23 = np.unique(lookup23, axis=0)
    lookup23 = lookup23[lookup23[:, 1].argsort()]
    return lookup12[:, 0], lookup23[:, 0]


reparam_config = {
    k: LocScaleReparam(0)
    for k in [
        "alpha_s1",
        "alpha_s2",
        "alpha_s3",
        "alpha_age_drift",
        "beta_s1",
        "beta_s2",
        "beta_s3",
        "beta_age_drift",
        "xi",
        "gamma_drift",
    ]
}


@numpyro.handlers.reparam(config=reparam_config)
def model(age, space, time, lookup12, lookup23, population, deaths=None):
    N_s1 = len(np.unique(lookup12))
    N_s2 = len(np.unique(lookup23))
    N_s3 = len(np.unique(space))
    N_age = len(np.unique(age))
    N_t = len(np.unique(time))
    N = len(population)

    # plates
    age_plate = numpyro.plate("age_groups", N_age, dim=-3)
    space_plate = numpyro.plate("space", N_s3, dim=-2)
    year_plate = numpyro.plate("year", N_t - 1, dim=-1)

    # hyperparameters
    sigma_alpha_s1 = numpyro.sample("sigma_alpha_s1", dist.HalfNormal(1.0))
    sigma_alpha_s2 = numpyro.sample("sigma_alpha_s2", dist.HalfNormal(1.0))
    sigma_alpha_s3 = numpyro.sample("sigma_alpha_s3", dist.HalfNormal(1.0))
    sigma_alpha_age = numpyro.sample("sigma_alpha_age", dist.HalfNormal(1.0))
    sigma_beta_s1 = numpyro.sample("sigma_beta_s1", dist.HalfNormal(1.0))
    sigma_beta_s2 = numpyro.sample("sigma_beta_s2", dist.HalfNormal(1.0))
    sigma_beta_s3 = numpyro.sample("sigma_beta_s3", dist.HalfNormal(1.0))
    sigma_beta_age = numpyro.sample("sigma_beta_age", dist.HalfNormal(1.0))
    sigma_xi = numpyro.sample("sigma_xi", dist.HalfNormal(1.0))
    sigma_gamma = numpyro.sample("sigma_gamma", dist.HalfNormal(1.0))

    # spatial hierarchy
    with numpyro.plate("s1", N_s1, dim=-2):
        alpha_s1 = numpyro.sample("alpha_s1", dist.Normal(0, sigma_alpha_s1))
        beta_s1 = numpyro.sample("beta_s1", dist.Normal(0, sigma_beta_s1))

    with numpyro.plate("s2", N_s2, dim=-2):
        alpha_s2 = numpyro.sample(
            "alpha_s2", dist.Normal(alpha_s1[lookup12], sigma_alpha_s2)
        )
        beta_s2 = numpyro.sample(
            "beta_s2", dist.Normal(beta_s1[lookup12], sigma_beta_s2)
        )

    with space_plate:
        alpha_s3 = numpyro.sample(
            "alpha_s3", dist.Normal(alpha_s2[lookup23], sigma_alpha_s3)
        )
        beta_s3 = numpyro.sample(
            "beta_s3", dist.Normal(beta_s2[lookup23], sigma_beta_s3)
        )
        beta_s3_cum = jnp.outer(beta_s3, jnp.arange(N_t))[jnp.newaxis, :, :]

    # age
    with age_plate:
        alpha_age_drift_scale = jnp.pad(
            jnp.broadcast_to(sigma_alpha_age, N_age - 1),
            (1, 0),
            constant_values=10.0,  # pad so first term is alpha0, prior N(0, 10)
        )[:, jnp.newaxis, jnp.newaxis]
        alpha_age_drift = numpyro.sample(
            "alpha_age_drift", dist.Normal(0, alpha_age_drift_scale)
        )
        alpha_age = jnp.cumsum(alpha_age_drift, -3)

        beta_age_drift_scale = jnp.pad(
            jnp.broadcast_to(sigma_beta_age, N_age - 1), (1, 0), constant_values=10.0
        )[:, jnp.newaxis, jnp.newaxis]
        beta_age_drift = numpyro.sample(
            "beta_age_drift", dist.Normal(0, beta_age_drift_scale)
        )
        beta_age = jnp.cumsum(beta_age_drift, -3)

    # age-space interaction
    with age_plate, space_plate:
        xi = numpyro.sample("xi", dist.Normal(0, sigma_xi))

    # age-time random walk
    with age_plate, year_plate:
        gamma_drift = numpyro.sample("gamma_drift", dist.Normal(beta_age, sigma_gamma))
        gamma = jnp.pad(jnp.cumsum(gamma_drift, -1), [(0, 0), (0, 0), (1, 0)])

    # likelihood
    latent_rate = alpha_s3 + alpha_age + beta_s3_cum + xi + gamma
    with numpyro.plate("N", N):
        mu_logit = latent_rate[age, space, time]
        numpyro.sample("deaths", dist.Binomial(population, logits=mu_logit), obs=deaths)


def run_inference(
    model, age, space, time, lookup12, lookup23, population, deaths, rng_key, args
):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thin,
        chain_method=args.chain_method,
        progress_bar=True,
    )
    mcmc.run(rng_key, age, space, time, lookup12, lookup23, population, deaths)
    # mcmc.print_summary()

    extra_fields = mcmc.get_extra_fields()
    print("Number of divergences: {}".format(jnp.sum(extra_fields["diverging"])))

    return mcmc.get_samples(group_by_chain=True)


def main(args):
    model_name = "{}_{}_{}".format(args.region, args.sex, "_nested_as_at")

    print("Fetching data...")
    a, s1, s2, s3, t, deaths, population = load_data(region=args.region, sex=args.sex)

    lookup12, lookup23 = create_lookup(s1, s2, s3)

    print("Starting inference...")
    rng_key = random.PRNGKey(args.rng_seed)
    samples = run_inference(
        model, a, s3, t, lookup12, lookup23, population, deaths, rng_key, args
    )

    with open(model_name + ".pkl", "wb+") as f:
        pickle.dump(dict([key, np.array(value)] for key, value in samples.items()), f)
        f.close()


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.10.0")

    parser = argparse.ArgumentParser(description="Mortality regression model")
    parser.add_argument("--region", default="LSOA", type=str, help='"LSOA" or "MSOA".')
    parser.add_argument("--sex", default="male", type=str, help='"male" or "female".')
    parser.add_argument("-n", "--num-samples", nargs="?", default=500, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=200, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--thin", nargs="?", default=1, type=int)
    parser.add_argument(
        "--chain-method",
        default="parallel",
        type=str,
        help='"parallel", "sequential" or "vectorized"',
    )
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--rng_seed", default=1, type=int, help="random number generator seed"
    )
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    numpyro.enable_x64()

    main(args)

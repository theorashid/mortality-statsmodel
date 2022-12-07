"""
Modelling mortality over space and time
=======================================

CAR mortality model with binomial likelihood.

"""

import argparse
import logging
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s:%(message)s")

print = logger.info

DATA_DIR = "/home/tar15/mortality-numpyro/data/"
OUTPUT_DIR = "/home/tar15/mortality-numpyro/output/"

causes_female = [
    "All_other_CVD",
    "All_other_NCD",
    "All_other_cancers",
    "All_other_infections_maternal_perinatal_and_nutritional_conditions",
    "Alzheimer_and_other_dementias",
    "Breast_cancer",
    "Cerebrovascular_disease",
    "Chronic_obstructive_pulmonary_disease",
    "Colon_and_rectum_cancers",
    "Diabetes_mellitus_nephritis_and_nephrosis",
    "External_causes",
    "Ischaemic_heart_disease",
    "Lower_respiratory_infections",
    "Lymphomas_multiple_myeloma",
    "Ovary_cancer",
    "Pancreas_cancer",
    "Trachea_bronchus_lung_cancers",
]

causes_male = [
    "All_other_CVD",
    "All_other_NCD",
    "All_other_cancers",
    "All_other_infections_maternal_perinatal_and_nutritional_conditions",
    "Alzheimer_and_other_dementias",
    "Cerebrovascular_disease",
    "Chronic_obstructive_pulmonary_disease",
    "Cirrhosis_of_the_liver",
    "Colon_and_rectum_cancers",
    "Diabetes_mellitus_nephritis_and_nephrosis",
    "External_causes",
    "Ischaemic_heart_disease",
    "Lower_respiratory_infections",
    "Lymphomas_multiple_myeloma",
    "Oesophagus_cancer",
    "Prostate_cancer",
    "Trachea_bronchus_lung_cancers",
]


def load_data(data_dir="", cause="", region="LAD", sex="male"):
    a = np.load(data_dir + region + "_" + "a.npy")
    s = np.load(data_dir + region + "_" + "s.npy")
    t = np.load(data_dir + region + "_" + "t.npy")
    adj = np.load(data_dir + region + "_" + "adj.npy")
    deaths = np.load(data_dir + region + "_" + sex + "_" + "deaths_" + cause + ".npy")
    population = np.load(data_dir + region + "_" + sex + "_" + "population.npy")
    return a, s, t, adj, deaths, population


reparam_config = {
    k: LocScaleReparam(0)
    for k in [
        "alpha_s",
        "alpha_age_drift",
        "beta_s",
        "beta_age_drift",
        "xi",
        "gamma_drift",
    ]
}


@numpyro.handlers.reparam(config=reparam_config)
def model(age, space, time, adj, population, deaths=None):
    N_s = len(np.unique(space))
    N_age = len(np.unique(age))
    N_t = len(np.unique(time))
    N = len(population)

    # plates
    age_plate = numpyro.plate("age_groups", N_age, dim=-3)
    space_plate = numpyro.plate("space", N_s, dim=-2)
    year_plate = numpyro.plate("year", N_t - 1, dim=-1)

    # hyperparameters
    sigma_alpha_s = numpyro.sample("sigma_alpha_s", dist.HalfNormal(1.0))
    sigma_alpha_age = numpyro.sample("sigma_alpha_age", dist.HalfNormal(1.0))
    sigma_beta_s = numpyro.sample("sigma_beta_s", dist.HalfNormal(1.0))
    sigma_beta_age = numpyro.sample("sigma_beta_age", dist.HalfNormal(1.0))
    sigma_xi = numpyro.sample("sigma_xi", dist.HalfNormal(1.0))
    sigma_gamma = numpyro.sample("sigma_gamma", dist.HalfNormal(1.0))

    # spatial
    alpha_s_raw = numpyro.sample(
        "alpha_s_raw",
        dist.CAR(
            loc=0.0,
            correlation=0.99,
            conditional_precision=1.0,
            adj_matrix=adj,
            is_sparse=True,
        ),
    )
    alpha_s = sigma_alpha_s * alpha_s_raw[:, jnp.newaxis]

    beta_s_raw = numpyro.sample(
        "beta_s_raw",
        dist.CAR(
            loc=0.0,
            correlation=0.99,
            conditional_precision=1.0,
            adj_matrix=adj,
            is_sparse=True,
        ),
    )
    beta_s_cum = jnp.outer(sigma_beta_s * beta_s_raw, jnp.arange(N_t))

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
    latent_rate = alpha_s + alpha_age + beta_s_cum + xi + gamma
    with numpyro.plate("N", N):
        mu_logit = latent_rate[age, space, time]
        numpyro.sample("deaths", dist.Binomial(population, logits=mu_logit), obs=deaths)


def print_model_shape(model, age, space, time, adj, population):
    with numpyro.handlers.seed(rng_seed=1):
        trace = numpyro.handlers.trace(model).get_trace(
            age=age,
            space=space,
            time=time,
            adj=adj,
            population=population,
        )
    print(numpyro.util.format_shapes(trace))


def run_inference(model, age, space, time, adj, population, deaths, rng_key, args):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thin,
        chain_method=args.chain_method,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, age, space, time, adj, population, deaths)
    # mcmc.print_summary()

    extra_fields = mcmc.get_extra_fields()
    print("Number of divergences: {}".format(jnp.sum(extra_fields["diverging"])))

    return mcmc.get_samples(group_by_chain=True)


def main(args):
    model_name = name + "_car_as_at"
    logger.info(model_name)

    cause_id = int(args.cause) - 1
    if args.sex == "female":
        cause = causes_female[cause_id]
    else:
        cause = causes_male[cause_id]

    cause_name = str(cause_id) + "_" + cause
    logger.info(cause_name)

    logger.info("Fetching data...")
    a, s, t, adj, deaths, population = load_data(
        data_dir=DATA_DIR, cause=cause_name, region=args.region, sex=args.sex
    )

    if args.device != "gpu":
        logger.info("Model shape:")
        print_model_shape(model, a, s, t, adj, population)

    logger.info("Starting inference...")
    rng_key = random.PRNGKey(args.rng_seed)
    samples = run_inference(model, a, s, t, adj, population, deaths, rng_key, args)

    with open(OUTPUT_DIR + model_name + "_" + cause_name + ".pkl", "wb+") as f:
        pickle.dump(dict([key, np.array(value)] for key, value in samples.items()), f)
        f.close()


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.10.0")

    parser = argparse.ArgumentParser(description="Mortality regression model")
    parser.add_argument("--cause", default="2_All_other_cancers", type=str)
    parser.add_argument("--region", default="LAD", type=str, help="LAD.")
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

    name = "{}_{}".format(args.region, args.sex)

    fh = logging.FileHandler(r"{}.log".format(name), "w+")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(jax.devices())
    logger.info(jax.local_device_count())
    logger.info(jax.device_count(args.device))

    main(args)

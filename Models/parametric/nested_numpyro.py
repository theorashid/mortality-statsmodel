"""
Modelling mortality over space and time
=======================================

Three-tier mortality model with beta-binomial likelihood.

"""

import argparse
import os

import numpy as np
import arviz as az

from jax import random
import jax.numpy as jnp
from jax.scipy.special import expit

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam

DATA_DIR = "Data/Mortality/"


def load_data(data_dir=""):
	a = np.load(data_dir + "LSOA_hf_a.npy")
	s1 = np.load(data_dir + "LSOA_hf_s1.npy")
	s2 = np.load(data_dir + "LSOA_hf_s2.npy")
	t = np.load(data_dir + "LSOA_hf_t.npy")
	deaths = np.load(data_dir + "LSOA_hf_deaths.npy")
	population = np.load(data_dir + "LSOA_hf_population.npy")
	return a, s1, s2, t, deaths, population


def create_lookup(s1, s2):
	"""
	Create a map between s1 indices and unique s2 indices
	"""
	lookup = np.column_stack([s1, s2])
	lookup = np.unique(lookup, axis=0)
	lookup = lookup[lookup[:, 1].argsort()]
	return lookup[:, 0]


reparam_config = {
	k: LocScaleReparam(0)
	for k in [
		"alpha_s1",
		"alpha_s2",
		"alpha_age_drift",
		"beta_s1",
		"beta_s2",
		"beta_age_drift",
		"xi",
		"nu_drift",
		"gamma_drift",
	]
}


@numpyro.handlers.reparam(config=reparam_config)
def model(age, space, time, lookup, population, deaths=None):
	N_s1 = len(np.unique(lookup))
	N_s2 = len(np.unique(space))
	N_age = len(np.unique(age))
	N_t = len(np.unique(time))
	N = len(population)

	# plates
	age_plate = numpyro.plate("age_groups", N_age, dim=-3)
	space_plate = numpyro.plate("space", N_s2, dim=-2)
	year_plate = numpyro.plate("year", N_t - 1, dim=-1)

	# hyperparameters
	sigma_alpha_s1 = numpyro.sample("sigma_alpha_s1", dist.HalfNormal(1.0))
	sigma_alpha_s2 = numpyro.sample("sigma_alpha_s2", dist.HalfNormal(1.0))
	sigma_alpha_age = numpyro.sample("sigma_alpha_age", dist.HalfNormal(1.0))
	sigma_beta_s1 = numpyro.sample("sigma_beta_s1", dist.HalfNormal(1.0))
	sigma_beta_s2 = numpyro.sample("sigma_beta_s2", dist.HalfNormal(1.0))
	sigma_beta_age = numpyro.sample("sigma_beta_age", dist.HalfNormal(1.0))
	sigma_xi = numpyro.sample("sigma_xi", dist.HalfNormal(1.0))
	sigma_nu = numpyro.sample("sigma_nu", dist.HalfNormal(1.0))
	sigma_gamma = numpyro.sample("sigma_gamma", dist.HalfNormal(1.0))
	theta = numpyro.sample("theta", dist.Exponential(0.1))

	# spatial hierarchy
	with numpyro.plate("s1", N_s1, dim=-2):
		alpha_s1 = numpyro.sample("alpha_s1", dist.Normal(0, sigma_alpha_s1))
		beta_s1 = numpyro.sample("beta_s1", dist.Normal(0, sigma_beta_s1))
	with space_plate:
		alpha_s2 = numpyro.sample(
			"alpha_s2", dist.Normal(alpha_s1[lookup], sigma_alpha_s2)
		)
		beta_s2 = numpyro.sample(
			"beta_s2", dist.Normal(beta_s1[lookup], sigma_beta_s2)
		)

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

	# age-space interactions
	with age_plate, space_plate:
		xi = numpyro.sample("xi", dist.Normal(alpha_age + alpha_s2, sigma_xi))

	# space-time random walk
	with space_plate, year_plate:
		nu_drift = numpyro.sample("nu_drift", dist.Normal(beta_s2, sigma_nu))
		nu = jnp.pad(jnp.cumsum(nu_drift, -1), [(0, 0), (1, 0)])

	# age-time random walk
	with age_plate, year_plate:
		gamma_drift = numpyro.sample("gamma_drift", dist.Normal(beta_age, sigma_gamma))
		gamma = jnp.pad(jnp.cumsum(gamma_drift, -1), [(0, 0), (0, 0), (1, 0)])

	# likelihood
	latent_rate = numpyro.deterministic("latent_rate", xi + nu + gamma)
	with numpyro.plate("N", N):
		mu_logit = latent_rate[age, space, time]
		numpyro.sample("deaths", dist.Binomial(population, logits=mu_logit), obs=deaths)
		# mu = numpyro.deterministic("mu", expit(latent_rate[age, space, time]))
		# numpyro.sample(
		# 	"deaths",
		# 	dist.BetaBinomial(mu * theta, (1 - mu) * theta, population),
		# 	obs=deaths,
		# )


def print_model_shape(model, age, space, time, lookup, population):
	with numpyro.handlers.seed(rng_seed=1):
		trace = numpyro.handlers.trace(model).get_trace(
			age=age,
			space=space,
			time=time,
			lookup=lookup,
			population=population,
		)
	print(numpyro.util.format_shapes(trace))


def run_inference(model, age, space, time, lookup, population, deaths, rng_key, args):
	kernel = NUTS(model)
	mcmc = MCMC(
		kernel,
		num_warmup=args.num_warmup,
		num_samples=args.num_samples,
		num_chains=args.num_chains,
		progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
	)
	mcmc.run(rng_key, age, space, time, lookup, population, deaths)
	# mcmc.print_summary()
	return mcmc.get_samples(group_by_chain=True)


def main(args):
	print("Fetching data...")
	a, s1, s2, t, deaths, population = load_data(data_dir=DATA_DIR)

	lookup = create_lookup(s1, s2)

	print("Model shape:")
	print_model_shape(model, a, s2, t, lookup, population)

	print("Starting inference...")
	rng_key = random.PRNGKey(args.rng_seed)
	samples = run_inference(model, a, s2, t, lookup, population, deaths, rng_key, args)

	az.to_netcdf(az.from_dict(samples), "posterior.nc")


if __name__ == "__main__":
	assert numpyro.__version__.startswith("0.9.2")

	parser = argparse.ArgumentParser(description="Mortality regression model")
	parser.add_argument("-n", "--num-samples", nargs="?", default=500, type=int)
	parser.add_argument("--num-warmup", nargs="?", default=200, type=int)
	parser.add_argument("--num-chains", nargs="?", default=1, type=int)
	parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
	parser.add_argument(
		"--rng_seed", default=1, type=int, help="random number generator seed"
	)
	args = parser.parse_args()

	numpyro.set_platform(args.device)
	numpyro.enable_x64()

	main(args)

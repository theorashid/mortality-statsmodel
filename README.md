# mortality-statsmodel

Mortality modelling using scalable Bayesian hierarchical models.

This code is used in:

- Rashid, T., Bennett, J.E. et al. (2021). [Life expectancy and risk of death in 6791 communities in England from 2002 to 2019: high-resolution spatiotemporal analysis of civil registration data](https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(21)00205-X/fulltext). _The Lancet Public Health_.

## nimble models

The models are fitted using [nimble](https://r-nimble.org). For ease of reading and to aid the user more familiar with other MCMC software, I've also added the model structure as BUGS code.

We are modelling the death rate per person in a given spatial unit, year and age group stratum. It is the _death rate per person_ that varies between models. The following spatial effects are used:

- The _nested_ model is designed for a three-level nested spatial hierarchy. In our case it follows the [ONS' hierarchical output area geographies](https://www.ons.gov.uk/methodology/geography/ukgeographies/censusgeography). Each Lower layer Super Output Area (LSOA) lies within a Middle layer Super Output Area (MSOA), which lies within a Local Authority District (LAD) (can also be used with MSOA, LAD and region). The spatial effects are modelled as IID.
- The _BYM_ model shares information between the nearest neighbours to each spatial unit.

An example invocation of the code from the command line is as follows:

```sh
Rscript run_model.R MSOA nested 1 10000 5000 --num_chains=4
```

For the full explanation of the options available, run

```sh
Rscript run_model.R --help
```

## numpyro models (experimental)

By porting the model to [numpyro](https://num.pyro.ai/), I have seen massive speedups, both in terms of run time and effective samples per second. This is thanks to numpyro's jax backend allowing sampling on a GPU, which is beneficial for large models, and using NUTS over nimble's conjugate Gibbs/RWMH samplers.

A simplified version of the model has been contributed as an [example](https://num.pyro.ai/en/latest/examples/mortality.html) for the numpyro documentation. The `car.py` model uses the ICAR distribution for spatial effects by setting the correlation parameter of the CAR distribution to 0.99. The model can be run as:

```sh
poetry run python car.py --region="MSOA" --sex=1 --num_samples=10000 --num_warmup=5000 --num_chains=4 --device="cpu"
```

## Table of models

file       | paper       | likelihood    | terms                                                                    | spatial effects
---------- | ----------- | ------------- | ------------------------------------------------------------------------------- | -------
nested.bug | Rashid 2021 | gamma-Poisson | $α_0 + β_0 t + α_{1s} + β_{1s} t+ α_{2a} + β_{2a} t + ξ_{as} + γ_{at} + ν_{st}$ | nested
BYM.bug    | Rashid 2021 | gamma-Poisson | $α_0 + β_0 t + α_{1s} + β_{1s} t+ α_{2a} + β_{2a} t + ξ_{as} + γ_{at} + ν_{st}$ | BYM
nested_bb.bug | -        | beta-binomial | $α_0 + β_0 t + α_{1s} + β_{1s} t+ α_{2a} + β_{2a} t + ξ_{as} + γ_{at} + ν_{st}$ | nested
nested.py     | -        | binomial      | $α_0 + β_0 t + α_{1s} + β_{1s} t+ α_{2a} + β_{2a} t + ξ_{as} + γ_{at}$          | nested
car.py        | -        | binomial      | $α_0 + β_0 t + α_{1s} + β_{1s} t+ α_{2a} + β_{2a} t + ξ_{as} + γ_{at}$          | ICAR

## Data availability

Data used in the analysis are controlled by the [Small Area Health Statistics Unit](https://www.imperial.ac.uk/school-public-health/epidemiology-and-biostatistics/small-area-health-statistics-unit/) who do not have permission to release data to third parties. Individual mortality data can be requested through the [Office for National Statistics](https://www.ons.gov.uk). If you would like a file containing simulated numbers that allow you to test the code, please contact [global.env.health@imperial.ac.uk](mailto:global.env.health@imperial.ac.uk?subject=mortality%20simulation).

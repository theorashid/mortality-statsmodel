# mortality-statsmodel
Mortality modelling using Bayesian hierarchical models and statistical machine learning methods.

The parametric model code requires `R` version 3.6 or higher and the following packages:
- `nimble` (>= 0.9.1)
- `tidyverse`
- `docopt`
- `here`
- `lme4`
- `geojsonio`
- `spdep`

And the analysis code requires the further packages:
- `reshape2`
- `rstan`
- `foreach`
- `doParallel`

## Parametric models
These are a collection of Bayesian hierarchical models. Deaths are modelled using a negative binomial likelihood. We are looking at the death rate per person in a given spatial unit, year and age group stratum. It is the _death rate per person_ that varies between models. The following model designs are used:
* The *nested* model is designed for a three-level nested spatial hierarchy. In our case it follows the [ONS' hierarchical output area geographies](https://www.ons.gov.uk/methodology/geography/ukgeographies/censusgeography). Each Lower layer Super Output Area (LSOA) lies within a Middle layer Super Output Area (MSOA), which lies within a Local Authority District (LAD) (can also be used with MSOA, LAD and region). The spatial effects are modelled as IID.
* The *BYM* model shares information between the nearest neighbours to each spatial unit.

The models here are fitted using [nimble](https://r-nimble.org). For ease of reading and to aid the user more familiar with other MCMC software, I've also added the basic model structure as BUGS code.

An example invocation of the code from the command line is as follows:
```
Rscript run_model.R MSOA nested 1 25000 20000 --num_chains=4 --test
```
Note, the `--test` flag is required as the full size datasets have not been uploaded. Please contact me if you need to try a larger dataset.

For the full explanation of the options available, run
```
Rscript run_model.R --help
```

## Data
The `Mortality/` folder contains two test datasets with simulated deaths (but real population data from [ONS](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates)). The LSOA file contains deaths for LSOAs in Hammersmith and Fulham, the test dataset for the London study. The MSOA file contains deaths for MSOAs in London, a test for the England study.

The `Inits/` folder contains inital values for the MCMC (made using the `inits.R` script). The `GIS/` folder contains TopoJSON files for the LSOA/MSOA geographies and population-weighted LSOA/MSOA centroids, both available from the ONS' [Open Geography Portal](https://geoportal.statistics.gov.uk).

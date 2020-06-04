# Mortality model building
Deaths are modelled as a negative binomial likelihood. We are looking at the death rate per person in a given spatial unit, year and age group stratum. It is the _death rate per person_ that varies between models.

The __linear__ model is a simple demonstration of a linear fit with a Poisson likelihood using NIMBLE.

The __hier__ model makes use of <a href="https://www.ons.gov.uk/methodology/geography/ukgeographies/censusgeography">ONS' hierarchical output area geographies</a>. Each Lower layer Super Output Area (LSOA) lies within a Middle layer Super Output Area (MSOA), which lies within a Local Authority District (LAD). The spatial effects modelled as IID.

In contrast, each spatial unit in the __BYM__ model is informed by its nearest neighbours.

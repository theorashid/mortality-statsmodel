# Mortality model building
Deaths are modelled as a Poisson likelihood, with the mean made up of the death rate per person and the population in a given spatial unit, year and age group stratum. It is the death rate that varies between models.

The models <b>linear</b>, <b>spatial_random</b>, <b>spatial_hier</b> and <b>spatial_hier_age</b> introduce progressive more complex structures, starting from a simple linear fit, adding spatial terms and then age effects.

The <b>hier</b> model makes use of <a href="https://www.ons.gov.uk/methodology/geography/ukgeographies/censusgeography">ONS' hierarchical output area geographies</a>. Each Lower layer Super Output Area (LSOA) lies within a Middle layer Super Output Area (MSOA), which lies within a Local Authority District (LAD). The spatial effects modelled as IID.

Conversely, each spatial unit in the <b>BYM</b> model is informed by its nearest neighbours.

# Mortality model building
Deaths are modelled as a negative binomial likelihood. We are looking at the death rate per person in a given spatial unit, year and age group stratum. It is the <i>death rate per person</i> that varies between models.

The models <b>linear</b>, <b>spatial_random</b>, <b>spatial_hier</b> and <b>spatial_hier_age</b> introduce progressively more complex structures, starting from a simple linear fit before adding IID spatial terms and age effects. These are built using a simpler Poisson likelihood.

The <b>hier</b> model makes use of <a href="https://www.ons.gov.uk/methodology/geography/ukgeographies/censusgeography">ONS' hierarchical output area geographies</a>. Each Lower layer Super Output Area (LSOA) lies within a Middle layer Super Output Area (MSOA), which lies within a Local Authority District (LAD). The spatial effects modelled as IID.

In contrast, each spatial unit in the <b>BYM</b> model is informed by its nearest neighbours.

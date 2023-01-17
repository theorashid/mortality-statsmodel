library(lme4)

# log(price) ~ theta_O and phi_O :- OA level intercepts and slopes
# theta_L and phi_L :- LSOA level intercepts and slopes
# theta_M and phi_M :- MSOA level intercepts and slopes
# theta_D and phi_D :- District level intercepts and slopes
# lamda_quarter :- Non-linear time
# eta_D,quarter :- Non-linear time, district offsets
# delta_1,type, :- Type intercept; flat, terraced house, semi-detached house, detached house
# delta_2,ownership :- Ownership intercept; leasehold, freehold
# delta_3,beds :- Beds intercept; studio, 1-, 2-, 3-, 4- and 5-bedrooms, 6/7/8-bedrooms
# delta_5,season :- Season intercept; spring (March-May), summer (June-August),autumn (September-November), winter (December-February))
# tau_1,type :- Type linear time slope; flat, terraced house, semi-detached house, detached house
# tau_2,beds :- Beds linear time slope; studio, 1-, 2-, 3-, 4- and 5-bedrooms, 6/7/8-bedrooms
# omega_D,beds :- Beds intercepts, district offsets

house_price_model <- lmer(
  Lprice ~ (1 + quart | oa_id) +
    (1 + quart | lsoa_id) +
    (1 + quart | msoa_id) +
    (1 + quart | lad_id) + q1 +
    (1 | q1_lad_id) +
    Type +
    FreeL +
    nBedrooms +
    NewOld +
    Season +
    Type:quart +
    nBedrooms:quart +
    (1 | nBedrooms_lad_id),
  data = dat
)

library(lme4)

house_price_model <- lmer(
  Lprice ~ (1 + quart | oa_id) + # log(price) ~ theta_O and phi_O :- OA level intercepts and slopes
    (1 + quart | lsoa_id) + # theta_L and phi_L :- LSOA level intercepts and slopes
    (1 + quart | msoa_id) + # theta_M and phi_M :- MSOA level intercepts and slopes
    (1 + quart | lad_id) + # theta_D and phi_D :- District level intercepts and slopes
    q1 + # lamda_quarter :- Non-linear time
    (1 | q1_lad_id) + # eta_D,quarter :- Non-linear time, district offsets
    Type + # delta_1,type, :- Type intercept; flat, terraced house, semi-detached house, detached house
    FreeL + # delta_2,ownership :- Ownership intercept; leasehold, freehold
    nBedrooms + # delta_3,beds :- Beds intercept; studio, 1-, 2-, 3-, 4- and 5-bedrooms, 6/7/8-bedrooms
    NewOld + # delta_4,newold :- New/old intercept; old
    Season + # delta_5,season :- Season intercept; spring (March-May), summer (June-August),autumn (September-November), winter (December-February))
    Type:quart + # tau_1,type :- Type linear time slope; flat, terraced house, semi-detached house, detached house
    nBedrooms:quart + # tau_2,beds :- Beds linear time slope; studio, 1-, 2-, 3-, 4- and 5-bedrooms, 6/7/8-bedrooms
    (1 | nBedrooms_lad_id), # omega_D,beds :- Beds intercepts, district offsets
  data = dat
)

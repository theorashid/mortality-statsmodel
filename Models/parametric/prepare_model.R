suppressPackageStartupMessages({
  library(dplyr)
  library(geojsonio)
  library(spdep)
  library(spdplyr)
})

#' Function to load the correct dataset based on region (LSOA | MSOA)
load_data <- function(data_path, region, sex, test = FALSE) {
  if (region == "MSOA") {
    if(!test) {
      mortality <- read.csv(file = paste0(data_path, "/Mortality/",
                                          "mortality_eng_ac_", region, ".csv"))
      print(paste0("LOADED ENGLAND DATA AT ", region, " LEVEL with DIMENSIONS "))
      print(dim(mortality))
    } else {
      mortality <- read.csv(file = paste0(data_path, "/Mortality/",
                                          "mortality_ldn_ac_", region, ".csv"))
      print(paste0("LOADED LONDON DATA AT ", region, " LEVEL with DIMENSIONS "))
      print(dim(mortality))
    }
    mortality <- mortality %>% 
      select(-X) %>%
      filter(sex == !!sex) %>%
      select(-sex) %>%
      arrange(MSOA2011, YEAR, age_group)
    mortality$GOR.id <- mortality %>% group_indices(GOR2011)
    mortality$LAD.id <- mortality %>% group_indices(LAD2011)
    mortality$MSOA.id <- mortality %>% group_indices(MSOA2011)
    mortality$age_group.id <- mortality %>% group_indices(age_group)
    mortality$YEAR.id <- mortality %>% group_indices(YEAR)
  } else if (region == "LSOA") {
    if(!test) {
      mortality <- read.csv(file = paste0(data_path, "/Mortality/",
                                          "mortality_ldn_ac_", region, ".csv"))
      print(paste0("LOADED LONDON DATA AT ", region, " LEVEL with DIMENSIONS "))
      print(dim(mortality))
    } else {
      mortality <- read.csv(file = paste0(data_path, "/Mortality/",
                                          "mortality_hf_ac_", region, ".csv"))
      print(paste0("LOADED HAMMERSMITH AND FULHAM DATA AT ", region, " LEVEL with DIMENSIONS "))
      print(dim(mortality))
    }
    mortality <- mortality %>%
      select(-X) %>%
      filter(sex == !!sex) %>%
      select(-sex) %>%
      arrange(LSOA2011, YEAR, age_group)
    mortality$GOR.id <- mortality %>% group_indices(GOR2011)
    mortality$LAD.id <- mortality %>% group_indices(LAD2011)
    mortality$MSOA.id <- mortality %>% group_indices(MSOA2011)
    mortality$LSOA.id <- mortality %>% group_indices(LSOA2011)
    mortality$age_group.id <- mortality %>% group_indices(age_group)
    mortality$YEAR.id <- mortality %>% group_indices(YEAR)
  } else stop("invalid region: MSOA or LSOA only")
  print(paste0("FILTERED MORTALITY DATA ROWS: ", dim(mortality)[1]))
  return(mortality)
}

#'  Prepare inputs for:
#'  - BYM model by matching shapedata to mortality data, output is reduced adjacency matrix
#'  - nested model by producing lookup tables linking hier1 to hier2 and hier3 geographies
prep_model <- function(data_path, mortality, region, model) {
  if (model == "BYM") {
    # Shape data for extent (not clipped) and merge with mortality
    if (region == "MSOA") {
      sp <- topojson_read(paste0(data_path, "/GIS/", "EW_", region, "2011_BFE.json"))
      sp <- as(sp, "Spatial")
      sp <- sp %>% rename(MSOA2011 = MSOA11CD)
      sp <- left_join(sp, distinct(select(mortality, MSOA2011, hier3.id)))
    } else if (region == "LSOA") {
      sp <- topojson_read(paste0(data_path, "/GIS/", "ldn_", region, "2011_BFE.json"))
      sp <- as(sp, "Spatial")
      sp <- sp %>% rename(LSOA2011 = code)
      sp <- left_join(sp, distinct(select(mortality, LSOA2011, hier3.id)))
    } else stop("invalid region: MSOA or LSOA only")
    
    sp <- sp[!is.na(sp$hier3.id) ,]  # remove NA rows for subsetting
    sp <- sp[order(sp$hier3.id),] # reorder on hier3
    # row.names(sp@data) <- sp$hier3.id
    
    # Extract adjacency matrix
    W.nb <- poly2nb(sp, row.names = sp$hier3.id)
    nbInfo <- nb2WB(W.nb)
    # adj = nbInfo$adj, weights = nbInfo$weights, num = nbInfo$num
    
    print("----- SHAPE DATA LOADED -----")
    return(nbInfo)
  } else if (model == "nested") {
    # lookup correct hier2 or hier3 for that hier1
    # grid.lookup[j, 2] for hier2, grid.lookup[j, 3] for hier1
    grid.lookup <- mortality %>% 
      select(hier3.id, hier2.id, hier1.id) %>%
      distinct() %>%
      arrange(hier3.id) # arrange ascending so matches loop order
     
    grid.lookup.s2 <- mortality %>%
      select(hier2.id, hier1.id) %>%
      distinct() %>%
      arrange(hier2.id)
     
    grid.lookup <- as.matrix(grid.lookup)
    grid.lookup.s2 <- as.matrix(grid.lookup.s2)
    print("----- LOOKUPS MADE -----")
    return(list(grid.lookup, grid.lookup.s2))
  } else stop("invalid model name: BYM or nested only")
} 

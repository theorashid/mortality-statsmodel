suppressPackageStartupMessages({
  library(tidyverse)
  library(geojsonio)
  library(spdep)
})

#' Function to load the correct dataset based on region (LSOA | MSOA)
load_data <- function(data_path, region, sex) {
  if (region == "MSOA") {
    mortality <- read_csv(
      file = str_c(data_path, "/mortality_ldn_ac_", region, ".csv")
    )
    print(str_c("Loaded England ", region, " data with dimensions "))
    print(dim(mortality))

    mortality <- mortality |>
      filter(sex == !!sex) |>
      select(-sex) |>
      arrange(MSOA2011, YEAR, age_group)
    mortality <- mortality |>
      mutate(
        GOR.id       = mortality |> group_by(GOR2011) |> group_indices(),
        LAD.id       = mortality |> group_by(LAD2020) |> group_indices(),
        MSOA.id      = mortality |> group_by(MSOA2011) |> group_indices(),
        age_group.id = mortality |> group_by(age_group) |> group_indices(),
        YEAR.id      = mortality |> group_by(YEAR) |> group_indices()
      )
  } else if (region == "LSOA") {
    mortality <- read_csv(
      file = str_c(data_path, "/mortality_ldn_ac_", region, ".csv")
    )
    print(str_c("Loaded London ", region, " data with dimensions "))
    print(dim(mortality))

    mortality <- mortality |>
      filter(sex == !!sex) |>
      select(-sex) |>
      arrange(LSOA2011, YEAR, age_group)
    mortality <- mortality |>
      mutate(
        GOR.id       = mortality |> group_by(GOR2011) |> group_indices(),
        LAD.id       = mortality |> group_by(LAD2020) |> group_indices(),
        MSOA.id      = mortality |> group_by(MSOA2011) |> group_indices(),
        LSOA.id      = mortality |> group_by(LSOA2011) |> group_indices(),
        age_group.id = mortality |> group_by(age_group) |> group_indices(),
        YEAR.id      = mortality |> group_by(YEAR) |> group_indices()
      )
  } else {
    stop("invalid region: MSOA or LSOA only")
  }
  print(str_c("Filtered mortality data rows: ", dim(mortality)[1]))
  return(mortality)
}

#' Prepare inputs for:
#' - BYM model by matching shapedata to mortality data, output is reduced adjacency matrix
#' - nested model by producing lookup tables linking hier1 to hier2 and hier3 geographies
prep_model <- function(data_path, mortality, region, model) {
  if (model == "BYM") {
    # Shape data for extent (not clipped) and merge with mortality
    if (region == "MSOA") {
      sf <- topojson_read(str_c(data_path, "EW_", region, "2011_BFC.json"))
      sf <- sf |> rename(MSOA2011 = MSOA11CD)
      sf <- sf |> left_join(mortality |> select(MSOA2011, hier3.id) |> distinct())
    } else if (region == "LSOA") {
      sf <- topojson_read(paste0(data_path, "ldn_", region, "2011_BFC.json"))
      sf <- sf |> rename(LSOA2011 = LSOA11CD)
      sf <- sf |> left_join(mortality |> select(LSOA2011, hier3.id) |> distinct())
    } else {
      stop("invalid region: MSOA or LSOA only")
    }

    sf <- sf |>
      filter(!is.na(hier3.id)) |> # remove NA rows for subsetting
      arrange(hier3.id) # reorder on hier3

    # Extract adjacency matrix
    W.nb <- poly2nb(sf, row.names = sf |> pull(hier3.id))
    nbInfo <- nb2WB(W.nb)
    # adj = nbInfo$adj, weights = nbInfo$weights, num = nbInfo$num

    if (region == "MSOA") {
      print("correcting island to mainland for MSOA level")
      # connect islands to the nearest MSOA (based on road/ferry connections)
      # Isle of Wight, E02003592  -- E02004801
      # Hayling Island, E02004776 -- E02004775
      # Scilly Isles, E02006781   -- E02003950
      # Canvey Island, E02004482  -- E02004479
      islands <- c("E02003592", "E02004776", "E02006781", "E02004482")
      mainlands <- c("E02004801", "E02004775", "E02003950", "E02004479")

      for (i in 1:length(islands)) {
        print(islands[i])
        head <- sf |>
          filter(MSOA2011 == islands[i]) |>
          pull(hier3.id)
        target <- sf |>
          filter(MSOA2011 == mainlands[i]) |>
          pull(hier3.id)

        # add connection from head to target
        hier3 <- head

        if (hier3 == 1) {
          pre_id <- 0
        } else {
          pre_id <- sum(nbInfo$num[1:(hier3 - 1)])
        }

        nn <- nbInfo$num[hier3] # number of neighbours
        if (nn > 0) {
          id <- sum(nbInfo$num[1:hier3]) # end position on neighbours list
          tmp <- nbInfo$adj[(pre_id + 1):id] # neighbours list

          # add target to neighbours list
          tmp <- append(tmp, target)
          tmp <- tmp[order(tmp)] # keep them in the right order
          pos <- which(tmp %in% target)

          nbInfo$adj <- append(nbInfo$adj, target, (pre_id + pos - 1))
          print(nbInfo$adj[(pre_id)])
          print(nbInfo$adj[(pre_id + pos)])

          nbInfo$num[hier3] <- nbInfo$num[hier3] + 1 # added one more neighbour
          nbInfo$weights <- append(nbInfo$weights, 1) # weights are all equal
        } else {
          # Scilly isles has no neighbours
          nbInfo$adj <- append(nbInfo$adj, target, pre_id)
          nbInfo$num[hier3] <- nbInfo$num[hier3] + 1 # added one more neighbour
          nbInfo$weights <- append(nbInfo$weights, 1) # weights are all equal
        }

        # add connection back from target to head
        hier3 <- target

        if (hier3 == 1) {
          pre_id <- 0
        } else {
          pre_id <- sum(nbInfo$num[1:(hier3 - 1)])
        }

        id <- sum(nbInfo$num[1:hier3]) # end position on neighbours list
        tmp <- nbInfo$adj[(pre_id + 1):id] # neighbours list

        # add head to neighbours list
        tmp <- append(tmp, head)
        tmp <- tmp[order(tmp)] # keep them in the right order
        pos <- which(tmp %in% head)

        nbInfo$adj <- append(nbInfo$adj, head, (pre_id + pos - 1))
        nbInfo$num[hier3] <- nbInfo$num[hier3] + 1 # added one more neighbour
        nbInfo$weights <- append(nbInfo$weights, 1) # weights are all equal
      }
    }
    print("Shape data loaded")
    return(nbInfo)
  } else if (model == "nested") {
    # lookup correct hier2 or hier3 for that hier1
    # grid.lookup[j, 2] for hier2, grid.lookup[j, 3] for hier1
    grid.lookup <- mortality |>
      select(hier3.id, hier2.id, hier1.id) |>
      distinct() |>
      arrange(hier3.id) # arrange ascending so matches loop order

    grid.lookup.s2 <- mortality |>
      select(hier2.id, hier1.id) |>
      distinct() |>
      arrange(hier2.id)

    grid.lookup <- as.matrix(grid.lookup)
    grid.lookup.s2 <- as.matrix(grid.lookup.s2)
    print("Lookups made")
    return(list(grid.lookup, grid.lookup.s2))
  } else {
    stop("invalid model name: BYM or nested only")
  }
}

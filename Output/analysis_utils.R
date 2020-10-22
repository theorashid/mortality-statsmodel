suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
})

#' Function to stack chains from MCMC output by row for sample (if samples = TRUE)
#' or samples2 (if samples = FALSE)
stack_chains <- function(mcmc.out, samples = TRUE) {
  n_chains <- length(mcmc.out)
  
  stacked <- list()
  if (samples) {
    for (i in 1:n_chains) {
      stacked[[i]] <- chain_output[[i]]$samples
    }
  } else {
    for (i in 1:n_chains) {
      stacked[[i]] <- chain_output[[i]]$samples2
    }
  }
  stacked <- do.call(rbind, stacked)
  
  return(stacked)
}

#' Calculates median and 95% confidence intervals of all samples,
#' which is a matrix of rows for iterations and a columns for each parameter
summarise_samples <- function(samples) {
  summary <- cbind(
    `Median`    = apply(samples, 2, median),
    `95%CI_low` = apply(samples, 2, function(x) quantile(x, 0.025)),
    `95%CI_upp` = apply(samples, 2, function(x) quantile(x, 0.975))
    )
  return(summary)
}

#' Function to replace rownames of lograte[a, s, t] in dataframe into
#' three rows at the end of the dataframe for year, hier3 region and age group
unlograte <- function(df) {
  df$node <- rownames(df)
  rownames(df) <- c()
  
  df <- df %>%
    mutate(tmp1 = str_split_fixed(df$node, ",", n=3)[,1]) %>% # "lograte[a"
    mutate(tmp2 = str_split_fixed(df$node, ",", n=3)[,2]) %>% # " s"
    mutate(tmp3 = str_split_fixed(df$node, ",", n=3)[,3]) # " t]"
  df <- df %>%
    mutate(age_group.id = str_sub(str_split_fixed(df$tmp1, "[a-z]+", n=2)[,2],2)) %>% # a (had to remove "[")
    mutate(YEAR.id = str_sub(df$tmp3,2,-2)) # a (had to remove first and last characters)
  df$age_group.id = as.numeric(df$age_group.id)
  df$hier3.id = as.numeric(df$tmp2) # as.numeric() deals with whitespace
  df$YEAR.id = as.numeric(df$YEAR.id)
  df <- df %>% # remove unnecessary columns
    select(-c(node, tmp1, tmp2, tmp3))
}
library(dplyr)
library(stringr)

unlograte <- function(df) {
  # Function to replace rownames of lograte[a, j, t] in dataframe into
  # three rows at the end of the dataframe for year, LSOA and age group
  df$node <- rownames(df)
  rownames(df) <- c()
  
  df <- df %>%
    mutate(tmp1 = str_split_fixed(df$node, ",", n=3)[,1]) %>% # "lograte[a"
    mutate(tmp2 = str_split_fixed(df$node, ",", n=3)[,2]) %>% # " j"
    mutate(tmp3 = str_split_fixed(df$node, ",", n=3)[,3]) # " t]"
  df <- df %>%
    mutate(age_group.id = str_sub(str_split_fixed(df$tmp1, "[a-z]+", n=2)[,2],2)) %>% # a (had to remove "[")
    mutate(YEAR.id = str_sub(df$tmp3,2,-2)) # a (had to remove first and last characters)
  df$age_group.id = as.numeric(df$age_group.id)
  df$LSOA.id = as.numeric(df$tmp2) # as.numeric() deals with whitespace
  df$YEAR.id = as.numeric(df$YEAR.id)
  df <- df %>% # remove unnecessary columns
    select(-c(node, tmp1, tmp2, tmp3))
}
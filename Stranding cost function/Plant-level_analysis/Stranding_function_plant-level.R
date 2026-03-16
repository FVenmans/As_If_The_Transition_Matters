# This code uses plant-level data on physical assets and emissions to fit a 'stranding cost function'
# This is part of "Optimal climate policy transition as if the transition matters"
# Authors: E. Campiglio, S. Dietz, F. Venmans

#*************************
# 0. Introduction and declarations ----
#*************************

# Install/load needed packages
pkgs <- c("DEoptim", "dplyr", "ggplot2", "janitor", "minpack.lm", "openxlsx", "Polychrome", "readr", "readxl", "rlang", "tidyr", "writexl", "xtable", "cowplot")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(pkgs, library, character.only = TRUE))

# Clear previous environment
rm(list = ls())

# Load functions from external script
source("Plant-level_analysis/Stranding_plant-level_FUNCTIONS.R")

# Choose the winsorisation level
winsor_level= 0.005
# Choose the minimum emission level (plants below emis_floor will be dropped)
emis_floor = 1e03
# Set depreciation rate \delta used in the model
delta<-0.04
# We calibrate LT to match asset half-life in the model
#Default value for \delta=0.04 is 34.65736
LT<-2*log(2)/delta

make_named_colors <- function(levels) {
  setNames(
    Polychrome::kelly.colors(length(levels)),
    levels
  )
}

#*************************
# 1. Data upload and manipulation ----
#*************************

plant_base_path <- "Plant-level_analysis"
plant_file <- "GEM-Global-Coal-Plant-Tracker-July-2024.xlsx"
plant_sheet <- 2

df_plants <- get_plant_data(
  base_path = plant_base_path,
  file_name = plant_file,
  sheet = plant_sheet
)


# Summary Statistics
df_plants %>%
  summarise(
    n_total = n(),
    n_used_for_avg_planned = sum(!is.na(plant_life_planned)),
    avg_plant_life_planned = mean(plant_life_planned, na.rm = TRUE),
    n_used_for_avg_actual = sum(!is.na(plant_life_actual)),
    avg_plant_life_actual = mean(plant_life_actual, na.rm = TRUE)
  )

# Run the function to obtain the data
df_plants_full <- process_gem_data(df_plants, LT)

analysis_region_levels <- df_plants_full %>%
  distinct(`Analysis Region`) %>%
  arrange(`Analysis Region`) %>%
  pull(`Analysis Region`)

analysis_region_cols <- make_named_colors(analysis_region_levels)

#*************************
# 2. SCCE/FLEI plots ----
#*************************

#--- Main stand-alone plots ---#

res_plot_SCCE <- plot_SCCE_plants(
  data = df_plants_full,
  winsor_level = winsor_level
)
p_SCCE<-res_plot_SCCE$plot

res_plot_FLEI <- plot_FLEI_plants(
  data = df_plants_full,
  yvar_scce = "SCCE",
  xvar_assets = "Net_Value",
  delta = delta,
  LT = LT,
  winsor_level = winsor_level
)
p_FLEI<-res_plot_FLEI$plot

# Keep cleaned / winsorised data for later use
df_FLEI_clean_wins <- res_plot_FLEI$data


#--- Simplified base plots ---#

# Create base plots with shorter labels, no legend, shorter titles
p_scce_simple <- res_plot_SCCE$plot +
  labs(y = "SCCE (USD/tCO2e)", title= "Stranding Cost per Cumulative Emission") +
  theme(legend.position = "none")
p_flei_simple <- res_plot_FLEI$plot +
  labs(y = "FLEI (kgCO2e/USD)", title = "Forward-Looking Emission Intensity") +
  theme(legend.position = "none")


#--- Stacked SCCE/FLEI plot ---#

main_panels <- plot_grid(
  p_scce_simple,
  p_flei_simple,
  ncol = 1,
  rel_heights = c(1, 1)
)

# Shared legend
shared_legend <- get_legend(
  res_plot_FLEI$plot +
    labs(color = NULL, fill = NULL) +
    guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
    theme(
      legend.position = "bottom",
      legend.box = "horizontal"
    )
)

# Stacked chart
plot_stacked <- plot_grid(
  main_panels,
  shared_legend,
  ncol = 1,
  rel_heights = c(1, 0.12)
)
plot_stacked


total_annual_emissions_mt <- df_plants_full %>%
  dplyr::summarise(
    total_annual_emissions_mt = sum(`Annual Emissions (MtCO2)`, na.rm = TRUE)
  ) %>%
  dplyr::pull(total_annual_emissions_mt)

#*************************
# 3. Function fitting ---- 
#*************************
# Fit plant-level FLEI curve with DEoptim (exp / double_exp / exp_shift)
# Uses df_FLEI_clean_wins produced by your plot_FLEI() (must contain: x_center, FLEI)
fit_plant_flei <- fit_flei_deoptim(
  data = df_FLEI_clean_wins,  
  xvar = "x_center",
  yvar = "FLEI",
  model = "exp_shift",  # exp, double_exp, exp_shift
  fixed_c = NULL,              # set e.g. 0 to force asymptote at 0
  x_fit_max = NULL,               # focus on first 5 tn USD
  penalty_floor_quantile = NULL,
  penalty_weight = NULL
)

print(fit_plant_flei$plot)
fit_plant_flei$params
fit_plant_flei$aic
fit_plant_flei$bic


# 4. Additional bits of analysis ----
#*************************

## 4.1 Total emissions per sector ----
summary_by_region <- plant_summary(df_plants_full, `Analysis Region`)
summary_by_region

summary_by_tech <- plant_summary(df_plants_full, `Combustion Technology`)
summary_by_tech

summary_by_coal <- plant_summary(df_plants_full, `Coal Type`)
summary_by_coal

total_emissions_2024 <- sum(df_plants_full$`Annual Emissions (MtCO2)`, na.rm = TRUE)

## 4.2 Analysis of plant age ----
p_age <- plot_age_distribution(
  df_plants_full,
  "Computed Remaining Lifetime (years)"
)
print(p_age[[1]])


























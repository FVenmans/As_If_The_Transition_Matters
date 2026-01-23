# This code uses company-level data on physical assets and emissions to fit an exponential intensity function 
# This is part of "Optimal climate policy transition as if the transition matters"
# E. Campiglio, S. Dietz, F. Venmans

#*************************
# 0. Introduction and declarations ----
#*************************
#*
# Install/load needed packages
pkgs <- c("DEoptim", "dplyr", "ggplot2", "janitor", "Polychrome", "readxl", "rlang")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(pkgs, library, character.only = TRUE))
# Clear previous environment
rm(list = ls())
# Load functions from external script
source("Capital_emission_intensity_FUNCTIONS.R")

# Choose the winsorisation level
winsor_level= 0.025
# Choose the minimum emission level (companies below emis_floor will be dropped)
emis_floor = 1e03
# Set depreciation rate \delta used in the model
delta<-0.04
# We calibrate LT (Orbis) to match asset half-life in the model
#Default value for \delta=0.04 is 34.65736
LT<-2*log(2)/delta

# Declare sectors to include in the analysis
sectors <- c(
  #"A01_Crop",
  "B5-10 Mining",
  "C10-12 Food",
  "C19 Coke",
  "C20 Chemicals",
  "C23 Non-metallic",
  "C24 Metals",
  "D35 Electricity",
  "E37-39 Waste",
  "F41-43 Construction",
  "H49 Land transport",
  "H50 Water transport",
  "H51 Air transport"
)

# Fix the colours assigned to sectors (for SCCE/FLEI charts)
# 1. Lock the sector order (also fixes legend order)
sector_levels <- sectors
# 2. Create a deterministic palette and name it by sector label
n <-length(sector_levels)
sector_cols <- setNames(Polychrome::kelly.colors(n), sector_levels)

# Choose scaling mode:
# "aggregate" = single global scale_factor 
# "sectoral"  = NACE-specific scaling factors
scaling_mode <- "sectoral"
# Choose the aggregate scaling factor to scale emissions 
agg_scale_factor = 1

sector_scale_factors <- c(
  # A01   = 5.954203481, # only A01 energy emissions
  # A01   = 131.603772292, # considering non-energy emissions
  "B5-10 Mining"         = 3.243871632,
  "C10-12 Food"          = 2.937160514,
  "C19 Coke"             = 1.477237726,
  "C20 Chemicals"        = 2.462214930,
  "C23 Non-metallic"     = 2.303791424,
  "C24 Metals"           = 1.005622784,
  "D35 Electricity"      = 5.595107938,
  "E37-39 Waste"         = 13.124664100,
  "F41-43 Construction"  = 0.470241819,
  "H49 Land transport"   = 10.716194590,
  "H50 Water transport"  = 2.270987151,
  "H51 Air transport"    = 2.379581910
)

#*************************
# 1. Data upload and manipulation ----
#*************************

# Create list with number of elements equal to sectors, storing data
df_list <- lapply(sectors, function(sec) {
  # Decide which scaling factor to use for this sector
  get_sector_data(
    sector_code  = sec,
    base_path = "."
  )%>%
    mutate(
      sector = sec
    )
})
names(df_list) <- sectors  # rename list elements using sector names

# Create an aggregate dataframe with raw data from all sectors
df_raw <- bind_rows(df_list)

# Create an imputed dataframe, which assigns values to gross/net capital when one of them is missing
df_raw_imp <- impute_sector_data(
  df_raw = df_raw,
  LT = LT
)

# Check number of companies for which data has been imputed 
imputation_summary <- attr(df_raw_imp, "imputation_summary")
# imputation_summary

# Create a dataframe where scaling factors are applied, key measures are computed (PPE, SCCE, FLEI), small firms are dropped, unnecessary columns are dropped for readability
# Note: we convert units of measure of emissions (tCO2e-->MtCO2e) and assets (thUSD-->milUSD)
df_full <- process_sector_data(
  df = df_raw_imp,
  LT = LT,
  emis_floor = emis_floor,
  scaling_mode = scaling_mode,
  agg_scale_factor = agg_scale_factor,
  sector_scale_factors = sector_scale_factors
)

# Create two dataframe ready to plot 
df_SCCE <- create_df_SCCE(
  df_full, 
  LT = LT, 
  winsor_level = winsor_level)

df_FLEI <- create_df_FLEI(
  df_full, 
  winsor_level = winsor_level)


#*************************
# 2. SCCE/FLEI Plots ----
#*************************

# Create SCCE plot
res_plot_SCCE <- plot_SCCE(df_SCCE, winsor_level = winsor_level)
print(res_plot_SCCE$plot)
# Chart without title if needed for paper
# result_plot_SCCE$plot + labs(title = NULL)

# Create FLEI plot
res_plot_FLEI <- plot_FLEI(df_FLEI, winsor_level = winsor_level)
print(res_plot_FLEI$plot)
# Chart without title if needed for paper
# res_plot_FLEI$plot + labs(title = NULL)


#*************************
# 3. Function fitting ----
#*************************

# Generic curve fitting algorithm using DEoptim
# - Fits models: exponential ("exp"), double exponential ("double_exp") or shifted exponential ("exp_shift")
# - Applies weighting schemes: Weighting scheme: "none" / exponential upweighting ("exp_up") / exponential decay ("exp_decay") / "logistic"
# The function minimizes the (weighted) sum of squared errors (SSE) between observed and predicted SCCE
# Returns best-fit parameters, SSE, AIC/BIC, residuals, and a ggplot of data + fitted curve.

fit_FLEI <- fit_curve_deoptim(
  data = df_FLEI,
  xvar = "x_center",
  yvar = "FLEI",
  model = "exp_shift", 
  weight_scheme = "logistic", # 
  w_x0 = 10,   # For "logistic": after w_x0 (in tn USD) observations are down-weighted by half
  w_s  = 1     # For "logistic" w_s controls how smooth the weight transition is. Smaller = sharper cutoff
)

# Adjustments before plotting
fit_FLEI$plot +
  labs(
    x = "Tangible assets (trillion USD)",
    y = "Forward-looking Emission Intensity (kgCO2e/USD)",
    title = "Shifted Exponential Fit: FLEI vs Tangible Assets"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10))
  )

# Inspect results
fit_FLEI$params
fit_FLEI$aic
fit_FLEI$bic

# Print results easy to copy
p <- fit_FLEI$params
txt <- sprintf("a=%s; b=%s; c=%s",
               formatC(unname(p[["a"]]), format = "f", digits = 7),
               formatC(unname(p[["b"]]), format = "f", digits = 7),
               formatC(unname(p[["c"]]), format = "f", digits = 7))
cat(txt, "\n")   # prints in console


#*************************
# 4. Additional bits of analysis ----
#*************************

## 4.1 Total emissions per sector ----
sectoral_info_df = sectoral_info(df_full)

## 4.2 Analysis of asset age ----

ggplot(df_full %>% filter(is.finite(age_PPE)), aes(x = age_PPE)) +
  geom_histogram(bins = 50) +
  labs(x = "Asset age (years)", y = "Count", title = paste0("Age distribution (N=", sum(is.finite(df_full$age_PPE)), ")")) +
  theme_bw()

## 4.3 Integral below the curve ----

# Set initial emissions (the area under the integral)
P0 <- 53.2

# This function uses the shifted exponential curve fitted above
# It returns the right extreme of the curve that matches the declared integral areas (P_0)
res <- x_from_integral(fit=fit_FLEI, P0 = P0, x0 = 0)
print(res$x)








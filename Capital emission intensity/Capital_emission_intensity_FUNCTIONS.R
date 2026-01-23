# This script is called by the Stranding function analysis main script

#*************************
## 0. Introductory functions ----
#*************************

# This function helps to deal with the presence of two alternative scaling approaches
get_scaling_label <- function() {
  # If sectoral scaling is active, show "sectoral"
  if (exists("scaling_mode", inherits = TRUE) &&
      identical(get("scaling_mode", inherits = TRUE), "sectoral")) {
    return("sectoral")
  }
  # Otherwise, fall back to the numeric aggregate scale_factor if available
  if (exists("scale_factor", inherits = TRUE)) {
    return(as.character(get("scale_factor", inherits = TRUE)))
  }
  # Fallback
  return("unknown")
}

#*************************
## 1. Extract and manipulate data ----
#*************************

# This function takes the Excel files with raw data and aggregates them into a dataframe 
# Companies with no emission or asset data are dropped
get_sector_data <- function(sector_code, 
                             base_path = ".") {
  
  # Build the Excel file path 
  file_path <- file.path(base_path, "Data_Sector", paste0("NACE_", sector_code, "_TA.xlsx"))
  # Read the Excel sheets
  df_company <- read_excel(file_path, sheet = "Results")   %>% janitor::clean_names()
  df_emissions <- read_excel(file_path, sheet = "Emissions") %>% janitor::clean_names()
  # Convert literal "NA" strings to true NA
  df_emissions <- df_emissions %>% mutate(across(where(is.character), ~na_if(.x, "NA")))
  # Join the tables using ISIN number to match
  df <- df_company %>%
    left_join(df_emissions, by = "isin_number")
  # Rename columns for easier handling and convert them to numeric
  df <- df %>%
    rename(
      Company              = company_name_latin_alphabet,
      Country              = country_iso_code,
      NACE_code            = nace_rev_2_core_code_4_digits,
      Turnover_th          = operating_revenue_turnover_th_usd_last_avail_yr,
      Employees            = number_of_employees_last_avail_yr,
      # Noncurrent_assets  = non_current_assets_th_usd_last_avail_yr, #unexplained weird error in renaming this column
      Total_assets_th      = total_assets_th_usd_last_avail_yr,
      CO2e1_t              = co2_equivalents_emission_direct,
      CO2etot_t            = co2_equivalents_emission_total,
      Net_TA_th            = net_property_plant_equipment_th_usd_last_avail_yr,
      Cum_Dep_NoBreak_th   = accumulated_depreciation_no_breakdown_th_usd_last_avail_yr,
      Land_th              = land_th_usd_last_avail_yr,
      Cum_Dep_Land_th      = accumulated_depreciation_on_land_th_usd_last_avail_yr,
      Net_Land_th          = net_land_th_usd_last_avail_yr,
      Buildings_th         = buildings_th_usd_last_avail_yr,
      Cum_Dep_Buildings_th = accumulated_depreciation_on_buildings_th_usd_last_avail_yr,
      Net_Buildings_th     = net_buildings_th_usd_last_avail_yr,
      PlantMach_th         = plant_machinery_th_usd_last_avail_yr,
      Cum_Dep_PlantMach_th = accumulated_depreciation_on_plant_machinery_th_usd_last_avail_yr,
      Net_PlantMach_th     = net_plant_machinery_th_usd_last_avail_yr,
      Transp_Equipment_th  = transportation_equipment_th_usd_last_avail_yr,
      Cum_Dep_Transp_th    = accumulated_depreciation_on_transportation_equipment_th_usd_last_avail_yr,
      Net_Transp_th        = net_transportation_equipment_th_usd_last_avail_yr,
      Leased_Assets_th     = leased_assets_th_usd_last_avail_yr,
      Cum_Dep_Leased_th    = accumulated_depreciation_on_leased_assets_th_usd_last_avail_yr,
      Net_Leased_th        = net_leased_assets_th_usd_last_avail_yr,
      Other_PPE_th         = other_property_plant_equipment_th_usd_last_avail_yr,
      Cum_Dep_OtherPPE_th  = accumulated_depreciation_on_other_property_plant_equipment_th_usd_last_avail_yr,
      Net_OtherPPE_th      = net_other_property_plant_equipment_th_usd_last_avail_yr
    ) %>%
    mutate(
      across(
        c(Turnover_th, Employees, Total_assets_th, Net_TA_th, CO2e1_t, CO2etot_t,
          Cum_Dep_NoBreak_th, Land_th, Cum_Dep_Land_th, Net_Land_th,
          Buildings_th, Cum_Dep_Buildings_th, Net_Buildings_th,
          PlantMach_th, Cum_Dep_PlantMach_th, Net_PlantMach_th,
          Transp_Equipment_th, Cum_Dep_Transp_th, Net_Transp_th,
          Leased_Assets_th, Cum_Dep_Leased_th, Net_Leased_th,
          Other_PPE_th, Cum_Dep_OtherPPE_th, Net_OtherPPE_th #, Non_current_assets
        ),
        as.numeric
      )
    )
  if ("n" %in% names(df)) df$n <- NULL  # remove the n. column with ordering from ORBIS
  
  # Drop companies for which there is no emission data
  df <- df %>% filter(!is.na(CO2e1_t), CO2e1_t > 0, !is.na(Total_assets_th))
  
  return(df)
}


impute_sector_data <- function(df_raw, LT) {
  
  assets <- list(
    Land        = c("Land_th", "Cum_Dep_Land_th", "Net_Land_th"),
    Buildings  = c("Buildings_th", "Cum_Dep_Buildings_th", "Net_Buildings_th"),
    PlantMach  = c("PlantMach_th", "Cum_Dep_PlantMach_th", "Net_PlantMach_th"),
    Transport  = c("Transp_Equipment_th", "Cum_Dep_Transp_th", "Net_Transp_th"),
    Leased     = c("Leased_Assets_th", "Cum_Dep_Leased_th", "Net_Leased_th"),
    OtherPPE   = c("Other_PPE_th", "Cum_Dep_OtherPPE_th", "Net_OtherPPE_th")
  )
  
  df <- df_raw
  
  # 1. Compute ages + eligibility flags
  for (a in names(assets)) {
    
    gross_col <- assets[[a]][1]
    net_col   <- assets[[a]][3]
    
    gross <- df[[gross_col]]
    net   <- df[[net_col]]
    
    gross_missing <- is.na(gross) | gross <= 0
    net_missing   <- is.na(net)   | net   <= 0
    
    share_dep <- ifelse(
      !gross_missing & !net_missing,
      pmin(pmax(1 - net / gross, 0), 1),
      NA_real_
    )
    
    df[[paste0("age_", a)]] <- LT * share_dep
    
    df[[paste0("can_impute_gross_", a)]] <- gross_missing & !net_missing
    df[[paste0("can_impute_net_", a)]]   <- net_missing   & !gross_missing
  }
  
  # 2. Mean age per category (from observed gross+net only)
  mean_age_category <- sapply(
    names(assets),
    function(a) {
      x <- df[[paste0("age_", a)]]
      if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
    }
  )
  
  # 3. Impute values + fill missing ages + imputation flags
  for (a in names(assets)) {
    
    gross_col <- assets[[a]][1]
    net_col   <- assets[[a]][3]
    age_col   <- paste0("age_", a)
    
    mean_age_a <- mean_age_category[a]
    if (is.na(mean_age_a)) next
    
    rem_share <- (LT - mean_age_a) / LT
    
    idx_net   <- df[[paste0("can_impute_net_", a)]]
    idx_gross <- df[[paste0("can_impute_gross_", a)]]
    
    # Impute values
    df[[net_col]][idx_net]     <- df[[gross_col]][idx_net] * rem_share
    df[[gross_col]][idx_gross] <- df[[net_col]][idx_gross] / rem_share
    
    # Actual imputation indicator (exactly one side missing)
    df[[paste0("imputed_", a)]] <- idx_net | idx_gross
    
    # Fill missing ages with mean age
    df[[age_col]][is.na(df[[age_col]])] <- mean_age_a
  }
  
  # 4. Asset-level imputation diagnostics
  n_total <- nrow(df)
  
  imputation_summary <- data.frame(
    asset = names(assets),
    n_imputed = NA_integer_,
    pct_total_firms = NA_real_,
    n_used_for_mean_age = NA_integer_,
    n_both_missing = NA_integer_
  )
  
  for (i in seq_along(names(assets))) {
    
    a <- names(assets)[i]
    
    gross_col <- assets[[a]][1]
    net_col   <- assets[[a]][3]
    
    gross <- df[[gross_col]]
    net   <- df[[net_col]]
    
    gross_missing <- is.na(gross) | gross <= 0
    net_missing   <- is.na(net)   | net   <= 0
    
    imputed_flag <- df[[paste0("imputed_", a)]]
    age_col      <- df[[paste0("age_", a)]]
    
    n_imputed <- sum(imputed_flag, na.rm = TRUE)
    
    used_for_mean <- !gross_missing & !net_missing & !is.na(age_col)
    n_used_for_mean <- sum(used_for_mean)
    
    both_missing <- gross_missing & net_missing
    n_both_missing <- sum(both_missing)
    
    imputation_summary$n_imputed[i] <- n_imputed
    imputation_summary$pct_total_firms[i] <-
      if (n_total > 0) n_imputed / n_total else NA_real_
    imputation_summary$n_used_for_mean_age[i] <- n_used_for_mean
    imputation_summary$n_both_missing[i] <- n_both_missing
  }
  
  imputed_cols <- paste0("imputed_", names(assets))
  
  df$imputed_any <- apply(
    df[, imputed_cols, drop = FALSE],
    1,
    function(x) any(x, na.rm = TRUE)
  )
  
  # firm-level: all PPE subcategories missing
  ppe_cols <- c(
    "Land_th", "Net_Land_th",
    "Buildings_th", "Net_Buildings_th",
    "PlantMach_th", "Net_PlantMach_th",
    "Transp_Equipment_th", "Net_Transp_th",
    "Leased_Assets_th", "Net_Leased_th",
    "Other_PPE_th", "Net_OtherPPE_th"
  )
  ppe_cols <- ppe_cols[ppe_cols %in% names(df)]
  
  df$all_PPE_missing <- apply(
    df[, ppe_cols, drop = FALSE],
    1,
    function(x) all(is.na(x) | x <= 0)
  )
  
  total_row <- data.frame(
    asset = "Total",
    n_imputed = sum(df$imputed_any, na.rm = TRUE),
    pct_total_firms =
      if (n_total > 0)
        sum(df$imputed_any, na.rm = TRUE) / n_total
    else NA_real_,
    n_used_for_mean_age = NA_integer_,
    n_both_missing = sum(df$all_PPE_missing, na.rm = TRUE)
  )
  
  imputation_summary <- rbind(imputation_summary, total_row)
  
  attr(df, "imputation_summary") <- imputation_summary
  
  # 5. Column ordering
  if (requireNamespace("dplyr", quietly = TRUE)) {
    for (a in names(assets)) {
      net_col <- assets[[a]][3]
      if (!net_col %in% names(df)) next
      
      df <- dplyr::relocate(
        df,
        dplyr::any_of(c(
          paste0("age_", a),
          paste0("can_impute_gross_", a),
          paste0("can_impute_net_", a),
          paste0("imputed_", a)
        )),
        .after = dplyr::all_of(net_col)
      )
    }
  }
  
  return(df)
}

# This function takes in pre-arranged excel spreadsheets and manipulates them to create df_list
process_sector_data     <- function(df,
                                    LT,
                                    emis_floor,
                                    scaling_mode = "sectoral",
                                    agg_scale_factor = 1,
                                    sector_scale_factors = NULL,
                                    cols_to_drop = NULL) {
  
  # --- 1) scale factor used ---
  if (scaling_mode == "aggregate") {
    df <- df %>%
      mutate(scale_factor = agg_scale_factor)
  } else {
    if (is.null(sector_scale_factors))
      stop("sector_scale_factors must be provided when scaling_mode == 'sectoral'.")
    
    # sector_scale_factors must be a named numeric vector with names matching df$sector
    df <- df %>%
      mutate(scale_factor = unname(sector_scale_factors[as.character(sector)]))
    
    if (any(is.na(df$scale_factor))) {
      bad <- sort(unique(as.character(df$sector[is.na(df$scale_factor)])))
      stop("Missing sectoral scaling factor for: ", paste(bad, collapse = ", "))
    }
  }
  
  # --- 2) scale the core variables (you can extend this list if you want) ---
  df <- df %>%
    mutate(
      CO2e1_t         = CO2e1_t         * scale_factor,
      CO2etot_t       = CO2etot_t       * scale_factor,
      Total_assets_th = Total_assets_th * scale_factor,
      Net_TA_th       = Net_TA_th       * scale_factor
    )
  
  # --- 3) compute PPE aggregates (treat NA components as 0, but set aggregate to NA if all components are NA) ---
  gross_cols <- c("Buildings_th", "PlantMach_th", "Transp_Equipment_th", "Leased_Assets_th", "Other_PPE_th")
  net_cols   <- c("Net_Buildings_th", "Net_PlantMach_th", "Net_Transp_th", "Net_Leased_th", "Net_OtherPPE_th")
  
  gross_cols <- gross_cols[gross_cols %in% names(df)]
  net_cols   <- net_cols[net_cols %in% names(df)]
  
  df <- df %>%
    mutate(
      n_gross_nonNA = if (length(gross_cols) > 0) rowSums(!is.na(across(all_of(gross_cols)))) else NA_integer_,
      n_net_nonNA   = if (length(net_cols)   > 0) rowSums(!is.na(across(all_of(net_cols))))   else NA_integer_, 
  
      Gross_PPE_th = if (length(gross_cols) > 0)
        if_else(
          n_gross_nonNA > 0,
          rowSums(across(all_of(gross_cols), ~ coalesce(.x, 0))),
          NA_real_
        ) else NA_real_,
      
      Net_PPE_th = if (length(net_cols) > 0)
        if_else(
          n_net_nonNA > 0,
          rowSums(across(all_of(net_cols), ~ coalesce(.x, 0))),
          NA_real_
        ) else NA_real_
    ) %>%
    mutate(
      # scale the aggregates (equivalent to scaling components)
      Gross_PPE_th = Gross_PPE_th * scale_factor,
      Net_PPE_th   = Net_PPE_th   * scale_factor
    )
  
  # --- 4) derive SCCE/FLEI variables (NO imputation here) ---
  df <- df %>%
    mutate(
      share_dep_PPE = if_else(!is.na(Gross_PPE_th) & Gross_PPE_th > 0,
                              pmin(pmax(1 - Net_PPE_th / Gross_PPE_th, 0), 1),
                              NA_real_),
      share_rem_lt_PPE = 1 - share_dep_PPE,
      age_PPE = LT * share_dep_PPE,
      
      # EI_PPE     = if_else(!is.na(Gross_PPE) & Gross_PPE > 0, CO2e1 / Gross_PPE, NA_real_),
      # EI_Net_PPE = if_else(!is.na(Net_PPE)   & Net_PPE   > 0, CO2e1 / Net_PPE,   NA_real_),
      
      # Remaining-life cumulative emissions (this is what your SCCE will use)
      # CO2e1Cum = CO2e1 * LT * share_rem_lt_PPE,
      
      # SCCE (Stranding cost per cumulative emissions) 
      SCCE = if_else(!is.na(CO2e1_t * LT * share_rem_lt_PPE) & CO2e1_t * LT * share_rem_lt_PPE > 0,
                     (Net_PPE_th * 1000) / (CO2e1_t * LT * share_rem_lt_PPE),
                     NA_real_),
      
      # FLEI: forward-looking emissions per $ of net PPE over remaining life
      FLEI = if_else(!is.na(Net_PPE_th) &  Net_PPE_th > 0, (CO2e1_t*(LT-age_PPE)*delta) / Net_PPE_th, NA_real_)
    )
  
  # --- 5) drop low-emission firms ---
  df <- df %>%
    filter(CO2e1_t >= emis_floor)
  
  # --- 6) change the unit of measure of key variables for dislay ---
  df <- df %>%
    mutate(
      CO2e1_Mt       = CO2e1_t / 1e6,
      Gross_PPE_mil  = Gross_PPE_th / 1e3,
      Net_PPE_mil    = Net_PPE_th   / 1e3
    )
  
  # --- 7) drop unnecessary columns ---
  if (is.null(cols_to_drop)) {
    cols_to_drop <- c(
      "inactive", "quoted", "Turnover_th", "branch", "own_data", "Employees", "woco", "NACE_code",
      "consolidation_code", "nace_rev_2_core_code_description", "last_avail_year",
      "Cum_Dep_NoBreak_th", "non_current_assets_th_usd_last_avail_yr",
      "Total_assets_th", "Net_TA_th",
      "Land_th", "Cum_Dep_Land_th", "Net_Land_th",
      "Buildings_th", "Cum_Dep_Buildings_th", "Net_Buildings_th",
      "PlantMach_th", "Cum_Dep_PlantMach_th", "Net_PlantMach_th",
      "Transp_Equipment_th", "Cum_Dep_Transp_th", "Net_Transp_th",
      "Leased_Assets_th", "Cum_Dep_Leased_th", "Net_Leased_th",
      "Other_PPE_th", "Cum_Dep_OtherPPE_th", "Net_OtherPPE_th",
      "orbis_id_number", "lei_legal_entity_identifier", "isin_number",
      "CO2etot_t",
      "share_dep_PPE", "share_rem_lt_PPE",
      "n_gross_nonNA", "n_net_nonNA",
      "CO2e1_t", "Gross_PPE_th", "Net_PPE_th", 
      "age_Land", "age_Buildings", "age_PlantMach", "age_Transport", "age_Leased", "age_OtherPPE",
      "can_impute_gross_Land", "can_impute_gross_Buildings", "can_impute_gross_PlantMach", 
      "can_impute_gross_Transport", "can_impute_gross_Leased", "can_impute_gross_OtherPPE",
      "can_impute_net_Land", "can_impute_net_Buildings", "can_impute_net_PlantMach", 
      "can_impute_net_Transport", "can_impute_net_Leased", "can_impute_net_OtherPPE", 
      "imputed_Land", "imputed_Buildings", "imputed_PlantMach", "imputed_Transport", "imputed_Leased",
      "imputed_OtherPPE", "imputed_any", "all_PPE_missing"
    )
  }
  
  df <- df %>% select(-any_of(cols_to_drop))
  
  # --- 8) aesthetics: rounded display columns + column order ---
  df <- df %>%
    select(
      Company,
      Country,
      sector,
      scale_factor,
      CO2e1_Mt,
      Gross_PPE_mil,
      Net_PPE_mil,
      age_PPE,
      SCCE,
      FLEI,
      everything()
    ) 
  
  return(df)
}


create_df_SCCE <- function(df_full, LT, winsor_level) {
  df_SCCE <- df_full %>%
    mutate(
      share_rem_lt_PPE = 1 - (age_PPE / LT),
      CO2e1Cum_t = (CO2e1_Mt * 1e6) * LT * share_rem_lt_PPE  # tonnes CO2e
    ) %>%
    filter(
      !is.na(SCCE), is.finite(SCCE),
      !is.na(CO2e1Cum_t), is.finite(CO2e1Cum_t), CO2e1Cum_t > 0
    ) %>%
    arrange(SCCE)
  
  ql <- quantile(df_SCCE$SCCE, winsor_level, na.rm = TRUE)
  qu <- quantile(df_SCCE$SCCE, 1 - winsor_level, na.rm = TRUE)
  
  df_SCCE %>%
    mutate(
      SCCE = pmin(pmax(SCCE, ql), qu),
      x_end    = cumsum(CO2e1Cum_t) / 1e9,   # GtCO2e
      x_start  = lag(x_end, default = 0),
      x_center = 0.5 * (x_start + x_end)
    )
}

create_df_FLEI <- function(df_full, winsor_level) {
  df_FLEI <- df_full %>%
    mutate(
      Net_PPE_trillion = Net_PPE_mil / 1e6   # million USD -> trillion USD
    ) %>%
    filter(
      !is.na(FLEI), is.finite(FLEI),
      !is.na(Net_PPE_trillion), is.finite(Net_PPE_trillion), Net_PPE_trillion > 0
    ) %>%
    arrange(desc(FLEI))  # high FLEI on the left (as in your rectangle plots)
  
  ql <- quantile(df_FLEI$FLEI, winsor_level, na.rm = TRUE)
  qu <- quantile(df_FLEI$FLEI, 1 - winsor_level, na.rm = TRUE)
  
  df_FLEI %>%
    mutate(
      FLEI     = pmin(pmax(FLEI, ql), qu),
      x_end    = cumsum(Net_PPE_trillion),
      x_start  = lag(x_end, default = 0),
      x_center = 0.5 * (x_start + x_end)
    )
}


#*************************
## 2. Plotting functions ----
#*************************

plot_SCCE <- function(data,
                      winsor_level,
                      scaling_label = NULL) {
  
  stopifnot(all(c("x_start", "x_end", "SCCE", "sector") %in% names(data)))
  
  if (is.null(scaling_label)) scaling_label <- get_scaling_label()
  
  p <- ggplot(data, aes(
    xmin = x_start, xmax = x_end,
    ymin = 0,       ymax = SCCE,
    fill = sector
  )) +
    geom_rect(color = NA, linewidth = 0.1) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(
      x = "Embodied emissions (GtCO2e)",
      y = "Stranding Cost per Cumulative Emission (USD/tCO2e)",
      title = paste0(
        "SCCE (winsorized at ", winsor_level * 100, "%; ",
        "Scaling factor = ", scaling_label, "; ",
        "N° obs = ", nrow(data), ")"
      ),
      fill = NULL
    ) +
    scale_fill_manual(values = sector_cols, drop = FALSE) +
    guides(fill = guide_legend(nrow = 2, byrow = TRUE)) +
    theme_bw() +
    theme(legend.position = "bottom")
  
  list(plot = p, data = data)
}


plot_FLEI <- function(data,
                      winsor_level,
                      scaling_label = NULL) {
  
  stopifnot(all(c("x_start", "x_end", "FLEI", "sector") %in% names(data)))
  
  if (is.null(scaling_label)) scaling_label <- get_scaling_label()
  
  p <- ggplot(data, aes(xmin = x_start, xmax = x_end, ymin = 0, ymax = FLEI, fill = sector)) +
    geom_rect(color = NA, linewidth = 0.1) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(
      x = "Tangible assets (Trillion USD)",
      y = "Forward-Looking Emission Intensity (kgCO2e/USD)",
      title = paste0(
        "FLEI (winsorized at ", winsor_level * 100, "%; ",
        "Scaling factor = ", scaling_label, "; ",
        "N° obs = ", nrow(data), ")"
      ),
      fill = NULL
    ) +
    scale_fill_manual(values = sector_cols, drop = FALSE, name = NULL) +
    guides(fill = guide_legend(nrow = 2, byrow = TRUE)) +
    theme_bw() +
    theme(legend.position = "bottom")
  
  list(
    plot = p,
    data = data
  )
}


#*************************
## 3. Function fitting ----
#*************************

# Generic curve fitting algorithm using DEoptim
# - Fits exp / double_exp / exp_shift to (x,y)
# - Optional x-dependent weights: none / exp_up / exp_decay / logistic

# --- Prediction for given params and model (to be used in fit_curve_deoptim) ---
.predict_curve <- function(x, params, model = c("exp", "double_exp", "exp_shift")) {
  model <- match.arg(model)
  
  if (model == "exp") {
    a <- params[1]; b <- params[2]
    a * exp(b * x)
    
  } else if (model == "double_exp") {
    a <- params[1]; b <- params[2]; c <- params[3]; d <- params[4]
    a * exp(b * x) + c * exp(d * x)
    
  } else { # exp_shift
    # y = a * exp(b * x) + c
    a <- params[1]; b <- params[2]; c <- params[3]
    a * exp(b * x) + c
  }
}

# --- Weighted SSE objective (to be used in fit_curve_deoptim) ---
.objective_curve <- function(params, x, y,
                             model = c("exp", "double_exp", "exp_shift"),
                             weights = NULL) {
  model <- match.arg(model)
  yhat  <- .predict_curve(x, params, model)
  
  if (is.null(weights)) {
    sum((y - yhat)^2)
  } else {
    sum(weights * (y - yhat)^2)
  }
}

# --- Main fitter: DEoptim wrapper ---
fit_curve_deoptim <- function(
    data,                   
    xvar,                   # Name (string) of x variable column 
    yvar,                   # Name (string) of y variable column
    model = c("exp", "double_exp", "exp_shift"),
    # - "exp":        y = a * exp(b * x)
    # - "double_exp": y = a * exp(b * x) + c * exp(d * x)
    # - "exp_shift":  y = a * exp(b * x) + c   (c is the asymptote)
    bounds = NULL,           #  If provided, must be a list with elements: bounds$lower (numeric vector) and bounds$upper (numeric vector) matching the number/order of parameters
    fixed_c = NULL,          # Only for "exp_shift": if set to a numeric value, c is fixed to this value (not estimated)
    x_fit_max = NULL,        # optional hard cutoff: only observations with x <= x_fit_max are used to estimate parameters
    weight_scheme = c("none", "exp_up", "exp_decay", "logistic"),
    w_alpha = 0,             # For "exp_up": controls the extra weight at x=0.
    w_tau   = NULL,          # For "exp_up"/"exp_decay": decay scale. Smaller w_tau => faster decay with x (more focus on the left).
    w_x0    = NULL,          # For "logistic": after w_x0 (in tn USD) observations are downweighted by half
    w_s     = NULL,          # For "logistic" w_s controls how smooth the weight transition is. Smaller = sharper cutoff
    de_control = DEoptim.control(itermax = 200000, NP = 50,
                                 trace = FALSE, reltol = 1e-9, steptol = 1000)
) {
  model <- match.arg(model)
  weight_scheme <- match.arg(weight_scheme)
  
  stopifnot(all(c(xvar, yvar) %in% names(data)))
  
  df <- data %>%
    dplyr::filter(!is.na(.data[[xvar]]), !is.na(.data[[yvar]]))
  
  x <- df[[xvar]]
  y <- df[[yvar]]
  
  # Optional hard restriction
  if (!is.null(x_fit_max)) {
    fit_mask <- x <= x_fit_max
  } else {
    fit_mask <- rep(TRUE, length(x))
  }
  
  x_fit <- x[fit_mask]
  y_fit <- y[fit_mask]
  
  # --- weights aligned with x_fit / y_fit ---
  weight_fun <- function(x) {
    if (weight_scheme == "none") {
      return(rep(1, length(x)))
    }
    
    if (weight_scheme == "exp_up") {
      # w(x) = 1 + alpha * exp(-x/tau)  (upweights left; tail weight >= 1)
      if (is.null(w_tau) || w_tau <= 0 || is.null(w_alpha) || w_alpha == 0) return(rep(1, length(x)))
      return(1 + w_alpha * exp(-x / w_tau))
    }
    
    if (weight_scheme == "exp_decay") {
      # w(x) = exp(-x/tau)  (downweights right tail toward 0)
      if (is.null(w_tau) || w_tau <= 0) return(rep(1, length(x)))
      return(exp(-x / w_tau))
    }
    
    if (weight_scheme == "logistic") {
      # w(x) = 1 / (1 + exp((x - x0)/s))  (soft x_fit_max; downweights right tail)
      if (is.null(w_x0) || is.null(w_s) || w_s <= 0) {
        stop("For weight_scheme='logistic', please provide w_x0 and positive w_s.")
      }
      return(1 / (1 + exp((x - w_x0) / w_s)))
    }
    
    rep(1, length(x))
  }
  
  w_fit <- weight_fun(x_fit)
  
  # Default bounds (scaled from data)
  ymin <- min(y_fit, na.rm = TRUE)
  ymax <- max(y_fit, na.rm = TRUE)
  
  if (is.null(bounds)) {
    if (model == "exp") {
      bounds <- list(lower = c(0, -1), upper = c(2 * ymax, 1))
      
    } else if (model == "double_exp") {
      bounds <- list(
        lower = c(0, -1, 0, -1),
        upper = c(2 * ymax, 1, 2 * ymax, 1)
      )
      
    } else { # exp_shift
      if (is.null(fixed_c)) {
        bounds <- list(
          lower = c(0,   -10,    ymin),
          upper = c(2*ymax, 1,  ymax)
        )
      } else {
        bounds <- list(
          lower = c(0,   -10,    fixed_c),
          upper = c(2*ymax, 1,  fixed_c)
        )
      }
    }
  } else {
    # If user supplied custom bounds and wants fixed c, clamp the 3rd param
    if (model == "exp_shift" && !is.null(fixed_c)) {
      bounds$lower[3] <- fixed_c
      bounds$upper[3] <- fixed_c
    }
  }
  
  # Wrap objective for DEoptim
  obj <- function(par) .objective_curve(par, x_fit, y_fit, model, weights = w_fit)
  
  # Run DEoptim
  de_out <- DEoptim(
    fn      = obj,
    lower   = bounds$lower,
    upper   = bounds$upper,
    control = de_control
  )
  
  best <- de_out$optim$bestmem
  sse  <- .objective_curve(best, x_fit, y_fit, model, weights = w_fit)
  
  # AIC/BIC (RSS-based; Gaussian errors; sigma^2 = SSE/n)
  n <- length(y_fit)
  k <- length(best)
  if (model == "exp_shift" && !is.null(fixed_c)) k <- k - 1
  aic <- n * log(sse / n) + 2 * k
  bic <- n * log(sse / n) + log(n) * k
  
  # Smooth prediction curve for plotting
  x_seq <- seq(min(x, na.rm = TRUE), max(x, na.rm = TRUE), length.out = 400)
  y_fit_curve <- .predict_curve(x_seq, best, model)
  fit_df <- data.frame(x = x_seq, y = y_fit_curve)
  
  # Residuals (on full df, not just fitting subset)
  y_hat_points <- .predict_curve(x, best, model)
  resid_all    <- y - y_hat_points
  
  # Plot
  title_txt <- paste0(
    dplyr::case_when(
      model == "exp"        ~ "Exponential",
      model == "double_exp" ~ "Double Exponential",
      model == "exp_shift"  ~ "Shifted Exponential"
    ),
    " Fit — ", yvar
  )
  
  p <- ggplot(df, aes(x = .data[[xvar]], y = .data[[yvar]])) +
    geom_point(color = "blue", alpha = 0.6) +
    geom_line(data = fit_df, aes(x = x, y = y), color = "red", linewidth = 1) +
    labs(title = title_txt, x = xvar, y = yvar) +
    theme_bw()
  
  list(
    model     = model,
    params    = setNames(
      as.numeric(best),
      if (model == "exp") {
        c("a","b")
      } else if (model == "double_exp") {
        c("a","b","c","d")
      } else {
        c("a","b","c")
      }
    ),
    sse       = sse,
    aic       = aic,
    bic       = bic,
    data_used = df,
    residuals = resid_all,
    pred_df   = fit_df,
    plot      = p,
    de_output = de_out,
    bounds    = bounds,
    weights   = list(scheme = weight_scheme, alpha = w_alpha, tau = w_tau, x0 = w_x0, s = w_s),
    x_fit_max = x_fit_max
  )
}






#*************************
## 4. Additional bits of analysis  ----
#*************************

# Compute emissions and tangible assets by sector
sectoral_info <- function(data) {
  
  out <- data %>%
    group_by(sector) %>%
    summarise(
      CO2e1_sector = sum(CO2e1_Mt, na.rm = TRUE),
      CO2e1_share  = NA_character_,  # filled below (display only)
      
      Net_PPE_sector = sum(Net_PPE_mil, na.rm = TRUE),
      Net_PPE_share  = NA_character_,  # filled below (display only)
      
      obs = n(),
      # non_missing_CO2e1 = sum(!is.na(CO2e1)),
      # missing_CO2e1 = sum(is.na(CO2e1)),
      .groups = "drop"
    )
  
  total_emissions <- sum(out$CO2e1_sector, na.rm = TRUE)
  total_net_ppe   <- sum(out$Net_PPE_sector, na.rm = TRUE)
  
  out <- out %>%
    mutate(
      CO2e1_share   = sprintf("%.2f", CO2e1_sector / total_emissions),
      Net_PPE_share = sprintf("%.2f", Net_PPE_sector / total_net_ppe)
    ) %>%
    relocate(CO2e1_share, .after = CO2e1_sector) %>%
    relocate(Net_PPE_share, .after = Net_PPE_sector)
  
  totals_row <- out %>%
    summarise(
      sector = "Total",
      CO2e1_sector   = sum(CO2e1_sector, na.rm = TRUE),
      CO2e1_share    = sprintf("%.2f", 1),
      
      Net_PPE_sector = sum(Net_PPE_sector, na.rm = TRUE),
      Net_PPE_share  = sprintf("%.2f", 1),
      
      obs = sum(obs, na.rm = TRUE)
    )
  
  bind_rows(out, totals_row)
}

# Integral inversion for shifted exponential: y(x)=a*exp(b*x)+c
# Find x such that ∫_{x0}^{x} y(t) dt = P0
# Input can be either:
#  - numeric a,b,c
#  - a fitted object 'fit' with fit$model=="exp_shift" and fit$params named c("a","b","c")
x_from_integral <- function(P0,
                            a = NULL, b = NULL, c = NULL,
                            fit = NULL,
                            x0 = 0,
                            x_upper_init = 1,
                            max_expand = 60,
                            tol = 1e-10) {
  
  # --- get parameters either from (a,b,c) or from fit ---
  if (!is.null(fit)) {
    if (is.null(fit$model) || fit$model != "exp_shift") {
      stop("If 'fit' is provided, fit$model must be 'exp_shift'.")
    }
    if (is.null(fit$params) || !all(c("a","b","c") %in% names(fit$params))) {
      stop("If 'fit' is provided, fit$params must be a named vector containing a, b, c.")
    }
    a <- as.numeric(fit$params[["a"]])
    b <- as.numeric(fit$params[["b"]])
    c <- as.numeric(fit$params[["c"]])
  } else {
    if (any(vapply(list(a,b,c), is.null, logical(1)))) {
      stop("Provide either 'fit' OR all of a, b, c.")
    }
    a <- as.numeric(a); b <- as.numeric(b); c <- as.numeric(c)
  }
  
  stopifnot(is.finite(a), is.finite(b), is.finite(c), is.finite(P0), is.finite(x0))
  
  # --- closed-form definite integral A(x) = ∫_{x0}^{x} (a*exp(b*t) + c) dt ---
  area_fn <- function(x) {
    if (abs(b) < 1e-12) {
      return((a + c) * (x - x0))
    }
    (a / b) * (exp(b * x) - exp(b * x0)) + c * (x - x0)
  }
  
  target_fn <- function(x) area_fn(x) - P0
  
  # If already at target
  if (abs(target_fn(x0)) < tol) {
    return(list(x = x0, area = area_fn(x0), x0 = x0, P0 = P0, params = c(a = a, b = b, c = c)))
  }
  
  # Bracket root for uniroot
  lower <- x0
  upper <- x0 + x_upper_init
  
  f_lower <- target_fn(lower)
  f_upper <- target_fn(upper)
  
  expand_i <- 0
  while (is.finite(f_lower) && is.finite(f_upper) &&
         sign(f_lower) == sign(f_upper) && expand_i < max_expand) {
    step <- (upper - x0)
    if (!is.finite(step) || step <= 0) step <- 1
    upper <- x0 + 2 * step
    f_upper <- target_fn(upper)
    expand_i <- expand_i + 1
  }
  
  if (!is.finite(f_upper) || sign(f_lower) == sign(f_upper)) {
    stop("Failed to bracket a root. Try changing x0/x_upper_init, or check if the target is reachable for these parameters.")
  }
  
  sol <- uniroot(target_fn, lower = lower, upper = upper, tol = tol)$root
  list(x = sol, area = area_fn(sol), x0 = x0, P0 = P0, params = c(a = a, b = b, c = c))
}
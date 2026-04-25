# This script is called by the Stranding function analysis main script

#*************************
## 1. Extract and manipulate data ----
#*************************

get_plant_data <- function(base_path, file_name, sheet = 2) {
  
  # --- 1) Build file path ---
  file_path <- file.path(base_path, file_name)
  
  # --- 2) Read raw data ---
  df <- readxl::read_excel(file_path, sheet = sheet) %>%
    janitor::clean_names()
  
  # --- 3) Core variable construction ---
  df <- df %>%
    dplyr::mutate(
      start_year         = as.numeric(start_year),
      planned_retirement = as.numeric(planned_retirement),
      retired_year       = as.numeric(retired_year),
      
      plant_life_planned = planned_retirement - start_year,
      plant_life_actual  = retired_year - start_year
    )
  
  # --- 4) Sanity filters
  df <- df %>%
    dplyr::filter(
      is.na(plant_life_planned) | plant_life_planned >= 0,
      is.na(plant_life_actual)  | plant_life_actual  >= 0
    )
  
  return(df)
}


process_coal_gem_data <- function(data, LT, delta = 0.04, emis_floor = 0) {
  
  # -------------------- Initial Data Cleaning --------------------
  
  data <- data %>%
    rename(
      Country      = country_area,
      Annual_co2   = annual_co2_million_tonnes_annum,
      Plant_age    = plant_age_years,
      Capacity_MW  = capacity_mw
    ) %>%
    filter(
      status == "operating",
      Annual_co2 * 1e6 >= emis_floor  # emis_floor in tCO2; Annual_co2 in MtCO2
    ) %>%
    mutate(
      Annual_co2  = as.numeric(Annual_co2),
      Plant_age   = as.numeric(Plant_age),
      Capacity_MW = as.numeric(Capacity_MW)
    )
  
  # -------------------- Remaining Lifetime --------------------
  
  data <- data %>%
    mutate(
      computed_remaining_years = LT - Plant_age
    ) %>%
    filter(computed_remaining_years > 0)
  
  # -------------------- Overnight Cost of Capital --------------------
  
  # OCC values (USD/kW) from von Dulong (2023)
  OCC_coal_values <- c(
    "Japan" = 2419,
    "Korea" = 1151,
    "United States" = 2100,
    "China" = 800,
    "India" = 1200,
    "Brazil" = 2189,
    "Australia" = 3095,
    "EU" = 2000,
    "Other OECD countries" = 2153,
    "Other non-OECD countries" = 1396
  )
  
  data <- data %>%
    mutate(
      OCC = case_when(
        Country == "China" ~ OCC_coal_values["China"],
        Country == "India" ~ OCC_coal_values["India"],
        Country == "United States" ~ OCC_coal_values["United States"],
        Country == "Brazil" ~ OCC_coal_values["Brazil"],
        Country == "Japan" ~ OCC_coal_values["Japan"],
        Country == "Korea" ~ OCC_coal_values["Korea"],
        region == "Europe" ~ OCC_coal_values["EU"],
        region == "Oceania" ~ OCC_coal_values["Australia"],
        Country %in% c(
          "Chile", "Colombia", "Canada",
          "Israel", "Mexico", "Türkiye"
        ) ~ OCC_coal_values["Other OECD countries"],
        TRUE ~ OCC_coal_values["Other non-OECD countries"]
      )
    )
  
  data <- data %>%
    mutate(
      analysis_region = case_when(
        region == "Asia" & Country == "China" ~ "China",
        region == "Asia" & Country == "India" ~ "India",
        region == "Asia"                     ~ "Rest of Asia",
        TRUE                                 ~ region
      )
    )
  
  # -------------------- Asset Values --------------------
  
  data <- data %>%
    mutate(
      Gross_Value = OCC * (Capacity_MW / 1000),
      Net_Value   = pmax(Gross_Value * (1 - Plant_age / LT), 0)
    )
  
  # -------------------- Emissions, SCCE, FLEI --------------------
  
  data <- data %>%
    mutate(
      CO2e1Cum_age = Annual_co2 * computed_remaining_years,
      
      SCCE = ifelse(
        Annual_co2 > 0 & computed_remaining_years > 0,
        Gross_Value / (Annual_co2 * LT),
        NA_real_
      ), 
      
      # FLEI: forward-looking emissions per $ of net PPE over remaining life
      FLEI = ifelse(!is.na(Net_Value) &  Net_Value > 0, (Annual_co2*computed_remaining_years*delta) / Net_Value, NA_real_)
    )
  
    cols_to_drop <- c(
      "gem_unit_phase_id", "gem_location_id", "wiki_url", "start_year", "subregion", "region",
      "unit_name", "plant_name_other", "plant_name_local", "owner", "parent", "status", "emission_factor_kg_of_co2_per_tj", 
      "retired_year", "planned_retirement", "coal_source", "alternate_fuel", "location", 
      "local_area_taluk_county", "major_area_prefecture_district", "subnational_unit_province_state", 
      "location_accuracy", "permits", "permit_date", "permit_parsed", "Plant_age",
      "captive", "captive_industry_use", "capacity_factor", "captive_residential_use", "heat_rate_btu_per_k_wh", 
      "plant_life_planned", "plant_life_actual", "lifetime_co2_million_tonnes"
    )
    data <- data %>% select(-any_of(cols_to_drop))
    
    
    # Reorder and rename columns for presentation
    data <- data %>%
      mutate(`Plant type` = "coal") %>% 
      dplyr::select(
        `Plant type`,
        Country,
        analysis_region,
        OCC,
        plant_name,
        Capacity_MW,
        combustion_technology,
        coal_type,
        Annual_co2,
        computed_remaining_years,
        CO2e1Cum_age,
        Gross_Value,
        Net_Value,
        SCCE,
        FLEI,
        longitude,
        latitude
      ) %>%
      dplyr::rename(
        `Analysis Region` = analysis_region,
        Plant             = plant_name,
        `Capacity (MW)`   = Capacity_MW,
        `Combustion Technology` = combustion_technology,
        `Coal Type`       = coal_type,
        `Annual Emissions (MtCO2)` = Annual_co2,
        `Computed Remaining Lifetime (years)` = computed_remaining_years,
        `Computed Lifetime Emissions (MtCO2)` = CO2e1Cum_age
      )

  
  return(data)
}




process_gas_gem_data <- function(data, LT, delta = 0.04, LCEF, emis_floor = 0, base_path, file_name, sheet = 1) {
  
  # -------------------- Initial Data Cleaning --------------------
  
  data <- data %>%
    rename(
      Country      = country_area,
      Capacity_MW  = capacity_mw,
      Start_Year   = start_year
    ) %>%
    filter(
      status == "operating",
    ) %>%
    mutate(
      Start_Year  = as.numeric(Start_Year),
      Plant_age   = 2024 - Start_Year,
      Capacity_MW = as.numeric(Capacity_MW)
    )
  # -------------------- Adding Emission Estimates --------------------
  
  # Build file path
  cf_file_path <- file.path(base_path, file_name)
  
  # Read & clean CF data 
  
  cf_table <- read_excel(cf_file_path, sheet = sheet) %>%
    janitor::clean_names() %>%
    rename(
      Country = country_area,
      CF = capacity_factor_to_use
    ) %>%
    mutate(
      CF = as.numeric(gsub("%", "", CF)),
      CF = pmin(CF, 1)   # cap at 1
    ) %>%
    select(Country, CF)
  
  
  # Calculate the capacity factors
  # Annual_co2: Capacity_MW * 1000 kW/MW * CF * 8760 h/yr * LCEF tCO2/kWh / 1e6 t/Mt = MW*CF*8760*LCEF/1000 MtCO2/yr
  data <- data %>%
    left_join(cf_table, by = "Country") %>%
    mutate(
      Annual_co2 = Capacity_MW * CF * 8760 * LCEF / 1000
    ) %>%
    filter(Annual_co2 * 1e6 >= emis_floor)  # emis_floor in tCO2; Annual_co2 in MtCO2
  
  # -------------------- Filtering out oil Plants -------------
  
  data <- data %>%
    filter(
      # Keep rows that are not pure oil: exclude "fossil liquids" unless they also contain "fossil gas"
      !str_detect(fuel, "fossil liquids") | str_detect(fuel, "fossil gas")
    )
  
  # -------------------- Remaining Lifetime --------------------
  
  data <- data %>%
    mutate(
      computed_remaining_years = LT - Plant_age
    ) %>%
    filter(computed_remaining_years > 0)
  
  # -------------------- Overnight Cost of Capital --------------------
  
  # OCC values (USD/kW) from von Dulong (2023)
  OCC_gas_values <- c(
    "Japan" = 1109,
    "United States" = 1000,
    "China" = 560,
    "India" = 700,
    "Brazil" = 847,
    "Korea" = 973,
    "Mexico" = 601,
    "Australia" = 812,
    "EU" = 1000,
    "Other OECD countries" = 914,
    "Other non-OECD countries" = 702
  )
  
  data <- data %>%
    mutate(
      OCC = case_when(
        Country == "China" ~ OCC_gas_values["China"],
        Country == "India" ~ OCC_gas_values["India"],
        Country == "United States" ~ OCC_gas_values["United States"],
        Country == "Brazil" ~ OCC_gas_values["Brazil"],
        Country == "Japan" ~ OCC_gas_values["Japan"],
        Country == "Mexico" ~ OCC_gas_values["Mexico"],
        Country == "South Korea" ~ OCC_gas_values["Korea"],
        region == "Europe" ~ OCC_gas_values["EU"],
        region == "Oceania" ~ OCC_gas_values["Australia"],
        Country %in% c(
          "Chile", "Colombia", "Canada",
          "Israel", "Türkiye"
        ) ~ OCC_gas_values["Other OECD countries"],
        TRUE ~ OCC_gas_values["Other non-OECD countries"]
      )
    )
  
  data <- data %>%
    mutate(
      analysis_region = case_when(
        region == "Asia" & Country == "China" ~ "China",
        region == "Asia" & Country == "India" ~ "India",
        region == "Asia"                     ~ "Rest of Asia",
        TRUE                                 ~ region
      )
    )
  
  # -------------------- Asset Values --------------------
  
  data <- data %>%
    mutate(
      Gross_Value = OCC * (Capacity_MW / 1000),
      Net_Value   = pmax(Gross_Value * (1 - Plant_age / LT), 0)
    )
  
  # -------------------- Emissions, SCCE, FLEI --------------------
  
  data <- data %>%
    mutate(
      CO2e1Cum_age = Annual_co2 * computed_remaining_years,
      
      SCCE = ifelse(
        Annual_co2 > 0 & computed_remaining_years > 0,
        Gross_Value / (Annual_co2 * LT),
        NA_real_
      ), 
      
      # FLEI: forward-looking emissions per $ of net PPE over remaining life
      FLEI = ifelse(!is.na(Net_Value) &  Net_Value > 0, (Annual_co2*computed_remaining_years*delta) / Net_Value, NA_real_)
    )
  
  cols_to_drop <- c(
    "gem_unit_phase_id", "gem_location_id", "wiki_url", "start_year", "subregion", "region",
    "unit_name", "plant_name_other", "plant_name_local", "owner", "parent", "status", "emission_factor_kg_of_co2_per_tj", 
    "retired_year", "planned_retirement", "coal_source", "alternate_fuel", "location", 
    "local_area_taluk_county", "major_area_prefecture_district", "subnational_unit_province_state", 
    "location_accuracy", "permits", "permit_date", "permit_parsed", "Plant_age",
    "captive", "captive_industry_use", "capacity_factor", "captive_residential_use", "heat_rate_btu_per_k_wh", 
    "plant_life_planned", "plant_life_actual", "lifetime_co2_million_tonnes"
  )
  data <- data %>% select(-any_of(cols_to_drop))
  
  
  # Reorder and rename columns for presentation
  data <- data %>%
    mutate(`Plant type` = "gas") %>% 
    dplyr::select(
      `Plant type`,
      Country,
      analysis_region,
      OCC,
      plant_name,
      Capacity_MW,
      CF,
      turbine_engine_technology,
      fuel,
      Annual_co2,
      computed_remaining_years,
      CO2e1Cum_age,
      Gross_Value,
      Net_Value,
      SCCE,
      FLEI,
      longitude,
      latitude
    ) %>%
    dplyr::rename(
      `Analysis Region` = analysis_region,
      Plant             = plant_name,
      `Capacity (MW)`   = Capacity_MW,
      `Turbine Engine Technology` = turbine_engine_technology,
      `Fuel Type`       = fuel,
      `Annual Emissions (MtCO2)` = Annual_co2,
      `Computed Remaining Lifetime (years)` = computed_remaining_years,
      `Computed Lifetime Emissions (MtCO2)` = CO2e1Cum_age
    )
  
  
  return(data)
}

#*************************
## 2. Plotting functions ----
#*************************

# Shared colour palette for region × plant-type combinations used in all plant-level plots
make_region_palette <- function() {
  region_colors <- c(
    "oceania"      = "#17BECF",
    "africa"       = "#D62728",
    "americas"     = "#1F77B4",
    "europe"       = "#2CA02C",
    "rest of asia" = "#E5671A",
    "india"        = "#9467BD",
    "china"        = "#E6A817"
  )

  lighten <- function(col, amount = 0.55) {
    col <- col2rgb(col) / 255
    col <- col + (1 - col) * amount
    rgb(col[1], col[2], col[3])
  }

  regions <- names(region_colors)

  region_type_levels <- c(
    paste(regions, "gas",  sep = "_"),
    paste(regions, "coal", sep = "_")
  )

  cols <- unlist(lapply(regions, function(r) {
    v <- c(region_colors[[r]], lighten(region_colors[[r]]))
    names(v) <- paste(r, c("coal", "gas"), sep = "_")
    v
  }))
  cols <- cols[region_type_levels]

  labels <- setNames(
    c(
      sapply(regions, function(r) paste(tools::toTitleCase(gsub("_", " ", r)), "— Gas")),
      sapply(regions, function(r) paste(tools::toTitleCase(gsub("_", " ", r)), "— Coal"))
    ),
    region_type_levels
  )

  list(region_type_levels = region_type_levels, cols = cols, labels = labels)
}

plot_SCCE_plants <- function(data, winsor_level, yvar = "SCCE") {
  data <- data %>%
    arrange(SCCE) %>%
    mutate(
      x_start_age  = lag(cumsum(`Computed Lifetime Emissions (MtCO2)`) / 1e3, default = 0),
      x_end_age    = cumsum(`Computed Lifetime Emissions (MtCO2)`) / 1e3,
      x_center_age = 0.5 * (x_start_age + x_end_age)
    )
  
  # Winsorization bounds
  q_lower <- quantile(data[[yvar]], winsor_level, na.rm = TRUE)
  q_upper <- quantile(data[[yvar]], 1 - winsor_level, na.rm = TRUE)
  
  n_lower <- sum(data[[yvar]] < q_lower, na.rm = TRUE)
  n_upper <- sum(data[[yvar]] > q_upper, na.rm = TRUE)
  
  # Winsorize, normalise region/type labels, and compute region_type key
  data_winsorized <- data %>%
    mutate(
      !!sym(yvar) := pmin(pmax(!!sym(yvar), q_lower), q_upper)
    ) %>%
    mutate(
      `Plant type`      = tolower(trimws(as.character(`Plant type`))),
      `Plant type`      = ifelse(is.na(`Plant type`), "unknown", `Plant type`),
      `Analysis Region` = tolower(trimws(`Analysis Region`))
    ) %>%
    mutate(region_type = paste(`Analysis Region`, `Plant type`, sep = "_"))
  
  pal               <- make_region_palette()
  region_type_levels <- pal$region_type_levels
  cols              <- pal$cols
  labels            <- pal$labels

  data_winsorized$region_type <- factor(
    data_winsorized$region_type,
    levels = rev(region_type_levels)
  )
  
  # Plot "merit-order" rectangles
  p <- ggplot(data_winsorized, aes(
    xmin = x_start_age,
    xmax = x_end_age,
    ymin = 0,
    ymax = .data[[yvar]],
    fill = region_type
  )) +
    geom_rect(linewidth = 0.1) +
    scale_fill_manual(values = cols, labels = labels, drop = FALSE) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(
      x = "Embodied Emissions (GtCO2e)",
      y = paste(yvar, "(USD/tCO2e)"),
      title = paste0(
        "SCCE ",
        "(Data winsorized at ", winsor_level * 100, "%; ",
        "N° obs = ", nrow(data_winsorized), ")"
      )
    ) +
    theme_bw() +
    theme(
      axis.text  = element_text(size = 12),
      axis.title = element_text(size = 13),
      legend.text  = element_text(size = 20),
      plot.title   = element_text(size = 14)
    )
  # -------------------- Return --------------------
  
  list(
    plot = p,
    data = data_winsorized,
    q_lower = q_lower,
    q_upper = q_upper,
    winsor_lower = n_lower,
    winsor_upper = n_upper
  )
}

plot_FLEI_plants <- function(data,
                             yvar_scce       = "SCCE",
                             xvar_assets     = "Net_Value",    
                             delta           = 0.04,
                             LT              = 2*log(2)/delta,
                             winsor_level    = 0.05) {
  
  stopifnot(all(c(yvar_scce, xvar_assets, "Annual Emissions (MtCO2)", "Gross_Value") %in% names(data)))
  
  df <- data %>%
    dplyr::filter(!is.na(.data[[yvar_scce]]),
                  !is.na(.data[[xvar_assets]]), .data[[xvar_assets]] > 0,
                  !is.na(Gross_Value), Gross_Value > 0) %>%
    mutate(
      # FLEI = delta / SCCE; ×1e3 converts SCCE from USD/tCO2 to USD/kgCO2, giving FLEI in kgCO2/USD
      FLEI_raw           = delta * 1e3 / (.data[[yvar_scce]]),
      Net_Value_trillion = .data[[xvar_assets]] / 1e6
    )
  
  ql <- quantile(df$FLEI_raw, winsor_level,     na.rm = TRUE)
  qu <- quantile(df$FLEI_raw, 1 - winsor_level, na.rm = TRUE)
  
  n_l <- sum(df$FLEI_raw < ql, na.rm = TRUE)
  n_u <- sum(df$FLEI_raw > qu, na.rm = TRUE)
  
  # -------------------- CLEANING --------------------
  
  df <- df %>%
    mutate(
      `Analysis Region` = tolower(trimws(`Analysis Region`)),
      `Plant type`      = tolower(trimws(as.character(`Plant type`))),
      `Plant type`      = ifelse(is.na(`Plant type`), "unknown", `Plant type`)
    ) %>%
    mutate(region_type = paste(`Analysis Region`, `Plant type`, sep = "_"))
  
  # -------------------- PALETTE --------------------

  pal               <- make_region_palette()
  region_type_levels <- pal$region_type_levels
  cols              <- pal$cols
  labels            <- pal$labels
  
  # -------------------- WINSORISE & ORDER --------------------
  
  df_w <- df %>%
    arrange(desc(FLEI_raw)) %>%
    mutate(
      FLEI     = pmin(pmax(FLEI_raw, ql), qu),
      x_start  = dplyr::lag(cumsum(Net_Value_trillion), default = 0),
      x_end    = cumsum(Net_Value_trillion),
      x_center = 0.5 * (x_start + x_end),
      region_type = factor(region_type, levels = rev(region_type_levels))
    )
  
  # -------------------- PLOT --------------------
  
  p <- ggplot(df_w, aes(
    xmin = x_start, xmax = x_end,
    ymin = 0,        ymax = FLEI,
    fill = region_type
  )) +
    geom_rect(linewidth = 0.1) +
    scale_fill_manual(values = cols, labels = labels, drop = FALSE) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(
      x     = "Power plant value (Trillion USD)",
      y     = "FLEI (kgCO2e/USD)",
      title = paste0(
        "FLEI ",
        "(Data winsorized at ", winsor_level * 100, "%; ",
        "N° obs = ", nrow(df_w), ")"
      )
    ) +
    theme_bw() +
    theme(
      axis.text  = element_text(size = 12),
      axis.title = element_text(size = 13),
      legend.text  = element_text(size = 20),
      plot.title   = element_text(size = 14)
    )
  
  list(
    plot      = p,
    data      = df_w,
    data_raw  = df,
    q_lower   = ql,
    q_upper   = qu,
    winsor_l  = n_l,
    winsor_u  = n_u
  )
}


#*************************
## 3. Function fitting ----
#*************************

# --- helper: winsorize (optional) ---
winsorize_var <- function(data, var, level) {
  if (is.null(level)) return(data)
  ql <- quantile(data[[var]], level, na.rm = TRUE)
  qu <- quantile(data[[var]], 1 - level, na.rm = TRUE)
  data[[var]] <- pmin(pmax(data[[var]], ql), qu)
  data
}

# --- prediction ---
.predict_flei <- function(x, params, model = c("exp", "double_exp", "exp_shift")) {
  model <- match.arg(model)
  if (model == "exp") {
    a <- params[1]; b <- params[2]
    a * exp(b * x)
  } else if (model == "double_exp") {
    a <- params[1]; b <- params[2]; c <- params[3]; d <- params[4]
    a * exp(b * x) + c * exp(d * x)
  } else {
    # a * exp(b*x) + c
    a <- params[1]; b <- params[2]; c <- params[3]
    a * exp(b * x) + c
  }
}

# --- objective (SSE) with optional low-end penalty ---
.objective_flei <- function(params, x, y,
                            model = c("exp", "double_exp", "exp_shift"),
                            penalty_floor = NULL,
                            penalty_x = NULL,
                            penalty_weight = 0,
                            weights = NULL) {
  model <- match.arg(model)
  yhat <- .predict_flei(x, params, model)
  
  if (is.null(weights)) {
    sse <- sum((y - yhat)^2)
  } else {
    sse <- sum(weights * (y - yhat)^2)
  }
  
  if (!is.null(penalty_floor) && penalty_weight > 0) {
    x0 <- if (is.null(penalty_x)) min(x, na.rm = TRUE) else penalty_x
    y0 <- .predict_flei(x0, params, model)
    viol <- max(0, penalty_floor - y0)
    sse <- sse + penalty_weight * viol^2
  }
  sse
}

# --- main fitter (DEoptim wrapper) ---
fit_flei_deoptim <- function(
    data,
    xvar = "x_center",
    yvar = "FLEI",
    model = c("exp", "double_exp", "exp_shift"),
    winsor_level = NULL,              # usually NULL because we already winsorized in plot_FLEI
    fixed_c = NULL,                   # for exp_shift only (fix asymptote)
    x_fit_max = NULL,                 # fit only for x <= x_fit_max (e.g. 5)
    penalty_floor = NULL,
    penalty_floor_quantile = NULL,    # e.g. 0.02
    penalty_weight = 0,
    bounds = NULL,
    de_control = DEoptim.control(itermax = 200000, NP = 60,
                                 trace = FALSE, reltol = 1e-9, steptol = 2000)
) {
  model <- match.arg(model)
  stopifnot(all(c(xvar, yvar) %in% names(data)))
  
  df <- data %>%
    dplyr::filter(!is.na(.data[[xvar]]), !is.na(.data[[yvar]]))
  
  df <- winsorize_var(df, yvar, winsor_level)
  
  x <- df[[xvar]]
  y <- df[[yvar]]
  
  # restrict fitting domain if requested
  fit_mask <- if (!is.null(x_fit_max)) x <= x_fit_max else rep(TRUE, length(x))
  x_fit <- x[fit_mask]
  y_fit <- y[fit_mask]
  
  # optional smooth weights (turn on if desired)
  w_alpha <- 0
  w_tau   <- 1
  weight_fun <- function(xx, alpha = w_alpha, tau = w_tau) 1 + alpha * exp(-xx / tau)
  w_fit <- weight_fun(x_fit)
  
  ymin <- min(y_fit, na.rm = TRUE)
  ymax <- max(y_fit, na.rm = TRUE)
  
  # default bounds
  if (is.null(bounds)) {
    if (model == "exp") {
      bounds <- list(lower = c(0, -10), upper = c(2*ymax, 10))
    } else if (model == "double_exp") {
      bounds <- list(lower = c(0, -10, 0, -10), upper = c(2*ymax, 10, 2*ymax, 10))
    } else { # exp_shift
      if (is.null(fixed_c)) {
        bounds <- list(lower = c(0, -10, 0), upper = c(2*ymax, 10, ymax))
      } else {
        bounds <- list(lower = c(0, -10, fixed_c), upper = c(2*ymax, 10, fixed_c))
      }
    }
  } else {
    if (model == "exp_shift" && !is.null(fixed_c)) {
      bounds$lower[3] <- fixed_c
      bounds$upper[3] <- fixed_c
    }
  }
  
  # penalty floor (fixed or from quantile)
  if (!is.null(penalty_floor_quantile)) {
    penalty_floor <- as.numeric(quantile(y_fit, probs = penalty_floor_quantile, na.rm = TRUE))
  }
  if (!is.null(penalty_floor) && penalty_weight <= 0) {
    stop("If you set a penalty floor, please provide a positive penalty_weight.")
  }
  
  obj <- function(par) .objective_flei(
    par, x_fit, y_fit, model = model,
    penalty_floor = penalty_floor,
    penalty_x = NULL,
    penalty_weight = penalty_weight,
    weights = w_fit
  )
  
  de_out <- DEoptim(fn = obj, lower = bounds$lower, upper = bounds$upper, control = de_control)
  best <- de_out$optim$bestmem
  
  sse <- .objective_flei(best, x_fit, y_fit, model,
                         penalty_floor = penalty_floor,
                         penalty_x = NULL,
                         penalty_weight = penalty_weight,
                         weights = w_fit)
  
  n <- length(y_fit)
  k <- length(best)
  if (model == "exp_shift" && !is.null(fixed_c)) k <- k - 1
  
  aic <- n * log(sse / n) + 2 * k
  bic <- n * log(sse / n) + log(n) * k
  
  # prediction curve
  x_seq <- seq(min(x, na.rm = TRUE), max(x, na.rm = TRUE), length.out = 500)
  y_seq <- .predict_flei(x_seq, best, model)
  pred_df <- data.frame(x = x_seq, y = y_seq)
  
  # plot
  title_txt <- paste0(
    dplyr::case_when(
      model == "exp"        ~ "Exponential",
      model == "double_exp" ~ "Double Exponential",
      model == "exp_shift"  ~ "Shifted Exponential"
    ),
    " Fit — Plant-level FLEI"
  )
  
  p <- ggplot(df, aes(x = .data[[xvar]], y = .data[[yvar]])) +
    geom_point(alpha = 0.35) +
    geom_line(data = pred_df, aes(x = x, y = y), linewidth = 1) +
    labs(
      title = title_txt,
      x = "Power plant value (Trillion USD)",
      y = "FLEI (kgCO2/USD)"
    ) +
    theme_bw()
  
  list(
    model = model,
    params = setNames(
      as.numeric(best),
      if (model == "exp") c("a","b") else if (model == "double_exp") c("a","b","c","d") else c("a","b","c")
    ),
    sse = sse,
    aic = aic,
    bic = bic,
    data_used = df,
    pred_df = pred_df,
    plot = p,
    de_output = de_out,
    bounds = bounds
  )
}

## 4. Additional bits of analysis  ----
#*************************

plot_age_distribution <- function(data, cols, bins = 35) {
  
  plot_list <- list()
  
  for (col in cols) {
    
    n_included <- sum(is.finite(data[[col]]))
    
    plot_list[[col]] <- ggplot(
      data %>% dplyr::filter(is.finite(.data[[col]])),
      aes(x = .data[[col]])
    ) +
      geom_histogram(bins = bins) +
      labs(
        x = "Plant age (years)",
        y = "Count",
        title = paste0("Age distribution (N=", n_included, ")")
      ) +
      theme_bw()
  }
  
  return(plot_list)
}

plant_summary <- function(data, group_var) {
  
  out <- data %>%
    group_by({{ group_var }}) %>%
    summarise(
      CO2e1_total = sum(`Annual Emissions (MtCO2)`, na.rm = TRUE),
      CO2e1_share = NA_character_,
      obs = n(),
      .groups = "drop"
    )
  
  total_emissions <- sum(out$CO2e1_total, na.rm = TRUE)
  
  out <- out %>%
    mutate(
      CO2e1_share = sprintf("%.2f", CO2e1_total / total_emissions)
    ) %>%
    relocate(CO2e1_share, .after = CO2e1_total)
  
  totals_row <- out %>%
    summarise(
      !!rlang::as_name(rlang::ensym(group_var)) := "Total",
      CO2e1_total = sum(CO2e1_total, na.rm = TRUE),
      CO2e1_share = sprintf("%.2f", 1),
      obs = sum(obs, na.rm = TRUE)
    )
  
  bind_rows(out, totals_row)
}








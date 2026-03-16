This folder contains the analysis of capital emission intensity used to calibrate key parameters and initial values in "Optimal climate policy transition as if the transition matters" (E. Campiglio, S. Dietz, F. Venmans)

Last updated: 13 March 2026

Main folder structure
- Stranding_cost_function.Rproj: main RStudio project file.
- Company-level_analysis: folder with company-level data and associated scripts
- Plant-level_analysis: folder with plant-level data and associated scripts

Within the Company-level_analysis folder:
- Stranding_function_company-level.R is the main script performing the analysis
- Stranding_company-level_FUNCTIONS.R contains all the necessary functions (the file is sourced by Stranding_function_company-level.R)
- Data_Sector is the folder containing all the raw data

Within the Plant-level_analysis folder:
- Stranding_function_plant-level.R is the main script performing the analysis
- Stranding_plant-level_FUNCTIONS.R contains all the necessary functions (the file is sourced by Stranding_function_plant-level.R)
- GEM-Global-Coal-Plant-Tracker-July-2024 is the file containing all the raw data

How to run
- Open Stranding_cost_function.Rproj (recommended).
- Open and run the desired main script from the Files tab within your IDE (e.g. RStudio):
	
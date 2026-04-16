This folder contains the analysis of capital emission intensity used to calibrate key parameters and initial values in "Optimal climate policy transition as if the transition matters" (E. Campiglio, S. Dietz, F. Venmans)

Last updated: 16 April 2026

Main folder structure
- Stranding_cost_function.Rproj: main RStudio project file.
- Company-level_analysis: folder with company-level data and associated scripts
- Plant-level_analysis: folder with plant-level data and associated scripts
- Results: folder containing all output files (Excel results and charts) produced by the analysis scripts

Within the Company-level_analysis folder:
- Stranding_function_company-level.R is the main script performing the analysis
- Stranding_company-level_FUNCTIONS.R contains all the necessary functions (the file is sourced by Stranding_function_company-level.R)
- Data_Sector is the folder containing all the raw data

Within the Plant-level_analysis folder:
- Stranding_function_plant-level.R is the main script performing the analysis
- Stranding_plant-level_FUNCTIONS.R contains all the necessary functions (the file is sourced by Stranding_function_plant-level.R)
- GEM-Global-Coal-Plant-Tracker-July-2024.xlsx contains the raw data on coal power plants (source: Global Energy Monitor)
- Global-Oil-and-Gas-Plant-Tracker-GOGPT-January-2026.xlsx contains the raw data on gas power plants (source: Global Energy Monitor)
- Capacity_factors_GEM_wiki.xlsx contains capacity factor estimates by country used to compute annual emissions for gas plants

Within the Results folder:
- Results_company.xlsx: output of the company-level analysis (FLEI and SCCE data)
- Results_plant.xlsx: output of the plant-level analysis (FLEI, SCCE, and asset data for coal and gas plants)
- FLEI-SCCE_stacked_plot.png: stacked FLEI/SCCE chart from the company-level analysis
- FLEI-SCCE_plant_stacked_plot.png: stacked FLEI/SCCE chart from the plant-level analysis

How to run
- Open Stranding_cost_function.Rproj (recommended).
- Open and run the desired main script from the Files tab within your IDE (e.g. RStudio)

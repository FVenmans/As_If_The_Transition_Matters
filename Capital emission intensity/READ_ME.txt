This folder replicates the company-level analysis of capital emission intensity used to calibrate key parameters and initial values in our model.

Folder structure
	•	Data_Sector/ contains the sector-specific Excel files (NACE_<sector>_TA.xlsx) used as input.
	•	Capital_emission_intensity.R is the main script.
	•	Capital_emission_intensity_FUNCTION.R contains the auxiliary functions and is sourced automatically by the main script.
	•	Capital_emission_intensity_PROJECT.Rproj is the RStudio project file (optional but convenient).

How to run
	1.	Open Capital_emission_intensity_PROJECT.Rproj (recommended).
	2.	Open and run Capital_emission_intensity.R.

The script:
	•	installs any missing R packages (if needed),
	•	loads sector-level input data from Data_Sector/,
	•	constructs the empirical emission-intensity curve,
	•	fits a shifted exponential function to the curve.

You do not need to open Capital_emission_intensity_FUNCTION.R unless you want to inspect or modify the helper functions.

Contact

If you spot inconsistencies or have questions, please contact: emanuele.campiglio@unibo.it
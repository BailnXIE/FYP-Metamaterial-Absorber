# FYP-Metamaterial-Absorber
Machine-learning-assisted inverse design for broadband plasmonic metasurface absorbers.

# AbsorberNetEnhanced: Machine-Learning-Assisted Metamaterial Absorber

This repository contains the source code and simulation scripts for my Final Year Project (FYP) at the City University of Hong Kong (Department of Physics). 

## Project Overview
This project proposes an efficient, broadband plasmonic metasurface absorber based on manganese (Mn). To overcome the massive computational cost of traditional numerical inverse design, a custom deep learning framework was implemented to map metasurface geometric parameters to their corresponding absorption spectra.

> ⚠️ **Important Note on File Paths:** > Before running the code, please remember to customize the file directories in the scripts. You will need to update the dataset input paths and model save paths (especially in LD scripts for the materials definition) to match your local environment.

## Repository Structure
* `ring_regression_study.py` : Defines the *AbsorberNetEnhanced* architecture (incorporating Residual Blocks and SiLU activation) to train the model from data collected in `ring_data.csv`
  
* `Simulation/` : Contains the MEEP/FDTD scripts used to generate the absorption spectra.
* `ring.py` : Original Simulation of rings-based pattern. Remember to change the path for saving data in the code before staring simulation.
* `ring_fuc.py` : A simplified programme for data generation in `ring_data.csv`. It includes a function for simulating with random parameters of geometry such as radii of rings and thickness of layers.
* `ring_auto.py` : Automatically running the `ring_fuc.py` in order for collecting various training data. There are about 800 spectrum data pre-generated in `ring_data.csv`.
  
* `materials/` : Contains complex refractive index and the Lorentz-Drude programme of various materials in `material_data/` and `material definition/`
* `Lorentz_Drude.py` : An example from MEEP library which can produce the frequency, gamma, and sigma parameters used for "E_susceptibilities" material definition. Refractive csv data can be found from https://refractiveindex.info.

## Key Technologies Used
* **Electromagnetic Simulation:** MEEP (FDTD method). Note that MEEP is not available via standard pip and must be installed via Conda. Please follow the official MEEP installation guide. https://meep.readthedocs.io/en/latest/Installation/
* **Deep Learning:** PyTorch, Scikit-Learn, Pandas, NumPy
* **Visualization:** Matplotlib

* ## Installation & Setup
Before running the code in this repository, you are suggest to install the required Python packages.

If you run into any execution issues, it is highly recommended to use AI-assisted coding tools (such as **[Cursor](https://cursor.sh/)**, ChatGPT, or GitHub Copilot) to help debug and adapt the code to your specific operating system. 

## Author
* **XIE Bailin** - *BSc in Physics, City University of Hong Kong*

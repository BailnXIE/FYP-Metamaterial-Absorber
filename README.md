# FYP-Metamaterial-Absorber
Machine-learning-assisted inverse design for broadband plasmonic metasurface absorbers.

# AbsorberNetEnhanced: Machine-Learning-Assisted Metamaterial Absorber

This repository contains the source code and simulation scripts for my Final Year Project (FYP) at the City University of Hong Kong (Department of Physics). 

## Project Overview
This project proposes an efficient, broadband plasmonic metasurface absorber based on manganese (Mn). To overcome the massive computational cost of traditional numerical inverse design, a custom deep learning framework was implemented to map metasurface geometric parameters to their corresponding absorption spectra.

## Repository Structure
* `ring_regression_study.py` : Defines the *AbsorberNetEnhanced* architecture (incorporating Residual Blocks and SiLU activation) to train the model from data collected in `ring_data.csv`
* 
* `Simulation/` : Contains the MEEP/FDTD scripts used to generate the absorption spectra.
* `ring.py` : Original Simulation of rings-based pattern. Remember to change the path for saving data in the code beforing staring simulation.
* `ring_fuc.py` : A simplified programme for data generation in `ring_data.csv`. It includes a function for simulating with random parameters of geometry such as radii of rings and thickness of layers.
* `ring_auto.py` : Automatically running the `ring_fuc.py` in order for collecting various trainning data. There are about 800 spectrum data pre-generated in `ring_data.csv`.
* 
* `materials/` : Contains complex refractive index and the Lorentz-Drude programme of various materials in `material_data/` and `material definition/`
* `Lorentz_Drude.py` : An example from MEEP library which can produce the frequency, gamma, and sigma parameters used for "E_susceptibilities" material definition. Refractive csv data can be found from https://refractiveindex.info.

## Key Technologies Used
* **Electromagnetic Simulation:** MEEP (FDTD method)
* **Deep Learning:** PyTorch
* **Data Processing:** NumPy, SciPy, Matplotlib

## Author
* **XIE Bailin** - *BSc in Physics, City University of Hong Kong*

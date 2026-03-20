"""
Meta Absorber simulation
based on Meep FDTD 
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import csv



# all the data and images saved in the same file
OUTPUT_DIR = "/Volumes/tremendous/FYP_data/rings" #Create a path for data saving

# =============================================================================
# Material Definition
# =============================================================================

nickel = mp.Medium(
    epsilon = 1.1,
    E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency = 3.9858250867132115,gamma = 2.511879899524022,sigma = 4.113838124625053),
        mp.LorentzianSusceptibility(frequency = 1.205259585418557,gamma = 2.1791021647942617,sigma = 33.8282537957956),
        mp.LorentzianSusceptibility(frequency = 5.473956202134901,gamma = 0.0,sigma = 0.6422654532224172),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 71.40722029347084,sigma = 4.2029657093411785),
        mp.LorentzianSusceptibility(frequency = 0.5391704583285774,gamma = 0.5710356956819119,sigma = 34.3101795523754),
        mp.LorentzianSusceptibility(frequency = 0.5162622607242847,gamma = 3.1517824141461036e-07, sigma = 5.087204485682729e-06),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 26.927018244146122),
    ],
)
SiO2 = mp.Medium(
    epsilon = 1.4734,
    E_susceptibilities=[
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 0.00025388935570505225),
        mp.LorentzianSusceptibility(frequency = 35.389765360298895,gamma = 7.127379294672169,sigma = 0.0),
        mp.LorentzianSusceptibility(frequency = 3.4770163700805936,gamma = 0.6710934044071608,sigma = 0.0004169766288133064),
        mp.LorentzianSusceptibility(frequency = 9.066558637023224,gamma = 0.0,sigma = 0.26650430221274624),
        mp.LorentzianSusceptibility(frequency = 9.06655863618851,gamma = 0.0,sigma = 0.4053568509089618),
    ]
)
MgF2 = mp.Medium(
    epsilon = 1.37,
    E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency = 41.55856434224302,gamma = 7.691238700911039,sigma = 0.0),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 0.004208991233482833),
        mp.LorentzianSusceptibility(frequency = 10.832666523360178,gamma = 0.0,sigma = 0.5164394025760769),
    ]
)
Al2O3 = mp.Medium(
    epsilon = 1.7449,
    E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency = 8.167622282458941,gamma = 0.009263608667566028,sigma = 0.8736049314588427),
        mp.LorentzianSusceptibility(frequency = 6.5801513245579955,gamma = 0.2377410025590789,sigma = 0.12804299304451808),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 3.9131293709833876,sigma = 0.0),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.004054404919851729,sigma = 0.018237564337179407),
        mp.LorentzianSusceptibility(frequency = 0.664460637582893,gamma = 0.38387488221702004,sigma = 0.0007773456682168302),
    ]
)
Pmma = mp.Medium(
    epsilon = 1.5021,
    E_susceptibilities=[
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 46.18302786117195,sigma = 0.0),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.004054404919851729,sigma = 0.0011328895244292689),
        mp.LorentzianSusceptibility(frequency = 7.352016911526541,gamma = 0.0,sigma = 0.6815670696049281),
    ]
)
LiF = mp.Medium(
    epsilon = 1.3969,
    E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency = 10.694242198779508,gamma = 0.0,sigma = 0.529716980032111),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 0.006664144278975116),
    ]
)
Si3N4 = mp.Medium(
    epsilon = 1.9966,
    E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency = 6.2329477642006355,gamma = 0.0,sigma = 2.030576371281131),
        mp.LorentzianSusceptibility(frequency = 3.2240786049669308,gamma = 0.0,sigma = 0.0000351468774624583),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 0.026383896884338384),
    ]
)
TiO2 = mp.Medium(
    epsilon = 2.41,
    E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency = 3.303769007745288,gamma = 0.1501319572468476,sigma = 0.16846281714670672),
        mp.LorentzianSusceptibility(frequency = 4.82749607708675,gamma = 0.4552927145342035,sigma = 0.27374257655317685),
        mp.LorentzianSusceptibility(frequency = 3.660289731242061,gamma = 0.21803033378394365,sigma = 0.252453521100252),
        mp.LorentzianSusceptibility(frequency = 3.4590119437461344,gamma = 6.266116336030089,sigma = 0.0),
        mp.LorentzianSusceptibility(frequency = 3.086333984769129,gamma = 0.12990489985651718,sigma = 0.07299566958232107),
        mp.LorentzianSusceptibility(frequency = 4.219127997277008,gamma = 0.37736668160457437,sigma = 0.2841766572881675),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 12.792490039214346,sigma = 0.0),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 0.499741642428046),
        mp.LorentzianSusceptibility(frequency = 3.198170421900659,gamma = 0.13883396238248982,sigma = 0.1248076945591825),
        mp.LorentzianSusceptibility(frequency = 3.8133038954899363,gamma = 0.2579228373743328,sigma = 0.2673522103903398),
        mp.LorentzianSusceptibility(frequency = 3.411910208370688,gamma = 0.16608522790547706,sigma = 0.2041103215399395),
        mp.LorentzianSusceptibility(frequency = 4.493565351323377,gamma = 0.4506986591203289,sigma = 0.28608681572439376),
        mp.LorentzianSusceptibility(frequency = 3.528763233502379,gamma = 0.1881537574493126,sigma = 0.2317897659561513),
        mp.LorentzianSusceptibility(frequency = 6.08760422435181,gamma = 0.0,sigma = 0.8884870988430073),
        mp.LorentzianSusceptibility(frequency = 3.9573419713422435,gamma = 2.151508030659981,sigma = 0.0),
        mp.LorentzianSusceptibility(frequency = 3.9961413329024413,gamma = 0.3104495637114701,sigma = 0.27763525114649484),
    ]
)
GaN = mp.Medium(
    epsilon = 2.3392,
    E_susceptibilities=[
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 0.01359390613237773),
        mp.LorentzianSusceptibility(frequency = 3.8842985569822135,gamma = 0.0,sigma = 1.688993509439387),
        mp.LorentzianSusceptibility(frequency = 22.196914265900865,gamma = 0.0,sigma = 1.3234783444254632),
    ]
)
GaF2 = mp.Medium(
    epsilon = 1.4292,
    E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency = 10.092758238836243,gamma = 0.0,sigma = 0.6102801449358649),
        mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 0.003443934059618324),
    ]
)

# =============================================================================
# Customized Parameter
# =============================================================================

Nickel = nickel
Air = mp.Medium(index=1.0)
pec = mp.perfect_electric_conductor
substrate_material = SiO2
ground_material = Nickel
pattern_material = ground_material
inter_pattern_material = Air
resolution = 181  # pixels/μm

# source intensity
source_amplitude = 10.0  # source amplitude

# Monitors position
monitor_reflection_x = None  # location of reflectance monitor, "None" = calculating automatically
monitor_transmission_x = None  # location of transmittion monitor
monitor_offset = 0.3  # offset distance from monitors to PML

# range of wavelength
min_wavelength = 0.3  # minimum wavelength (μm)
max_wavelength = 2.0  # maximum wavelength (μm)
n_wavelengths = 200  # point number of wavelength

t_r=0.092 # pattern's height
t_s=0.091 # spacer's height
t_g=0.194 # ground's height

radii_group = [0.140,0.137,0.096,0.086,0.067,0.057,0.031,0.023,0.020]
period=0.298
# boundary 
period_y = period  # period in y direction (μm)
period_z = period_y  # period in z direction (μm)

# geometric height
Height_geometry = t_r + t_s + t_g
pattern_height = t_r  
substrate_thickness = t_s 

# PML 厚度
dpml = 0.6*max_wavelength  # PML thickness (μm)
dpad = 0.8 * max_wavelength   # packing layer thickness (μm)
dgap = Height_geometry * 1.1 

# =============================================================================
# Calculation of Domain Size
# =============================================================================

# Total length in x direction：PML + 填充 + 幾何 + 填充 + PML
sx = 2 * (dpml + dpad + dgap)
sy = period_y  # unit cell size in y direction
sz = period_z  # unit cell size in zdirection

cell_size = mp.Vector3(sx, sy, sz)

# =============================================================================
# functions for geometry definition
# =============================================================================
def base_geometry(spacer_hight, ground_height, pattern_height):
    # Add ground and spacer at the beginning
    objs = []
    objs.append(
        mp.Block(
            material=substrate_material,
            size=mp.Vector3(spacer_hight, sy, sz),
            center=mp.Vector3(-spacer_hight/2 - pattern_height/2, 0, 0),
        )
    )
    objs.append(
        mp.Block(
            material=ground_material,
            size=mp.Vector3(ground_height, sy, sz),
            center=mp.Vector3(-ground_height/2 - spacer_hight - pattern_height/2, 0, 0),
        )
    )
    return objs

def disk_geometry(radius, material, pattern_height):
    # Create Disk geometry. Ring geometry can be generated by two disks with different radii and materials.
    objs = []
    objs.append(
        mp.Cylinder(
            radius=radius,
            height=pattern_height,
            axis=mp.Vector3(1, 0, 0),
            center=mp.Vector3(0, 0, 0),
            material=material,
        )
    )
    return objs

def create_geometry():
    # Create spacer, ground, and pattern geometries
    geometry = []
    geometry.extend(base_geometry(t_s, t_g, t_r))
    # Rings pattern created by disks with radii from large to small and alternated with Nickel/Air
    radii = sorted([round(float(r), 3) for r in radii_group], reverse=True)
    for i, radius in enumerate(radii):
        material = Nickel if i % 2 == 0 else inter_pattern_material
        geometry.extend(disk_geometry(radius, material, pattern_height))
    return geometry


# =============================================================================
# Setting of Light Source
# =============================================================================

# Light source position: in the middle of the filling layer
source_x = (dgap) + (0.75 * dpad)
source_position = mp.Vector3(source_x, 0, 0)

# Central wavelength and frequency
lcen = (min_wavelength + max_wavelength) / 2
fcen = 1.0 / lcen
df = 2.0 * (1.0 / min_wavelength - 1.0 / max_wavelength)

# Gaussian pulse light source
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        component=mp.Ey,  # Ey polarization
        center=source_position,
        size=mp.Vector3(0, sy, sz),  # plane source, covers the whole Y-Z plane
        amplitude=source_amplitude,  # intensity of the source
    )
]

# =============================================================================
# Setting of boundary condition
# =============================================================================

# PML setting (x direction only, periodic boundary in y-z direction)
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

# periodic boundary condition
k_point = mp.Vector3(0, 0, 0)

# =============================================================================
# Creating Simulation objects
# =============================================================================

geometry = create_geometry()

# calculation monitor position 
if monitor_reflection_x is None:
    monitor_reflection_x = (dgap) + (0.25 * dpad)
if monitor_transmission_x is None:
    monitor_transmission_x = -(dgap) - (0.25 * dpad)

# Wavelength and frequency aries
wavelengths = np.linspace(min_wavelength, max_wavelength, n_wavelengths)
frequencies = 1.0 / wavelengths

# =============================================================================
# Reference simulation: unstructured
# =============================================================================

print("Conducting reference simulation (unstructured) to obtain the incident intensity...")

# Creating reference simulation (unstructured)
sim_ref = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=[],  # unstructured geometry
    k_point=k_point,
    sources=sources,
    default_material=mp.Medium(index=1.0),  # default material is air
    # eps_averaging=True,
)

# Adding a reflectance monitor (used to obtain the incidence intensity)
flux_ref_reflection = sim_ref.add_flux(
    frequencies,
    mp.FluxRegion(
        center=mp.Vector3(monitor_reflection_x, 0, 0), 
        size=mp.Vector3(0, sy, sz),
    )
)

# Running reference simulation
sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ey, mp.Vector3(0, 0, 0), 1e-4
))
# run_time = sx + 50 
# sim_ref.run(until=run_time)

# obtain the incidence intensity
incident_refl_field_data = sim_ref.get_flux_data(flux_ref_reflection)
incident_intensity = np.array(mp.get_fluxes(flux_ref_reflection))
print(f"Reference range of incident intensity: {incident_intensity.min():.6e} - {incident_intensity.max():.6e}")

# Reset the simulation
sim_ref.reset_meep()

# =============================================================================
# main simulation (structured)
# =============================================================================

print("\nrunning main simulation (structured)...")

# Creating main simulation (structured)
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    k_point=k_point,
    sources=sources,
    default_material=mp.Medium(index=1.0),  
    eps_averaging=True,
    Courant = 0.2,
)

# =============================================================================
# Adding Flux Monitors
# =============================================================================

print("Setting Flux Monitors...")

# Reflectance monitor 
flux_reflection = sim.add_flux(
    frequencies,
    mp.FluxRegion(
        center=mp.Vector3(monitor_reflection_x, 0, 0),
        size=mp.Vector3(0, sy, sz),  # plane monitor
    )
)

# Transmission monitor
flux_transmission = sim.add_flux(
    frequencies,
    mp.FluxRegion(
        center=mp.Vector3(monitor_transmission_x, 0, 0),
        size=mp.Vector3(0, sy, sz),  # plane monitor
    )
)

# =============================================================================
# Adding Mode Monitor for S-parameter
# =============================================================================
# Mode monitor for calculation S11 (reflectance)
mode_monitor_reflection = sim.add_mode_monitor(
    frequencies,
    mp.ModeRegion(
        center=mp.Vector3(monitor_reflection_x, 0, 0),
        size=mp.Vector3(0, sy, sz),
        direction=mp.X,
    ),
)

# Mode monitor for calculation S21 (Transmittance)
mode_monitor_transmission = sim.add_mode_monitor(
    frequencies,
    mp.ModeRegion(
        center=mp.Vector3(monitor_transmission_x, 0, 0),
        size=mp.Vector3(0, sy, sz),
        direction=mp.X,
    ),
)

# =============================================================================
# Adding DFT field monitor for field distribution
# =============================================================================

dft_freqs = [
    1.0 / min_wavelength,
    fcen,
    1.0 / max_wavelength,
]

dft_fields = sim.add_dft_fields(
    [mp.Ey],  # obtain Ey
    dft_freqs,
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(sx, sy, sz),  
)

print(f"Monitor setting is completed:")
print(f"- Reflectanve monitor: x={monitor_reflection_x:.3f} μm")
print(f"- Transmittance monitor: x={monitor_transmission_x:.3f} μm")
print(f"- Monitoring frequencies: {len(frequencies)} frequency points")
print(f"- Using incident intensity from reference simulation for normalization")

# =============================================================================
# Runnig Simulation
# =============================================================================

print("\nStart running the simulation...")
print("=" * 50)

# Before running the simulation, prompt the reflection monitor needs to subtract the incident data.
sim.load_minus_flux_data(flux_reflection, incident_refl_field_data)

# Run the simulation until the field decays
sim.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ey, mp.Vector3(0, 0, 0), 1e-4
))

print("Simulation is done!")

# =============================================================================
# Obtain Flux data
# =============================================================================

print("\nResult Analysing...")

# Obtain Flux data
reflection_flux = np.array(mp.get_fluxes(flux_reflection))
transmission_flux = np.array(mp.get_fluxes(flux_transmission))

# Using incident intensity from reference simulation for normalization
# Calculating TRA
T = np.abs(transmission_flux / incident_intensity)  # Reflectance rate
R = np.abs(reflection_flux / incident_intensity)  # Transmittance rate
A = 1 - R - T  # Absorption rate

# making sure the A at rational range
A = np.clip(A, 0, 1)

# =============================================================================
# determine the absorption peak
# =============================================================================

# find the local maximum
if len(A) > 2:
    candidate_idx = np.where((A[1:-1] > A[:-2]) & (A[1:-1] > A[2:]))[0] + 1
    if candidate_idx.size > 0:
        # peak
        peak_idx = candidate_idx[np.argmax(A[candidate_idx])]
    else:
        peak_idx = np.argmax(A)
else:
    peak_idx = np.argmax(A)

peak_wavelength = wavelengths[peak_idx]
peak_absorption = A[peak_idx]

print(f"\nabsorption peak:")
print(f"- wavelength: {peak_wavelength:.4f} μm")
print(f"- Absorption: {peak_absorption*100:.2f}%")

# =============================================================================
# drawing TRA cure 
# =============================================================================

plt.figure(figsize=(10, 6))
plt.plot(wavelengths, T, label="Transmission", color="green", linewidth=2)
plt.plot(wavelengths, R, label="Reflection", color="blue", linewidth=2)
plt.plot(wavelengths, A, label="Absorption", color="red", linewidth=2)

peak_label = f"Max Peak: {peak_wavelength:.3f} μm ({peak_absorption*100:.1f}%)"
plt.scatter(peak_wavelength, peak_absorption, color="black", s=100, zorder=5, label=peak_label)

plt.legend(
    loc="best", 
    fontsize=11, 
    frameon=True, 
    fancybox=True, 
    framealpha=0.9, 
    edgecolor="gray",
    # title="Simulation Results" # optional
)
plt.xlabel("Wavelength (μm)", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("TRA vs Wavelength (Meta Absorber)", fontsize=14, fontweight="bold")
plt.legend(loc="best", fontsize=11)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.ylim(0.0, 1.05)
plt.xlim(min_wavelength, max_wavelength)
plt.tight_layout()
#save
output_folder = OUTPUT_DIR #TRA saving path
# ensure the existance of the file
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"have created a new file: {output_folder}")
# 2. 設定基礎檔名
base_name = "TRA_ring"
file_ext = ".png"
final_path = os.path.join(output_folder, base_name + file_ext)
# 3. 檢查重名並自動改名 (核心邏輯)
counter = 1
while os.path.exists(final_path):
    # 如果檔案已經存在，就加上 _1, _2, _3...
    new_name = f"{base_name}_{counter}{file_ext}"
    final_path = os.path.join(output_folder, new_name)
    counter += 1
plt.savefig(final_path, dpi=300, bbox_inches="tight")
print(f"TRA curve has been stored at: {final_path}")
plt.close()

# =============================================================================
# Drawing TRA curvature
# =============================================================================

plt.figure(figsize=(10, 6))
plt.plot(wavelengths, np.abs(incident_intensity), label="Incident Intensity (Reference)", color="red", linewidth=2)
plt.plot(wavelengths, np.abs(reflection_flux), label="Reflection Intensity", color="blue", linewidth=2)
plt.plot(wavelengths, np.abs(transmission_flux), label="Transmission Intensity", color="green", linewidth=2)

plt.xlabel("Wavelength (μm)", fontsize=12)
plt.ylabel("Intensity (a.u.)", fontsize=12)
plt.title("Incident, Reflection, and Transmission Intensity", fontsize=14, fontweight="bold")
plt.legend(loc="best", fontsize=11)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.xlim(min_wavelength, max_wavelength)
plt.tight_layout()

output_folder = OUTPUT_DIR 
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created a new file at: {output_folder}")
#  setting intensity file name
base_name = "intensity_ring"
file_ext = ".png"
final_path = os.path.join(output_folder, base_name + file_ext)
counter = 1
while os.path.exists(final_path):
    new_name = f"{base_name}_{counter}{file_ext}"
    final_path = os.path.join(output_folder, new_name)
    counter += 1
plt.savefig(final_path, dpi=300, bbox_inches="tight")
print(f"TRA curve has been stored at: {final_path}")
plt.close()

# =============================================================================
# drawing graphs of cross section
# =============================================================================

# XY cross section 
plt.figure(figsize=(10, 10))
sim.plot2D(
    output_plane=mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx, sy, 0)),
)
plt.title("XY Cross-section (Horizontal, Z=0)", fontsize=12, fontweight="bold")
plt.xlabel("X (μm)", fontsize=11)
plt.ylabel("Y (μm)", fontsize=11)
plt.tight_layout()
output_folder = OUTPUT_DIR # path of XY cross section
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"new file has been created at: {output_folder}")
base_name = "XY_cross_section_ring"
file_ext = ".png"
final_path = os.path.join(output_folder, base_name + file_ext)
counter = 1
while os.path.exists(final_path):
    new_name = f"{base_name}_{counter}{file_ext}"
    final_path = os.path.join(output_folder, new_name)
    counter += 1
plt.savefig(final_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"XY cross section stored at {final_path}")

# XZ cross section
sim.plot2D(
    output_plane=mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx, 0, sz)),
)
plt.title("XZ Cross-section (Vertical, Y=0)", fontsize=12, fontweight="bold")
plt.xlabel("X (μm)", fontsize=11)
plt.ylabel("Z (μm)", fontsize=11)
plt.tight_layout()
output_folder = OUTPUT_DIR 
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"new file has been created at: {output_folder}")
base_name = "XZ_cross_section_ring"
file_ext = ".png"
final_path = os.path.join(output_folder, base_name + file_ext)
counter = 1
while os.path.exists(final_path):
    new_name = f"{base_name}_{counter}{file_ext}"
    final_path = os.path.join(output_folder, new_name)
    counter += 1
plt.savefig(final_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"XZ cross section stored at {final_path}")

# YZ cross section
sim.plot2D(
    output_plane=mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(0, sy, sz)),
)
plt.title("YZ Cross-section After Simulation (Structure)", fontsize=12, fontweight="bold")
plt.xlabel("Y (μm)", fontsize=11)
plt.ylabel("Z (μm)", fontsize=11)
plt.tight_layout()
output_folder = OUTPUT_DIR 
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"new file has been created at: {output_folder}")
base_name = "YZ_cross_section_ring"
file_ext = ".png"
final_path = os.path.join(output_folder, base_name + file_ext)
counter = 1
while os.path.exists(final_path):
    new_name = f"{base_name}_{counter}{file_ext}"
    final_path = os.path.join(output_folder, new_name)
    counter += 1
plt.savefig(final_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"YZ cross section stored at {final_path}")




# =============================================================================
# Drawing the field distribution at peak absorption
# =============================================================================

print("\nStart drawing the distribution graph of electric field (Auto-Reshape Mode)...")

try:
    peak_freq = 1.0 / peak_wavelength
    dft_freq_array = np.array(dft_freqs)
    freq_idx = int(np.argmin(np.abs(dft_freq_array - peak_freq)))

    # get_dft_array returns complex field
    Ey_data_full = sim.get_dft_array(dft_fields, mp.Ey, freq_idx)
    # get_epsilon returns real field
    eps_data_full = sim.get_epsilon()

    def plot_cross_section(slice_Ey, slice_eps, axis1_name, axis2_name, axis1_range, axis2_range, filename, xlim_range=None):
        # obtain the shape of eletric field data
        n1, n2 = slice_Ey.shape
        
        if slice_eps.shape[0] > n1: slice_eps = slice_eps[:n1, :]
        if slice_eps.shape[1] > n2: slice_eps = slice_eps[:, :n2]
        if slice_eps.shape[0] < n1: 
            slice_Ey = slice_Ey[:slice_eps.shape[0], :]
            n1 = slice_Ey.shape[0]
        if slice_eps.shape[1] < n2: 
            slice_Ey = slice_Ey[:, :slice_eps.shape[1]]
            n2 = slice_Ey.shape[1]

        ax1 = np.linspace(axis1_range[0], axis1_range[1], n1)
        ax2 = np.linspace(axis2_range[0], axis2_range[1], n2)
        X_grid, Y_grid = np.meshgrid(ax1, ax2, indexing='ij')

        fig, ax = plt.subplots(figsize=(10, 8))
        
        pcm = ax.contourf(X_grid, Y_grid, np.abs(slice_Ey), levels=100, cmap='inferno')
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('|Ey| Amplitude (V/m)', fontsize=12, rotation=270, labelpad=20)
        
        ax.contour(X_grid, Y_grid, slice_eps, levels=[1.1], colors='white', linewidths=1.0, alpha=0.7)

        ax.set_xlabel(f"{axis1_name} (μm)", fontsize=12)
        ax.set_ylabel(f"{axis2_name} (μm)", fontsize=12)
        ax.set_title(f"{axis1_name}{axis2_name} Plane @ λ={peak_wavelength:.3f} μm", fontsize=14, fontweight="bold")

        if xlim_range is not None:
          ax.set_xlim(xlim_range)
        
        output_folder = OUTPUT_DIR 
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        
        save_path = os.path.join(output_folder, filename)
        if os.path.exists(save_path):
             save_path = save_path.replace(".png", f"_{int(time.time())}.png")
             
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> graph saved at: {save_path}")

    # ==========================================
    # esculating drawing
    # ==========================================
    
    nx_full, ny_full, nz_full = Ey_data_full.shape

    my_x_limit = (-dgap, 0.2 * dgap)

    # --- 1. XY cross section (Z=0) ---
    z_mid = nz_full // 2
    plot_cross_section(
        Ey_data_full[:, :, z_mid], 
        eps_data_full[:, :, z_mid],
        "X", "Y", (-sx/2, sx/2), (-sy/2, sy/2),
        f"XY_Ey_peak_{peak_wavelength:.3f}um.png",
        xlim_range=my_x_limit
    )

    # --- 2. XZ cross section (Y=0) ---
    y_mid = ny_full // 2
    plot_cross_section(
        Ey_data_full[:, y_mid, :], 
        eps_data_full[:, y_mid, :],
        "X", "Z", (-sx/2, sx/2), (-sz/2, sz/2),
        f"XZ_Ey_peak_{peak_wavelength:.3f}um.png",
        xlim_range=my_x_limit
    )

    # --- 3. YZ cross section (X=Center) ---
    x_idx = nx_full // 2
    plot_cross_section(
        Ey_data_full[x_idx, :, :],
        eps_data_full[x_idx, :, :],
        "Y", "Z", (-sy/2, sy/2), (-sz/2, sz/2),
        f"YZ_Ey_peak_{peak_wavelength:.3f}um.png",
        xlim_range=None
    )

    print("Complete the filed graphs drawing！")

except Exception as e:
    print(f"there are some mistakes during the drawing: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Calculating S parameter
# =============================================================================

print("\n計算S參數...")

# obtain the mindex of eigen mode
eig_parity = mp.NO_PARITY
eps = 1e-12 

# initialize the dataset
S11_mag = np.full(len(wavelengths), np.nan)
S11_phase = np.full(len(wavelengths), np.nan)

try:
    # S11 
    mode_data_ref = sim.get_eigenmode_coefficients(
        mode_monitor_reflection, [1], eig_parity=eig_parity
    )
    
    if mode_data_ref is not None:
        forward_coeff_ref = mode_data_ref.alpha[0, :, 0]  # forward mode
        backward_coeff_ref = mode_data_ref.alpha[0, :, 1]  # backward mode
        
        S11_complex = np.where(
            np.abs(forward_coeff_ref) > eps,
            backward_coeff_ref / forward_coeff_ref,
            np.nan
        )
        S11_phase = np.unwrap(np.angle(S11_complex))
        
        R_clipped = np.clip(R, 0, 1)
        S11_mag = np.sqrt(R_clipped)
    else:
        S11_mag = np.full(len(wavelengths), np.nan)
        S11_phase = np.full(len(wavelengths), np.nan)
        print("Warning: S11 data cannot be obtained from reflectance monitor")
except Exception as e:
    print(f"Mistake happened during the S11 calculation: {e}")

try:
    # S21 
    mode_data_tran = sim.get_eigenmode_coefficients(
        mode_monitor_transmission, [1], eig_parity=eig_parity
    )
    
    if mode_data_tran is not None:
        T_clipped = np.clip(T, 0, 1)
        S21_mag = np.sqrt(T_clipped)
    else:
        S21_mag = np.full(len(wavelengths), np.nan)
        print("Warning: S21 data cannot be obtained from transmittance monitor")
except Exception as e:
    print(f"Mistake happened during the S11 calculation: {e}")
    S21_mag = np.full(len(wavelengths), np.nan)

# =============================================================================
# Drawing the curvature of S parameter
# =============================================================================
output_folder = OUTPUT_DIR    
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# S11 
ax1.plot(wavelengths, S11_mag, color='green', linestyle='-', linewidth=2, label='|S11| (Reflection)')

# S21 
ax1.plot(wavelengths, S21_mag, color='purple', linestyle='-', linewidth=2, label='|S21| (Transmission)')

ax1.set_ylabel("Magnitude (Linear)", fontsize=12)
ax1.set_title("S-Parameters Magnitude (|S11| vs |S21|)", fontsize=14, fontweight="bold")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax1.legend(loc='best', fontsize=11)

ax1.set_ylim(0, 1.1)

# S11 phase
ax1_phase_plot = ax2.plot(wavelengths, S11_phase, color='black', linestyle='--', linewidth=1.5, label='Phase(S11)')

ax2.set_xlabel("Wavelength (μm)", fontsize=12)
ax2.set_ylabel("Phase [rad]", fontsize=12)
ax2.set_title("S-Parameters Phase", fontsize=14, fontweight="bold")
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax2.legend(loc='best', fontsize=11)
ax2.set_xlim(min_wavelength, max_wavelength)

plt.tight_layout()
base_name = "S_Parameters_Combined"
file_ext = ".png"
final_path = os.path.join(output_folder, base_name + file_ext)
counter = 1
while os.path.exists(final_path):
    new_name = f"{base_name}_{counter}{file_ext}"
    final_path = os.path.join(output_folder, new_name)
    counter += 1

plt.savefig(final_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"he merged S-parameter plot has been saved as: {final_path}")

# =============================================================================
# save TRA (csv)
# =============================================================================
data_folder = OUTPUT_DIR 

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f"TRA_ring_{timestamp}.csv"
tra_file_path = os.path.join(data_folder, filename)

A = np.clip(A, 0, 1)

data_to_save = np.column_stack((wavelengths, T, R, A))

np.savetxt(
    tra_file_path,
    data_to_save,
    delimiter=",",
    header="Wavelength(um), Transmission(T), Reflection(R), Absorption(A)",
    comments="", 
    fmt='%.6e'   
)

print("-" * 40)
print(f"✅ Data has been successfully saved.！")
print(f"📂 file path: {tra_file_path}")
print("-" * 40)

# =============================================================================
# save S parameter data (c s v)
# =============================================================================
data_folder = OUTPUT_DIR 

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f"S_parameters_{timestamp}.csv"
s_param_path = os.path.join(data_folder, filename)

data_to_save = np.column_stack((
    wavelengths, 
    S11_mag, 
    S11_phase, 
    S21_mag
))

np.savetxt(
    s_param_path,
    data_to_save,
    delimiter=",",
    header="Wavelength(um),|S11|(Mag),Phase(S11)[rad],|S21|(Mag)",
    comments="", 
    fmt='%.6e'
)

print("-" * 40)
print(f"✅ S-parameter data has been successfully saved.")
print(f"📂 file path: {s_param_path}")
print("-" * 40)

# =============================================================================
# conclusion
# =============================================================================

print("\n" + "=" * 50)
print("Simulation complete! All results have been saved.")
print(f"  wavelength: {peak_wavelength:.4f} μm")
print(f"  Absorption: {peak_absorption*100:.2f}%")
print(f"  resolution: {resolution}pixels/μm")
print("=" * 50)

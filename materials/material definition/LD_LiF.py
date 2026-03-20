from typing import Tuple

import matplotlib
import meep
import nlopt
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt


def lorentzfunc(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Returns the complex ε profile given a set of Lorentzian parameters p
    (σ_0, ω_0, γ_0, σ_1, ω_1, γ_1, ...) for a set of frequencies x.
    """
    N = len(p) // 3
    y = np.zeros(len(x))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        y = y + A_n / (np.square(x_n) - np.square(x) - 1j * x * g_n)
    return y


def lorentzerr(p: np.ndarray, x: np.ndarray, y: np.ndarray, grad: np.ndarray) -> float:
    """
    Returns the error (or residual or loss) as the L2 norm
    of the difference of ε(p,x) and y over a set of frequencies x as
    well as the gradient of this error with respect to each Lorentzian
    polarizability parameter in p and saving the result in grad.
    """
    N = len(p) // 3
    yp = lorentzfunc(p, x)
    val = np.sum(np.square(abs(y - yp)))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        d = 1 / (np.square(x_n) - np.square(x) - 1j * x * g_n)
        if grad.size > 0:
            grad[3 * n + 0] = 2 * np.real(np.dot(np.conj(yp - y), d))
            grad[3 * n + 1] = (
                -4 * x_n * A_n * np.real(np.dot(np.conj(yp - y), np.square(d)))
            )
            grad[3 * n + 2] = (
                -2 * A_n * np.imag(np.dot(np.conj(yp - y), x * np.square(d)))
            )
    return val


def lorentzfit(
    p0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    alg=nlopt.LD_LBFGS,
    tol: float = 1e-25,
    maxeval: float = 10000,
) -> Tuple[np.ndarray, float]:
    """
    Returns the optimal Lorentzian polarizability parameters and error
    which minimize the error in ε(p0,x) relative to y for an initial
    set of Lorentzian polarizability parameters p0 over a set of
    frequencies x using the NLopt algorithm alg for a relative
    tolerance tol and a maximum number of iterations maxeval.
    """
    opt = nlopt.opt(alg, len(p0))
    opt.set_ftol_rel(tol)
    opt.set_maxeval(maxeval)
    opt.set_lower_bounds(np.zeros(len(p0)))
    opt.set_upper_bounds(float("inf") * np.ones(len(p0)))
    opt.set_min_objective(lambda p, grad: lorentzerr(p, x, y, grad))
    local_opt = nlopt.opt(nlopt.LD_LBFGS, len(p0))
    local_opt.set_ftol_rel(1e-10)
    local_opt.set_xtol_rel(1e-8)
    opt.set_local_optimizer(local_opt)
    popt = opt.optimize(p0)
    minf = opt.last_optimum_value()
    return popt, minf


def load_material_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load material refractive index data from CSV file.
    The file format has two sections:
    - First section: wavelength (μm), real(n)
    - Second section: wavelength (μm), imag(n)
    """
    # Read the entire file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the separator line (empty line)
    separator_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == '':
            separator_idx = i
            break
    
    if separator_idx == -1:
        raise ValueError("Could not find separator between n and k data")
    
    # Parse n data (first section)
    n_data = []
    for i in range(1, separator_idx):  # Skip header
        parts = lines[i].strip().split(',')
        if len(parts) == 2:
            wl, n_real = float(parts[0]), float(parts[1])
            n_data.append([wl, n_real])
    
    # Parse k data (second section)
    k_data = []
    for i in range(separator_idx + 2, len(lines)):  # Skip separator and header
        parts = lines[i].strip().split(',')
        if len(parts) == 2:
            wl, k_imag = float(parts[0]), float(parts[1])
            k_data.append([wl, k_imag])
    
    n_data = np.array(n_data)
    k_data = np.array(k_data)
    
    # Ensure wavelengths match
    if not np.allclose(n_data[:, 0], k_data[:, 0]):
        raise ValueError("Wavelengths in n and k data do not match")
    
    # Create complex refractive index
    wavelengths = n_data[:, 0]  # in μm
    n_complex = n_data[:, 1] + 1j * k_data[:, 1]
    
    return wavelengths, n_complex


if __name__ == "__main__":
    # Load material refractive index data
    print("Loading material refractive index data...")
    wavelengths, n_complex = load_material_data("Material_LiF.csv")
    
    print(f"Loaded {len(wavelengths)} data points")
    print(f"Wavelength range: {wavelengths[0]:.3f} - {wavelengths[-1]:.3f} μm")
    print(f"Refractive index range: n = {np.min(np.real(n_complex)):.2f} - {np.max(np.real(n_complex)):.2f}")
    print(f"Extinction coefficient range: k = {np.min(np.imag(n_complex)):.2f} - {np.max(np.imag(n_complex)):.2f}")

    # Fitting parameter: the instantaneous (infinite frequency) dielectric.
    # Should be > 1.0 for stability and chosen such that
    # np.amin(np.real(eps)) is ~1.0. eps is defined below.
    eps_inf = 1.3969  # For metals, this is typically 1.0

    eps = np.square(n_complex) - eps_inf

    # Fit only the data in the wavelength range of [wl_min, wl_max].
    wl_min = 0.2  # minimum wavelength (units of μm)
    wl_max = 4.5  # maximum wavelength (units of μm)
    start_idx = np.where(wavelengths >= wl_min)
    idx_start = start_idx[0][0]
    end_idx = np.where(wavelengths <= wl_max)
    idx_end = end_idx[0][-1] + 1

    # The fitting function is ε(f) where f is the frequency, rather than ε(λ).
    # Note: an equally spaced grid of wavelengths results in the larger
    #       wavelengths having a finer frequency grid than smaller ones.
    #       This feature may impact the accuracy of the fit.
    freqs = 1.0 / wavelengths  # units of 1/μm
    freqs_reduced = freqs[idx_start:idx_end]
    wl_reduced = wavelengths[idx_start:idx_end]
    eps_reduced = eps[idx_start:idx_end]

    print(f"Fitting range: {wl_reduced[0]:.3f} - {wl_reduced[-1]:.3f} μm")
    print(f"Frequency range: {freqs_reduced[0]:.3f} - {freqs_reduced[-1]:.3f} 1/μm")

    # Fitting parameter: number of Lorentzian terms to use in the fit
    num_lorentzians = 2  # Increased for better fit of metal data

    # Number of times to repeat local optimization with random initial values.
    num_repeat = 100  # Increased for better optimization

    print(f"Starting optimization with {num_lorentzians} Lorentzian terms...")
    print(f"Running {num_repeat} optimization attempts...")

    ps = np.zeros((num_repeat, 3 * num_lorentzians))
    mins = np.zeros(num_repeat)
    for m in range(num_repeat):
        # Initial values for the Lorentzian polarizability terms. Each term
        # consists of three parameters (σ, ω, γ) and is chosen randomly.
        # Note: for metals, γ should be non-zero for absorption.
        # Better initialization for metal fitting
        p_rand = []
        for i in range(num_lorentzians):
            # σ: strength parameter (0.1 to 100)
            p_rand.append(10 ** (2 * np.random.random() - 1))
            # ω: frequency parameter (0.1 to 5.0)
            p_rand.append(0.1 + 4.9 * np.random.random())
            # γ: damping parameter (0.01 to 2.0)
            p_rand.append(0.01 + 1.99 * np.random.random())
        # Try different algorithms for better convergence
        algorithms = [nlopt.LD_MMA, nlopt.LD_LBFGS, nlopt.GN_DIRECT_L]
        best_error = float('inf')
        best_params = p_rand
        
        for alg in algorithms:
            try:
                params, error = lorentzfit(
                    p_rand, freqs_reduced, eps_reduced, alg, 1e-25, 10000
                )
                if error < best_error:
                    best_error = error
                    best_params = params
            except:
                continue
        
        ps[m, :] = best_params
        mins[m] = best_error
        ps_str = "( " + ", ".join(f"{prm:.4f}" for prm in ps[m, :]) + " )"
        print(f"iteration: {m:3d}, params: {ps_str}, error: {mins[m]:.6f}")

    # Find the best performing set of parameters.
    idx_opt = np.where(np.min(mins) == mins)[0][0]
    popt_str = "( " + ", ".join(f"{prm:.4f}" for prm in ps[idx_opt]) + " )"
    print(f"optimal: {popt_str}, error: {mins[idx_opt]:.6f}")

    # Define a `Medium` class object using the optimal fitting parameters.
    E_susceptibilities = []

    for n in range(num_lorentzians):
        mymaterial_freq = ps[idx_opt][3 * n + 1]
        mymaterial_gamma = ps[idx_opt][3 * n + 2]

        if mymaterial_freq == 0:
            mymaterial_sigma = ps[idx_opt][3 * n + 0]
            # For now, just store the parameters
            E_susceptibilities.append({
                'type': 'Drude',
                'frequency': 1.0,
                'gamma': mymaterial_gamma,
                'sigma': mymaterial_sigma
            })
        else:
            mymaterial_sigma = ps[idx_opt][3 * n + 0] / mymaterial_freq**2
            # For now, just store the parameters
            E_susceptibilities.append({
                'type': 'Lorentzian',
                'frequency': mymaterial_freq,
                'gamma': mymaterial_gamma,
                'sigma': mymaterial_sigma
            })

    # Calculate the fitted epsilon values using our Lorentz function
    material_eps_fitted = lorentzfunc(ps[idx_opt], freqs_reduced) + eps_inf

    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

    ax[0].plot(wl_reduced, np.real(eps_reduced) + eps_inf, "bo-", label="actual", markersize=3)
    ax[0].plot(wl_reduced, np.real(material_eps_fitted), "ro-", label="fit", markersize=3)
    ax[0].set_xlabel("wavelength (μm)")
    ax[0].set_ylabel(r"real($\epsilon$)")
    ax[0].set_xlim(0, 0.6)  # 设置x轴范围
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(wl_reduced, np.imag(eps_reduced), "bo-", label="actual", markersize=3)
    ax[1].plot(wl_reduced, np.imag(material_eps_fitted), "ro-", label="fit", markersize=3)
    ax[1].set_xlabel("wavelength (μm)")
    ax[1].set_ylabel(r"imag($\epsilon$)")
    ax[1].set_xlim(0, 0.6)  # 设置x轴范围
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"material: Comparison of Actual Material Data and Fit\n"
        f"using Drude-Lorentzian Susceptibility ({num_lorentzians} terms)"
    )

    fig.subplots_adjust(wspace=0.3)
    fig.savefig("eps_fit_LiF.png", dpi=150, bbox_inches="tight")
    print("Plot saved as 'eps_fit_LiF.png'")

    # Print the final Medium definition for use in Meep
    print("\n" + "="*60)
    print("Meep Medium Definition for material:")
    print("="*60)
    print(f"# Meep Medium Definition for material:")
    print(f"# epsilon_inf = {eps_inf}")
    if E_susceptibilities:
        print("# Susceptibility parameters:")
        for i, sus in enumerate(E_susceptibilities):
            if sus['type'] == 'Drude':
                print(f"# Drude term {i+1}:")
                print(f"#   frequency = {sus['frequency']}")
                print(f"#   gamma = {sus['gamma']}")
                print(f"#   sigma = {sus['sigma']}")
            else:  # Lorentzian
                print(f"# Lorentzian term {i+1}:")
                print(f"#   frequency = {sus['frequency']}")
                print(f"#   gamma = {sus['gamma']}")
                print(f"#   sigma = {sus['sigma']}")
    
    print("\n# Fitted parameters for manual Meep definition:")
    print(f"# Optimal parameters: {popt_str}")
    print(f"# Final error: {mins[idx_opt]:.6f}")
    print("="*60)


#cd /Users/xiebailin/meep_projects
#source meep_env/bin/activate
#python LD_Ni.py
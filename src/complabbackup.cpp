/* ============================================================================
 * CompLaB3D - Three-Dimensional Lattice Boltzmann Reactive Transport Solver
 * ============================================================================
 * 
 * Author:      Shahram Asgari
 * Advisor:     Dr. Christof Meile
 * Laboratory:  Meile Lab
 * Institution: University of Georgia (UGA)
 * 
 * ============================================================================
 * CALCULATION FLOW (13 Steps):
 * ───────────────────────────────────────────────────────────────────────────
 * STEP 1:  L = nx - 1 (domain length)              [Matches PressureGradient]
 * STEP 2:  ν_NS = (1/3) × (τ_NS - 0.5)             [NS viscosity, D3Q19]
 * STEP 3:  D_lattice = (1/4) × (τ_ADE - 0.5)       [ADE diffusivity, D3Q7, τ_ADE=0.8 FIXED]
 * STEP 4:  First NS simulation with ΔP₀ → u₀      [Measure geometry response]
 * STEP 5:  k = (u₀ × ν_NS × L) / ΔP₀              [Permeability from Darcy's Law]
 * STEP 6:  u_target = (Pe × D_lattice) / L        [Target velocity for desired Pe]
 * STEP 7:  ΔP_new = (u_target × ν_NS × L) / k     [Corrected pressure gradient]
 * STEP 8:  Second NS simulation with ΔP_new       [Achieve target velocity]
 * STEP 9:  Extract u_final, PoreMaxUx             [Final velocities]
 * STEP 10: Ma = PoreMaxUx / √(1/3)                [Stability checks]
 * STEP 11: Pe_achieved = (u_final × L) / D        [Verify Péclet number]
 * STEP 12: Δt = (D_lattice × dx²) / D_physical    [Physical time step]
 * STEP 13: Run ADE simulation with adaptive stop  [Main transport loop]
 * ───────────────────────────────────────────────────────────────────────────
 * 
 * OUTPUT FILES:
 *   - Porous medium geometry (VTI)
 *   - Flow field / NS lattice (VTI)
 *   - Concentration fields (VTI)
 *   - BTC_timeseries.csv (breakthrough curve data)
 *   - Domain_properties.csv (single-row summary)
 *   - Moments_summary.csv (M1, CV, etc.)
 * 
 * ADAPTIVE STOPPING:
 *   Simulation stops automatically when C_flux_weighted >= 0.99
 *   for 3 consecutive checks, ensuring complete BTC without wasting iterations.
 * 
 * ============================================================================
 */

#include "complab_functions.hh"
#include "complab_processors.hh"
#include <chrono>
#include <string>
#include <iostream>
#include <cstring>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

// NOTE: The following are already defined in complab_functions.hh:
//   typedef double T;
//   #define DESCRIPTOR descriptors::D3Q19Descriptor
//   #define BGK descriptors::AdvectionDiffusionD3Q7Descriptor
//   #define thrd 1e-14

// ============================================================================
// STABILITY CHECK STRUCTURE
// ============================================================================
struct StabilityReport {
    T Ma;              // Mach number = u_max / cs
    bool Ma_ok;        // true if Ma < 1.0
    bool Ma_warning;   // true if Ma > 0.3
    T CFL;             // CFL number
    bool CFL_ok;       // true if CFL < 1.0
    T tau_NS;
    T tau_ADE;
    bool tau_NS_ok;    // true if 0.5 < tau_NS < 2.0
    bool tau_ADE_ok;   // true if 0.5 < tau_ADE < 2.0
    T Pe_grid;
    bool Pe_grid_ok;   // true if Pe_grid < 2.0
    bool all_ok;
    bool has_warnings;
};

// ============================================================================
// STRUCTURE: BTC Data Point (for storing timeseries data)
// ============================================================================
struct BTCDataPoint {
    plint iteration;
    T time_seconds;
    T time_lattice;
    T C_average;
    T C_flux_weighted;
    T pore_volume;
};

// ============================================================================
// FUNCTION: performStabilityChecks  [Used in STEP 10]
// ============================================================================
StabilityReport performStabilityChecks(T u_max, T tau_NS, T tau_ADE, T D_lattice) {
    StabilityReport report;
    
    T cs = std::sqrt(1.0 / 3.0);
    
    report.Ma = u_max / cs;
    report.Ma_ok = (report.Ma < 1.0);
    report.Ma_warning = (report.Ma > 0.3);
    
    report.CFL = u_max;
    report.CFL_ok = (report.CFL < 1.0);
    
    report.tau_NS = tau_NS;
    report.tau_NS_ok = (tau_NS > 0.5 && tau_NS < 2.0);
    report.tau_ADE = tau_ADE;
    report.tau_ADE_ok = (tau_ADE > 0.5 && tau_ADE < 2.0);
    
    report.Pe_grid = u_max / D_lattice;
    report.Pe_grid_ok = (report.Pe_grid < 2.0);
    
    report.all_ok = report.Ma_ok && report.CFL_ok && report.tau_NS_ok && report.tau_ADE_ok;
    report.has_warnings = report.Ma_warning || !report.Pe_grid_ok;
    
    return report;
}

// ============================================================================
// FUNCTION: printStabilityReport
// ============================================================================
void printStabilityReport(const StabilityReport& report) {
    pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║                    STABILITY CHECK REPORT                        ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ Ma = " << std::scientific << std::setprecision(4) << report.Ma 
          << (report.Ma_ok ? (report.Ma_warning ? " ⚠" : " ✓") : " ✗") << "  ";
    pcout << "CFL = " << report.CFL << (report.CFL_ok ? " ✓" : " ✗") << "           ║\n";
    pcout << "║ τ_NS = " << std::fixed << std::setprecision(4) << report.tau_NS 
          << (report.tau_NS_ok ? " ✓" : " ✗") << "    ";
    pcout << "τ_ADE = " << report.tau_ADE << (report.tau_ADE_ok ? " ✓" : " ✗") << "                    ║\n";
    pcout << "║ Pe_grid = " << report.Pe_grid << (report.Pe_grid_ok ? " ✓" : " ⚠") << "                                            ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    if (report.all_ok && !report.has_warnings) {
        pcout << "║              ✓ ALL STABILITY CHECKS PASSED                      ║\n";
    } else if (report.all_ok) {
        pcout << "║         ⚠ SIMULATION CAN PROCEED WITH WARNINGS                  ║\n";
    } else {
        pcout << "║         ✗ CRITICAL ERRORS - SIMULATION MAY FAIL                 ║\n";
    }
    pcout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
}

// ============================================================================
// FUNCTION: calculatePorosity
// ============================================================================
// Matches Python: np.sum(geometry == 0) / geometry.size
// ============================================================================
T calculatePorosity(
    MultiScalarField3D<int>& geometry,
    plint nx, plint ny, plint nz,
    std::vector<plint>& pore_dynamics)
{
    plint total_cells = nx * ny * nz;
    plint pore_cells = 0;
    
    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            for (plint iZ = 0; iZ < nz; ++iZ) {
                plint geom_tag = geometry.get(iX, iY, iZ);
                
                for (size_t p = 0; p < pore_dynamics.size(); ++p) {
                    if (geom_tag == pore_dynamics[p]) {
                        pore_cells++;
                        break;
                    }
                }
            }
        }
    }
    
    return (T)pore_cells / (T)total_cells;
}

// ============================================================================
// FUNCTION: calculateFluxWeightedConcentration
// ============================================================================
// Matches Python: np.sum(C * u) / np.sum(u) at outlet pore cells
// From fluxweighted3D.py lines 23-28
// ============================================================================
T calculateFluxWeightedConcentration(
    MultiBlockLattice3D<T, BGK>& adeLattice,
    MultiBlockLattice3D<T, DESCRIPTOR>& nsLattice,
    MultiScalarField3D<int>& geometry,
    plint nx, plint ny, plint nz,
    std::vector<plint>& pore_dynamics)
{
    plint outletX = nx - 2;  // Same as Python: outletX = nx - 2
    
    T sum_C_times_u = 0.0;
    T sum_u = 0.0;
    
    for (plint iY = 0; iY < ny; ++iY) {
        for (plint iZ = 0; iZ < nz; ++iZ) {
            plint geom_tag = geometry.get(outletX, iY, iZ);
            
            // Check if pore cell
            bool is_pore = false;
            for (size_t p = 0; p < pore_dynamics.size(); ++p) {
                if (geom_tag == pore_dynamics[p]) {
                    is_pore = true;
                    break;
                }
            }
            
            if (is_pore) {
                // Get concentration (density in LBM)
                T C = adeLattice.get(outletX, iY, iZ).computeDensity();
                C = std::min(C, (T)1.0);  // Cap at 1.0, same as Python: np.minimum(..., 1.0)
                
                // Get velocity magnitude (velocityNorm)
                Array<T, 3> vel;
                nsLattice.get(outletX, iY, iZ).computeVelocity(vel);
                T u = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
                
                sum_C_times_u += C * u;
                sum_u += u;
            }
        }
    }
    
    if (sum_u > thrd) {
        return sum_C_times_u / sum_u;
    } else {
        return 0.0;
    }
}

// ============================================================================
// FUNCTION: calculateAverageConcentration
// ============================================================================
// Matches Python: np.mean(outlet_concentrations) at outlet pore cells
// From fluxweighted3D.py lines 30-34
// ============================================================================
T calculateAverageConcentration(
    MultiBlockLattice3D<T, BGK>& adeLattice,
    MultiScalarField3D<int>& geometry,
    plint nx, plint ny, plint nz,
    std::vector<plint>& pore_dynamics)
{
    plint outletX = nx - 2;  // Same as Python
    
    T sum_C = 0.0;
    plint count = 0;
    
    for (plint iY = 0; iY < ny; ++iY) {
        for (plint iZ = 0; iZ < nz; ++iZ) {
            plint geom_tag = geometry.get(outletX, iY, iZ);
            
            bool is_pore = false;
            for (size_t p = 0; p < pore_dynamics.size(); ++p) {
                if (geom_tag == pore_dynamics[p]) {
                    is_pore = true;
                    break;
                }
            }
            
            if (is_pore) {
                T C = adeLattice.get(outletX, iY, iZ).computeDensity();
                C = std::min(C, (T)1.0);  // Cap at 1.0
                sum_C += C;
                count++;
            }
        }
    }
    
    if (count > 0) {
        return sum_C / (T)count;
    } else {
        return 0.0;
    }
}

// ============================================================================
// FUNCTION: calculatePoreVolume
// ============================================================================
// Matches Python: (darcy_velocity * time) / (length * porosity)
// From Gau_pore_volume.py lines 38-43
// ============================================================================
T calculatePoreVolume(T u_Darcy, T time, T L, T porosity)
{
    if (porosity > thrd && L > thrd) {
        return (u_Darcy * time) / (L * porosity);
    } else {
        return 0.0;
    }
}

// ============================================================================
// FUNCTION: calculateMoments
// ============================================================================
// Matches Python: Moment_calculator_for_age_and_BTC_01.py lines 359-397
// Uses trapezoidal integration (np.trapezoid)
// ============================================================================
void calculateMoments(
    const std::vector<BTCDataPoint>& btc_data,
    T& M0, T& M1, T& M2_central, T& std_dev, T& cv)
{
    size_t n = btc_data.size();
    if (n < 2) {
        M0 = 0; M1 = 0; M2_central = 0; std_dev = 0; cv = 0;
        return;
    }
    
    // Extract time and concentration arrays
    std::vector<T> time(n);
    std::vector<T> conc(n);
    
    for (size_t i = 0; i < n; ++i) {
        time[i] = btc_data[i].time_seconds;
        conc[i] = btc_data[i].C_flux_weighted;
    }
    
    // M0: Zeroth moment using trapezoidal rule
    // np.trapezoid(conc, time)
    M0 = 0.0;
    for (size_t i = 0; i < n - 1; ++i) {
        T dt = time[i+1] - time[i];
        M0 += 0.5 * (conc[i] + conc[i+1]) * dt;
    }
    
    if (M0 <= 0) {
        M1 = 0; M2_central = 0; std_dev = 0; cv = 0;
        return;
    }
    
    // M1: First moment (mean arrival time)
    // M1_raw = np.trapezoid(time * conc, time)
    // M1 = M1_raw / M0
    T M1_raw = 0.0;
    for (size_t i = 0; i < n - 1; ++i) {
        T dt = time[i+1] - time[i];
        T integrand_i = time[i] * conc[i];
        T integrand_ip1 = time[i+1] * conc[i+1];
        M1_raw += 0.5 * (integrand_i + integrand_ip1) * dt;
    }
    M1 = M1_raw / M0;
    
    // M2: Second moment
    // M2_raw = np.trapezoid(time**2 * conc, time)
    // M2_normalized = M2_raw / M0
    // M2_central = max(0, M2_normalized - M1**2)
    T M2_raw = 0.0;
    for (size_t i = 0; i < n - 1; ++i) {
        T dt = time[i+1] - time[i];
        T integrand_i = time[i] * time[i] * conc[i];
        T integrand_ip1 = time[i+1] * time[i+1] * conc[i+1];
        M2_raw += 0.5 * (integrand_i + integrand_ip1) * dt;
    }
    T M2_normalized = M2_raw / M0;
    M2_central = std::max((T)0.0, M2_normalized - M1 * M1);
    
    // Standard deviation
    std_dev = std::sqrt(M2_central);
    
    // Coefficient of variation
    if (M1 > thrd) {
        cv = std_dev / M1;
    } else {
        cv = 0.0;
    }
}

// ============================================================================
// FUNCTION: writeBTCTimeseries
// ============================================================================
void writeBTCTimeseries(
    const std::string& filepath,
    const std::vector<BTCDataPoint>& btc_data,
    T porosity, T u_Darcy_lattice, T u_Darcy_physical,
    T permeability_lattice, T permeability_m2)
{
    std::ofstream file(filepath);
    if (!file.is_open()) {
        pcout << "ERROR: Could not open file " << filepath << " for writing.\n";
        return;
    }
    
    // Header
    file << "Iteration,Time_seconds,Time_lattice,C_average,C_flux_weighted,Pore_Volume,"
         << "Porosity,u_Darcy_lattice,u_Darcy_m_s,Permeability_lattice,Permeability_m2\n";
    
    // Data rows
    file << std::scientific << std::setprecision(8);
    for (size_t i = 0; i < btc_data.size(); ++i) {
        const BTCDataPoint& pt = btc_data[i];
        file << pt.iteration << ","
             << pt.time_seconds << ","
             << pt.time_lattice << ","
             << pt.C_average << ","
             << pt.C_flux_weighted << ","
             << pt.pore_volume << ","
             << porosity << ","
             << u_Darcy_lattice << ","
             << u_Darcy_physical << ","
             << permeability_lattice << ","
             << permeability_m2 << "\n";
    }
    
    file.close();
    pcout << "  ✓ Saved BTC timeseries: " << filepath << "\n";
}

// ============================================================================
// FUNCTION: writeDomainProperties
// ============================================================================
void writeDomainProperties(
    const std::string& filepath,
    plint nx, plint ny, plint nz, T dx, T dt,
    T porosity, T permeability_lattice, T permeability_m2,
    T u_Darcy_lattice, T u_Darcy_physical,
    T Pe_target, T Pe_achieved,
    T tau_NS, T tau_ADE, T nu_NS, T D_lattice, T L_domain)
{
    std::ofstream file(filepath);
    if (!file.is_open()) {
        pcout << "ERROR: Could not open file " << filepath << " for writing.\n";
        return;
    }
    
    // Header
    file << "nx,ny,nz,dx_m,dt_s,Porosity,Permeability_lattice,Permeability_m2,"
         << "u_Darcy_lattice,u_Darcy_m_s,Pe_target,Pe_achieved,"
         << "tau_NS,tau_ADE,nu_NS,D_lattice,L_domain\n";
    
    // Single data row
    file << std::scientific << std::setprecision(8);
    file << nx << "," << ny << "," << nz << ","
         << dx << "," << dt << ","
         << porosity << ","
         << permeability_lattice << "," << permeability_m2 << ","
         << u_Darcy_lattice << "," << u_Darcy_physical << ","
         << Pe_target << "," << Pe_achieved << ","
         << tau_NS << "," << tau_ADE << ","
         << nu_NS << "," << D_lattice << "," << L_domain << "\n";
    
    file.close();
    pcout << "  ✓ Saved domain properties: " << filepath << "\n";
}

// ============================================================================
// FUNCTION: writeMomentsSummary
// ============================================================================
void writeMomentsSummary(
    const std::string& filepath,
    T M0, T M1_seconds, T M1_lattice,
    T M2_central, T sigma_seconds, T sigma_lattice, T cv,
    T porosity, T Pe_achieved,
    T permeability_m2, T u_Darcy_physical)
{
    std::ofstream file(filepath);
    if (!file.is_open()) {
        pcout << "ERROR: Could not open file " << filepath << " for writing.\n";
        return;
    }
    
    // Header
    file << "M0,M1_seconds,M1_lattice,M2_central,Sigma_seconds,Sigma_lattice,CV,"
         << "Porosity,Pe_achieved,Permeability_m2,u_Darcy_m_s\n";
    
    // Single data row
    file << std::scientific << std::setprecision(8);
    file << M0 << ","
         << M1_seconds << "," << M1_lattice << ","
         << M2_central << ","
         << sigma_seconds << "," << sigma_lattice << ","
         << cv << ","
         << porosity << "," << Pe_achieved << ","
         << permeability_m2 << "," << u_Darcy_physical << "\n";
    
    file.close();
    pcout << "  ✓ Saved moments summary: " << filepath << "\n";
}

// ============================================================================
//                              MAIN FUNCTION
// ============================================================================
int main(int argc, char** argv) {
    
    plbInit(&argc, &argv);
    global::timer("total").start();
    ImageWriter<T> image("leeloo");

    // Variable declarations
    plint kns_count = 0, lb_count = 0;
    char* main_path = (char*)malloc(256 * sizeof(char));
    getcwd(main_path, 256 * sizeof(char));
    char* src_path = (char*)malloc(256 * sizeof(char));
    char* input_path = (char*)malloc(256 * sizeof(char));
    char* output_path = (char*)malloc(256 * sizeof(char));
    char* ns_filename = (char*)malloc(256 * sizeof(char));
    
    plint nx, ny, nz, num_of_substrates;
    T dx, dy, dz, deltaP, Pe, charcs_length;
    std::string geom_filename, mask_filename, ade_filename;
    
    std::vector<bool> vec_left_btype, vec_right_btype;
    std::vector<T> vec_c0, vec_left_bcondition, vec_right_bcondition;
    std::vector<std::vector<int>> vec_const_loc;
    std::vector<std::vector<T>> vec_Kc, vec_Kc_kns, vec_const_lb, vec_const_ub;

    bool read_NS_file = false, read_ADE_file = false, soluteDindex = false;
    bool track_performance = false, halfflag = false, eqflag = false;

    plint no_dynamics = 0, bounce_back = 1, ns_rerun_iT0 = 0;
    plint ns_update_interval = 1, ade_update_interval = 1;
    plint ns_maxiTer_1, ns_maxiTer_2;
    plint ade_rerun_iT0 = 0, ade_maxiTer = 10000000;
    plint ade_VTK_iTer = 1000, ade_CHK_iTer = 1000000;

    T tau = 0.8;
    T ns_converge_iT1 = 1e-8, ns_converge_iT2 = 1e-4, ade_converge_iT = 1e-8;

    std::vector<bool> vec_fixC, vec_fixLB;
    std::vector<plint> pore_dynamics, solver_type, reaction_type, vec_sense;
    std::vector<T> vec_solute_poreD;
    std::vector<std::string> vec_subs_names;

    std::string str_mainDir = main_path;
    if (str_mainDir.back() != '/') { str_mainDir += "/"; }
    std::srand(std::time(nullptr));

    // ========================================================================
    // LOAD XML INPUT FILE
    // ========================================================================
    pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║                         CompLaB3D                                ║\n";
    pcout << "║          Three-Dimensional Reactive Transport Solver             ║\n";
    pcout << "║                  (with CSV output & adaptive stop)               ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║  Author:  Shahram Asgari                                         ║\n";
    pcout << "║  Advisor: Dr. Christof Meile                                     ║\n";
    pcout << "║  Lab:     Meile Lab, University of Georgia                       ║\n";
    pcout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    int erck = 0;
    try {
        erck = initialize_complab(
            main_path, src_path, input_path, output_path, ns_filename, ade_filename, 
            geom_filename, mask_filename,
            read_NS_file, ns_rerun_iT0, ns_converge_iT1, ns_converge_iT2, 
            ns_maxiTer_1, ns_maxiTer_2, ns_update_interval, ade_update_interval,
            read_ADE_file, ade_rerun_iT0, ade_VTK_iTer, ade_CHK_iTer, 
            ade_converge_iT, ade_maxiTer, 
            nx, ny, nz, dx, dy, dz, deltaP, tau,
            Pe, charcs_length, vec_solute_poreD, soluteDindex, 
            pore_dynamics, bounce_back, no_dynamics, num_of_substrates, vec_subs_names,
            solver_type, lb_count, kns_count, reaction_type, 
            vec_c0, vec_left_btype, vec_right_btype, vec_left_bcondition, vec_right_bcondition,
            vec_Kc, vec_Kc_kns, vec_fixLB, vec_fixC, vec_sense, 
            vec_const_loc, vec_const_lb, vec_const_ub, 
            track_performance, halfflag, eqflag);
    }
    catch (PlbIOException& exception) {
        pcout << "ERROR: " << exception.what() << "\nTerminating.\n" << std::endl;
        return -1;
    }
    if (erck != 0) { return -1; }

    std::string str_inputDir = input_path;
    std::string str_outputDir = output_path;
    if (str_inputDir.back() != '/') { str_inputDir += "/"; }
    if (str_outputDir.back() != '/') { str_outputDir += "/"; }

    struct stat statStruct;
    stat(output_path, &statStruct);
    if (!S_ISDIR(statStruct.st_mode)) { mkdir(output_path, 0777); }
    global::directories().setOutputDir(str_outputDir);

    pcout << "CompLaB main directory = " << str_mainDir << std::endl;
    pcout << "CompLaB input directory = " << main_path << "/" << input_path << std::endl;
    pcout << "CompLaB output directory = " << main_path << "/" << output_path << std::endl << std::endl;

    if (track_performance == 1) {
        pcout << "Performance tracker has been activated. Skipping all the non-essential IO.\n";
    }

    // ════════════════════════════════════════════════════════════════════════
    // STEP 1: DOMAIN LENGTH (for permeability calculation)
    // ════════════════════════════════════════════════════════════════════════
    // L = nx - 1 to match PressureGradient class which divides deltaP by (nx-1)
    // ────────────────────────────────────────────────────────────────────────
    T L_lattice = (T)(nx - 1);  // STEP 1: Domain length (matches PressureGradient)

    // ════════════════════════════════════════════════════════════════════════
    // STEP 3: SET FIXED τ_ADE
    // ════════════════════════════════════════════════════════════════════════
    T tau_ADE = 0.8;  // FIXED value

    pcout << "╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║                    SIMULATION PARAMETERS                         ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ Domain: " << nx << " × " << ny << " × " << nz << " lattice units\n";
    pcout << "║ Grid spacing: dx = " << dx << " m\n";
    pcout << "║ Domain length L = " << L_lattice << " lattice units (nx - 1)\n";
    pcout << "║ Target Pe: " << Pe << "\n";
    pcout << "║ τ_NS (from XML): " << tau << "\n";
    pcout << "║ τ_ADE (FIXED): " << tau_ADE << "\n";
    pcout << "║ D_physical: " << vec_solute_poreD[0] << " m²/s\n";
    pcout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 1: Domain Length                                          │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ L = nx - 1 = " << nx << " - 1 = " << L_lattice << " lattice units           │\n";
    pcout << "│ Physical length: L × dx = " << L_lattice * dx * 1e6 << " um                  │\n";
    pcout << "│ NOTE: Matches (nx-1) divisor in PressureGradient class         │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // Read geometry
    MultiScalarField3D<int> geometry(nx, ny, nz);
    readGeometry(str_inputDir + geom_filename, geometry);
    saveGeometry("inputGeom", geometry);

    // ════════════════════════════════════════════════════════════════════════
    // CALCULATE POROSITY
    // ════════════════════════════════════════════════════════════════════════
    T porosity = calculatePorosity(geometry, nx, ny, nz, pore_dynamics);
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ POROSITY CALCULATION                                           │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ Porosity φ = " << porosity << "                                        │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // STEP 2: NS KINEMATIC VISCOSITY
    // ════════════════════════════════════════════════════════════════════════
    T nsLatticeTau = tau;
    T nsLatticeOmega = 1.0 / nsLatticeTau;
    T nu_NS = DESCRIPTOR<T>::cs2 * (nsLatticeTau - 0.5);
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 2: NS Kinematic Viscosity                                 │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ ν_NS = cs² × (τ_NS - 0.5) = " << nu_NS << "                        │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // STEP 3: LATTICE DIFFUSIVITY
    // ════════════════════════════════════════════════════════════════════════
    T D_lattice = BGK<T>::cs2 * (tau_ADE - 0.5);
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 3: Lattice Diffusivity                                    │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ D_lattice = cs² × (τ_ADE - 0.5) = " << D_lattice << "                  │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // Setup auxiliary domains
    MultiScalarField3D<int> distanceDomain(nx, ny, nz);
    distanceDomain = geometry;
    std::vector<std::vector<std::vector<plint>>> distVec(
        nx, std::vector<std::vector<plint>>(ny, std::vector<plint>(nz)));
    calculateDistanceFromSolid3D(distanceDomain, no_dynamics, bounce_back, distVec);
    applyProcessingFunctional(new createDistanceDomain3D<int>(distVec), 
        distanceDomain.getBoundingBox(), distanceDomain);

    MultiScalarField3D<int> ageDomain(nx, ny, nz);
    ageDomain = geometry;
    applyProcessingFunctional(new createAgeDomain3D<int>(pore_dynamics, bounce_back, no_dynamics), 
        ageDomain.getBoundingBox(), ageDomain);

    // ════════════════════════════════════════════════════════════════════════
    // STEP 4: FIRST NS SIMULATION
    // ════════════════════════════════════════════════════════════════════════
    T PoreMeanU = 0, PoreMaxUx = 0, DarcyOutletUx = 0, DarcyMiddleUx = 0, DarcyInletUx = 0;
    
    plint iT = 0;
    MultiBlockLattice3D<T, DESCRIPTOR> nsLattice(nx, ny, nz, 
        new IncBGKdynamics<T, DESCRIPTOR>(nsLatticeOmega));
    util::ValueTracer<T> ns_convg1(1.0, 1000.0, ns_converge_iT1);

    pcout << "Initializing the fluid lattice (deltaP = " << deltaP << ").\n";

    NSdomainSetup(nsLattice, createLocalBoundaryCondition3D<T, DESCRIPTOR>(), 
        geometry, deltaP, nsLatticeOmega, pore_dynamics, bounce_back, no_dynamics);

    global::timer("NS").start();

    if (Pe == 0) {
        pcout << "Peclet number is set to 0. Skipping lattice Boltzmann flow solver.\n";
    } else {
        pcout << "nsLatticeTau = " << nsLatticeTau << ", nsLatticeOmega = " << nsLatticeOmega 
              << ", nsLatticeNu = " << nu_NS << std::endl;
        pcout << "\n========== LBM NS simulation begins ==========\n";
    }

    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 4: First NS Simulation (with initial ΔP₀)                 │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n";

    if (Pe > thrd) {
        for (; iT < ns_maxiTer_1; ++iT) {
            nsLattice.collideAndStream();
            ns_convg1.takeValue(getStoredAverageEnergy(nsLattice), true);
            if (ns_convg1.hasConverged()) { break; }
        }
        pcout << "  NS converged at iteration " << iT << "\n";
    }

    PoreMeanU = computeAverage(*computeVelocityNorm(nsLattice, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1)));
    PoreMaxUx = computeMax(*computeVelocityComponent(nsLattice, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1), 0));
    DarcyOutletUx = computeAverage(*computeVelocityComponent(nsLattice, Box3D(nx - 2, nx - 2, 0, ny - 1, 0, nz - 1), 0));
    DarcyMiddleUx = computeAverage(*computeVelocityComponent(nsLattice, Box3D((nx - 1) / 2, (nx - 1) / 2, 0, ny - 1, 0, nz - 1), 0));
    DarcyInletUx = computeAverage(*computeVelocityComponent(nsLattice, Box3D(1, 1, 0, ny - 1, 0, nz - 1), 0));
    
    T u0 = DarcyOutletUx;

    // ════════════════════════════════════════════════════════════════════════
    // STEP 5: CALCULATE PERMEABILITY
    // ════════════════════════════════════════════════════════════════════════
    T permeability = u0 * nu_NS * L_lattice / deltaP;
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 5: Calculate Permeability                                 │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ k = (u₀ × ν_NS × L) / ΔP₀                                      │\n";
    pcout << "│ k = " << permeability << " (lattice²)                              │\n";
    pcout << "│ k = " << permeability * dx * dx << " m²                              │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // STEP 6: TARGET VELOCITY
    // ════════════════════════════════════════════════════════════════════════
    T u_target = Pe * D_lattice / L_lattice;
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 6: Target Velocity                                        │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ u_target = (Pe × D_lattice) / L = " << u_target << "                │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // STEP 7: CORRECTED PRESSURE GRADIENT
    // ════════════════════════════════════════════════════════════════════════
    T deltaP_new = u_target * nu_NS * L_lattice / permeability;
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 7: Corrected Pressure Gradient                            │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ ΔP_new = " << deltaP_new << "                                       │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // STEP 8: SECOND NS SIMULATION
    // ════════════════════════════════════════════════════════════════════════
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 8: Second NS Simulation (with corrected ΔP_new)           │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n";
    
    NSdomainSetup(nsLattice, createLocalBoundaryCondition3D<T, DESCRIPTOR>(), 
        geometry, deltaP_new, nsLatticeOmega, pore_dynamics, bounce_back, no_dynamics);
    
    ns_convg1.resetValues();
    for (iT = 0; iT < ns_maxiTer_1; ++iT) {
        nsLattice.collideAndStream();
        ns_convg1.takeValue(getStoredAverageEnergy(nsLattice), true);
        if (ns_convg1.hasConverged()) { break; }
    }
    pcout << "  NS converged at iteration " << iT << "\n\n";
    
    // ════════════════════════════════════════════════════════════════════════
    // STEP 9: EXTRACT VELOCITIES
    // ════════════════════════════════════════════════════════════════════════
    PoreMeanU = computeAverage(*computeVelocityNorm(nsLattice, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1)));
    PoreMaxUx = computeMax(*computeVelocityComponent(nsLattice, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1), 0));
    DarcyOutletUx = computeAverage(*computeVelocityComponent(nsLattice, Box3D(nx - 2, nx - 2, 0, ny - 1, 0, nz - 1), 0));
    DarcyMiddleUx = computeAverage(*computeVelocityComponent(nsLattice, Box3D((nx - 1) / 2, (nx - 1) / 2, 0, ny - 1, 0, nz - 1), 0));
    DarcyInletUx = computeAverage(*computeVelocityComponent(nsLattice, Box3D(1, 1, 0, ny - 1, 0, nz - 1), 0));
    
    T u_final = DarcyOutletUx;
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 9: Extract Velocities                                     │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ u_Darcy (outlet) = " << u_final << "                                │\n";
    pcout << "│ u_max (pore)     = " << PoreMaxUx << "                                │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // STEP 10: STABILITY CHECKS
    // ════════════════════════════════════════════════════════════════════════
    StabilityReport stability = performStabilityChecks(PoreMaxUx, nsLatticeTau, tau_ADE, D_lattice);
    printStabilityReport(stability);
    
    if (!stability.Ma_ok) {
        pcout << "CRITICAL ERROR: Mach number too high. Terminating.\n";
        return -1;
    }
    
    if (!stability.tau_NS_ok) {
        pcout << "CRITICAL ERROR: τ_NS out of stable range. Terminating.\n";
        return -1;
    }
    
    // Additional NS stabilization
    ns_convg1.resetValues();
    for (plint iT2 = 0; iT2 < ns_maxiTer_1; ++iT2) {
        nsLattice.collideAndStream();
        ns_convg1.takeValue(getStoredAverageEnergy(nsLattice), false);
        if (ns_convg1.hasConverged()) { break; }
    }
    
    if (track_performance == 0) {
        writeNsVTI(nsLattice, ns_maxiTer_1, "nsLatticeFinal_");
        saveBinaryBlock(nsLattice, str_outputDir + ns_filename + ".chk");
    }
    
    global::timer("NS").stop();
    T nstime = global::timer("NS").getTime();

    // ════════════════════════════════════════════════════════════════════════
    // STEP 11: VERIFY PÉCLET NUMBER
    // ════════════════════════════════════════════════════════════════════════
    T Pe_achieved = u_final * L_lattice / D_lattice;
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 11: Verify Péclet Number                                  │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ Pe_achieved = " << Pe_achieved << "                                    │\n";
    pcout << "│ Pe_target   = " << Pe << "                                          │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // STEP 12: PHYSICAL TIME STEP
    // ════════════════════════════════════════════════════════════════════════
    T D_physical = vec_solute_poreD[0];
    T ade_dt = D_lattice * dx * dx / D_physical;
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ STEP 12: Physical Time Step                                    │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ Δt = " << ade_dt << " seconds per lattice step                  │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";
    
    // Physical Darcy velocity
    T u_Darcy_physical = u_final * dx / ade_dt;
    pcout << "  Physical Darcy velocity = " << u_Darcy_physical << " m/s\n";
    pcout << "  NS simulation time: " << nstime << " seconds\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // WRITE DOMAIN PROPERTIES CSV
    // ════════════════════════════════════════════════════════════════════════
    writeDomainProperties(
        str_outputDir + "Domain_properties.csv",
        nx, ny, nz, dx, ade_dt,
        porosity, permeability, permeability * dx * dx,
        u_final, u_Darcy_physical,
        Pe, Pe_achieved,
        nsLatticeTau, tau_ADE, nu_NS, D_lattice, L_lattice);

    // ════════════════════════════════════════════════════════════════════════
    // ADE LATTICE SETUP
    // ════════════════════════════════════════════════════════════════════════
    T refTau = tau_ADE;
    T refOmega = 1.0 / refTau;
    
    std::vector<T> substrNUinPore(num_of_substrates);
    std::vector<T> substrTAUinPore(num_of_substrates);
    std::vector<T> substrOMEGAinPore(num_of_substrates);
    
    for (plint iS = 0; iS < num_of_substrates; ++iS) {
        if (iS == 0) {
            substrNUinPore[iS] = D_lattice;
            substrTAUinPore[iS] = refTau;
            substrOMEGAinPore[iS] = refOmega;
        } else {
            substrNUinPore[iS] = substrNUinPore[0] * vec_solute_poreD[iS] / vec_solute_poreD[0];
            substrTAUinPore[iS] = substrNUinPore[iS] * BGK<T>::invCs2 + 0.5;
            substrOMEGAinPore[iS] = 1.0 / substrTAUinPore[iS];
        }
    }

    pcout << "Initializing substrate lattices (" << nx << "x" << ny << "x" << nz << ")..." << std::endl;
    MultiBlockLattice3D<T, BGK> substrLattice(nx, ny, nz, 
        new AdvectionDiffusionBGKdynamics<T, BGK>(refOmega));
    std::vector<MultiBlockLattice3D<T, BGK>> vec_substr_lattices(num_of_substrates, substrLattice);
    std::vector<MultiBlockLattice3D<T, BGK>> dC(num_of_substrates, substrLattice);
    std::vector<MultiBlockLattice3D<T, BGK>> dC0(num_of_substrates, substrLattice);

    for (plint iS = 0; iS < num_of_substrates; ++iS) {
        soluteDomainSetup(vec_substr_lattices[iS], 
            createLocalAdvectionDiffusionBoundaryCondition3D<T, BGK>(), 
            geometry, substrOMEGAinPore[iS], pore_dynamics, bounce_back, no_dynamics,
            vec_c0[iS], vec_left_btype[iS], vec_right_btype[iS], 
            vec_left_bcondition[iS], vec_right_bcondition[iS]);
        soluteDomainSetup(dC[iS], 
            createLocalAdvectionDiffusionBoundaryCondition3D<T, BGK>(), 
            geometry, substrOMEGAinPore[iS], pore_dynamics, bounce_back, no_dynamics,
            0., vec_left_btype[iS], vec_right_btype[iS], 
            vec_left_bcondition[iS], vec_right_bcondition[iS]);
    }
    dC0 = dC;

    MultiBlockLattice3D<T, BGK> maskLattice(nx, ny, nz, 
        new AdvectionDiffusionBGKdynamics<T, BGK>(0.));
    MultiBlockLattice3D<T, BGK> distLattice(nx, ny, nz, 
        new AdvectionDiffusionBGKdynamics<T, BGK>(0.));
    
    defineMaskLatticeDynamics(substrLattice, maskLattice, thrd);
    applyProcessingFunctional(new CopyGeometryScalar2maskLattice3D<T, BGK, int>(pore_dynamics), 
        maskLattice.getBoundingBox(), maskLattice, geometry);
    applyProcessingFunctional(new CopyGeometryScalar2distLattice3D<T, BGK, int>(), 
        distLattice.getBoundingBox(), distLattice, distanceDomain);

    std::vector<MultiBlockLattice3D<T, BGK>*> ptr_kns_lattices;
    for (plint iS = 0; iS < num_of_substrates; ++iS) {
        ptr_kns_lattices.push_back(&vec_substr_lattices[iS]);
    }
    for (plint iS = 0; iS < num_of_substrates; ++iS) {
        ptr_kns_lattices.push_back(&dC[iS]);
    }
    ptr_kns_lattices.push_back(&maskLattice);

    // Couple NS velocity to ADE
    if (Pe > thrd) {
        pcout << "Coupling NS velocity field to ADE lattices..." << std::endl;
        for (plint iS = 0; iS < num_of_substrates; ++iS) {
            latticeToPassiveAdvDiff(nsLattice, vec_substr_lattices[iS], 
                vec_substr_lattices[iS].getBoundingBox());
        }
        
        pcout << "Stabilizing ADE lattices (10000 iterations)..." << std::endl;
        for (plint stab_iT = 0; stab_iT < 10000; ++stab_iT) {
            for (plint iS = 0; iS < num_of_substrates; ++iS) {
                vec_substr_lattices[iS].collideAndStream();
            }
        }
        
        for (plint iS = 0; iS < num_of_substrates; ++iS) {
            applyProcessingFunctional(
                new stabilizeADElattice<T, BGK, int>(vec_c0[iS], pore_dynamics), 
                vec_substr_lattices[iS].getBoundingBox(), vec_substr_lattices[iS], geometry);
        }
        pcout << "ADE stabilization complete." << std::endl;
    }

    // Save initial VTI files
    if (track_performance == 0) {
        pcout << "Saving initial VTI files..." << std::endl;
        if (Pe > thrd) {
            writeNsVTI(nsLattice, 0, "nsLattice_initial_");
        }
        for (plint iS = 0; iS < num_of_substrates; ++iS) {
            writeAdvVTI(vec_substr_lattices[iS], 0, ade_filename + std::to_string(iS) + "_");
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // STEP 13: ADE MAIN SIMULATION LOOP WITH ADAPTIVE STOPPING
    // ════════════════════════════════════════════════════════════════════════
    pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║              ADE SIMULATION WITH ADAPTIVE STOPPING              ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ Stop condition: C_flux >= 0.95 for 3 consecutive checks         ║\n";
    pcout << "║ VTI saved at: Initial, Early (5%), BT (50%), Near (90%), Final  ║\n";
    pcout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
    
    // Adaptive stopping parameters
    T C_threshold = 0.95;           // Stop when C_flux >= this
    plint stable_count_needed = 3;  // Must stay above for this many checks
    plint stable_count = 0;
    bool btc_complete = false;
    
    // Key moment thresholds for VTI saving
    T threshold_early = 0.05;       // Early time / first arrival
    T threshold_BT = 0.50;          // Breakthrough (mid-point)
    T threshold_near = 0.90;        // Near complete
    
    // Flags to track which key moments have been saved
    bool saved_early = false;
    bool saved_BT = false;
    bool saved_near = false;
    
    // ════════════════════════════════════════════════════════════════════════
    // BTC RECORDING INTERVAL (separate from VTI saving)
    // ════════════════════════════════════════════════════════════════════════
    // Record BTC frequently for smooth curves & accurate M1/CV
    // VTI files saved only at key milestones
    // ────────────────────────────────────────────────────────────────────────
    plint btc_record_interval = ade_VTK_iTer / 10;  // 10x more points than console output
    if (btc_record_interval < 100) btc_record_interval = 100;  // Minimum 100 iterations
    
    pcout << "┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ BTC RECORDING SETTINGS                                         │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ BTC record interval:  " << btc_record_interval << " iterations (for CSV)             │\n";
    pcout << "│ Console/VTI interval: " << ade_VTK_iTer << " iterations                       │\n";
    pcout << "│ Expected BTC points:  ~100-200 (for smooth curve)              │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n\n";
    
    // BTC data storage
    std::vector<BTCDataPoint> btc_data;
    
    global::timer("ade").restart();
    iT = 0;
    
    // Record initial BTC point
    T C_flux_latest = calculateFluxWeightedConcentration(vec_substr_lattices[0], nsLattice, geometry, nx, ny, nz, pore_dynamics);
    T C_avg_latest = calculateAverageConcentration(vec_substr_lattices[0], geometry, nx, ny, nz, pore_dynamics);
    {
        BTCDataPoint pt;
        pt.iteration = 0;
        pt.time_seconds = 0;
        pt.time_lattice = 0;
        pt.C_average = C_avg_latest;
        pt.C_flux_weighted = C_flux_latest;
        pt.pore_volume = 0;
        btc_data.push_back(pt);
    }
    
    // Main ADE loop
    for (; iT < ade_maxiTer; ++iT) {
        
        // ════════════════════════════════════════════════════════════════
        // RECORD BTC DATA (frequent - for CSV, smooth plots, accurate M1)
        // ════════════════════════════════════════════════════════════════
        if (iT > 0 && iT % btc_record_interval == 0) {
            C_avg_latest = calculateAverageConcentration(vec_substr_lattices[0], geometry, nx, ny, nz, pore_dynamics);
            C_flux_latest = calculateFluxWeightedConcentration(vec_substr_lattices[0], nsLattice, geometry, nx, ny, nz, pore_dynamics);
            T time_s = iT * ade_dt;
            T pore_vol = calculatePoreVolume(u_final, (T)iT, L_lattice, porosity);
            
            BTCDataPoint pt;
            pt.iteration = iT;
            pt.time_seconds = time_s;
            pt.time_lattice = (T)iT;
            pt.C_average = C_avg_latest;
            pt.C_flux_weighted = C_flux_latest;
            pt.pore_volume = pore_vol;
            btc_data.push_back(pt);
        }
        
        // ════════════════════════════════════════════════════════════════
        // CONSOLE OUTPUT & VTI MILESTONES (less frequent)
        // ════════════════════════════════════════════════════════════════
        if (ade_VTK_iTer > 0 && iT % ade_VTK_iTer == 0 && iT > 0) {
            
            T time_s = iT * ade_dt;
            T pore_vol = calculatePoreVolume(u_final, (T)iT, L_lattice, porosity);
            
            pcout << "iT=" << iT << " | t=" << std::scientific << std::setprecision(3) << time_s 
                  << "s | PV=" << std::fixed << std::setprecision(3) << pore_vol 
                  << " | C_flux=" << std::setprecision(4) << C_flux_latest
                  << " | pts=" << btc_data.size();
            
            // ════════════════════════════════════════════════════════════════
            // SAVE VTI AT KEY MILESTONES ONLY
            // ════════════════════════════════════════════════════════════════
            
            // Early time (C_flux crosses 0.05)
            if (!saved_early && C_flux_latest >= threshold_early && track_performance == 0) {
                pcout << " | 📸 EARLY";
                for (plint iS = 0; iS < num_of_substrates; ++iS) {
                    writeAdvVTI(vec_substr_lattices[iS], iT, ade_filename + std::to_string(iS) + "_early_");
                }
                saved_early = true;
            }
            
            // Breakthrough (C_flux crosses 0.50)
            if (!saved_BT && C_flux_latest >= threshold_BT && track_performance == 0) {
                pcout << " | 📸 BT";
                for (plint iS = 0; iS < num_of_substrates; ++iS) {
                    writeAdvVTI(vec_substr_lattices[iS], iT, ade_filename + std::to_string(iS) + "_BT_");
                }
                saved_BT = true;
            }
            
            // Near complete (C_flux crosses 0.90)
            if (!saved_near && C_flux_latest >= threshold_near && track_performance == 0) {
                pcout << " | 📸 NEAR";
                for (plint iS = 0; iS < num_of_substrates; ++iS) {
                    writeAdvVTI(vec_substr_lattices[iS], iT, ade_filename + std::to_string(iS) + "_near_");
                }
                saved_near = true;
            }
            
            // Check for breakthrough completion (adaptive stopping)
            if (C_flux_latest >= C_threshold) {
                stable_count++;
                pcout << " | ✓ (" << stable_count << "/" << stable_count_needed << ")";
                
                if (stable_count >= stable_count_needed) {
                    pcout << "\n\n╔══════════════════════════════════════════════════════════════════╗\n";
                    pcout << "║ ✓ BREAKTHROUGH COMPLETE - STOPPING SIMULATION                   ║\n";
                    pcout << "║   C_flux = " << C_flux_latest << " >= " << C_threshold << "                                ║\n";
                    pcout << "║   BTC data points: " << btc_data.size() << " (for smooth curve & accurate M1)   ║\n";
                    pcout << "╚══════════════════════════════════════════════════════════════════╝\n";
                    btc_complete = true;
                    ++iT;
                    break;
                }
            } else {
                stable_count = 0;
            }
            
            pcout << std::endl;
        }
        
        // Checkpoint saving (for restart capability)
        if (ade_CHK_iTer > 0 && iT % ade_CHK_iTer == 0 && iT > 0 && track_performance == 0) {
            for (plint iS = 0; iS < num_of_substrates; ++iS) {
                saveBinaryBlock(vec_substr_lattices[iS], 
                    str_outputDir + ade_filename + std::to_string(iS) + "_" + std::to_string(iT) + ".chk");
            }
        }
        
        // Collision
        for (plint iS = 0; iS < num_of_substrates; ++iS) {
            vec_substr_lattices[iS].collide();
        }
        
        // Kinetics
        dC = dC0;
        applyProcessingFunctional(
            new run_kinetics<T, BGK>(nx, num_of_substrates, ade_dt, 
                vec_Kc_kns, no_dynamics, bounce_back),
            vec_substr_lattices[0].getBoundingBox(), ptr_kns_lattices);
        
        // Streaming
        for (plint iS = 0; iS < num_of_substrates; ++iS) {
            vec_substr_lattices[iS].stream();
        }
    }
    
    global::timer("ade").stop();
    T adetime = global::timer("ade").getTime();

    // ════════════════════════════════════════════════════════════════════════
    // SAVE FINAL VTI FILES
    // ════════════════════════════════════════════════════════════════════════
    if (track_performance == 0) {
        pcout << "\n  Saving final VTI files...\n";
        
        // Final concentration field
        for (plint iS = 0; iS < num_of_substrates; ++iS) {
            writeAdvVTI(vec_substr_lattices[iS], iT, ade_filename + std::to_string(iS) + "_final_");
            saveBinaryBlock(vec_substr_lattices[iS], 
                str_outputDir + ade_filename + std::to_string(iS) + "_final.chk");
        }
        
        // Final mask (optional)
        saveBinaryBlock(maskLattice, str_outputDir + mask_filename + "_final.chk");
        
        // NS checkpoint (velocity field already saved as nsLatticeFinal)
        if (Pe > thrd) {
            saveBinaryBlock(nsLattice, str_outputDir + ns_filename + "_final.chk");
        }
    }
    
    // Print VTI files summary
    pcout << "\n┌────────────────────────────────────────────────────────────────┐\n";
    pcout << "│ VTI FILES SAVED AT KEY MOMENTS                                 │\n";
    pcout << "├────────────────────────────────────────────────────────────────┤\n";
    pcout << "│ ✓ inputGeom.vti           - Porous medium geometry             │\n";
    pcout << "│ ✓ nsLatticeFinal_*.vti    - Flow velocity field                │\n";
    if (saved_early) {
        pcout << "│ ✓ subsLattice0_early_*.vti - Concentration at C≈0.05          │\n";
    }
    if (saved_BT) {
        pcout << "│ ✓ subsLattice0_BT_*.vti    - Concentration at C≈0.50          │\n";
    }
    if (saved_near) {
        pcout << "│ ✓ subsLattice0_near_*.vti  - Concentration at C≈0.95          │\n";
    }
    pcout << "│ ✓ subsLattice0_final_*.vti - Concentration at end              │\n";
    pcout << "└────────────────────────────────────────────────────────────────┘\n";

    // ════════════════════════════════════════════════════════════════════════
    // WRITE BTC TIMESERIES CSV
    // ════════════════════════════════════════════════════════════════════════
    writeBTCTimeseries(
        str_outputDir + "BTC_timeseries.csv",
        btc_data,
        porosity, u_final, u_Darcy_physical,
        permeability, permeability * dx * dx);

    // ════════════════════════════════════════════════════════════════════════
    // CALCULATE AND WRITE MOMENTS
    // ════════════════════════════════════════════════════════════════════════
    T M0, M1, M2_central, std_dev, cv;
    calculateMoments(btc_data, M0, M1, M2_central, std_dev, cv);
    
    // Convert M1 and sigma to lattice units
    T M1_lattice = M1 / ade_dt;
    T sigma_lattice = std_dev / ade_dt;
    
    writeMomentsSummary(
        str_outputDir + "Moments_summary.csv",
        M0, M1, M1_lattice,
        M2_central, std_dev, sigma_lattice, cv,
        porosity, Pe_achieved,
        permeability * dx * dx, u_Darcy_physical);

    // ════════════════════════════════════════════════════════════════════════
    // FINAL SUMMARY
    // ════════════════════════════════════════════════════════════════════════
    T TET = global::timer("total").getTime();
    global::timer("total").stop();

    pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║                      FINAL SUMMARY                               ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ TIMING:                                                          ║\n";
    pcout << "║   Total time: " << TET << " s (" << TET/60 << " min)\n";
    pcout << "║   NS time:    " << nstime << " s\n";
    pcout << "║   ADE time:   " << adetime << " s\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ ITERATIONS:                                                      ║\n";
    pcout << "║   Completed:  " << iT << " iterations\n";
    pcout << "║   BTC complete: " << (btc_complete ? "YES" : "NO (max iterations reached)") << "\n";
    pcout << "║   Simulated time: " << iT * ade_dt << " s\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ KEY RESULTS:                                                     ║\n";
    pcout << "║   Porosity φ = " << porosity << "\n";
    pcout << "║   Permeability k = " << permeability * dx * dx << " m²\n";
    pcout << "║   Darcy velocity = " << u_Darcy_physical << " m/s\n";
    pcout << "║   Pe_achieved = " << Pe_achieved << "\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ MOMENTS (from BTC):                                              ║\n";
    pcout << "║   M₀ = " << M0 << "\n";
    pcout << "║   M₁ = " << M1 << " s (mean arrival time)\n";
    pcout << "║   σ  = " << std_dev << " s (std deviation)\n";
    pcout << "║   CV = " << cv << " (coefficient of variation)\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ OUTPUT FILES:                                                    ║\n";
    pcout << "║   CSV:                                                           ║\n";
    pcout << "║     - BTC_timeseries.csv                                         ║\n";
    pcout << "║     - Domain_properties.csv                                      ║\n";
    pcout << "║     - Moments_summary.csv                                        ║\n";
    pcout << "║   VTI (key moments only):                                        ║\n";
    pcout << "║     - inputGeom.vti (geometry)                                   ║\n";
    pcout << "║     - nsLatticeFinal_*.vti (velocity)                            ║\n";
    pcout << "║     - subsLattice0_early/BT/near/final_*.vti (concentration)     ║\n";
    pcout << "╚══════════════════════════════════════════════════════════════════╝\n";
    pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║                    Simulation Finished!                          ║\n";
    pcout << "║                                                                  ║\n";
    pcout << "║  Author:  Shahram Asgari                                         ║\n";
    pcout << "║  Advisor: Dr. Christof Meile                                     ║\n";
    pcout << "║  Lab:     Meile Lab, University of Georgia                       ║\n";
    pcout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    free(main_path);
    free(src_path);
    free(input_path);
    free(output_path);
    free(ns_filename);

    return 0;
}

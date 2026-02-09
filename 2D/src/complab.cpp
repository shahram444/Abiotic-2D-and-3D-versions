/* ============================================================================
 * CompLaB2D - Two-Dimensional Lattice Boltzmann Reactive Transport Solver
 * ============================================================================
 *
 * Author:      Shahram Asgari
 * Advisor:     Dr. Christof Meile
 * Laboratory:  Meile Lab
 * Institution: University of Georgia (UGA)
 *
 * ============================================================================
 * BATCH PROCESSING VERSION (WITH FULL DIAGNOSTICS)
 * ============================================================================
 * Features:
 *   - Processes ALL .dat files in input directory automatically
 *   - Creates separate output folder for each geometry
 *   - Generates comprehensive summary CSV with all results
 *   - Skips failed simulations and reports failures at end
 *   - CSV files named with geometry name and Pe number
 *   - Adaptive stopping based on C_flux threshold
 *   - VTI files saved only at key milestones (5%, 50%, 90%, final)
 *   - FULL DIAGNOSTIC PRINTS restored for debugging
 *
 * ============================================================================
 * CALCULATION FLOW (13 Steps):
 * ─────────────────────────────────────────────────────────────────────────────
 * STEP 1:  L = nx - 1 (domain length)              [Matches PressureGradient]
 * STEP 2:  ν_NS = (1/3) × (τ_NS - 0.5)             [NS viscosity, D2Q9]
 * STEP 3:  D_lattice = (1/3) × (τ_ADE - 0.5)       [ADE diffusivity, D2Q5]
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
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * OUTPUT FILES (per geometry):
 *   - {geom_name}_Pe{Pe}/BTC_{geom_name}_Pe{Pe}.csv
 *   - {geom_name}_Pe{Pe}/Domain_{geom_name}_Pe{Pe}.csv
 *   - {geom_name}_Pe{Pe}/Moments_{geom_name}_Pe{Pe}.csv
 *   - {geom_name}_Pe{Pe}/ *.vti (key milestones only)
 *
 * SUMMARY OUTPUT:
 *   - BATCH_SUMMARY.csv (all geometries, all results)
 *   - FAILED_SIMULATIONS.txt (list of failures with reasons)
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
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

// NOTE: The following are already defined in complab_functions.hh:
//   typedef double T;
//   #define DESCRIPTOR descriptors::D2Q9Descriptor
//   #define BGK descriptors::AdvectionDiffusionD2Q5Descriptor
//   #define thrd 1e-14

// ============================================================================
// STRUCTURE: Simulation Result (for batch summary)
// ============================================================================
struct SimulationResult {
    std::string geom_name;
    bool success;
    std::string failure_reason;

    // Domain properties
    plint nx, ny;
    T dx;
    T porosity;
    T permeability_lattice;
    T permeability_m2;
    T u_Darcy_lattice;
    T u_Darcy_m_s;

    // Simulation parameters
    T Pe_target;
    T Pe_achieved;
    T tau_NS;
    T tau_ADE;
    T dt_seconds;

    // BTC results
    T M0;
    T M1_seconds;
    T M2_central;
    T sigma_seconds;
    T CV;

    // Timing
    T ns_time_seconds;
    T ade_time_seconds;
    T total_time_seconds;
    plint total_iterations;

    // Initialize with defaults
    SimulationResult() : success(false), nx(0), ny(0), dx(0), porosity(0),
        permeability_lattice(0), permeability_m2(0), u_Darcy_lattice(0), u_Darcy_m_s(0),
        Pe_target(0), Pe_achieved(0), tau_NS(0), tau_ADE(0), dt_seconds(0),
        M0(0), M1_seconds(0), M2_central(0), sigma_seconds(0), CV(0),
        ns_time_seconds(0), ade_time_seconds(0), total_time_seconds(0), total_iterations(0) {}
};

// ============================================================================
// STRUCTURE: BTC Data Point
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
// STRUCTURE: Stability Report
// ============================================================================
struct StabilityReport {
    T Ma;
    bool Ma_ok;
    bool Ma_warning;
    T CFL;
    bool CFL_ok;
    T tau_NS;
    T tau_ADE;
    bool tau_NS_ok;
    bool tau_ADE_ok;
    T Pe_grid;
    bool Pe_grid_ok;
    bool all_ok;
    bool has_warnings;
};

// ============================================================================
// FUNCTION: Get all .dat files from directory
// ============================================================================
std::vector<std::string> getDatFiles(const std::string& directory) {
    std::vector<std::string> datFiles;
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        pcout << "ERROR: Cannot open directory: " << directory << std::endl;
        return datFiles;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".dat") {
            datFiles.push_back(filename);
        }
    }
    closedir(dir);
    std::sort(datFiles.begin(), datFiles.end());
    return datFiles;
}

// ============================================================================
// FUNCTION: Get base name (remove .dat extension)
// ============================================================================
std::string getBaseName(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

// ============================================================================
// FUNCTION: Create output folder name with Pe
// ============================================================================
std::string createOutputFolderName(const std::string& baseName, T Pe) {
    std::ostringstream oss;
    oss << baseName << "_Pe" << std::fixed << std::setprecision(0) << Pe;
    return oss.str();
}

// ============================================================================
// FUNCTION: performStabilityChecks
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
T calculatePorosity(
    MultiScalarField2D<int>& geometry,
    plint nx, plint ny,
    std::vector<plint>& pore_dynamics)
{
    plint total_cells = nx * ny;
    plint pore_cells = 0;

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            plint geom_tag = geometry.get(iX, iY);

            for (size_t p = 0; p < pore_dynamics.size(); ++p) {
                if (geom_tag == pore_dynamics[p]) {
                    pore_cells++;
                    break;
                }
            }
        }
    }

    return (T)pore_cells / (T)total_cells;
}

// ============================================================================
// FUNCTION: calculateFluxWeightedConcentration
// ============================================================================
T calculateFluxWeightedConcentration(
    MultiBlockLattice2D<T, BGK>& adeLattice,
    MultiBlockLattice2D<T, DESCRIPTOR>& nsLattice,
    MultiScalarField2D<int>& geometry,
    plint nx, plint ny,
    std::vector<plint>& pore_dynamics)
{
    plint outletX = nx - 2;

    T sum_C_times_u = 0.0;
    T sum_u = 0.0;

    for (plint iY = 0; iY < ny; ++iY) {
        plint geom_tag = geometry.get(outletX, iY);

        bool is_pore = false;
        for (size_t p = 0; p < pore_dynamics.size(); ++p) {
            if (geom_tag == pore_dynamics[p]) {
                is_pore = true;
                break;
            }
        }

        if (is_pore) {
            T C = adeLattice.get(outletX, iY).computeDensity();
            C = std::min(C, (T)1.0);

            Array<T, 2> vel;
            nsLattice.get(outletX, iY).computeVelocity(vel);
            T u = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1]);

            sum_C_times_u += C * u;
            sum_u += u;
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
T calculateAverageConcentration(
    MultiBlockLattice2D<T, BGK>& adeLattice,
    MultiScalarField2D<int>& geometry,
    plint nx, plint ny,
    std::vector<plint>& pore_dynamics)
{
    plint outletX = nx - 2;

    T sum_C = 0.0;
    plint count = 0;

    for (plint iY = 0; iY < ny; ++iY) {
        plint geom_tag = geometry.get(outletX, iY);

        bool is_pore = false;
        for (size_t p = 0; p < pore_dynamics.size(); ++p) {
            if (geom_tag == pore_dynamics[p]) {
                is_pore = true;
                break;
            }
        }

        if (is_pore) {
            T C = adeLattice.get(outletX, iY).computeDensity();
            C = std::min(C, (T)1.0);
            sum_C += C;
            count++;
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
void calculateMoments(
    const std::vector<BTCDataPoint>& btc_data,
    T& M0, T& M1, T& M2_central, T& std_dev, T& cv)
{
    size_t n = btc_data.size();
    if (n < 2) {
        M0 = 0; M1 = 0; M2_central = 0; std_dev = 0; cv = 0;
        return;
    }

    std::vector<T> time(n);
    std::vector<T> conc(n);

    for (size_t i = 0; i < n; ++i) {
        time[i] = btc_data[i].time_seconds;
        conc[i] = btc_data[i].C_flux_weighted;
    }

    // M0: Zeroth moment using trapezoidal rule
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
    T M1_raw = 0.0;
    for (size_t i = 0; i < n - 1; ++i) {
        T dt = time[i+1] - time[i];
        T integrand_i = time[i] * conc[i];
        T integrand_ip1 = time[i+1] * conc[i+1];
        M1_raw += 0.5 * (integrand_i + integrand_ip1) * dt;
    }
    M1 = M1_raw / M0;

    // M2: Second moment
    T M2_raw = 0.0;
    for (size_t i = 0; i < n - 1; ++i) {
        T dt = time[i+1] - time[i];
        T integrand_i = time[i] * time[i] * conc[i];
        T integrand_ip1 = time[i+1] * time[i+1] * conc[i+1];
        M2_raw += 0.5 * (integrand_i + integrand_ip1) * dt;
    }
    T M2_normalized = M2_raw / M0;
    M2_central = std::max((T)0.0, M2_normalized - M1 * M1);

    std_dev = std::sqrt(M2_central);

    if (M1 > thrd) {
        cv = std_dev / M1;
    } else {
        cv = 0.0;
    }
}

// ============================================================================
// FUNCTION: writeBTCTimeseries (with geometry name in filename)
// ============================================================================
void writeBTCTimeseries(
    const std::string& outputDir,
    const std::string& geomName,
    T Pe,
    const std::vector<BTCDataPoint>& btc_data,
    T porosity, T u_Darcy_lattice, T u_Darcy_physical,
    T permeability_lattice, T permeability_m2)
{
    std::ostringstream filename;
    filename << outputDir << "BTC_" << geomName << "_Pe" << std::fixed << std::setprecision(0) << Pe << ".csv";

    std::ofstream file(filename.str());
    if (!file.is_open()) {
        pcout << "ERROR: Could not open file " << filename.str() << " for writing.\n";
        return;
    }

    file << "Iteration,Time_seconds,Time_lattice,C_average,C_flux_weighted,Pore_Volume,"
         << "Porosity,u_Darcy_lattice,u_Darcy_m_s,Permeability_lattice,Permeability_m2\n";

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
    pcout << "  ✓ Saved: " << filename.str() << "\n";
}

// ============================================================================
// FUNCTION: writeDomainProperties (with geometry name in filename)
// ============================================================================
void writeDomainProperties(
    const std::string& outputDir,
    const std::string& geomName,
    T Pe,
    plint nx, plint ny, T dx, T dt,
    T porosity, T permeability_lattice, T permeability_m2,
    T u_Darcy_lattice, T u_Darcy_physical,
    T Pe_target, T Pe_achieved,
    T tau_NS, T tau_ADE, T nu_NS, T D_lattice, T L_domain)
{
    std::ostringstream filename;
    filename << outputDir << "Domain_" << geomName << "_Pe" << std::fixed << std::setprecision(0) << Pe << ".csv";

    std::ofstream file(filename.str());
    if (!file.is_open()) {
        pcout << "ERROR: Could not open file " << filename.str() << " for writing.\n";
        return;
    }

    file << "Geometry,nx,ny,dx_m,dt_s,Porosity,Permeability_lattice,Permeability_m2,"
         << "u_Darcy_lattice,u_Darcy_m_s,Pe_target,Pe_achieved,"
         << "tau_NS,tau_ADE,nu_NS,D_lattice,L_domain\n";

    file << std::scientific << std::setprecision(8);
    file << geomName << ","
         << nx << "," << ny << ","
         << dx << "," << dt << ","
         << porosity << ","
         << permeability_lattice << "," << permeability_m2 << ","
         << u_Darcy_lattice << "," << u_Darcy_physical << ","
         << Pe_target << "," << Pe_achieved << ","
         << tau_NS << "," << tau_ADE << ","
         << nu_NS << "," << D_lattice << "," << L_domain << "\n";

    file.close();
    pcout << "  ✓ Saved: " << filename.str() << "\n";
}

// ============================================================================
// FUNCTION: writeMomentsSummary (with geometry name in filename)
// ============================================================================
void writeMomentsSummary(
    const std::string& outputDir,
    const std::string& geomName,
    T Pe,
    T M0, T M1_seconds, T M1_lattice,
    T M2_central, T sigma_seconds, T sigma_lattice, T cv,
    T porosity, T Pe_achieved,
    T permeability_m2, T u_Darcy_physical)
{
    std::ostringstream filename;
    filename << outputDir << "Moments_" << geomName << "_Pe" << std::fixed << std::setprecision(0) << Pe << ".csv";

    std::ofstream file(filename.str());
    if (!file.is_open()) {
        pcout << "ERROR: Could not open file " << filename.str() << " for writing.\n";
        return;
    }

    file << "Geometry,M0,M1_seconds,M1_lattice,M2_central,Sigma_seconds,Sigma_lattice,CV,"
         << "Porosity,Pe_achieved,Permeability_m2,u_Darcy_m_s\n";

    file << std::scientific << std::setprecision(8);
    file << geomName << ","
         << M0 << ","
         << M1_seconds << "," << M1_lattice << ","
         << M2_central << ","
         << sigma_seconds << "," << sigma_lattice << ","
         << cv << ","
         << porosity << "," << Pe_achieved << ","
         << permeability_m2 << "," << u_Darcy_physical << "\n";

    file.close();
    pcout << "  ✓ Saved: " << filename.str() << "\n";
}

// ============================================================================
// FUNCTION: writeBatchSummary (comprehensive summary of ALL simulations)
// ============================================================================
void writeBatchSummary(
    const std::string& outputDir,
    const std::vector<SimulationResult>& results)
{
    std::string filename = outputDir + "BATCH_SUMMARY.csv";

    std::ofstream file(filename);
    if (!file.is_open()) {
        pcout << "ERROR: Could not open file " << filename << " for writing.\n";
        return;
    }

    // Header
    file << "Geometry,Success,Failure_Reason,"
         << "nx,ny,dx_m,Porosity,"
         << "Permeability_lattice,Permeability_m2,"
         << "u_Darcy_lattice,u_Darcy_m_s,"
         << "Pe_target,Pe_achieved,tau_NS,tau_ADE,dt_s,"
         << "M0,M1_seconds,M2_central,Sigma_seconds,CV,"
         << "NS_time_s,ADE_time_s,Total_time_s,Total_iterations\n";

    file << std::scientific << std::setprecision(8);
    for (size_t i = 0; i < results.size(); ++i) {
        const SimulationResult& r = results[i];
        file << r.geom_name << ","
             << (r.success ? "YES" : "NO") << ","
             << "\"" << r.failure_reason << "\","
             << r.nx << "," << r.ny << "," << r.dx << ","
             << r.porosity << ","
             << r.permeability_lattice << "," << r.permeability_m2 << ","
             << r.u_Darcy_lattice << "," << r.u_Darcy_m_s << ","
             << r.Pe_target << "," << r.Pe_achieved << ","
             << r.tau_NS << "," << r.tau_ADE << "," << r.dt_seconds << ","
             << r.M0 << "," << r.M1_seconds << "," << r.M2_central << ","
             << r.sigma_seconds << "," << r.CV << ","
             << r.ns_time_seconds << "," << r.ade_time_seconds << ","
             << r.total_time_seconds << "," << r.total_iterations << "\n";
    }

    file.close();
    pcout << "\n  ✓ Saved batch summary: " << filename << "\n";
}

// ============================================================================
// FUNCTION: writeFailedSimulations (report of all failures)
// ============================================================================
void writeFailedSimulations(
    const std::string& outputDir,
    const std::vector<SimulationResult>& results)
{
    std::string filename = outputDir + "FAILED_SIMULATIONS.txt";

    std::ofstream file(filename);
    if (!file.is_open()) {
        pcout << "ERROR: Could not open file " << filename << " for writing.\n";
        return;
    }

    int failCount = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].success) {
            failCount++;
        }
    }

    file << "╔══════════════════════════════════════════════════════════════════╗\n";
    file << "║                    FAILED SIMULATIONS REPORT                     ║\n";
    file << "╠══════════════════════════════════════════════════════════════════╣\n";
    file << "║ Total simulations: " << results.size() << "\n";
    file << "║ Failed: " << failCount << "\n";
    file << "║ Success rate: " << std::fixed << std::setprecision(1)
         << (100.0 * (results.size() - failCount) / results.size()) << "%\n";
    file << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    if (failCount > 0) {
        file << "FAILED SIMULATIONS:\n";
        file << "───────────────────────────────────────────────────────────────────\n";
        for (size_t i = 0; i < results.size(); ++i) {
            if (!results[i].success) {
                file << "  " << (i+1) << ". " << results[i].geom_name << "\n";
                file << "     Reason: " << results[i].failure_reason << "\n\n";
            }
        }
    } else {
        file << "✓ ALL SIMULATIONS COMPLETED SUCCESSFULLY!\n";
    }

    file.close();
    pcout << "  ✓ Saved failure report: " << filename << "\n";
}

// ============================================================================
//                              MAIN FUNCTION
// ============================================================================
int main(int argc, char** argv) {

    plbInit(&argc, &argv);
    global::timer("batch_total").start();
    ImageWriter<T> image("leeloo");

    // Variable declarations
    plint kns_count = 0, lb_count = 0;
    char* main_path = (char*)malloc(256 * sizeof(char));
    getcwd(main_path, 256 * sizeof(char));
    char* src_path = (char*)malloc(256 * sizeof(char));
    char* input_path = (char*)malloc(256 * sizeof(char));
    char* output_path = (char*)malloc(256 * sizeof(char));
    char* ns_filename = (char*)malloc(256 * sizeof(char));

    plint nx, ny, num_of_substrates;
    T dx, dy, deltaP, Pe, charcs_length;
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
    pcout << "║                         CompLaB2D                                ║\n";
    pcout << "║           Two-Dimensional Reactive Transport Solver              ║\n";
    pcout << "║                     BATCH PROCESSING MODE                        ║\n";
    pcout << "║                   (WITH FULL DIAGNOSTICS)                        ║\n";
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
            nx, ny, dx, dy, deltaP, tau,
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
    std::string str_outputDir_base = output_path;
    if (str_inputDir.back() != '/') { str_inputDir += "/"; }
    if (str_outputDir_base.back() != '/') { str_outputDir_base += "/"; }

    // Create base output directory
    struct stat statStruct;
    stat(output_path, &statStruct);
    if (!S_ISDIR(statStruct.st_mode)) { mkdir(output_path, 0777); }

    pcout << "CompLaB main directory = " << str_mainDir << std::endl;
    pcout << "CompLaB input directory = " << main_path << "/" << input_path << std::endl;
    pcout << "CompLaB output directory = " << main_path << "/" << output_path << std::endl << std::endl;

    // ========================================================================
    // FIND ALL .DAT FILES
    // ========================================================================
    std::vector<std::string> datFiles = getDatFiles(str_inputDir);

    if (datFiles.empty()) {
        pcout << "ERROR: No .dat files found in: " << str_inputDir << std::endl;
        return -1;
    }

    pcout << "╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║                    BATCH PROCESSING INFO                         ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ Found " << datFiles.size() << " .dat files to process                              ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    for (size_t i = 0; i < datFiles.size(); ++i) {
        pcout << "║   " << (i+1) << ". " << datFiles[i] << "\n";
    }
    pcout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // Storage for all simulation results
    std::vector<SimulationResult> allResults;

    // ========================================================================
    // MAIN BATCH LOOP
    // ========================================================================
    for (size_t fileIdx = 0; fileIdx < datFiles.size(); ++fileIdx) {

        std::string currentDatFile = datFiles[fileIdx];
        std::string baseName = getBaseName(currentDatFile);
        std::string folderName = createOutputFolderName(baseName, Pe);
        std::string str_outputDir = str_outputDir_base + folderName + "/";

        // Create output folder for this geometry
        mkdir(str_outputDir.c_str(), 0755);
        global::directories().setOutputDir(str_outputDir);

        SimulationResult result;
        result.geom_name = baseName;
        result.Pe_target = Pe;
        result.nx = nx;
        result.ny = ny;
        result.dx = dx;
        result.tau_NS = tau;
        result.tau_ADE = 0.8;  // FIXED value (OLD METHOD uses fixed tau_ADE)

        pcout << "\n";
        pcout << "╔══════════════════════════════════════════════════════════════════╗\n";
        pcout << "║ PROCESSING [" << (fileIdx+1) << "/" << datFiles.size() << "]: " << currentDatFile << "\n";
        pcout << "║ Output folder: " << folderName << "\n";
        pcout << "╚══════════════════════════════════════════════════════════════════╝\n";

        global::timer("sim").start();

        // ====================================================================
        // TRY-CATCH BLOCK FOR ERROR HANDLING
        // ====================================================================
        try {

            // Read geometry
            MultiScalarField2D<int> geometry(nx, ny);
            readGeometry(str_inputDir + currentDatFile, geometry);

            if (track_performance == 0) {
                saveGeometry("inputGeom", geometry);
            }

            // ═══════════════════════════════════════════════════════════════════
            // STEP 1: DOMAIN LENGTH
            // ═══════════════════════════════════════════════════════════════════
            // For flow/transport calculations: L = nx - 1 (distance between inlet and outlet)
            // For Pore Volume calculation (Python method): L = nx (total domain length)
            // ═══════════════════════════════════════════════════════════════════
            T L_lattice = (T)(nx - 1);  // For Pe, permeability calculations
            T L_for_PV = (T)nx;         // For Pore Volume calculation (matches Python: length = nx * dx)

            // NOTE: τ_ADE will be calculated in STEP 7 based on velocity and target Pe
            // (OLD METHOD - not fixed like in correct method)

            // ═══════════════════════════════════════════════════════════════════
            // CALCULATE POROSITY
            // ═══════════════════════════════════════════════════════════════════
            T porosity = calculatePorosity(geometry, nx, ny, pore_dynamics);
            result.porosity = porosity;

            // *** RESTORED PRINTS FROM SINGLE-FILE VERSION ***
            pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
            pcout << "║                    SIMULATION PARAMETERS                         ║\n";
            pcout << "║      (OLD METHOD - deltaP adjustment + τ_ADE adjustment)         ║\n";
            pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
            pcout << "║ Domain: " << nx << " × " << ny << " lattice units\n";
            pcout << "║ Grid spacing: dx = " << dx << " m\n";
            pcout << "║ Domain length L = " << L_lattice << " lattice units (nx - 1)\n";
            pcout << "║ charcs_length = " << charcs_length << " (from XML)\n";
            pcout << "║ Target Pe: " << Pe << "\n";
            pcout << "║ τ_NS (from XML): " << tau << "\n";
            pcout << "║ τ_ADE: Will be CALCULATED (refNu = PoreMeanU * charcs_length / Pe)\n";
            pcout << "║ D_physical: " << vec_solute_poreD[0] << " m²/s\n";
            pcout << "║ Initial deltaP (from XML): " << deltaP << "\n";
            pcout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

            pcout << "┌────────────────────────────────────────────────────────────────┐\n";
            pcout << "│ STEP 1: Domain Length                                          │\n";
            pcout << "├────────────────────────────────────────────────────────────────┤\n";
            pcout << "│ L = nx - 1 = " << nx << " - 1 = " << L_lattice << " lattice units\n";
            pcout << "│ Physical length: L × dx = " << L_lattice * dx * 1e6 << " um\n";
            pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

            pcout << "┌────────────────────────────────────────────────────────────────┐\n";
            pcout << "│ POROSITY CALCULATION                                           │\n";
            pcout << "├────────────────────────────────────────────────────────────────┤\n";
            pcout << "│ Porosity φ = " << porosity << "\n";
            pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

            if (porosity < 0.01) {
                throw std::runtime_error("Porosity too low (< 1%). Check geometry file.");
            }

            // ═══════════════════════════════════════════════════════════════════
            // STEP 2: NS KINEMATIC VISCOSITY
            // ═══════════════════════════════════════════════════════════════════
            T nsLatticeTau = tau;
            T nsLatticeOmega = 1.0 / nsLatticeTau;
            T nu_NS = DESCRIPTOR<T>::cs2 * (nsLatticeTau - 0.5);
            T nsNu = nu_NS;  // Alias for compatibility with computePermeability

            pcout << "┌────────────────────────────────────────────────────────────────┐\n";
            pcout << "│ STEP 2: NS Kinematic Viscosity                                 │\n";
            pcout << "├────────────────────────────────────────────────────────────────┤\n";
            pcout << "│ Kinematic Viscosity (nsNu) = " << nsNu << "\n";
            pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

            pcout << "┌────────────────────────────────────────────────────────────────┐\n";
            pcout << "│ STEP 3: OLD METHOD OVERVIEW                                    │\n";
            pcout << "├────────────────────────────────────────────────────────────────┤\n";
            pcout << "│ 1. First NS → get permeability (actually returns meanU)        │\n";
            pcout << "│ 2. deltaP = Pe × μ × D / (meanU × dx²)                         │\n";
            pcout << "│ 3. Second NS with adjusted deltaP                              │\n";
            pcout << "│ 4. refNu = PoreMeanU × charcs_length / Pe                      │\n";
            pcout << "│ 5. τ_ADE = refNu × invCs² + 0.5                                │\n";
            pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

            // Setup auxiliary domains
            MultiScalarField2D<int> distanceDomain(nx, ny);
            distanceDomain = geometry;
            std::vector<std::vector<plint>> distVec(
                nx, std::vector<plint>(ny));
            calculateDistanceFromSolid2D(distanceDomain, no_dynamics, bounce_back, distVec);
            applyProcessingFunctional(new createDistanceDomain2D<int>(distVec),
                distanceDomain.getBoundingBox(), distanceDomain);

            MultiScalarField2D<int> ageDomain(nx, ny);
            ageDomain = geometry;
            applyProcessingFunctional(new createAgeDomain2D<int>(pore_dynamics, bounce_back, no_dynamics),
                ageDomain.getBoundingBox(), ageDomain);

            // ═══════════════════════════════════════════════════════════════════
            // STEP 4: FIRST NS SIMULATION (to measure permeability)
            // ═══════════════════════════════════════════════════════════════════
            // OLD METHOD: Uses physical units for deltaP calculation
            // ═══════════════════════════════════════════════════════════════════
            T PoreMeanU = 0, PoreMaxUx = 0, DarcyOutletUx = 0, DarcyMiddleUx = 0, DarcyInletUx = 0;

            plint iT = 0;
            MultiBlockLattice2D<T, DESCRIPTOR> nsLattice(nx, ny,
                new IncBGKdynamics<T, DESCRIPTOR>(nsLatticeOmega));
            util::ValueTracer<T> ns_convg1(1.0, 1000.0, ns_converge_iT1);

            pcout << "Initializing the fluid lattice (deltaP = " << deltaP << ").\n";

            NSdomainSetup(nsLattice, createLocalBoundaryCondition2D<T, DESCRIPTOR>(),
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
            pcout << "│ STEP 4: First NS Simulation (with initial ΔP from XML)         │\n";
            pcout << "│ Purpose: Measure geometry permeability                         │\n";
            pcout << "└────────────────────────────────────────────────────────────────┘\n";

            if (Pe > thrd) {
                pcout << "Run a new simulation for nsLattice" << std::endl;
                for (; iT < ns_maxiTer_1; ++iT) {
                    nsLattice.collideAndStream();
                    ns_convg1.takeValue(getStoredAverageEnergy(nsLattice), true);
                    if (ns_convg1.hasConverged()) { break; }
                }

                // Check for divergence
                T energy_check = getStoredAverageEnergy(nsLattice);
                if (std::isnan(energy_check) || std::isinf(energy_check) || energy_check > 1e10) {
                    pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
                    pcout << "║ ✗ FLOW DIVERGENCE DETECTED IN FIRST NS SIMULATION!              ║\n";
                    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
                    pcout << "║ REASON: Initial deltaP from XML is too high for this geometry.  ║\n";
                    pcout << "║   ΔP (from XML) = " << deltaP << "\n";
                    pcout << "║ SOLUTIONS:                                                       ║\n";
                    pcout << "║   1. Reduce <deltaP> in XML file (try " << deltaP * 0.1 << ")\n";
                    pcout << "║   2. Increase <tau> in XML file (try " << nsLatticeTau + 0.2 << ")\n";
                    pcout << "╚══════════════════════════════════════════════════════════════════╝\n";
                    throw std::runtime_error("First NS simulation diverged - initial deltaP too high");
                }

                pcout << "  NS converged at iteration " << iT << "\n";
            }

            // ═══════════════════════════════════════════════════════════════════
            // STEP 5: CALCULATE INITIAL PERMEABILITY
            // ═══════════════════════════════════════════════════════════════════
            PoreMeanU = computeAverage(*computeVelocityNorm(nsLattice, Box2D(1, nx - 2, 0, ny - 1)));
            PoreMaxUx = computeMax(*computeVelocityComponent(nsLattice, Box2D(1, nx - 2, 0, ny - 1), 0));
            DarcyOutletUx = computeAverage(*computeVelocityComponent(nsLattice, Box2D(nx - 2, nx - 2, 0, ny - 1), 0));

            // Calculate permeability using computePermeability function
            T permeability = computePermeability(nsLattice, nu_NS, deltaP, nsLattice.getBoundingBox());
            pcout << "Initial Permeability: " << permeability << std::endl;

            // Convert permeability to metric units
            T permeabilityMetric = permeability * dx * dx;
            pcout << "Permeability in metric units: " << permeabilityMetric << " m^2" << std::endl;

            result.permeability_lattice = permeability;
            result.permeability_m2 = permeabilityMetric;

            // ═══════════════════════════════════════════════════════════════════
            // STEP 6-7: AUTOMATIC DELTA_P ADJUSTMENT
            // ═══════════════════════════════════════════════════════════════════
            //
            // OLD METHOD (currently active):
            //   deltaP = Pe * mu * D / (k * dx²)
            //   Uses physical units (mu=1e-3 Pa·s, D in m²/s, k in m²)
            //   Then τ_ADE is calculated from velocity: refNu = PoreMeanU * charcs_length / Pe
            //
            // ───────────────────────────────────────────────────────────────────
            // CORRECT METHOD (commented out for now):
            //   T tau_ADE = 0.8;  // Fixed
            //   T D_lattice = BGK<T>::cs2 * (tau_ADE - 0.5);
            //   T u_target = Pe * D_lattice / L_lattice;
            //   T deltaP_new = u_target * nu_NS * L_lattice / permeability;
            //   This keeps τ_ADE fixed and adjusts deltaP in lattice units
            // ───────────────────────────────────────────────────────────────────

            // OLD METHOD: Physical units deltaP calculation
            T viscosity = 1e-3;  // Dynamic viscosity in Pa·s (water at ~20°C)
            T diffusionCoefficient = vec_solute_poreD[0];  // Diffusion coefficient from XML (m²/s)
            T deltaP_adjusted = Pe * viscosity * diffusionCoefficient / permeabilityMetric;

            pcout << "\n========== AUTO DELTA_P CALCULATOR ==========\n";
            pcout << "Target Pe: " << Pe << std::endl;
            pcout << "Diffusion coefficient: " << diffusionCoefficient << " m^2/s" << std::endl;
            pcout << "Adjusted deltaP: " << deltaP_adjusted << std::endl;
            pcout << "==============================================\n";

            // ═══════════════════════════════════════════════════════════════════
            // STEP 8: SECOND NS SIMULATION (with adjusted deltaP)
            // ═══════════════════════════════════════════════════════════════════
            // Rerun NS with adjusted deltaP
            NSdomainSetup(nsLattice, createLocalBoundaryCondition2D<T, DESCRIPTOR>(),
                geometry, deltaP_adjusted, nsLatticeOmega, pore_dynamics, bounce_back, no_dynamics);

            pcout << "\n========== LBM NS simulation begins with adjusted delta_P ==========\n";
            ns_convg1.resetValues();
            for (iT = 0; iT < ns_maxiTer_1; ++iT) {
                nsLattice.collideAndStream();
                ns_convg1.takeValue(getStoredAverageEnergy(nsLattice), true);
                if (ns_convg1.hasConverged()) { break; }
            }

            // Check for divergence in second NS
            T energy_check2 = getStoredAverageEnergy(nsLattice);
            if (std::isnan(energy_check2) || std::isinf(energy_check2) || energy_check2 > 1e10) {
                pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
                pcout << "║ ✗ FLOW DIVERGENCE DETECTED IN SECOND NS SIMULATION!             ║\n";
                pcout << "║ Adjusted ΔP = " << deltaP_adjusted << "\n";
                pcout << "║ SOLUTIONS: Reduce <Pe> or increase <tau> in XML                 ║\n";
                pcout << "╚══════════════════════════════════════════════════════════════════╝\n";
                throw std::runtime_error("Second NS simulation diverged - adjusted deltaP too high");
            }

            // ═══════════════════════════════════════════════════════════════════
            // STEP 9: RECALCULATE VELOCITIES WITH ADJUSTED DELTAP
            // ═══════════════════════════════════════════════════════════════════
            // Get final permeability
            permeability = computePermeability(nsLattice, nu_NS, deltaP_adjusted, nsLattice.getBoundingBox());
            pcout << "Final Permeability: " << permeability << std::endl;

            PoreMeanU = computeAverage(*computeVelocityNorm(nsLattice, Box2D(1, nx - 2, 0, ny - 1)));
            PoreMaxUx = computeMax(*computeVelocityComponent(nsLattice, Box2D(1, nx - 2, 0, ny - 1), 0));
            DarcyOutletUx = computeAverage(*computeVelocityComponent(nsLattice, Box2D(nx - 2, nx - 2, 0, ny - 1), 0));
            DarcyMiddleUx = computeAverage(*computeVelocityComponent(nsLattice, Box2D((nx - 1) / 2, (nx - 1) / 2, 0, ny - 1), 0));
            DarcyInletUx = computeAverage(*computeVelocityComponent(nsLattice, Box2D(1, 1, 0, ny - 1), 0));

            // ═══════════════════════════════════════════════════════════════════
            // DARCY VELOCITY CALCULATION (Python method)
            // ═══════════════════════════════════════════════════════════════════
            // Python script: darcy_velocity = np.mean(velocities) where velocities = velocityNorm over entire domain
            // This is equivalent to: average of velocity magnitude over entire domain (x: 1 to nx-2)
            // ═══════════════════════════════════════════════════════════════════
            T DarcyVelocity_PythonMethod = computeAverage(*computeVelocityNorm(nsLattice, Box2D(1, nx - 2, 0, ny - 1)));
            // Note: This is the same as PoreMeanU!

            pcout << "PoreMeanU = " << PoreMeanU << "." << std::endl;
            pcout << "PoreMaxUx = " << PoreMaxUx << "." << std::endl;
            pcout << "DarcyOutletUx = " << DarcyOutletUx << "." << std::endl;
            pcout << "DarcyMiddleUx = " << DarcyMiddleUx << "." << std::endl;
            pcout << "DarcyInletUx = " << DarcyInletUx << "." << std::endl;
            pcout << "DarcyVelocity (Python method) = " << DarcyVelocity_PythonMethod << "." << std::endl;

            // Use Python method for u_final (average velocityNorm over entire domain)
            T u_final = DarcyVelocity_PythonMethod;
            result.u_Darcy_lattice = u_final;
            result.permeability_lattice = permeability;
            result.permeability_m2 = permeability * dx * dx;

            // Check Mach number
            T Ma = PoreMaxUx / std::sqrt(BGK<T>::cs2);
            pcout << "CFL number (= maximum local lattice velocity)= " << PoreMaxUx << "." << std::endl;
            pcout << "Mach number = " << Ma << "." << std::endl;

            if (Ma > 1) {
                pcout << "Ma must be << 1. Terminating the simulation." << std::endl;
                throw std::runtime_error("Ma > 1 - flow unstable");
            }

            // ═══════════════════════════════════════════════════════════════════
            // STEP 10: CALCULATE τ_ADE (OLD METHOD - from velocity)
            // ═══════════════════════════════════════════════════════════════════
            //
            // OLD METHOD (currently active):
            //   refNu = PoreMeanU * charcs_length / Pe
            //   refTau = refNu * invCs² + 0.5
            //   This adjusts τ_ADE based on the resulting velocity
            //
            // ───────────────────────────────────────────────────────────────────
            // CORRECT METHOD (commented out for now):
            //   T tau_ADE = 0.8;  // Fixed
            //   T D_lattice = BGK<T>::cs2 * (tau_ADE - 0.5);
            //   This keeps τ_ADE fixed at 0.8
            // ───────────────────────────────────────────────────────────────────

            // OLD METHOD: Calculate τ_ADE from velocity
            T refNu;
            T refTau;

            if (Pe > thrd) {
                refNu = PoreMeanU * charcs_length / Pe;
                refTau = refNu * BGK<T>::invCs2 + 0.5;

                if (refTau > 2) {
                    pcout << "Reference relaxation time is > 2 (refTau = " << refTau << "). Consider reducing it for numerical accuracy by reducing average flow velocity (e.g. reduce delta_P).\n";
                    throw std::runtime_error("refTau > 2 - reduce deltaP");
                }
                else if (refTau <= 0.5) {
                    pcout << "Reference relaxation time does not satisfy a necessary stability condition for the BGK operator. (tau must be > 0.5, but refTau = " << refTau << ").\n";
                    pcout << "Consider increasing average flow velocity (e.g. increase delta_P).\n";
                    throw std::runtime_error("refTau <= 0.5 - increase deltaP");
                }
            }
            else {
                refTau = tau;
                refNu = BGK<T>::cs2 * (refTau - 0.5);
            }

            T tau_ADE = refTau;
            T D_lattice = refNu;
            result.tau_ADE = tau_ADE;

            // ═══════════════════════════════════════════════════════════════════
            // STEP 11: VERIFY ACHIEVED Pe
            // ═══════════════════════════════════════════════════════════════════
            // Old code formula: Pe = PoreMeanU × charcs_length / refNu
            // This will always equal Pe_target because refNu = PoreMeanU × charcs_length / Pe
            // ═══════════════════════════════════════════════════════════════════
            T Pe_achieved = PoreMeanU * charcs_length / refNu;
            result.Pe_achieved = Pe_achieved;

            pcout << "Peclet Number (meanU) = " << Pe_achieved << ", Grid Peclet Number (maxU) = " << PoreMaxUx / refNu << std::endl;

            // Stability check
            StabilityReport stability = performStabilityChecks(PoreMaxUx, nsLatticeTau, tau_ADE, D_lattice);
            printStabilityReport(stability);

            if (!stability.Ma_ok) {
                throw std::runtime_error("Mach number too high (Ma >= 1.0).");
            }

            if (track_performance == 0) {
                writeNsVTI(nsLattice, ns_maxiTer_1, "nsLatticeFinal_");
            }

            global::timer("NS").stop();
            T nstime = global::timer("NS").getTime();
            result.ns_time_seconds = nstime;

            // ═══════════════════════════════════════════════════════════════════
            // STEP 12: PHYSICAL TIME STEP
            // ═══════════════════════════════════════════════════════════════════
            T refOmega = 1.0 / refTau;
            T D_physical = vec_solute_poreD[0];  // D in m²/s
            T ade_dt = refNu * dx * dx / D_physical;
            result.dt_seconds = ade_dt;

            pcout << "dt = " << ade_dt << ".\n";

            // Calculate mean velocity in metric units
            T meanUMetric = PoreMeanU * dx / ade_dt;
            T domainLength = nx * dx;
            T flowDuration = domainLength / meanUMetric;
            plint suggestedMaxIter = static_cast<plint>(flowDuration / ade_dt);

            pcout << "\n========== SIMULATION PARAMETERS ==========\n";
            pcout << "Mean velocity (metric): " << meanUMetric << " m/s" << std::endl;
            pcout << "Domain length: " << domainLength << " m" << std::endl;
            pcout << "Flow duration: " << flowDuration << " s" << std::endl;
            pcout << "Suggested ade_maxiTer: " << suggestedMaxIter << std::endl;
            pcout << "Actual Pe: " << meanUMetric * charcs_length / vec_solute_poreD[0] << std::endl;
            pcout << "============================================\n";

            T u_Darcy_physical = u_final * dx / ade_dt;
            result.u_Darcy_m_s = u_Darcy_physical;

            pcout << "  NS simulation time: " << nstime << " seconds\n\n";

            // Write domain properties CSV
            writeDomainProperties(
                str_outputDir, baseName, Pe,
                nx, ny, dx, ade_dt,
                porosity, permeability, permeability * dx * dx,
                u_final, u_Darcy_physical,
                Pe, Pe_achieved,
                nsLatticeTau, tau_ADE, nu_NS, D_lattice, L_lattice);

            // ═══════════════════════════════════════════════════════════════════
            // ADE LATTICE SETUP (using calculated refTau from old method)
            // ═══════════════════════════════════════════════════════════════════
            std::vector<T> substrNUinPore(num_of_substrates);
            std::vector<T> substrTAUinPore(num_of_substrates);
            std::vector<T> substrOMEGAinPore(num_of_substrates);

            for (plint iS = 0; iS < num_of_substrates; ++iS) {
                if (iS == 0) {
                    substrNUinPore[iS] = refNu;
                    substrTAUinPore[iS] = refTau;
                    substrOMEGAinPore[iS] = refOmega;
                } else {
                    substrNUinPore[iS] = substrNUinPore[0] * vec_solute_poreD[iS] / vec_solute_poreD[0];
                    substrTAUinPore[iS] = substrNUinPore[iS] * BGK<T>::invCs2 + 0.5;
                    substrOMEGAinPore[iS] = 1.0 / substrTAUinPore[iS];
                }
            }

            pcout << "\nInitializing the reaction lattices... \n";
            pcout << "substrTAUinPore = ";
            for (plint iS = 0; iS < num_of_substrates; ++iS) {
                pcout << substrTAUinPore[iS] << " ";
            }
            pcout << std::endl;

            if (Pe > thrd) {
                pcout << "Peclet Number (meanU) = " << PoreMeanU * charcs_length / refNu
                      << ", Grid Peclet Number (maxU) = " << PoreMaxUx / refNu << std::endl;
            }
            pcout << "ade_dt = " << ade_dt << " s/ts" << std::endl;

            pcout << "Initializing substrate lattices (" << nx << "x" << ny << ")..." << std::endl;

            MultiBlockLattice2D<T, BGK> substrLattice(nx, ny,
                new AdvectionDiffusionBGKdynamics<T, BGK>(refOmega));
            std::vector<MultiBlockLattice2D<T, BGK>> vec_substr_lattices(num_of_substrates, substrLattice);
            std::vector<MultiBlockLattice2D<T, BGK>> dC(num_of_substrates, substrLattice);
            std::vector<MultiBlockLattice2D<T, BGK>> dC0(num_of_substrates, substrLattice);

            for (plint iS = 0; iS < num_of_substrates; ++iS) {
                soluteDomainSetup(vec_substr_lattices[iS],
                    createLocalAdvectionDiffusionBoundaryCondition2D<T, BGK>(),
                    geometry, substrOMEGAinPore[iS], pore_dynamics, bounce_back, no_dynamics,
                    vec_c0[iS], vec_left_btype[iS], vec_right_btype[iS],
                    vec_left_bcondition[iS], vec_right_bcondition[iS]);
                soluteDomainSetup(dC[iS],
                    createLocalAdvectionDiffusionBoundaryCondition2D<T, BGK>(),
                    geometry, substrOMEGAinPore[iS], pore_dynamics, bounce_back, no_dynamics,
                    0., vec_left_btype[iS], vec_right_btype[iS],
                    vec_left_bcondition[iS], vec_right_bcondition[iS]);
            }
            dC0 = dC;

            MultiBlockLattice2D<T, BGK> maskLattice(nx, ny,
                new AdvectionDiffusionBGKdynamics<T, BGK>(0.));
            MultiBlockLattice2D<T, BGK> distLattice(nx, ny,
                new AdvectionDiffusionBGKdynamics<T, BGK>(0.));

            defineMaskLatticeDynamics(substrLattice, maskLattice, thrd);
            applyProcessingFunctional(new CopyGeometryScalar2maskLattice2D<T, BGK, int>(pore_dynamics),
                maskLattice.getBoundingBox(), maskLattice, geometry);
            applyProcessingFunctional(new CopyGeometryScalar2distLattice2D<T, BGK, int>(),
                distLattice.getBoundingBox(), distLattice, distanceDomain);

            std::vector<MultiBlockLattice2D<T, BGK>*> ptr_kns_lattices;
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

            // ═══════════════════════════════════════════════════════════════════
            // STEP 13: ADE SIMULATION WITH ADAPTIVE STOPPING
            // ═══════════════════════════════════════════════════════════════════
            pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
            pcout << "║              ADE SIMULATION WITH ADAPTIVE STOPPING              ║\n";
            pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
            pcout << "║ Stop condition: C_flux >= 0.95 for 3 consecutive checks         ║\n";
            pcout << "║ VTI saved at: Initial, Early (5%), BT (50%), Near (90%), Final  ║\n";
            pcout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

            // Adaptive stopping parameters
            T C_threshold = 0.95;
            plint stable_count_needed = 3;
            plint stable_count = 0;

            // Key moment thresholds for VTI saving
            T threshold_early = 0.05;
            T threshold_BT = 0.50;
            T threshold_near = 0.90;
            bool saved_early = false;
            bool saved_BT = false;
            bool saved_near = false;

            // BTC recording interval
            plint btc_record_interval = ade_VTK_iTer / 10;
            if (btc_record_interval < 100) btc_record_interval = 100;

            pcout << "┌────────────────────────────────────────────────────────────────┐\n";
            pcout << "│ BTC RECORDING SETTINGS                                         │\n";
            pcout << "├────────────────────────────────────────────────────────────────┤\n";
            pcout << "│ BTC record interval:  " << btc_record_interval << " iterations (for CSV)\n";
            pcout << "│ Console/VTI interval: " << ade_VTK_iTer << " iterations\n";
            pcout << "│ Expected BTC points:  ~100-200 (for smooth curve)              │\n";
            pcout << "└────────────────────────────────────────────────────────────────┘\n\n";

            std::vector<BTCDataPoint> btc_data;

            global::timer("ade").restart();
            iT = 0;

            // Record initial BTC point
            T C_flux_latest = calculateFluxWeightedConcentration(vec_substr_lattices[0], nsLattice, geometry, nx, ny, pore_dynamics);
            T C_avg_latest = calculateAverageConcentration(vec_substr_lattices[0], geometry, nx, ny, pore_dynamics);
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

                // Record BTC data frequently
                if (iT > 0 && iT % btc_record_interval == 0) {
                    C_avg_latest = calculateAverageConcentration(vec_substr_lattices[0], geometry, nx, ny, pore_dynamics);
                    C_flux_latest = calculateFluxWeightedConcentration(vec_substr_lattices[0], nsLattice, geometry, nx, ny, pore_dynamics);
                    T time_s = iT * ade_dt;
                    T pore_vol = calculatePoreVolume(u_final, (T)iT, L_for_PV, porosity);

                    BTCDataPoint pt;
                    pt.iteration = iT;
                    pt.time_seconds = time_s;
                    pt.time_lattice = (T)iT;
                    pt.C_average = C_avg_latest;
                    pt.C_flux_weighted = C_flux_latest;
                    pt.pore_volume = pore_vol;
                    btc_data.push_back(pt);
                }

                // Console output and VTI milestones
                if (ade_VTK_iTer > 0 && iT % ade_VTK_iTer == 0 && iT > 0) {

                    T time_s = iT * ade_dt;
                    T pore_vol = calculatePoreVolume(u_final, (T)iT, L_for_PV, porosity);

                    pcout << "iT=" << iT << " | t=" << std::scientific << std::setprecision(3) << time_s
                          << "s | PV=" << std::fixed << std::setprecision(3) << pore_vol
                          << " | C_flux=" << std::setprecision(4) << C_flux_latest
                          << " | pts=" << btc_data.size();

                    // Save VTI at key milestones
                    if (!saved_early && C_flux_latest >= threshold_early && track_performance == 0) {
                        pcout << " | 📸 EARLY";
                        for (plint iS = 0; iS < num_of_substrates; ++iS) {
                            writeAdvVTI(vec_substr_lattices[iS], iT, ade_filename + std::to_string(iS) + "_early_");
                        }
                        saved_early = true;
                    }

                    if (!saved_BT && C_flux_latest >= threshold_BT && track_performance == 0) {
                        pcout << " | 📸 BT";
                        for (plint iS = 0; iS < num_of_substrates; ++iS) {
                            writeAdvVTI(vec_substr_lattices[iS], iT, ade_filename + std::to_string(iS) + "_BT_");
                        }
                        saved_BT = true;
                    }

                    if (!saved_near && C_flux_latest >= threshold_near && track_performance == 0) {
                        pcout << " | 📸 NEAR";
                        for (plint iS = 0; iS < num_of_substrates; ++iS) {
                            writeAdvVTI(vec_substr_lattices[iS], iT, ade_filename + std::to_string(iS) + "_near_");
                        }
                        saved_near = true;
                    }

                    // Check stopping condition
                    if (C_flux_latest >= C_threshold) {
                        stable_count++;
                        pcout << " | ✓ (" << stable_count << "/" << stable_count_needed << ")";

                        if (stable_count >= stable_count_needed) {
                            pcout << "\n\n╔══════════════════════════════════════════════════════════════════╗\n";
                            pcout << "║ ✓ BREAKTHROUGH COMPLETE - STOPPING SIMULATION                   ║\n";
                            pcout << "║   C_flux = " << C_flux_latest << " >= " << C_threshold << "\n";
                            pcout << "║   BTC data points: " << btc_data.size() << "\n";
                            pcout << "╚══════════════════════════════════════════════════════════════════╝\n";
                            ++iT;
                            break;
                        }
                    } else {
                        stable_count = 0;
                    }

                    pcout << std::endl;
                }

                // Collision, kinetics, streaming
                for (plint iS = 0; iS < num_of_substrates; ++iS) {
                    vec_substr_lattices[iS].collide();
                }

                dC = dC0;
                applyProcessingFunctional(
                    new run_kinetics<T, BGK>(nx, num_of_substrates, ade_dt,
                        vec_Kc_kns, no_dynamics, bounce_back),
                    vec_substr_lattices[0].getBoundingBox(), ptr_kns_lattices);

                for (plint iS = 0; iS < num_of_substrates; ++iS) {
                    vec_substr_lattices[iS].stream();
                }
            }

            global::timer("ade").stop();
            T adetime = global::timer("ade").getTime();
            result.ade_time_seconds = adetime;
            result.total_iterations = iT;

            // Save final VTI
            if (track_performance == 0) {
                pcout << "\n  Saving final VTI files...\n";
                for (plint iS = 0; iS < num_of_substrates; ++iS) {
                    writeAdvVTI(vec_substr_lattices[iS], iT, ade_filename + std::to_string(iS) + "_final_");
                }
            }

            // Write BTC CSV
            writeBTCTimeseries(
                str_outputDir, baseName, Pe,
                btc_data,
                porosity, u_final, u_Darcy_physical,
                permeability, permeability * dx * dx);

            // Calculate and write moments
            T M0, M1, M2_central, std_dev, cv;
            calculateMoments(btc_data, M0, M1, M2_central, std_dev, cv);

            result.M0 = M0;
            result.M1_seconds = M1;
            result.M2_central = M2_central;
            result.sigma_seconds = std_dev;
            result.CV = cv;

            T M1_lattice = M1 / ade_dt;
            T sigma_lattice = std_dev / ade_dt;

            writeMomentsSummary(
                str_outputDir, baseName, Pe,
                M0, M1, M1_lattice,
                M2_central, std_dev, sigma_lattice, cv,
                porosity, Pe_achieved,
                permeability * dx * dx, u_Darcy_physical);

            // Mark as successful
            result.success = true;
            result.failure_reason = "";

            pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
            pcout << "║                      SIMULATION COMPLETE                          ║\n";
            pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
            pcout << "║ M₀ = " << M0 << "\n";
            pcout << "║ M₁ = " << M1 << " s (mean arrival time)\n";
            pcout << "║ σ  = " << std_dev << " s (std deviation)\n";
            pcout << "║ CV = " << cv << " (coefficient of variation)\n";
            pcout << "╚══════════════════════════════════════════════════════════════════╝\n";

        }
        catch (const std::exception& e) {
            result.success = false;
            result.failure_reason = e.what();
            pcout << "\n  ✗ SIMULATION FAILED: " << e.what() << "\n";
            pcout << "    Skipping to next geometry...\n";
        }
        catch (...) {
            result.success = false;
            result.failure_reason = "Unknown error";
            pcout << "\n  ✗ SIMULATION FAILED: Unknown error\n";
            pcout << "    Skipping to next geometry...\n";
        }

        global::timer("sim").stop();
        result.total_time_seconds = global::timer("sim").getTime();

        allResults.push_back(result);

        pcout << "\n  Time for this simulation: " << result.total_time_seconds << " seconds\n";
    }

    // ========================================================================
    // WRITE BATCH SUMMARY
    // ========================================================================
    writeBatchSummary(str_outputDir_base, allResults);
    writeFailedSimulations(str_outputDir_base, allResults);

    // ========================================================================
    // FINAL REPORT
    // ========================================================================
    global::timer("batch_total").stop();
    T total_batch_time = global::timer("batch_total").getTime();

    int successCount = 0;
    int failCount = 0;
    for (size_t i = 0; i < allResults.size(); ++i) {
        if (allResults[i].success) successCount++;
        else failCount++;
    }

    pcout << "\n";
    pcout << "╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║                    BATCH PROCESSING COMPLETE                     ║\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ Total geometries: " << allResults.size() << "\n";
    pcout << "║ Successful:       " << successCount << "\n";
    pcout << "║ Failed:           " << failCount << "\n";
    pcout << "║ Success rate:     " << std::fixed << std::setprecision(1)
          << (100.0 * successCount / allResults.size()) << "%\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ Total time: " << total_batch_time << " s (" << total_batch_time/60 << " min)\n";
    pcout << "╠══════════════════════════════════════════════════════════════════╣\n";
    pcout << "║ Output files:                                                    ║\n";
    pcout << "║   - BATCH_SUMMARY.csv (all results)                              ║\n";
    pcout << "║   - FAILED_SIMULATIONS.txt (failure report)                      ║\n";
    pcout << "║   - Individual folders per geometry                              ║\n";
    pcout << "╚══════════════════════════════════════════════════════════════════╝\n";

    if (failCount > 0) {
        pcout << "\n⚠ FAILED SIMULATIONS:\n";
        for (size_t i = 0; i < allResults.size(); ++i) {
            if (!allResults[i].success) {
                pcout << "  - " << allResults[i].geom_name << ": " << allResults[i].failure_reason << "\n";
            }
        }
    }

    pcout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    pcout << "║                    Batch Processing Finished!                    ║\n";
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

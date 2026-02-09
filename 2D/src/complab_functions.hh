



#include <climits>
#include <cfloat>
#include "palabos2D.h"
#include "palabos2D.hh"
//#include "complab_processors.hh"

#include <map>





using namespace plb;
typedef double T;


#define DESCRIPTOR descriptors::D2Q9Descriptor
#define BGK descriptors::AdvectionDiffusionD2Q5Descriptor
#define thrd 1e-14


// a function to initially distribute the pressure linearly.
// This is used only to initializing the flow field
class PressureGradient {
public:
    PressureGradient(T deltaP_, plint nx_) : deltaP(deltaP_), nx(nx_)
    { }
    void operator() (plint iX, plint iY, T& density, Array<T, 2>& velocity) const
    {
        velocity.resetToZero();
        density = (T)1 - deltaP * DESCRIPTOR<T>::invCs2 / (T)(nx - 1) * (T)iX;

    }
private:
    T deltaP;
    plint nx;
};

// This is a function called "writeNsVTI" that writes the fluid flow data in the format of VTK (Visualization Toolkit) image file.

void writeNsVTI(MultiBlockLattice2D<T, DESCRIPTOR>& lattice, plint iter, std::string nameid)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    VtkImageOutput2D<T> vtkOut(createFileName(nameid, iter, 7), 1.);

    vtkOut.writeData<float>(*computeVelocityNorm(lattice, Box2D(1, nx - 2, 0, ny - 1)), "velocityNorm", 1.);
    vtkOut.writeData<2, float>(*computeVelocity(lattice, Box2D(1, nx - 2, 0, ny - 1)), "velocity", 1.);
}


// This function writes the MultiScalarField2D<int> "geometry" to a VTK file format.
void writePorousMediumVTI(MultiScalarField2D<int>& geometry, plint iter, std::string nameid)
{
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();
    VtkImageOutput2D<T> vtkOut(createFileName(nameid, iter, 7), 1.);
    vtkOut.writeData<float>(*copyConvert<int, T>(geometry, Box2D(1, nx - 2, 0, ny - 1)), "tag", 1.0);
}

// writing output files of the SOLUTE domain

void writeAdvVTI(MultiBlockLattice2D<T, BGK>& lattice, plint iter, std::string nameid)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();

    VtkImageOutput2D<T> vtkOut(createFileName(nameid, iter, 7), 1.);
    vtkOut.writeData<T>(*computeDensity(lattice, Box2D(1, nx - 2, 0, ny - 1)), "Density", 1.);
}


void writeScalarVTI(MultiScalarField2D<int>& field)
{
    const plint nx = field.getNx();
    const plint ny = field.getNy();

    VtkImageOutput2D<T> vtkOut("distanceDomain", 1.);
    vtkOut.writeData<float>(*copyConvert<int, T>(field, Box2D(0, nx - 1, 0, ny - 1)), "tag", 1.0);
}


// load a geometry file with predefined material numbers
void readGeometry(std::string fNameIn, MultiScalarField2D<int>& geometry)
{
    pcout << "Reading the geometry file (" << fNameIn << ").\n";
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();

    Box2D sliceBox(0, 0, 0, ny - 1);
    std::unique_ptr<MultiScalarField2D<int> > slice = generateMultiScalarField<int>(geometry, sliceBox);

    plb_ifstream geometryFile(fNameIn.c_str());
    if (!geometryFile.is_open()) {
        pcout << "Error: could not open geometry file " << fNameIn << std::endl;
        exit(EXIT_FAILURE);
    }
    for (plint iX = 0; iX < nx - 1; ++iX) {
        geometryFile >> *slice;
        if (iX == 1) {
            copy(*slice, slice->getBoundingBox(), geometry, Box2D(0, 0, 0, ny - 1));
            copy(*slice, slice->getBoundingBox(), geometry, Box2D(iX, iX, 0, ny - 1));
        }
        else if (iX == nx - 2) {
            copy(*slice, slice->getBoundingBox(), geometry, Box2D(iX, iX, 0, ny - 1));
            copy(*slice, slice->getBoundingBox(), geometry, Box2D(nx - 1, nx - 1, 0, ny - 1));
        }
        else {
            copy(*slice, slice->getBoundingBox(), geometry, Box2D(iX, iX, 0, ny - 1));
        }
    }
}

void saveGeometry(std::string fNameIn, MultiScalarField2D<int>& geometry)
{
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();


    pcout << "Save geometry vti file (" << fNameIn << ").\n";
    VtkImageOutput2D<T> vtkOut(fNameIn, 1.0);
    vtkOut.writeData<float>(*copyConvert<int, T>(geometry, Box2D(1, nx - 2, 0, ny - 1)), "tag", 1.0);
}


// This function, calculateDistanceFromSolid2D, calculates the distance from each point in a 2D lattice to the nearest solid cell.

void calculateDistanceFromSolid2D(MultiScalarField2D<int> distance, plint nodymcs, plint bb, std::vector<std::vector<plint>>& distVec)

{
    const plint nx = distance.getNx();
    const plint ny = distance.getNy();


    for (plint iX = 0; iX < nx - 1; ++iX) {
        for (plint iY = 0; iY < ny - 1; ++iY) {
            plint mask = distance.get(iX, iY);
            if (mask == nodymcs) { distVec[iX][iY] = -1; }
            else if (mask == bb) { distVec[iX][iY] = 0; }
            else { distVec[iX][iY] = 1; }
        }
    }



    for (plint iX = 0; iX < nx - 1; ++iX) {
        for (plint iY = 0; iY < ny - 1; ++iY) {
            if ( distVec[iX][iY] == 1 ) {
                plint lp = 1, r = 0, dist = 0;
                while (lp == 1) {
                    ++r;
                    std::vector<plint> vx(r + 1), vy(r + 1);
                    for (plint tmp = 0; tmp < r + 1; ++tmp) { vx[tmp] = tmp; vy[tmp] = r - tmp; }
                    for (plint it = 0; it < r + 1; ++it) {
                        plint xp = iX + vx[it], yp = iY + vy[it], xn = iX - vx[it], yn = iY - vy[it];
                        if (xp >= 0 && yp >= 0 && xp < nx && yp < ny) {
                            if ( distVec[xp][yp] == 0 ) {
                                dist = r; lp = 0; break;
                            }
                        }
                        if (xp >= 0 && yn >= 0 && xp < nx && yn < ny) {
                            if (distVec[xp][yn] == 0 ) {
                                dist = r; lp = 0; break;
                            }
                        }
                        if (xn >= 0 && yp >= 0 && xn < nx && yp < ny) {
                            if ( distVec[xn][yp] == 0) {
                                dist = r; lp = 0; break;
                            }
                        }
                        if (xn >= 0 && yn >= 0 && xn < nx && yn < ny) {
                            if ( distVec[xn][yn] == 0 ) {
                                dist = r; lp = 0; break;
                            }
                        }
                    }
                }
                if (lp == 0) { distVec[iX][iY] = dist; }
            }
        }
    }

}


void NSdomainSetup(MultiBlockLattice2D<T, DESCRIPTOR>& lattice, OnLatticeBoundaryCondition2D<T, DESCRIPTOR>* boundaryCondition, MultiScalarField2D<int>& geometry, T deltaP,

    T fluidOmega, std::vector<plint> pore, plint bounceback, plint nodymcs)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();



    Box2D west(0, 0, 0, ny - 1);
    Box2D east(nx - 1, nx - 1, 0, ny - 1);



    // default. initialize the entire domain. may be redundant
    defineDynamics(lattice, lattice.getBoundingBox(), new IncBGKdynamics<T, DESCRIPTOR>(fluidOmega));

    // pore space
    for (size_t iP = 0; iP < pore.size(); ++iP) {
        if (pore[iP] > 0) { defineDynamics(lattice, geometry, new IncBGKdynamics<T, DESCRIPTOR>(fluidOmega), pore[iP]); }
    }
    // bounce-back boundary
    if (bounceback > 0) {
        defineDynamics(lattice, geometry, new BounceBack<T, DESCRIPTOR>(), bounceback);
    }
    // no dynamics
    if (nodymcs >= 0) {
        defineDynamics(lattice, geometry, new NoDynamics<T, DESCRIPTOR>(), nodymcs);
    }



    boundaryCondition->addPressureBoundary0N(west, lattice);
    setBoundaryDensity(lattice, west, (T)1.);
    boundaryCondition->addPressureBoundary0P(east, lattice);
    setBoundaryDensity(lattice, east, (T)1. - deltaP * DESCRIPTOR<T>::invCs2);


    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), PressureGradient(deltaP, nx));

    lattice.initialize();
    delete boundaryCondition;
}


// solute domain boundary conditions. No flow boundaries at the top and bottom.
void soluteDomainSetup(MultiBlockLattice2D<T, BGK>& lattice, OnLatticeAdvectionDiffusionBoundaryCondition2D<T, BGK>* boundaryCondition, MultiScalarField2D<int>& geometry,
    T substrOmega, std::vector<plint> pore, plint bounceback, plint nodymcs,
    T rho0, bool left_btype, bool right_btype, T left_BC, T right_BC)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();


    Box2D west(0, 0, 0, ny - 1);
    Box2D east(nx - 1, nx - 1, 0, ny - 1);
    plint processorLevelBC = 1;

    // default. initialize the entire domain. may be redundant
    defineDynamics(lattice, lattice.getBoundingBox(), new AdvectionDiffusionBGKdynamics<T, BGK>(substrOmega));

    // pore space
    for (size_t iP = 0; iP < pore.size(); ++iP) {
        if (pore[iP] > 0) { defineDynamics(lattice, geometry, new AdvectionDiffusionBGKdynamics<T, BGK>(substrOmega), pore[iP]); }
    }
    // bounceback boundary
    if (bounceback > 0) { defineDynamics(lattice, geometry, new BounceBack<T, BGK>(), bounceback); }
    // no dynamics
    if (nodymcs >= 0) { defineDynamics(lattice, geometry, new NoDynamics<T, BGK>(), nodymcs); }

    // Set the boundary-conditions
    boundaryCondition->addTemperatureBoundary0N(west, lattice);
    if (left_btype == 0) { setBoundaryDensity(lattice, west, left_BC); }
    else { integrateProcessingFunctional(new FlatAdiabaticBoundaryFunctional2D<T, BGK, 0, -1>, west, lattice, processorLevelBC); }

    boundaryCondition->addTemperatureBoundary0P(east, lattice);
    if (right_btype == 0) { setBoundaryDensity(lattice, east, right_BC); }
    else { integrateProcessingFunctional(new FlatAdiabaticBoundaryFunctional2D<T, BGK, 0, +1>, east, lattice, processorLevelBC); }

    // Init lattice
    Array<T, 2> u0(0., 0.);
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), rho0, u0);


    lattice.initialize();
    delete boundaryCondition;
}


T computePermeability(MultiBlockLattice2D<T, DESCRIPTOR>& nsLattice, T nsNu, T deltaP, Box2D domain)
{
    pcout << "Computing the permeability." << std::endl;

    // Compute only the x-direction of the velocity (direction of the flow).
    plint xComponent = 0;
    plint nx = nsLattice.getNx();
    plint ny = nsLattice.getNy();

    T meanU = computeAverage(*computeVelocityComponent(nsLattice, Box2D(1,nx-2,0,ny-1), xComponent));

    pcout << "Average velocity (meanU) = " << meanU                                 << std::endl;
    pcout << "Grad P               = " << deltaP / (T)(nx - 1)                      << std::endl;
    pcout << "Permeability         = " << nsNu * meanU / (deltaP / (T)(nx - 1))     << std::endl;
    pcout << "Lattice viscosity nu = " << nsNu                                      << std::endl;

    return nsNu * meanU / (deltaP / (T)(nx - 1));
}




T computeResidenceTime(MultiBlockLattice2D<T, DESCRIPTOR>& nsLattice, MultiScalarField2D<int>& geometry, T meanU, const std::vector<long int>& pore_dynamics, T dx)
{
    pcout << "Computing the residence time." << std::endl;

    plint nx = nsLattice.getNx();
    plint ny = nsLattice.getNy();
    T porosity = 0;
    plint num_pore_cells = 0;

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            for (long int pore_dynamic : pore_dynamics) {
                if (geometry.get(iX, iY) == pore_dynamic) {
                    num_pore_cells++;
                    break;
                }
            }
        }
    }
    porosity = (T)num_pore_cells / (nx * ny);

    T length = nx * dx;

    T residence_time = porosity * length / meanU;

    pcout << "Porosity: " << porosity << std::endl;
    pcout << "System Length: " << length << std::endl;
    pcout << "Mean Fluid Velocity: " << meanU << std::endl;
    pcout << "Residence Time: " << residence_time << " s" << std::endl;

    return residence_time;
}



void scalarDomainDynamicsSetupFromVectors(MultiBlockLattice2D<T, BGK>& lattice, MultiScalarField2D<int>& geometry, std::vector<plint> mtrvec, std::vector<T> omegavec)
{
    // default. initialize the entire domain. may be redundant
    defineDynamics(lattice, lattice.getBoundingBox(), new AdvectionDiffusionBGKdynamics<T, BGK>(0.));

    if (mtrvec.size() != omegavec.size()) {
        pcout << "ERROR: the length of input vectors (mtrvec and omegavec) must be the same.\n";
        exit(EXIT_FAILURE);
    }
    // assign lattice omegas (dynamics) for each mask number
    for (size_t iT = 0; iT < mtrvec.size(); ++iT) {
        defineDynamics(lattice, geometry, new AdvectionDiffusionBGKdynamics<T, BGK>(omegavec[iT]), mtrvec[iT]);
    }
    // Init lattice
    Array<T, 2> jEq(0., 0.);
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), 0., jEq);

    lattice.initialize();
}


void scalarDomainDynamicsSetupFromGeometry(MultiBlockLattice2D<T, BGK>& lattice, MultiScalarField2D<int>& geometry, plint nx, plint ny)
{
    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            plint geom = geometry.get(iX, iY);
            defineDynamics(lattice, iX, iY, new AdvectionDiffusionBGKdynamics<T, BGK>((T)geom));
        }
    }
    // Init lattice
    Array<T, 2> jEq(0., 0.);
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), 0., jEq);

    lattice.initialize();
}


void gridSearch2D(MultiScalarField2D<int> geometry, std::vector< std::vector<plint> >& distVec, plint bb, plint solid, std::vector<plint> pore)
{
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            plint geom = geometry.get(iX, iY);
            bool flag0 = false;

            if (!flag0) {
                plint iR = 0;
                bool flag1 = false;
                while (!flag1) {
                    ++iR;
                    for (plint rx = 0; rx < iR; ++rx) {
                        for (plint ry = 0; ry < iR; ++ry) {
                            std::vector<std::vector<plint>> neighbors = {
                                {iX + rx, iY + ry},
                                {iX + rx, iY - ry},
                                {iX - rx, iY + ry},
                                {iX - rx, iY - ry},
                            };

                            for (const auto& neighbor : neighbors) {
                                plint nx_ = neighbor[0];
                                plint ny_ = neighbor[1];

                                if (nx_ >= 0 && nx_ < geometry.getNx() && ny_ >= 0 && ny_ < geometry.getNy()) {
                                    plint mask = geometry.get(nx_, ny_);

                                    for (size_t iP = 0; iP < pore.size(); ++iP) {
                                        if (mask == pore[iP]) {
                                            flag1 = true;
                                            distVec[iX][iY] = iR;
                                            break;
                                        }
                                    }
                                    if (flag1) { break; }
                                }
                            }
                            if (flag1) { break; }
                        }
                        if (flag1) { break; }
                    }
                }
            }
            else if (geom == bb || geom == solid) { distVec[iX][iY] = -1; }
            else { distVec[iX][iY] = 0; }
        }
    }
}



void initSubstrateMaskLatticeDensity(MultiBlockLattice2D<T, BGK>& lattice1, MultiBlockLattice2D<T, BGK>& lattice2)
{
    const plint nx = lattice1.getNx();
    const plint ny = lattice1.getNy();

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            T substrateConcentration = lattice1.get(iX, iY).computeDensity();
            Array<T, 5> g;
            lattice1.get(iX, iY).getPopulations(g);

            lattice2.get(iX, iY).setPopulations(g);
        }
    }
}

void defineMaskLatticeDynamics(MultiBlockLattice2D<T, BGK>& lattice1, MultiBlockLattice2D<T, BGK>& lattice2, T fbM)
{
    const plint nx = lattice1.getNx();
    const plint ny = lattice1.getNy();

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            T substrateConcentration = lattice1.get(iX, iY).computeDensity();
            T omega = 0.;

            if (substrateConcentration > fbM) {
                omega = 1.;
            }

            defineDynamics(lattice2, iX, iY, new AdvectionDiffusionBGKdynamics<T, BGK>(omega));
        }
    }

    // Init lattice
    Array<T, 2> jEq(0., 0.);
    initializeAtEquilibrium(lattice2, lattice2.getBoundingBox(), 0., jEq);

    lattice2.initialize();
}



int initialize_complab(char *&main_path, char *&src_path, char *&input_path, char *&output_path, char *&ns_filename, std::string &ade_filename, std::string &geom_filename,
    std::string &mask_filename, bool &read_NS_file, plint &ns_rerun_iT0, T &ns_converge_iT1, T &ns_converge_iT2, plint &ns_maxiTer_1, plint &ns_maxiTer_2, plint& ns_update_interval, plint& ade_update_interval,
    bool& read_ADE_file, plint& ade_rerun_iT0, plint& ade_VTK_iTer, plint& ade_CHK_iTer, T& ade_converge_iT, plint& ade_maxiTer, plint& nx, plint& ny, T &dx, T& dy, T& delta_P, T& tau,
    T &Pe, T &charcs_length, std::vector<T> &solute_poreD, bool &soluteDindex, std::vector<plint> &pore_dynamics, plint &bounce_back, plint &no_dynamics, plint &num_of_substrates, std::vector<std::string> &vec_subs_names, std::vector<plint> &solver_type,
    plint &lb_count, plint &kns_count, std::vector<plint> &reaction_type, std::vector<T> &vec_c0, std::vector<bool>& left_btype, std::vector<bool>& right_btype, std::vector<T>& vec_leftBC, std::vector<T>& vec_rightBC,
    std::vector< std::vector<T> > &vec_Kc, std::vector< std::vector<T>> &vec_Kc_kns,  std::vector<bool> &vec_fixLB, std::vector<bool> &vec_fixC,
    std::vector<plint> &vec_sense, std::vector<std::vector<int>> &vec_const_loc, std::vector<std::vector<T>> &vec_const_lb, std::vector<std::vector<T>> &vec_const_ub, bool &track_performance, bool &halfflag, bool &eqflag)

{


    try {
        std::string fin("CompLaB.xml");
        XMLreader doc(fin);
        pcout << "Successfully opened XML file: " << fin << std::endl;

        // terminate the simulation if inputs are undefined.
        try {

            // 1. Read domain parameters (LB_numerics)
            doc["parameters"]["LB_numerics"]["domain"]["nx"].read(nx);
            pcout << "Read nx: " << nx << std::endl;
            doc["parameters"]["LB_numerics"]["domain"]["ny"].read(ny);
            pcout << "Read ny: " << ny << std::endl;
            doc["parameters"]["LB_numerics"]["domain"]["dx"].read(dx);
            pcout << "Read dx: " << dx << std::endl;
            doc["parameters"]["LB_numerics"]["domain"]["filename"].read(geom_filename);
            pcout << "Read geom_filename: " << geom_filename << std::endl;

            // 2. chemistry Read substrates

            doc["parameters"]["chemistry"]["number_of_substrates"].read(num_of_substrates);
            pcout << "Read number_of_substrates: " << num_of_substrates << std::endl;

            // Loop through each substrate
            soluteDindex = 0;
            lb_count = 0;
            for (plint iT = 0; iT < num_of_substrates; ++iT) {
                T D0, c0, bc0, bc1;
                std::string chemname = "substrate" + std::to_string(iT);
                std::string tmp0, tmp1;

                try { doc["parameters"]["chemistry"][chemname]["name_of_substrates"].read(vec_subs_names);
                pcout << "Read name_of_substrates for " << chemname << ": " << vec_subs_names.back() << std::endl;}
                catch (PlbIOException& exception) { vec_subs_names.push_back("substrate_" + std::to_string(iT)); }

                try { doc["parameters"]["chemistry"][chemname]["substrate_diffusion_coefficients"]["in_pore"].read(D0); solute_poreD.push_back(D0);
                pcout << "Read substrate_diffusion_coefficients in_pore for " << chemname << ": " << D0 << std::endl;}
                catch (PlbIOException& exception) { solute_poreD.push_back(1e-9); }
                if (std::abs(D0) > thrd) { soluteDindex = 1; }

                try { doc["parameters"]["chemistry"][chemname]["initial_concentration"].read(c0); vec_c0.push_back(c0);
                pcout << "Read initial_concentration for " << chemname << ": " << c0 << std::endl;}
                catch (PlbIOException& exception) { vec_c0.push_back(0.0); }

                doc["parameters"]["chemistry"][chemname]["left_boundary_type"].read(tmp0);
                pcout << "Read left_boundary_type for " << chemname << ": " << tmp0 << std::endl;
                std::transform(tmp0.begin(), tmp0.end(), tmp0.begin(), [](unsigned char c) { return std::tolower(c); });
                if (tmp0.compare("dirichlet") == 0) { left_btype.push_back(0); }
                else if (tmp0.compare("neumann") == 0) { left_btype.push_back(1); }
                else { pcout << "left_boundary_type (" << tmp0 << ") should be either Dirichlet or Neumann. Terminating the simulation.\n"; return -1; }

                doc["parameters"]["chemistry"][chemname]["right_boundary_type"].read(tmp1);
                pcout << "Read right_boundary_type for " << chemname << ": " << tmp1 << std::endl;
                std::transform(tmp1.begin(), tmp1.end(), tmp1.begin(), [](unsigned char c) { return std::tolower(c); });
                if (tmp1.compare("dirichlet") == 0) { right_btype.push_back(0); }
                else if (tmp1.compare("neumann") == 0) { right_btype.push_back(1); }
                else { pcout << "right_boundary_type (" << tmp1 << ") should be either Dirichlet or Neumann. Terminating the simulation.\n"; return -1; }

                doc["parameters"]["chemistry"][chemname]["left_boundary_condition"].read(bc0); vec_leftBC.push_back(bc0);
                pcout << "Read left_boundary_condition for " << chemname << ": " << bc0 << std::endl;
                doc["parameters"]["chemistry"][chemname]["right_boundary_condition"].read(bc1); vec_rightBC.push_back(bc1);
                pcout << "Read right_boundary_condition for " << chemname << ": " << bc1 << std::endl;

                doc["parameters"]["chemistry"][chemname]["reaction_type"].read(tmp0);
                pcout << "Read reaction_type for " << chemname << ": " << tmp0 << std::endl;
                std::transform(tmp0.begin(), tmp0.end(), tmp0.begin(), [](unsigned char c) { return std::tolower(c); });
                if (tmp0.compare("kinetics") == 0) { reaction_type.push_back(1); ++kns_count; }
                else { pcout << "reaction_type " << tmp0 << " is not implemented. Only 'kinetics' is supported for substrates. Terminating the simulation.\n"; return -1; }

                doc["parameters"]["chemistry"][chemname]["solver_type"].read(tmp1);
                pcout << "Read solver_type for " << chemname << ": " << tmp1 << std::endl;
                std::transform(tmp1.begin(), tmp1.end(), tmp1.begin(), [](unsigned char c) { return std::tolower(c); });
                if (tmp1.compare("lbm") == 0 || tmp1.compare("lattice boltzmann") == 0 || tmp1.compare("lattice_boltzmann") == 0) { solver_type.push_back(3); ++lb_count; }
                else { pcout << "Palabos IO exception: Element solver_type " << tmp1 << " is not defined. Only 'LBM' is supported for substrates. Terminating the simulation.\n"; return -1; }

            }

            if (left_btype.size() != (unsigned)num_of_substrates) { pcout << "The length of left_boundary_type vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
            if (right_btype.size() != (unsigned)num_of_substrates) { pcout << "The length of right_boundary_type vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
            if (vec_leftBC.size() != (unsigned)num_of_substrates) { pcout << "The length of left_boundary_condition vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
            if (vec_rightBC.size() != (unsigned)num_of_substrates) { pcout << "The length of right_boundary_condition vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
            if (reaction_type.size() != (unsigned)num_of_substrates) { pcout << "The length of reaction_type vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
            if (solver_type.size() != (unsigned)num_of_substrates) { pcout << "The length of solver_type vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
            if (vec_c0.size() != (unsigned)num_of_substrates) { pcout << "The length of initial_concentration vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
            if (solute_poreD.size() != (unsigned)num_of_substrates) { pcout << "The length of substrate_diffusion_coefficients in_pore vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
            if (vec_subs_names.size() != (unsigned)num_of_substrates) { pcout << "The length of name_of_substrates vector does not match the num_of_substrates. Terminating the simulation.\n"; return -1; }
        }
        catch (PlbIOException& exception) {
            pcout << exception.what() << " Terminating the simulation.\n" << std::endl;
            return -1;
        }


        // parameters with default values
        // define paths
        try {
            std::string item;
            doc["parameters"]["path"]["src_path"].read(item);
            src_path = (char*)calloc(item.size() + 1, sizeof(char));
            for (size_t i = 0; i < item.size(); ++i) { src_path[i] = item[i]; }
            src_path[item.size() + 1] = '\0';
        }
        catch (PlbIOException& exception) {
            std::string item = "src";
            src_path = (char*)calloc(item.size() + 2, sizeof(char));
            for (size_t i = 0; i < item.size(); ++i) { src_path[i] = item[i]; }
            src_path[item.size() + 1] = '\0';
        }
        try {
            std::string item;
            doc["parameters"]["path"]["input_path"].read(item);
            input_path = (char*)calloc(item.size() + 1, sizeof(char));
            for (size_t i = 0; i < item.size(); ++i) { input_path[i] = item[i]; }
            input_path[item.size() + 1] = '\0';
        }
        catch (PlbIOException& exception) {
            std::string item = "input";
            input_path = (char*)calloc(item.size() + 1, sizeof(char));
            for (size_t i = 0; i < item.size(); ++i) { input_path[i] = item[i]; }
            input_path[item.size() + 1] = '\0';
        }
        try {
            std::string item;
            doc["parameters"]["path"]["output_path"].read(item);
            output_path = (char*)calloc(item.size() + 1, sizeof(char));
            for (size_t i = 0; i < item.size(); ++i) { output_path[i] = item[i]; }
            output_path[item.size() + 1] = '\0';
        }
        catch (PlbIOException& exception) {
            std::string item = "output";
            output_path = (char*)calloc(item.size() + 1, sizeof(char));
            for (size_t i = 0; i < item.size(); ++i) { output_path[i] = item[i]; }
            output_path[item.size() + 1] = '\0';
        }

        // LB_numerics
        try { doc["parameters"]["LB_numerics"]["delta_P"].read(delta_P); }
        catch (PlbIOException& exception) { delta_P = 0; }
        try {
            std::string tmp;
            doc["parameters"]["LB_numerics"]["track_performance"].read(tmp);
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) { return std::tolower(c); });
            if (tmp.compare("no") == 0 || tmp.compare("false") == 0 || tmp.compare("0") == 0) { track_performance = 0; }
            else if (tmp.compare("yes") == 0 || tmp.compare("true") == 0 || tmp.compare("1") == 0) { track_performance = 1; }
            else { pcout << "track_performance (" << tmp << ") should be either true or false. Terminating the simulation.\n"; return -1; }
        }
        catch (PlbIOException& exception) { track_performance = 0; }
        try { doc["parameters"]["LB_numerics"]["Peclet"].read(Pe); }
        catch (PlbIOException& exception) { Pe = 0; }
        if (delta_P < thrd) { Pe = 0; }
        try { doc["parameters"]["LB_numerics"]["tau"].read(tau); }
        catch (PlbIOException& exception) { tau = 0.8; }
        try { doc["parameters"]["LB_numerics"]["domain"]["dy"].read(dy); }
        catch (PlbIOException& exception) { dy = dx; }
        try { doc["parameters"]["LB_numerics"]["domain"]["characteristic_length"].read(charcs_length); }
        catch (PlbIOException& exception) {
            charcs_length = 0;
            if (Pe > thrd) {
                pcout << "charcs_length must be defined when for transport simulations (Pe > 0). Terminating the simulation.\n"; return -1;
            }
        }
        try {
            std::string unit;
            doc["parameters"]["LB_numerics"]["domain"]["unit"].read(unit);
            if (unit == "m") { charcs_length /= dx; }
            else if (unit == "mm") { charcs_length /= dx; dx *= 1e-3; dy *= 1e-3; }
            else if (unit == "um") { charcs_length /= dx; dx *= 1e-6; dy *= 1e-6; }
            else { pcout << "unit (" << unit << ") must be either m, mm, or um. Terminating the simulation.\n"; return -1; }

        }
        catch (PlbIOException& exception) { charcs_length /= dx; dx *= 1e-6; }

        try {
            doc["parameters"]["LB_numerics"]["domain"]["material_numbers"]["pore"].read(pore_dynamics);
            pcout << "Pore dynamics read from xml file: ";
            for (auto val : pore_dynamics) {
                pcout << val << " ";
            }
            pcout << std::endl;
        }
        catch (PlbIOException& exception) {
            pore_dynamics.push_back(2);
            pcout << "Exception caught for pore dynamics. Assigning default value: 2" << std::endl;
        }

        try {
            doc["parameters"]["LB_numerics"]["domain"]["material_numbers"]["solid"].read(no_dynamics);
            pcout << "Solid dynamics read from xml file: " << no_dynamics << std::endl;
        }
        catch (PlbIOException& exception) {
            no_dynamics = 0;
            pcout << "Exception caught for solid dynamics. Assigning default value: 0" << std::endl;
        }

        try {
            doc["parameters"]["LB_numerics"]["domain"]["material_numbers"]["bounce_back"].read(bounce_back);
            pcout << "Bounce back dynamics read from xml  file: " << bounce_back << std::endl;
        }
        catch (PlbIOException& exception) {
            bounce_back = 1;
            pcout << "Exception caught for bounce back dynamics. Assigning default value: 1" << std::endl;
        }


        try {
            std::string tmp;
            doc["parameters"]["IO"]["read_NS_file"].read(tmp);
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) { return std::tolower(c); });
            if (tmp.compare("no") == 0 || tmp.compare("false") == 0 || tmp.compare("0") == 0) { read_NS_file = 0; }
            else if (tmp.compare("yes") == 0 || tmp.compare("true") == 0 || tmp.compare("1") == 0) { read_NS_file = 1; }
            else { pcout << "read_NS_file (" << tmp << ") should be either true or false. Terminating the simulation.\n"; return -1; }
        }
        catch (PlbIOException& exception) { read_NS_file = 0; }
        try {
            doc["parameters"]["LB_numerics"]["iteration"]["ns_rerun_iT0"].read(ns_rerun_iT0);
            if (ns_rerun_iT0 < 0) {
                pcout << "ns_rerun_iT0 (" << ns_rerun_iT0 << ") must be a positive number. Terminating the simulation.\n";
                return -1;
            }
        }
        catch (PlbIOException& exception) { if (read_NS_file == 1) { pcout << "WARNING: NS checkpoint file is loaded but ns_rerun_iT0 is not provided. Assume no further flow simulation.\n"; ns_rerun_iT0 = 0; } }
        try { doc["parameters"]["LB_numerics"]["iteration"]["ns_update_interval"].read(ns_update_interval); }
        catch (PlbIOException& exception) { ns_update_interval = 1; }
        try { doc["parameters"]["LB_numerics"]["iteration"]["ade_update_interval"].read(ade_update_interval); }
        catch (PlbIOException& exception) { ade_update_interval = 1; }
        try { doc["parameters"]["LB_numerics"]["iteration"]["ns_max_iT1"].read(ns_maxiTer_1); }
        catch (PlbIOException& exception) { ns_maxiTer_1 = 100000; }
        try { doc["parameters"]["LB_numerics"]["iteration"]["ns_max_iT2"].read(ns_maxiTer_2); }
        catch (PlbIOException& exception) { ns_maxiTer_2 = 100000; }
        try { doc["parameters"]["LB_numerics"]["iteration"]["ns_converge_iT1"].read(ns_converge_iT1); }
        catch (PlbIOException& exception) { ns_converge_iT1 = 1e-8; }
        try { doc["parameters"]["LB_numerics"]["iteration"]["ns_converge_iT2"].read(ns_converge_iT2); }
        catch (PlbIOException& exception) { ns_converge_iT2 = 1e-6; }
        try {
            doc["parameters"]["LB_numerics"]["iteration"]["ade_rerun_iT0"].read(ade_rerun_iT0);
            if (ade_rerun_iT0 < 0) {
                pcout << "ade_rerun_iT0 (" << ade_rerun_iT0 << ") must be a positive number. Terminating the simulation.\n";
                return -1;
            }
        }
        catch (PlbIOException& exception) { if (read_ADE_file == 1) { pcout << "WARNING: ADE checkpoint file is loaded but ade_rerun_iT0 is not provided. Assume no further flow simulation.\n"; ade_rerun_iT0 = 0; } }
        try { doc["parameters"]["LB_numerics"]["iteration"]["ade_max_iT"].read(ade_maxiTer); }
        catch (PlbIOException& exception) { ade_maxiTer = 10000000; }
        try { doc["parameters"]["LB_numerics"]["iteration"]["ade_converge_iT"].read(ade_converge_iT); }
        catch (PlbIOException& exception) { ade_converge_iT = 1e-8; }

        // IO
        try {
            std::string tmp;
            doc["parameters"]["IO"]["read_ADE_file"].read(tmp);
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) { return std::tolower(c); });
            if (tmp.compare("no") == 0 || tmp.compare("false") == 0 || tmp.compare("0") == 0) { read_ADE_file = 0; }
            else if (tmp.compare("yes") == 0 || tmp.compare("true") == 0 || tmp.compare("1") == 0) { read_ADE_file = 1; }
            else { pcout << "read_ADE_file (" << tmp << ") should be either true or false. Terminating the simulation.\n"; return -1; }
        }
        catch (PlbIOException& exception) { read_ADE_file = 0; }
        try {
            std::string item;
            doc["parameters"]["IO"]["ns_filename"].read(item);
            ns_filename = (char*)calloc(item.size() + 1, sizeof(char));
            for (size_t i = 0; i < item.size(); ++i) { ns_filename[i] = item[i]; }
            ns_filename[item.size() + 1] = '\0';
        }
        catch (PlbIOException& exception) {
            std::string item = "nsLattice";
            ns_filename = (char*)calloc(item.size() + 1, sizeof(char));
            for (size_t i = 0; i < item.size(); ++i) { ns_filename[i] = item[i]; }
            ns_filename[item.size() + 1] = '\0';
        }
        try {
            doc["parameters"]["IO"]["mask_filename"].read(mask_filename);
        }
        catch (PlbIOException& exception) { mask_filename = "maskLattice"; }

        try {
            doc["parameters"]["IO"]["subs_filename"].read(ade_filename);
        }
        catch (PlbIOException& exception) { ade_filename = "subsLattice"; }

        try {
            doc["parameters"]["IO"]["save_VTK_interval"].read(ade_VTK_iTer);
        }
        catch (PlbIOException& exception) { ade_VTK_iTer = 1000; }

        try {
            doc["parameters"]["IO"]["save_CHK_interval"].read(ade_CHK_iTer);
        }
        catch (PlbIOException& exception) { ade_CHK_iTer = 1000000; }
    }

    catch (PlbIOException& exception) {
        pcout << exception.what() << " Terminating the simulation.\n" << std::endl;
        return -1;
    }

    return 0;

}

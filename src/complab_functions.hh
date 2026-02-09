


#include <climits> 
#include <cfloat> 
#include "palabos3D.h"
#include "palabos3D.hh"
//#include "complab_processors.hh"

#include <map>





using namespace plb;
typedef double T;


#define DESCRIPTOR descriptors::D3Q19Descriptor  
#define BGK descriptors::AdvectionDiffusionD3Q7Descriptor 
#define thrd 1e-14


// a function to initially distribute the pressure linearly.
// This is used only to initializing the flow field
class PressureGradient {
public:
    PressureGradient(T deltaP_, plint nx_) : deltaP(deltaP_), nx(nx_)
    { }
    void operator() (plint iX, plint iY, plint iZ, T& density, Array<T, 3>& velocity) const
    {
        velocity.resetToZero();
        density = (T)1 - deltaP * DESCRIPTOR<T>::invCs2 / (T)(nx - 1) * (T)iX;

    }
private:
    T deltaP;
    plint nx;
};

// This is a function called "writeNsVTI" that writes the fluid flow data in the format of VTK (Visualization Toolkit) image file. The function takes in a MultiBlockLattice3D object, which represents the lattice for simulating fluid flow using the lattice Boltzmann method.

// The function takes in three arguments:

//     lattice: the MultiBlockLattice3D object that contains fluid flow data
//     iter: an integer that represents the current iteration of the simulation
//     nameid: a string that is used as a prefix for the VTK image file name

// Inside the function, first the dimensions of the lattice are obtained and then a VtkImageOutput3D object is created with a specified output file name. The function "computeVelocityNorm" is used to compute the velocity norm of the fluid flow in the lattice, and the function "computeVelocity" is used to compute the three components of the velocity.
// These velocity components and the velocity norm are then written to the VTK file using the "writeData" method of the VtkImageOutput3D object.

void writeNsVTI(MultiBlockLattice3D<T, DESCRIPTOR>& lattice, plint iter, std::string nameid)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    const plint nz = lattice.getNz();
    VtkImageOutput3D<T> vtkOut(createFileName(nameid, iter, 7), 1.);

    vtkOut.writeData<float>(*computeVelocityNorm(lattice, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1)), "velocityNorm", 1.);
    vtkOut.writeData<3, float>(*computeVelocity(lattice, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1)), "velocity", 1.);
}


// This function writes the MultiScalarField3D<int> "geometry" to a VTK file format with the extension .vti. The VTK file contains a 3D grid of points and a scalar value at each point representing the value of the "tag" field in the "geometry" object. The grid has dimensions (nx, ny, nz), which are determined by the size of the "geometry" object. The function creates a VtkImageOutput3D object with the given file name, iter, and precision (1.0). 
// Then, it writes the "tag" field of the "geometry" object to the VTK file using the writeData method of the VtkImageOutput3D object.
void writePorousMediumVTI(MultiScalarField3D<int>& geometry, plint iter, std::string nameid)
{
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();
    const plint nz = geometry.getNz();
    VtkImageOutput3D<T> vtkOut(createFileName(nameid, iter, 7), 1.);
    vtkOut.writeData<float>(*copyConvert<int, T>(geometry, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1)), "tag", 1.0);
}

// writing output files of the SOLUTE domain
  // This function writes the density data of a lattice into a VTK (.vti) file format for visualization purposes.

  // The function takes in a MultiBlockLattice3D<T, BGK> reference, which is the lattice for which the density data is to be written. The second argument is an integer variable iter, which specifies the iteration number of the simulation. The last argument is a string variable nameid, which is used to create the name of the output VTI file.

  // The function starts by getting the dimensions of the lattice through the lattice's getNx() method. Then, it creates a VtkImageOutput3D object with the specified output file name using the createFileName() function. Finally, it writes the density data of the lattice using the computeDensity() method, which is passed the lattice and a 
  // Box3D object specifying the sub-domain from which to compute the density data. The output data is written as a single scalar data component named "Density".

void writeAdvVTI(MultiBlockLattice3D<T, BGK>& lattice, plint iter, std::string nameid)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    const plint nz = lattice.getNz();

    VtkImageOutput3D<T> vtkOut(createFileName(nameid, iter, 7), 1.);
    vtkOut.writeData<T>(*computeDensity(lattice, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1)), "Density", 1.);

    //VtkImageOutput3D<T> vtkOut(createFileName("vtkDensity" + nameid, iter, 6), 1.);
    //vtkOut.writeData<T>(*computeDensity(lattice), "Density", 1.);

}


void writeScalarVTI(MultiScalarField3D<int>& field)
{
    const plint nx = field.getNx();
    const plint ny = field.getNy();
    const plint nz = field.getNz();

    VtkImageOutput3D<T> vtkOut("distanceDomain", 1.);
    vtkOut.writeData<float>(*copyConvert<int, T>(field, Box3D(0, nx - 1, 0, ny - 1, 0, nz - 1)), "tag", 1.0);
}


// load a geometry file with predefined material numbers
void readGeometry(std::string fNameIn, MultiScalarField3D<int>& geometry)
{
    pcout << "Reading the geometry file (" << fNameIn << ").\n";
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();
    const plint nz = geometry.getNz();

    Box3D sliceBox(0, 0, 0, ny - 1, 0, nz - 1);
    std::unique_ptr<MultiScalarField3D<int> > slice = generateMultiScalarField<int>(geometry, sliceBox);

    plb_ifstream geometryFile(fNameIn.c_str());
    if (!geometryFile.is_open()) {
        pcout << "Error: could not open geometry file " << fNameIn << std::endl;
        exit(EXIT_FAILURE);
    }
    for (plint iX = 0; iX < nx - 1; ++iX) {
        geometryFile >> *slice;
        if (iX == 1) {
            copy(*slice, slice->getBoundingBox(), geometry, Box3D(0, 0, 0, ny - 1, 0, nz - 1));
            copy(*slice, slice->getBoundingBox(), geometry, Box3D(iX, iX, 0, ny - 1, 0, nz - 1));
        }
        else if (iX == nx - 2) {
            copy(*slice, slice->getBoundingBox(), geometry, Box3D(iX, iX, 0, ny - 1, 0, nz - 1));
            copy(*slice, slice->getBoundingBox(), geometry, Box3D(nx - 1, nx - 1, 0, ny - 1, 0, nz - 1));
        }
        else {
            copy(*slice, slice->getBoundingBox(), geometry, Box3D(iX, iX, 0, ny - 1, 0, nz - 1));
        }
    }
}

void saveGeometry(std::string fNameIn, MultiScalarField3D<int>& geometry)
{
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();
    const plint nz = geometry.getNz();


    pcout << "Save geometry vti file (" << fNameIn << ").\n";
    VtkImageOutput3D<T> vtkOut(fNameIn, 1.0);
    vtkOut.writeData<float>(*copyConvert<int, T>(geometry, Box3D(1, nx - 2, 0, ny - 1, 0, nz - 1)), "tag", 1.0);
}

// void calculateDistanceFromSolid3D(MultiScalarField3D<int>& distance, plint nodymcs, plint bb, std::vector<std::vector<std::vector<plint>>>& distVec)

// {
//     const plint nx = distance.getNx();
//     const plint ny = distance.getNy();
//     const plint nz = distance.getNz();

//     for (plint iX = 0; iX < nx; ++iX) {
//         for (plint iY = 0; iY < ny; ++iY) {
//             for (plint iZ = 0; iZ < nz; ++iZ) {
//                 plint mask = distance.get(iX, iY, iZ);
//                 if (mask == nodymcs) { distVec[iX][iY][iZ] = -1; }
//                 else if (mask == bb) { distVec[iX][iY][iZ] = 0; }
//                 else { distVec[iX][iY][iZ] = 1; }
//             }
//         }
//     }
//     for (plint iX = 0; iX < nx; ++iX) {
//         for (plint iY = 0; iY < ny; ++iY) {
//             for (plint iZ = 0; iZ < nz; ++iZ) {
//                 if (distVec[iX][iY][iZ] == 1) {
//                     plint lp = 1, r = 0, dist = 0;
//                     while (lp == 1) {
//                         ++r;
//                         std::vector<plint> vx(r + 1), vy(r + 1), vz(r + 1);
//                         for (plint tmp = 0; tmp < r + 1; ++tmp) {
//                             vx[tmp] = tmp;
//                             vy[tmp] = r - tmp;
//                             vz[tmp] = r - tmp;
//                         }
//                         for (plint itx = 0; itx < r + 1; ++itx) {
//                             for (plint ity = 0; ity < r + 1; ++ity) {
//                                 for (plint itz = 0; itz < r + 1; ++itz) {
//                                     plint xp = iX + vx[itx], yp = iY + vy[ity], zp = iZ + vz[itz];
//                                     plint xn = iX - vx[itx], yn = iY - vy[ity], zn = iZ - vz[itz];
//                                     if (xp >= 0 && yp >= 0 && zp >= 0 && xp < nx && yp < ny && zp < nz) {
//                                         if (distVec[xp][yp][zp] == 0) {
//                                             dist = r; lp = 0; break;
//                                         }
//                                     }
//                                     if (xn >= 0 && yn >= 0 && zn >= 0 && xn < nx && yn < ny && zn < nz) {
//                                         if (distVec[xn][yn][zn] == 0) {
//                                             dist = r; lp = 0; break;
//                                         }
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                     if (lp == 0) { distVec[iX][iY][iZ] = dist; }
//                 }
//             }
//         }
//     }
// }

// This function, calculateDistanceFromSolid3D, calculates the distance from each point in a 3D lattice to the nearest solid cell (where the solid is represented by the bb value). It stores the calculated distance in a 3D vector distVec.

// Here's a step-by-step explanation of the function:

//     The function first initializes the variables nx, ny, and nz to the dimensions of the input distance MultiScalarField3D object.

//     It then iterates over all the points in the 3D lattice and checks the value of mask (which is obtained from the distance object) at each point. If the value of mask is equal to nodymcs, the corresponding distVec entry is set to -1. If the value of mask is equal to bb (representing a solid), the corresponding distVec entry is set to 0. In all other cases, the distVec entry is set to 1.

//     The function then iterates over the 3D lattice again. For each point with a distVec value of 1 (which represents non-solid and non-nodymcs cells), it calculates the distance to the nearest solid cell using a while loop. The loop increments the search radius r and checks the neighboring cells at each radius. If a solid cell (with a distVec value of 0) is found at any of the neighboring cells, the distance is set to the current value of r, and the loop is terminated.

//    Once the distance is found for a particular point, the distVec entry for that point is updated with the calculated distance value.

void calculateDistanceFromSolid3D(MultiScalarField3D<int> distance, plint nodymcs, plint bb, std::vector<std::vector<std::vector<plint>>>& distVec)

{
    const plint nx = distance.getNx();
    const plint ny = distance.getNy();
    const plint nz = distance.getNz();

    
    for (plint iX = 0; iX < nx - 1; ++iX) {
        for (plint iY = 0; iY < ny - 1; ++iY) {
            for (plint iZ = 0; iZ < nz - 1; ++iZ) {
                plint mask = distance.get(iX, iY, iZ);
                if (mask == nodymcs) { distVec[iX][iY][iZ] = -1; }
                else if (mask == bb) { distVec[iX][iY][iZ] = 0; }
             
                else { distVec[iX][iY][iZ] = 1; }
 
            } 
        }
    }
    

    
    for (plint iX = 0; iX < nx - 1; ++iX) {
        for (plint iY = 0; iY < ny - 1; ++iY) {
            for (plint iZ = 0; iZ < nz - 1; ++iZ) {
                if ( distVec[iX][iY][iZ] == 1 ) {
                    plint lp = 1, r = 0, dist = 0;
                    while (lp == 1) {
                        ++r;
                        std::vector<plint> vx(r + 1), vy(r + 1), vz(r + 1);
                        for (plint tmp = 0; tmp < r + 1; ++tmp) { vx[tmp] = tmp; vy[tmp] = r - tmp; vz[tmp] = r - tmp; }
                        for (plint it = 0; it < r + 1; ++it) {
                            plint xp = iX + vx[it], yp = iY + vy[it], zp = iZ + vz[it], xn = iX - vx[it], yn = iY - vy[it], zn = iZ - vz[it];
                            if (xp >= 0 && yp >= 0 && zp >= 0 && xp < nx && yp < ny && zp < nz) {
                                if ( distVec[xp][yp][zp] == 0 ) {
                                    dist = r; lp = 0; break;
                                }
                            }
                            if (xp >= 0 && yn >= 0 && zn >= 0 && xp < nx && yn < ny && zn < nz) {
                                if (distVec[xp][yn][zn] == 0 ) {
                                    dist = r; lp = 0; break;
                                }
                            }
                            if (xp >= 0 && yp >= 0 && zn >= 0 && xp < nx && yp < ny && zn < nz) {
                                if (distVec[xp][yp][zn] == 0) {
                                    dist = r; lp = 0; break;
                                }
                            }
                            if (xp >= 0 && yn >= 0 && zp >= 0 && xp < nx && yn < ny && zp < nz) {
                                if (distVec[xp][yn][zp] == 0) {
                                    dist = r; lp = 0; break;
                                }
                            }
                            if (xn >= 0 && yp >= 0 && zp >= 0 && xn < nx && yp < ny && zp < nz) {
                                if ( distVec[xn][yp][zp] == 0) {
                                    dist = r; lp = 0; break;
                                }
                            }
                            if (xn >= 0 && yp >= 0 && zn >= 0 && xn < nx && yp < ny && zn < nz) {
                                if (distVec[xn][yp][zn] == 0) {
                                    dist = r; lp = 0; break;
                                }
                            }
                            if (xn >= 0 && yn >= 0 && zp >= 0 && xn < nx && yp < ny && zp < nz) {
                                if (distVec[xn][yn][zp] == 0) {
                                    dist = r; lp = 0; break;
                                }
                            }
                            if (xn >= 0 && yn >= 0 && zn >= 0 && xn < nx && yn < ny && zn < nz) {
                                if ( distVec[xn][yn][zn] == 0 ) {
                                    dist = r; lp = 0; break;
                                }
                            }
                        }
                    }
                    if (lp == 0) { distVec[iX][iY][iZ] = dist; }
                }
            }


        }
    }
    
}

// This function sets up the lattice for fluid flow simulation using the lattice Boltzmann method. It takes in a MultiBlockLattice3D and a MultiScalarField3D for the lattice and geometry respectively, and several other parameters such as pressure difference (deltaP), fluidOmega, pore space, bounceback boundary, and no dynamics boundary.

// The function initializes the dynamics of the lattice using the IncBGKdynamics, BounceBack, and NoDynamics classes for fluid, boundary, and no dynamics regions respectively. It also sets up pressure boundaries at the west and east sides of the domain using addPressureBoundary0N and addPressureBoundary0P functions from OnLatticeBoundaryCondition3D class.

// The pressure difference is set using setBoundaryDensity function, and the lattice is initialized at equilibrium using initializeAtEquilibrium function. Finally, the lattice is initialized, and the boundaryCondition object is deleted.



void NSdomainSetup(MultiBlockLattice3D<T, DESCRIPTOR>& lattice, OnLatticeBoundaryCondition3D<T, DESCRIPTOR>* boundaryCondition, MultiScalarField3D<int>& geometry, T deltaP,

    T fluidOmega, std::vector<plint> pore, plint bounceback, plint nodymcs)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    const plint nz = lattice.getNz();




    Box3D west(0, 0, 0, ny - 1, 0, nz - 1);
    Box3D east(nx - 1, nx - 1, 0, ny - 1, 0, nz - 1);



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


// The soluteDomainSetup function is used to set up the lattice for the advection-diffusion of solute in the simulation domain. The function takes in several arguments, including a reference to the lattice object, a pointer to the boundaryCondition object, a reference to the geometry object, several parameters related to the simulation (e.g. substrOmega for relaxation time, rho0 for initial density), and information about the boundary conditions (e.g. left_btype and right_btype for boundary type, left_BC and right_BC for boundary concentration).

// The function first calculates the dimensions of the lattice using getNx(), getNy(), and getNz() and defines two boxes (west and east) for the west and east boundaries respectively. It then sets up the dynamics for the lattice, including the advection-diffusion dynamics in the entire domain (lattice.getBoundingBox()) and any specific dynamics for the pore space or boundary conditions. The function then sets up the boundary conditions for the lattice using addTemperatureBoundary0N() and addTemperatureBoundary0P(), with the type and concentration of each boundary specified based on the input arguments. Finally, 
// the lattice is initialized using initialize() and the boundaryCondition pointer is deleted to free up memory.

// solute domain boundary conditions. No flow boundaries at the top and bottom.
void soluteDomainSetup(MultiBlockLattice3D<T, BGK>& lattice, OnLatticeAdvectionDiffusionBoundaryCondition3D<T, BGK>* boundaryCondition, MultiScalarField3D<int>& geometry,
    T substrOmega, std::vector<plint> pore, plint bounceback, plint nodymcs,
    T rho0, bool left_btype, bool right_btype, T left_BC, T right_BC)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    const plint nz = lattice.getNz();


    Box3D west(0, 0, 0, ny - 1, 0, nz - 1);
    Box3D east(nx - 1, nx - 1, 0, ny - 1, 0, nz - 1);
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
    else { integrateProcessingFunctional(new FlatAdiabaticBoundaryFunctional3D<T, BGK, 0, -1>, west, lattice, processorLevelBC); }

    boundaryCondition->addTemperatureBoundary0P(east, lattice);
    if (right_btype == 0) { setBoundaryDensity(lattice, east, right_BC); }
    else { integrateProcessingFunctional(new FlatAdiabaticBoundaryFunctional3D<T, BGK, 0, +1>, east, lattice, processorLevelBC); }

    // Init lattice
    Array<T, 3> u0(0., 0., 0.);
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), rho0, u0);


    lattice.initialize();
    delete boundaryCondition;
}

// This code defines a function computePermeability that takes as input a MultiBlockLattice3D object nsLattice, a scalar nsNu, a scalar deltaP, and a Box3D object domain. The function computes the permeability of the domain in the x-direction, which is the direction of fluid flow.

// The function first computes the mean velocity of the fluid in the x-direction by calling the computeVelocityComponent function, which extracts the x-component of the velocity from nsLattice within the specified domain. The average of this velocity component is then computed using the computeAverage function.

// Next, the function prints the mean velocity, the pressure gradient deltaP divided by the length of the domain in the x-direction, and the computed permeability, which is calculated by multiplying the kinematic viscosity nsNu by the mean velocity and dividing by the pressure gradient.




T computePermeability(MultiBlockLattice3D<T, DESCRIPTOR>& nsLattice, T nsNu, T deltaP, Box3D domain)
{
    pcout << "Computing the permeability." << std::endl;

    // Compute only the x-direction of the velocity (direction of the flow).
    plint xComponent = 0;           //considered in the x-direction
    plint nx = nsLattice.getNx();  //Retrieves the number of lattice points in the x-direction.
    plint ny = nsLattice.getNy();
    plint nz = nsLattice.getNz();

    // This function calculates the x-component of velocity for each node within the specified domain of the lattice.
    // It accesses the lattice's distribution functions to compute the local velocity at each point, then aggregates these
    // velocities and returns a pointer to the resulting data. The '*' indicates dereferencing the returned pointer to access the actual velocity data.

    //// This averaging over total area (including solid regions) gives Darcy velocity

    T meanU = computeAverage(*computeVelocityComponent(nsLattice, Box3D (1,nx-2,0,ny-1,0,nz-1), xComponent));

    pcout << "Average velocity (meanU) = " << meanU                                 << std::endl;
    // This line calculates the pressure gradient across the lattice in the x-direction. 'deltaP' represents the total pressure difference,
    // and 'nx - 1' is the number of intervals between nodes in the x-direction. The result is the pressure change per unit length, which
    // is used in calculating the permeability.
    pcout << "Grad P               = " << deltaP / (T)(nx - 1)                      << std::endl; 
    pcout << "Permeability         = " << nsNu * meanU / (deltaP / (T)(nx - 1))     << std::endl;
    pcout << "Lattice viscosity nu = " << nsNu                                      << std::endl;

    return nsNu * meanU / (deltaP / (T)(nx - 1));
}





// This code defines a function computeResidenceTime that takes as input a MultiBlockLattice3D object nsLattice, 
// a MultiScalarField3D object geometry, and a scalar meanU. The function computes the residence time of the fluid in the domain.

// To derive the equation t = φL/ν from the general equation for fluid residence time, t = V/A, we need to express the volume (V) and the cross-sectional area (A) in terms of the system's porosity (φ), length (L), and average linear velocity of the fluid (ν).
// 
// V = A * L
// where A is the cross-sectional area .
// However, not all of the volume in the system is available for fluid flow due to the porosity (φ). The effective volume for fluid flow (V_eff) is given by:
// V_eff = φ * V                    V_eff = φ * A * L
// Now, let's consider the average linear velocity of the fluid (ν). This velocity is defined as the volume flow rate (Q) divided by the cross-sectional area (A):
// ν = Q/A
// We need to find an expression for the volume flow rate (Q) in terms of the effective volume (V_eff). Since the residence time (t) is the time it takes for the fluid to travel through the entire system, we can write:
// t = V_eff / Q
// By rearranging the equation, we get:
// Q = V_eff / t
// Now, let's substitute the expression for the volume flow rate (Q) into the equation for average linear velocity (ν):
// ν = (V_eff / t) / A
// Rearranging the equation to solve for the residence time (t), we get:
// t = V_eff / (ν * A)
// Now, substitute the expression for the effective volume (V_eff) that we derived earlier:
// t = (φ * A * L) / (ν * A)
//
// t = φL/ν
// This is the simplified equation for fluid residence time in a porous system, where t is the residence time, φ is the porosity, L is the length of the system, and ν is the average linear velocity of the fluid.


T computeResidenceTime(MultiBlockLattice3D<T, DESCRIPTOR>& nsLattice, MultiScalarField3D<int>& geometry, T meanU, const std::vector<long int>& pore_dynamics, T dx)
{
    pcout << "Computing the residence time." << std::endl;

    plint nx = nsLattice.getNx();
    plint ny = nsLattice.getNy();
    plint nz = nsLattice.getNz();
    T porosity = 0;
    plint num_pore_cells = 0;

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            for (plint iZ = 0; iZ < nz; ++iZ) {


                // Code identifies pore cells by comparing each cell's value in the geometry field to the values in the 
                // pore_dynamics vector. If a match is found, the cell is considered a pore.

                for (long int pore_dynamic : pore_dynamics) {
                    if (geometry.get(iX, iY, iZ) == pore_dynamic) {
                        num_pore_cells++;
                        break;
                    }
                }
            }
        }
    }
    // This line calculates the porosity by dividing the number of pore cells by the total number of cells in the lattice.
    porosity = (T)num_pore_cells / (nx * ny * nz);

    // This line calculates the system length in the x-direction by multiplying nx by the appropriate conversion factor (1 in this case).
    // the conversion factor is set to 1. This means that the length of the system in the x-direction is assumed to be directly proportional
    // to the number of cells in the x-direction (nx). In this case, the length of the system is equal to the number of cells in the x-direction.

    //T length = nx * 1; // Use the appropriate conversion factor if needed

    T length = nx * dx;

    // This line calculates the residence time by multiplying the porosity by the system length and dividing by the mean fluid velocity.
    T residence_time = porosity * length / meanU;

    pcout << "Porosity: " << porosity << std::endl;
    pcout << "System Length: " << length << std::endl;
    pcout << "Mean Fluid Velocity: " << meanU << std::endl;
    pcout << "Residence Time: " << residence_time << " s" << std::endl;   

    return residence_time;
}



// This function sets up the dynamics of a scalar field lattice based on a given vector of mask values and corresponding vector of dynamics (omegas).

// The function takes as input a MultiBlockLattice3D object, a MultiScalarField3D object, a vector of mask values (mtrvec), and a vector of dynamics values (omegavec).

// The function first sets the default dynamics of the lattice to be AdvectionDiffusionBGKdynamics with a parameter of 0.0. Then, for each mask value in the mtrvec vector, the corresponding dynamics value from the omegavec vector is assigned to the lattice using the defineDynamics function.

// Finally, the function initializes the lattice at equilibrium with 0 concentration and a zero flux.

// Overall, this function sets up the dynamics for a scalar field lattice based on user-defined mask and dynamics vectors.


void scalarDomainDynamicsSetupFromVectors(MultiBlockLattice3D<T, BGK>& lattice, MultiScalarField3D<int>& geometry, std::vector<plint> mtrvec, std::vector<T> omegavec)
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
    Array<T, 3> jEq(0., 0., 0.);
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), 0., jEq);

    lattice.initialize();
}


// This function sets up the dynamics of the scalar field in the lattice based on the geometry of the domain. It takes as input a MultiBlockLattice3D object (lattice) and a MultiScalarField3D object (geometry) that defines the domain geometry.

// The function iterates through all the cells in the domain (iX, iY, iZ) and gets the geometry value at each location. It then assigns the corresponding dynamics to the cell using the defineDynamics function. The AdvectionDiffusionBGKdynamics class is used to define the dynamics of the scalar field. The dynamics are set to zero if the geometry value is zero, and to a non-zero value (given by the geometry value) if the geometry value is non-zero.

// Finally, the function initializes the lattice and equilibrium of the scalar field.

void scalarDomainDynamicsSetupFromGeometry(MultiBlockLattice3D<T, BGK>& lattice, MultiScalarField3D<int>& geometry, plint nx, plint ny, plint nz)
{
    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            for (plint iZ = 0; iZ < nz; ++iZ) {
                plint geom = geometry.get(iX, iY, iZ);
                defineDynamics(lattice, iX, iY, iZ, new AdvectionDiffusionBGKdynamics<T, BGK>((T)geom));

            }
        }
    }
    // Init lattice
    Array<T, 3> jEq(0., 0., 0.);
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), 0., jEq);

    lattice.initialize();
}

// The function gridSearch3D performs a grid search algorithm to find the distance between each point in a 3D scalar field and the nearest point belonging to a set of specified masks. The function takes as input a 3D scalar field geometry, a vector of 3D vectors distVec to store the resulting distances, two integer values bb and solid representing the values for the bounding box and solid regions respectively, and a vector of integer values pore representing the masks to search for.

// The algorithm iterates over each point in the scalar field geometry. If the current point is already part of the bounding box or solid regions, the distance is set to -1 in the corresponding location in distVec. If not, the algorithm iteratively searches for the nearest point belonging to one of the specified masks. It does so by expanding a search radius around the current point, and checking all points within the search radius. Once a point belonging to one of the specified masks is found, the search is stopped and the distance is stored in the corresponding location in distVec.

// The resulting distVec vector can be used for various purposes, such as determining the distance to the nearest pore in a porous medium, or generating a distance-based mask for use in simulations.


void gridSearch3D(MultiScalarField3D<int> geometry, std::vector< std::vector< std::vector<plint> > >& distVec, plint bb, plint solid, std::vector<plint> pore)
{
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();
    const plint nz = geometry.getNz();

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            for (plint iZ = 0; iZ < nz; ++iZ) {
                plint geom = geometry.get(iX, iY, iZ);
                bool flag0 = false;

                if (!flag0) {
                    plint iR = 0;
                    bool flag1 = false;
                    while (!flag1) {
                        ++iR;
                        for (plint rx = 0; rx < iR; ++rx) {
                            for (plint ry = 0; ry < iR; ++ry) {
                                for (plint rz = 0; rz < iR; ++rz) {
                                    std::vector<std::vector<plint>> neighbors = {
                                        {iX + rx, iY + ry, iZ + rz},
                                        {iX + rx, iY - ry, iZ + rz},
                                        {iX - rx, iY + ry, iZ + rz},
                                        {iX - rx, iY - ry, iZ + rz},
                                        {iX + rx, iY + ry, iZ - rz},
                                        {iX + rx, iY - ry, iZ - rz},
                                        {iX - rx, iY + ry, iZ - rz},
                                        {iX - rx, iY - ry, iZ - rz},
                                    };

                                    for (const auto& neighbor : neighbors) {
                                        plint nx = neighbor[0];
                                        plint ny = neighbor[1];
                                        plint nz = neighbor[2];

                                        if (nx >= 0 && nx < geometry.getNx() && ny >= 0 && ny < geometry.getNy() && nz >= 0 && nz < geometry.getNz()) {
                                            plint mask = geometry.get(nx, ny, nz);

                                            for (size_t iP = 0; iP < pore.size(); ++iP) {
                                                if (mask == pore[iP]) {
                                                    flag1 = true;
                                                    distVec[iX][iY][iZ] = iR;
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
                            if (flag1) { break; }
                        }
                    }
                }
                else if (geom == bb || geom == solid) { distVec[iX][iY][iZ] = -1; }
                else { distVec[iX][iY][iZ] = 0; }
            }
        }
    }
}



// void gridSearch3D(MultiScalarField3D<int> geometry, std::vector<std::vector<std::vector<plint>>> &distVec, plint bb, plint solid, std::vector<plint> pore)
// {
//     const plint nx = geometry.getNx();
//     const plint ny = geometry.getNy();
//     const plint nz = geometry.getNz();
//     for (plint iX = 0; iX < nx; ++iX) {
//         for (plint iY = 0; iY < ny; ++iY) {
//             for (plint iZ = 0; iZ < nz; ++iZ) {
//                 plint geom = geometry.get(iX, iY, iZ);
//                 if (geom == bb || geom == solid) {
//                     distVec[iX][iY][iZ] = -1;
//                 }
//                 else {
//                     bool flag1 = 0;
//                     plint iR = 0;
//                     while (flag1 == 0) {
//                         ++iR;
//                         for (plint rx = 0; rx < iR; ++rx) {
//                             for (plint ry = 0; ry < iR; ++ry) {
//                                 for (plint rz = 0; rz < iR; ++rz) {
//                                     std::vector<plint> dirs = {1, -1};
//                                     for (plint dx : dirs) {
//                                         for (plint dy : dirs) {
//                                             for (plint dz : dirs) {
//                                                 plint x = iX + dx * rx;
//                                                 plint y = iY + dy * ry;
//                                                 plint z = iZ + dz * rz;
//                                                 if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
//                                                     plint mask = geometry.get(x, y, z);
//                                                     if (std::find(pore.begin(), pore.end(), mask) != pore.end()) {
//                                                         flag1 = 1;
//                                                         distVec[iX][iY][iZ] = iR;
//                                                         break;
//                                                     }
//                                                 }
//                                             }
//                                             if (flag1 == 1) break;
//                                         }
//                                         if (flag1 == 1) break;
//                                     }
//                                     if (flag1 == 1) break;
//                                 }
//                                 if (flag1 == 1) break;
//                             }
//                             if (flag1 == 1) break;
//                         }
//                     }
//                     if (flag1 == 0) distVec[iX][iY][iZ] = 0;
//                 }
//             }
//         }
//     }
// }



// This function initializes the density of a second lattice (lattice2) based on the density of a first lattice (lattice1).

// The function iterates through all lattice sites in lattice1 and computes the density s at each site. It then retrieves the populations g at the corresponding site in lattice2 and adds a fraction of the density s to each population. The population fractions are based on the weights of the discrete velocity directions in the lattice Boltzmann method.

// The effect of this function is to initialize lattice2 with a non-zero density, based on the density in lattice1, in order to start the simulation with non-zero velocity and fluid flow.


// void initializeLatticeDensity(MultiBlockLattice3D<T, BGK>& lattice1, MultiBlockLattice3D<T, BGK>& lattice2)
// {
//     const plint nx = lattice1.getNx();
//     const plint ny = lattice1.getNy();
//     const plint nz = lattice1.getNz();

//     for (plint iX = 0; iX < nx; ++iX) {
//         for (plint iY = 0; iY < ny; ++iY) {
//             for (plint iZ = 0; iZ < nz; ++iZ) {
//                 T s = lattice1.get(iX, iY, iZ).computeDensity();
//                 Array<T, 7> g;
//                 lattice2.get(iX, iY, iZ).getPopulations(g);
//                 g[0] += (T)(s) / 4; g[1] += (T)(s) / 8; g[2] += (T)(s) / 8; g[3] += (T)(s) / 8; g[4] += (T)(s) / 8; g[5] += (T)(s) / 8; g[6] += (T)(s) / 8;
//                 lattice2.get(iX, iY, iZ).setPopulations(g);
//             }
//         }
//     }
// }


// This code defines the dynamics of a mask lattice based on the density values of another lattice. The first lattice (lattice1) is used to calculate the density at each point in the domain, and if the density at that point is greater than a given threshold (fbM), then the dynamics of the corresponding point in the second lattice (lattice2) are set to AdvectionDiffusionBGKdynamics with omega = 1, indicating that the point is part of the fluid phase. Otherwise, the dynamics are set to AdvectionDiffusionBGKdynamics with omega = 0, indicating that the point is part of the solid phase.

// The function takes in two MultiBlockLattice3D objects (lattice1 and lattice2), representing the two lattices with different dynamics, as well as a threshold value (fbM) used to determine whether a point is part of the fluid or solid phase. It first iterates through all the points in the domain of the lattices, calculating the density at each point from lattice1. Then, if the density at a given point is greater than the threshold, it sets the dynamics of the corresponding point in lattice2 to fluid dynamics (omega = 1). Finally, it initializes lattice2 to equilibrium and initializes it.

// Overall, this code can be useful for defining the fluid-solid interface in simulations, where the dynamics of each point are dependent on whether it is part of the fluid or solid phase.


// void defineMaskLatticeDynamics(MultiBlockLattice3D<T, BGK>& lattice1, MultiBlockLattice3D<T, BGK>& lattice2, T fbM)
// {
//     const plint nx = lattice1.getNx();
//     const plint ny = lattice1.getNy();
//     const plint nz = lattice1.getNz();

//      // Set a constant value for omega.
//     T omega = 1.0;

//     for (plint iX = 0; iX < nx; ++iX) {
//         for (plint iY = 0; iY < ny; ++iY) {
//             for (plint iZ = 0; iZ < nz; ++iZ) {
        
//                 defineDynamics(lattice2, iX, iY, iZ, new AdvectionDiffusionBGKdynamics<T, BGK>(omega));
//             }
//         }
//     }
//     // Init lattice
//     Array<T, 3> jEq(0., 0., 0.);
//     initializeAtEquilibrium(lattice2, lattice2.getBoundingBox(), 0., jEq);

//     lattice2.initialize();


// }

void initSubstrateMaskLatticeDensity(MultiBlockLattice3D<T, BGK>& lattice1, MultiBlockLattice3D<T, BGK>& lattice2)
{
    const plint nx = lattice1.getNx();
    const plint ny = lattice1.getNy();
    const plint nz = lattice1.getNz();

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            for (plint iZ = 0; iZ < nz; ++iZ) {
                T substrateConcentration = lattice1.get(iX, iY, iZ).computeDensity();
                Array<T, 7> g;
                lattice1.get(iX, iY, iZ).getPopulations(g);

                // Modify this logic based on your requirements
                //g[0] += (T)(substrateConcentration) / 4;  g[1] += (T)((substrateConcentration)/8; g[2]+=(T) (substrateConcentration)/8; g[3]+=(T) (substrateConcentration)/8; g[4]+=(T) (substrateConcentration)/8; g[5] += (T)(substrateConcentration) / 8; g[6] += (T)(substrateConcentration) / 8; 
                

                lattice2.get(iX, iY, iZ).setPopulations(g);
            }
        }
    }
}

void defineMaskLatticeDynamics(MultiBlockLattice3D<T, BGK>& lattice1, MultiBlockLattice3D<T, BGK>& lattice2, T fbM)
{
    const plint nx = lattice1.getNx();
    const plint ny = lattice1.getNy();
    const plint nz = lattice1.getNz();

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            for (plint iZ = 0; iZ < nz; ++iZ) {
                T substrateConcentration = lattice1.get(iX, iY, iZ).computeDensity();
                T omega = 0.;

                if (substrateConcentration > fbM) {
                    omega = 1.;
                }

                defineDynamics(lattice2, iX, iY, iZ, new AdvectionDiffusionBGKdynamics<T, BGK>(omega));
            }
        }
    }

    // Init lattice
    Array<T, 3> jEq(0., 0., 0.); // Adjusted for 3D
    initializeAtEquilibrium(lattice2, lattice2.getBoundingBox(), 0., jEq);

    lattice2.initialize();
}



int initialize_complab(char *&main_path, char *&src_path, char *&input_path, char *&output_path, char *&ns_filename, std::string &ade_filename, std::string &geom_filename,
    std::string &mask_filename, bool &read_NS_file, plint &ns_rerun_iT0, T &ns_converge_iT1, T &ns_converge_iT2, plint &ns_maxiTer_1, plint &ns_maxiTer_2, plint& ns_update_interval, plint& ade_update_interval,
    bool& read_ADE_file, plint& ade_rerun_iT0, plint& ade_VTK_iTer, plint& ade_CHK_iTer, T& ade_converge_iT, plint& ade_maxiTer, plint& nx, plint& ny, plint& nz, T &dx, T& dy, T& dz, T& delta_P, T& tau,
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
            doc["parameters"]["LB_numerics"]["domain"]["nz"].read(nz);
            pcout << "Read nz: " << nz << std::endl;
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

                // Read the name of the substrates and add them to the vector
                try { doc["parameters"]["chemistry"][chemname]["name_of_substrates"].read(vec_subs_names); 
                pcout << "Read name_of_substrates for " << chemname << ": " << vec_subs_names.back() << std::endl;}
                catch (PlbIOException& exception) { vec_subs_names.push_back("substrate_" + std::to_string(iT)); }

                // Read the substrate diffusion coefficients in the pore and add them to the vector
                try { doc["parameters"]["chemistry"][chemname]["substrate_diffusion_coefficients"]["in_pore"].read(D0); solute_poreD.push_back(D0); 
                pcout << "Read substrate_diffusion_coefficients in_pore for " << chemname << ": " << D0 << std::endl;}
                catch (PlbIOException& exception) { solute_poreD.push_back(1e-9); }
                if (std::abs(D0) > thrd) { soluteDindex = 1; }

                try { doc["parameters"]["chemistry"][chemname]["initial_concentration"].read(c0); vec_c0.push_back(c0); 
                pcout << "Read initial_concentration for " << chemname << ": " << c0 << std::endl;}
                catch (PlbIOException& exception) { vec_c0.push_back(0.0); }

                // Read left and right boundary types and conditions
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

                // Read reaction type and solver type
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

            // Check if the size of each vector matches the number of substrates
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


        //3. Use substrates as species
        

       


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
            //src_path = (char*)calloc(item.size() + 1, sizeof(char));
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
        try { doc["parameters"]["LB_numerics"]["domain"]["dz"].read(dz); }
        catch (PlbIOException& exception) { dz = dx; }
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
            if (unit == "m") { charcs_length /= dx; /* do nothing */ }

            // else if (unit == "mm") { charcs_length /= dx; dx *= 1e-3; }
            // else if (unit == "um") { charcs_length /= dx; dx *= 1e-6; }

            //suggested modification for xml reading conflict of dx  with dy and dz

            else if (unit == "mm") { charcs_length /= dx; dx *= 1e-3; dy *= 1e-3; dz *= 1e-3; }
            else if (unit == "um") { charcs_length /= dx; dx *= 1e-6; dy *= 1e-6; dz *= 1e-6; }

            else { pcout << "unit (" << unit << ") must be either m, mm, or um. Terminating the simulation.\n"; return -1; }

        }
        catch (PlbIOException& exception) { charcs_length /= dx; dx *= 1e-6; }
        //try { doc["parameters"]["LB_numerics"]["domain"]["material_numbers"]["pore"].read(pore_dynamics); }
        //catch (PlbIOException& exception) { pore_dynamics.push_back(2); }
        //try { doc["parameters"]["LB_numerics"]["domain"]["material_numbers"]["solid"].read(no_dynamics); }
        //catch (PlbIOException& exception) { no_dynamics = 0; }
        //try { doc["parameters"]["LB_numerics"]["domain"]["material_numbers"]["bounce_back"].read(bounce_back); }
        //catch (PlbIOException& exception) { bounce_back = 1; }

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

        // chemistry

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












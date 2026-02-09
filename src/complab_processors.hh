#include "../defineKinetics.hh"
#include <random>
#include <iomanip>
//#include "equilibrium_solver.h"

using namespace plb;
typedef double T;

#define DESCRIPTOR descriptors::D3Q19Descriptor  // Cs2 = 1/3
#define BGK descriptors::AdvectionDiffusionD3Q7Descriptor // Cs2 = 1/4


/* ===============================================================================================================
   =============================================== DATA PROCESSORS ===============================================
   =============================================================================================================== */

// The run_kinetics class is a subclass of LatticeBoxProcessingFunctional3D. This class is responsible for 
// calculating the reaction kinetics of the system, i.e., it computes how the concentrations of the different 
// substrates in the system change due to the reactions in the system. Unlike update_rxnLattices, which applies 
// changes in concentrations to the lattice, run_kinetics only computes these changes based on the current 
// state of the system and the reaction kinetics. These changes are then applied to the lattice in the 
// update_rxnLattices step.

// The constructor of the run_kinetics class initializes several member variables. These include the size of 
// the simulation domain (nx), the number of substrates (subsNum), the time step size (dt), a two-dimensional 
// vector of kinetics constants (vec2_Kc_kns), and mask numbers for solid and boundary (solid, bb).

// The process method of this class calculates the reaction kinetics for each cell in the domain. It does this 
// by first obtaining the absolute location of the cell in the global simulation domain. Then, it iterates over 
// each cell in the domain, skipping those at the boundary. For each valid cell, it computes the relative 
// displacement to the mask lattice, rounds the mask value to the nearest integer, and only proceeds if the 
// mask is neither solid nor boundary. It then calculates the relative displacements for all lattices, constructs 
// the concentration vector, defines the reaction kinetics with the given concentration, substrate rates, and mask, 
// and finally modifies the populations of the lattice based on the calculated substrate rates. 

// The appliesTo method overrides the parent class method to specify that this functional applies to both the 
// bulk and the envelope of the domain.

// The clone method allows for creating a copy of this object, which is useful in scenarios where the exact same 
// computations need to be performed on multiple different domains or data sets.

// The getTypeOfModification method specifies what kind of data this functional modifies. In this case, it modifies 
// static variables, i.e., the populations of the lattice.

template<typename T, template<typename U> class Descriptor>
class run_kinetics : public LatticeBoxProcessingFunctional3D<T, Descriptor>
{
public:
   // Constructor: initialize the members variables
   run_kinetics(plint nx_, plint subsNum_, T dt_, std::vector< std::vector<T>> vec2_Kc_kns_, plint solid_, plint bb_)
       : nx(nx_), subsNum(subsNum_), dt(dt_), vec2_Kc_kns(vec2_Kc_kns_), solid(solid_), bb(bb_), dCloc(subsNum_), maskLloc(2 * (subsNum_))
   {}

          virtual void process(Box3D domain, std::vector<BlockLattice3D<T, Descriptor>*> lattices) {
            // Get the location (origin or reference point) of the first lattice in the vector.
            Dot3D absoluteOffset = lattices[0]->getLocation();

            // Start a loop iterating over the x-axis of the given domain (Box3D).
            for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
                // Calculate the absolute x-position by adding the current iX to the x-offset from the first lattice.
                plint absX = iX + absoluteOffset.x;

                // Check if the absolute x-position is within the boundaries (excluding the outermost layers).
                if (absX > 0 && absX < nx - 1) {
                    // Start a loop iterating over the y-axis of the given domain.
                    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                        // Start a loop iterating over the z-axis of the given domain.
                        for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
                                // Calculate the relative displacement between the first lattice and the lattice represented by maskLloc.
                                // This might be used to get positions from a reference lattice compared to another.
                                Dot3D maskOffset = computeRelativeDisplacement(*lattices[0], *lattices[maskLloc]);
                                // Fetch the density from the lattice at maskLloc, adjusted for the displacement, and round it to the nearest integer.
                                // This potentially retrieves a value (like a label or marker) for the current cell being processed.
                                plint mask = util::roundToInt(lattices[maskLloc]->get(iX + maskOffset.x, iY + maskOffset.y, iZ + maskOffset.z).computeDensity());
                                if (mask != solid && mask != bb) {

                                    // Kinetics calculations

                                    // For a particular cell (iX, iY, iZ), the kinetics calculations first determine the current concentrations.
                                    std::vector<Dot3D> vec_offset;
                                    for (plint iT = 0; iT < maskLloc; ++iT) {
                                        vec_offset.push_back(computeRelativeDisplacement(*lattices[0], *lattices[iT]));
                                    }
                                
                                
                                    std::vector<T> conc, subs_rate(subsNum, 0);
                        

                                    for (plint iS = 0; iS < subsNum; ++iS) {
                                        plint iXs = iX + vec_offset[iS].x, iYs = iY + vec_offset[iS].y, iZs = iZ + vec_offset[iS].z;
                                        T c0 = lattices[iS]->get(iXs, iYs, iZs).computeDensity();
                                        if (c0 < thrd) { c0 = 0; }
                                        conc.push_back(c0);
                                        //pcout << "Substance " << iS << " concentration at offset [" << iXs << ", " << iYs << ", " << iZs << "] is: " << c0 << std::endl;
                                    }

                                    // Before calling defineRxnKinetics
                                    //std::cout << "Concentration before reaction: " << conc[0] << std::endl;

                                    // Using the current concentrations, reaction kinetics are defined 
                                    defineRxnKinetics(conc, subs_rate, mask);

                                    // After calling defineRxnKinetics
                                    //std::cout << "Concentration after reaction: " << conc[0] << std::endl;
                                    
                                
                                    // then the populations of the lattice are modified based on calculated substrate rates.
                                    // Updating Concentrations (dC):
                                    // Each substrate's concentration change (dC) is calculated by multiplying the reaction rate (subs_rate[iS]) with a time step (dt).
                                    // If the magnitude of dC is significant (greater than thrd), the lattice populations are updated to reflect this change in concentration. 
                                    // The update is distributed among different components of the lattice population (g[0] to g[6]).

                                    // Based on the code, this model shows the updated concentrations in the lattice after implementing the concentration change dC, not just dC itself.

                                    // Specifically:

                                    //     dC is calculated based on the reaction rates and current concentrations. It represents the local concentration change for each substrate due to reactions.
                                    //     dC is then distributed across the lattice populations g[0]-g[6] for that substrate's lattice.
                                    //     The lattice populations are updated to incorporate this change dC.
                                    //     After streaming and collision steps, the updated lattice holds the new concentrations reflecting the changes dC.

                                    // So dC acts as an intermediate variable that affects how the lattice populations are updated. But the end result is that the lattice stores the updated concentrations after applying dC.

                                    // The concentrations are not explicitly stored or updated in a separate vector. They are implicitly represented in the lattice populations.

                                    // You could think of it like:

                                    // Original concentration = f(original lattice populations)

                                    // Updated concentration = f(updated lattice populations)

                                    // Where f() represents the mapping from lattice populations to concentration.

                                    // The populations are updated according to dC, and then represent the new concentrations after the reactions and updates have occurred.

                                    // The model shows the final concentrations in the lattice after implementing dC, not just dC itself. The lattice holds the state of the system reflecting the concentration changes.


                                    // for (plint iS = 0; iS < subsNum; ++iS) {
                                        
                                    // // Print the time step size 'dt'
                                    // //std::cout << "Time step size (dt): " << dt << std::endl;
                                    // T dC = subs_rate[iS] * dt;
                                    // std::cout << "Reaction rate (subs_rate) for substrate " << iS << ": " << subs_rate[iS] << std::endl;
                                    // std::cout << "dC for substrate " << iS << ": " << dC << std::endl;
                                    // if (dC > thrd || dC < -thrd) {
                                    //     //Avoiding Negative Concentrations 
                                    //     //if (conc[iS]+dC <= 0) { dC = -conc[iS]; }
                                    //     // Array<T, 7> g;
                                    //     // lattices[iS + dCloc]->get(iX + vec_offset[iS + dCloc].x, iY + vec_offset[iS + dCloc].y, iZ + vec_offset[iS + dCloc].z).getPopulations(g);
                                    //     // g[0] += (T)(dC) / 4; g[1] += (T)(dC) / 8; g[2] += (T)(dC) / 8; g[3] += (T)(dC) / 8; g[4] += (T)(dC) / 8; g[5] += (T)(dC) / 8; g[6] += (T)(dC) / 8;
                                    //     // lattices[iS + dCloc]->get(iX + vec_offset[iS + dCloc].x, iY + vec_offset[iS + dCloc].y, iZ + vec_offset[iS + dCloc].z).setPopulations(g);
                                    //     // std::cout << "Updated lattice populations for substrate " << iS << " at [" << iX << ", " << iY << ", " << iZ << "] with dC: " << dC << std::endl;
                                    //     Array<T, 7> g;
                                    //     lattices[iS + dCloc]->get(iX + vec_offset[iS + dCloc].x, iY + vec_offset[iS + dCloc].y, iZ + vec_offset[iS + dCloc].z).getPopulations(g);
                                    //     g[0]+= (T)(dC) / 4; g[1]+= (T)(dC) / 8; g[2]+= (T)(dC) / 8; g[3]+= (T)(dC) / 8; g[4]+= (T)(dC) / 8; g[5]+= (T)(dC) / 8; g[6]+= (T)(dC) / 8;

                                    //     // Print the old populations (if needed, you can move this part before updating the g array)
                                    //     std::cout << "Original lattice populations for substrate " << iS << " at [" << iX << ", " << iY << ", " << iZ << "]: ";
                                    //     for (int j = 0; j < 7; ++j) {
                                    //         //std::cout << g[j] << " ";
                                    //         std::cout << std::fixed << std::setprecision(15) << g[j] << " ";
                                    //     }
                                    //     std::cout << std::endl;

                                    //     lattices[iS + dCloc]->get(iX + vec_offset[iS + dCloc].x, iY + vec_offset[iS + dCloc].y, iZ + vec_offset[iS + dCloc].z).setPopulations(g);

                                    //     // Print the updated populations
                                    //     std::cout << "Updated lattice populations for substrate " << iS << " at [" << iX << ", " << iY << ", " << iZ << "] with dC: " << dC << ": ";
                                    //     for (int j = 0; j < 7; ++j) {
                                    //         //std::cout << g[j] << " ";
                                    //         std::cout << std::fixed << std::setprecision(15) << g[j] << " ";
                                    //     }
                                    //     std::cout << std::endl;

                                    // }


                                   // update concentration
                                    for (plint iS=0; iS<subsNum; ++iS) {
                                        // forward-Euler method
                                        T dC = subs_rate[iS]*dt;

                                        //std::cout << "Reaction rate (subs_rate) for substrate " << iS << ": " << subs_rate[iS] << std::endl;
                                        //std::cout << "dC for substrate " << iS << ": " << dC << std::endl;

                                        if (dC > thrd || dC < -thrd) {
                                            Array<T,7> g;
                                            plint iXt = iX+vec_offset[iS].x, iYt = iY+vec_offset[iS].y, iZt = iZ+vec_offset[iS].z;
                                            lattices[iS]->get(iXt,iYt,iZt).getPopulations(g);

                                            // Print the original populations
                                            // std::cout << "Original lattice populations for substrate " << iS << " at [" << iXt << ", " << iYt << ", " << iZt << "]: ";
                                            // for (int j = 0; j < 7; ++j) {
                                            //     std::cout << std::fixed << std::setprecision(15) << g[j] << " ";
                                            // }
                                            // std::cout << std::endl;

                                            // Update the populations
                                            g[0]+=(T) (dC)/4; g[1]+=(T) (dC)/8; g[2]+=(T) (dC)/8; g[3]+=(T) (dC)/8; g[4]+=(T) (dC)/8; g[5]+=(T) (dC)/8; g[6]+=(T) (dC)/8;
                                            lattices[iS]->get(iXt,iYt,iZt).setPopulations(g);

                                            // Print the updated populations
                                            // std::cout << "Updated lattice populations for substrate " << iS << " at [" << iXt << ", " << iYt << ", " << iZt << "] with dC: " << dC << ": ";
                                            // for (int j = 0; j < 7; ++j) {
                                            //     std::cout << std::fixed << std::setprecision(15) << g[j] << " ";
                                            // }
                                            // std::cout << std::endl;
                                        }
                                    }


                                    



















                                    
                                }



                        }


                                      
                       
                    }
                }
            }
          }

    virtual BlockDomain::DomainT appliesTo() const {
        return BlockDomain::bulkAndEnvelope;
    }

    virtual run_kinetics<T, Descriptor>* clone() const {
        return new run_kinetics<T, Descriptor>(*this);
    }

  

    // void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
    //    modified[0] = modif::staticVariables;
    //    for (plint iT = 1; iT < subsNum; ++iT) {
    //        modified[iT] = modif::nothing;
    //    }
    // }



    /**
     * `getTypeOfModification` is setting modification statuses to indicate that only the static 
     * parts of certain cells are being altered. In this context, it refers typically to updates 
     * to species concentrations or similar static variables following the reaction kinetics and Equilibrium calculations.
     */
    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        for (plint iT = dCloc; iT < maskLloc; ++iT) {
            modified[iT] = modif::staticVariables;
        }
      
    }


    private:
    // The nx field denotes the length of the lattice in the x-dimension
    plint nx;

    // The subsNum field denotes the number of substrates
    plint subsNum;

    // The dt field denotes the time step size
    T dt;

    // The vec2_Kc_kns field holds the two-dimensional kinetics constants
    std::vector<std::vector<T>> vec2_Kc_kns;

    // The solid and bb fields are specific states of the lattice cells
    plint solid, bb;


    // The maskLloc field holds the index for the mask lattice
    plint dCloc, maskLloc;



};

// This class, update_rxnLattices, is used to update the substrate concentrations in the lattice based on 
// the changes calculated during the kinetic reactions. After the reaction kinetics are computed (in the run_kinetics step), this class is invoked 
// to apply these changes to the substrate concentrations in the lattice.
// This class, update_rxnLattices, performs this 
// crucial step of actually updating the substrate concentrations in the lattice, thereby reflecting the 
// temporal evolution of the reaction system in the lattice.

// The run_kinetics class is responsible for implementing the mathematical model of reactions.
// It uses the rate constants and the current substrate concentrations to compute the changes in concentrations.
// It does not apply these changes to the lattice, but rather computes the changes (dC) that should be applied.

// The update_rxnLattices class takes the computed changes (dC) from the run_kinetics class 
// and applies them to the lattice populations.
// This class does not concern itself with how these changes are calculated, 
// its role is simply to update the lattices based on these changes.

// template<typename T, template<typename U> class Descriptor>
// class update_rxnLattices : public LatticeBoxProcessingFunctional3D<T, Descriptor>
// {
// public:
//     // The constructor of the class. It initializes the class with the given parameters.
//     update_rxnLattices(plint nx_, plint subsNum_, plint solid_, plint bb_)
//         : nx(nx_), subsNum(subsNum_), solid(solid_), bb(bb_), dCloc(subsNum_ ), maskLloc(2 * (subsNum_ ))
//     {}

//     // The process method of this class. This is the main function where the updating of the substrate concentrations is performed.
//     virtual void process(Box3D domain, std::vector<BlockLattice3D<T, Descriptor>*> lattices) {
//         // Get the absolute location of the first lattice (this is used to ensure that the computations are performed within the bounds of the lattice).
//         // This is necessary because the lattice could be part of a larger simulation where multiple lattices are used, and each lattice could be offset from the origin.
//         Dot3D absoluteOffset = lattices[0]->getLocation();

//         // The following nested for loop iterates over each node in the domain of the lattice.
//         for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
//             // Convert the x-coordinate from relative to the domain to absolute in the lattice.
//             plint absX = iX + absoluteOffset.x;

//             // Check if the x-coordinate is within the bounds of the lattice. This is to avoid computation on boundary nodes which may not be part of the physical simulation domain.
//             if (absX > 0 && absX < nx - 1) {
//                 for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
//                     for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
//                         // Get the offset of the mask lattice. This is used to access the mask lattice, which contains information about the physical state of each lattice node (e.g., whether it's solid or pore space).
//                         Dot3D maskOffset = computeRelativeDisplacement(*lattices[0], *lattices[maskLloc]);

//                         // Get the mask value at the current node. The mask value indicates whether the node is a solid node, a pore space node, or a boundary node.
//                         plint mask = util::roundToInt(lattices[maskLloc]->get(iX + maskOffset.x, iY + maskOffset.y, iZ + maskOffset.z).computeDensity());

//                         // Check if the node is not solid and not a boundary. Only perform computations on pore space nodes.
//                         if (mask != solid && mask != bb) {
//                             // Declare a vector to store the offsets of the substrate lattices.
//                             std::vector<Dot3D> vec_offset;

//                             // Compute the offset of each substrate lattice and store it in vec_offset.
//                             for (plint iT = 0; iT < maskLloc; ++iT) { vec_offset.push_back(computeRelativeDisplacement(*lattices[0], *lattices[iT])); }

//                             // Iterate over each substrate.
//                             for (plint iS = 0; iS < subsNum; ++iS) {
//                                 // Compute the absolute coordinates of the node in the substrate lattice.
//                                 plint iXs = iX + vec_offset[iS + dCloc].x, iYs = iY + vec_offset[iS + dCloc].y, iZs = iZ + vec_offset[iS + dCloc].z;

//                                 // Get the change in concentration at the node. This is computed by the run_kinetics class.
//                                 T dC = lattices[iS + dCloc]->get(iXs, iYs, iZs).computeDensity();

//                                 // Check if the change in concentration is above a certain threshold. If it is, then update the concentration at the node in the substrate lattice.
//                                 if (dC > thrd || dC < -thrd) {
//                                     // Get the current populations at the node in the substrate lattice.
//                                     Array<T, 7> g;
//                                     lattices[iS]->get(iX + vec_offset[iS].x, iY + vec_offset[iS].y, iZ + vec_offset[iS].z).getPopulations(g);

//                                     // Update the populations with the change in concentration.
//                                     g[0] += (T)(dC) / 4; g[1] += (T)(dC) / 8; g[2] += (T)(dC) / 8; g[3] += (T)(dC) / 8; g[4] += (T)(dC) / 8; g[5] += (T)(dC) / 8; g[6] += (T)(dC) / 8;

//                                     // Set the updated populations back into the substrate lattice at the current node.
//                                     lattices[iS]->get(iX + vec_offset[iS].x, iY + vec_offset[iS].y, iZ + vec_offset[iS].z).setPopulations(g);
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     // The appliesTo method of this class. This method returns the type of domain that this class applies to. In this case, it is the bulk and envelope of the lattice.
//     virtual BlockDomain::DomainT appliesTo() const {
//         return BlockDomain::bulkAndEnvelope;
//     }

//     // The clone method of this class. This method returns a new instance of this class that is a copy of this instance. This is used when you need a new instance of this class but don't want to run the constructor.
//     virtual update_rxnLattices<T, Descriptor>* clone() const {
//         return new update_rxnLattices<T, Descriptor>(*this);
//     }

//     // The getTypeOfModification method of this class. This method returns the type of modification that this class performs on the lattice. In this case, it modifies static variables.
//     void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
//         for (plint iT = 0; iT < subsNum; ++iT) {
//             modified[iT] = modif::staticVariables;
//         }
//     }

// private:
//     // Private member variables of the class. These store the parameters passed to the constructor.
//     plint nx, subsNum, solid, bb, dCloc, maskLloc;
// };




//update the flow field

// template<typename T1, template<typename U> class Descriptor1, typename T2, template<typename U> class Descriptor2>
// class updateNsLatticesDynamics3D : public BoxProcessingFunctional3D_LL<T1, Descriptor1, T2, Descriptor2>
// {
// public:
//     updateNsLatticesDynamics3D(T nsOmega_, std::vector<plint> pore_, plint solid_, plint bb_)
//         : nsOmega(nsOmega_), pore(pore_), solid(solid_), bb(bb_)
//     {}
//     // lattice0 = flow field lattice
//     // lattice1 = mask field lattice
//     virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor1>& lattice0, BlockLattice3D<T2, Descriptor2>& lattice1) {
//         Dot3D offset_12 = computeRelativeDisplacement(lattice0, lattice1);
//         for (plint iX0 = domain.x0; iX0 <= domain.x1; ++iX0) {
//             plint iX1 = iX0 + offset_12.x;
//             for (plint iY0 = domain.y0; iY0 <= domain.y1; ++iY0) {
//                 plint iY1 = iY0 + offset_12.y;
//                 for (plint iZ0 = domain.z0; iZ0 <= domain.z1; ++iZ0) {
//                     plint iZ1 = iZ0 + offset_12.z;
//                     plint mask = util::roundToInt(lattice1.get(iX1, iY1, iZ1).computeDensity());
//                     T currentOmega = lattice0.get(iX0, iY0, iZ0).getDynamics().getOmega();
//                     if (mask != bb && mask != solid) {
//                         bool poreflag = 0;
//                         for (size_t iP = 0; iP < pore.size(); ++iP) { if (mask == pore[iP]) { poreflag = 1; break; } }
//                         // update pore dynamics
//                         if (poreflag && std::abs(currentOmega - nsOmega) > thrd) {
//                             lattice0.attributeDynamics(iX0, iY0, iZ0, new IncBGKdynamics<T1, Descriptor1>(nsOmega));
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     virtual updateNsLatticesDynamics3D<T1, Descriptor1, T2, Descriptor2>* clone() const {
//         return new updateNsLatticesDynamics3D<T1, Descriptor1, T2, Descriptor2>(*this);
//     }
//     virtual BlockDomain::DomainT appliesTo() const {
//         return BlockDomain::bulkAndEnvelope;
//     }
//     void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
//         modified[0] = modif::dataStructure;
//         modified[1] = modif::nothing;
//     }
// private:
//     T nsOmega;
//     std::vector<plint> pore;
//     plint solid, bb;
// };




// template<typename T, template<typename U> class Descriptor>
// class CopyGeometryMaterialNumbers : public BoxProcessingFunctional3D {
// public:
//     virtual void processGenericBlocks(Box3D domain, std::vector<AtomicBlock3D*> blocks) {
//         PLB_ASSERT(blocks.size() == 2);

//         MultiScalarField3D<int>& geometry = dynamic_cast<MultiScalarField3D<int>&>(*blocks[0]);
//         MultiBlockLattice3D<T, Descriptor>& maskLattice = dynamic_cast<MultiBlockLattice3D<T, Descriptor>&>(*blocks[1]);

//         for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
//             for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
//                 for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
//                     int material = geometry.get(iX, iY, iZ);
//                     maskLattice.get(iX, iY, iZ).defineDensity((T)material);
//                 }
//             }
//         }
//     }

//     virtual CopyGeometryMaterialNumbers<T, Descriptor>* clone() const {
//         return new CopyGeometryMaterialNumbers<T, Descriptor>(*this);
//     }

//     virtual void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
//         modified.resize(2);
//         modified[0] = modif::nothing;
//         modified[1] = modif::dataStructure;
//     }
// };

// These codes are used to link scalar values from a scalar field to a lattice in a lattice-based simulation. In the first code, "CopyGeometryScalar2maskLattice3D", the scalar field represents the mask numbers of the simulation domain, and the lattice represents the flow field. The code links the mask numbers to the lattice by allocating the corresponding population values. 
//  The goal of these codes is to ensure that the scalar values in the simulation domain are properly represented in the lattice-based simulation.
// Link geometry scalar numbers and maskLattice
// template<typename T1, template<typename U> class Descriptor, typename T2>
// class CopyGeometryScalar2maskLattice3D : public BoxProcessingFunctional3D_LS<T1, Descriptor, T2>
// {
// public:
//     CopyGeometryScalar2maskLattice3D(std::vector< std::vector<plint> > mask0_) : mask0(mask0_)
//     {}
//     virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor>& lattice, ScalarField3D<T2>& field) {
//         Dot3D offset = computeRelativeDisplacement(lattice, field);
//         for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
//             plint iX1 = iX + offset.x;
//             for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
//                 plint iY1 = iY + offset.y;
//                 for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
//                     plint iZ1 = iZ + offset.z;
//                     bool flag = 0; T mask1 = field.get(iX1, iY1, iZ1); T mask2 = -1;
//                     // for (size_t iM = 0; iM < mask0.size(); ++iM) {
//                     //     for (size_t iN = 0; iN < mask0[iM].size(); ++iN) {
//                     //         if (mask1 == mask0[iM][iN]) {
//                     //             flag = 1;
//                     //             mask2 = mask0[iM][0];
//                     //             break;
//                     //         }
//                     //     }
//                     //     if (flag == 1) {
//                     //         break;
//                     //     }
//                     // }
//                     Array<T, 7> g;
//                     if (flag == 0) {
//                         g[0] = (T)(mask1 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask1 - 1) / 8;
//                     }
//                     else {
//                         g[0] = (T)(mask2 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask2 - 1) / 8;
//                     }
//                     lattice.get(iX, iY, iZ).setPopulations(g); // allocate the mask number
//                 }
//             }
//         }
//     }
//     virtual CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>* clone() const {
//         return new CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>(*this);
//     }
//     virtual BlockDomain::DomainT appliesTo() const {
//         return BlockDomain::bulkAndEnvelope;
//     }
//     void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
//         modified[0] = modif::staticVariables;
//         modified[1] = modif::nothing;
//     }
// private:
//     std::vector<std::vector<plint>> mask0;
// };


// template<typename T1, template<typename U> class Descriptor, typename T2>
// class CopyGeometryScalar2maskLattice3D : public BoxProcessingFunctional3D_LS<T1, Descriptor, T2>
// {
// public:
//     CopyGeometryScalar2maskLattice3D(std::vector< std::vector<plint> > mask0_) : mask0(mask0_)
//     {}
//     virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor>& lattice, ScalarField3D<T2>& field) {
//         Dot3D offset = computeRelativeDisplacement(lattice, field);
//         for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
//             plint iX1 = iX + offset.x;
//             for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
//                 plint iY1 = iY + offset.y;
//                 for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
//                     plint iZ1 = iZ + offset.z;
//                     bool flag = 0; T mask1 = field.get(iX1, iY1, iZ1); T mask2 = -1;
//                     // for (size_t iM = 0; iM < mask0.size(); ++iM) {
//                     //     for (size_t iN = 0; iN < mask0[iM].size(); ++iN) {
//                     //         if (mask1 == mask0[iM][iN]) {
//                     //             flag = 1;
//                     //             mask2 = mask0[iM][0];
//                     //             break;
//                     //         }
//                     //     }
//                     //     if (flag == 1) {
//                     //         break;
//                     //     }
//                     // }
//                     Array<T, 7> g;
//                     if (flag == 0) {
//                         g[0] = (T)(mask1 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask1 - 1) / 8;
//                     }
//                     else {
//                         g[0] = (T)(mask2 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask2 - 1) / 8;
//                     }
//                     lattice.get(iX, iY, iZ).setPopulations(g); // allocate the mask number
//                 }
//             }
//         }
//     }
//     virtual CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>* clone() const {
//         return new CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>(*this);
//     }
//     virtual BlockDomain::DomainT appliesTo() const {
//         return BlockDomain::bulkAndEnvelope;
//     }
//     void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
//         modified[0] = modif::staticVariables;
//         modified[1] = modif::nothing;
//     }
// private:
//     std::vector<std::vector<plint>> mask0;
// };


 //template<typename t1, template<typename u> class descriptor, typename t2>
 //class copygeometryscalar2masklattice3d : public boxprocessingfunctional3d_ls<t1, descriptor, t2>
 //{
 //public:
 //    copygeometryscalar2masklattice3d(const std::vector<plint> &mask0_) : mask0(mask0_)
 //    {}
 //    virtual void process(box3d domain, blocklattice3d<t1, descriptor>& lattice, scalarfield3d<t2>& field) {
 //        dot3d offset = computerelativedisplacement(lattice, field);
 //        for (plint ix = domain.x0; ix <= domain.x1; ++ix) {
 //            plint ix1 = ix + offset.x;
 //            for (plint iy = domain.y0; iy <= domain.y1; ++iy) {
 //                plint iy1 = iy + offset.y;
 //                for (plint iz = domain.z0; iz <= domain.z1; ++iz) {
 //                    plint iz1 = iz + offset.z;
 //                    t mask1 = field.get(ix1, iy1, iz1);
 //                    t mask2 = -1;
 //                    for (size_t im = 0; im < mask0.size(); ++im) {
 //                        if (mask1 == mask0[im]) {
 //                            mask2 = mask0[im];
 //                            break;
 //                        }
 //                    }
 //                    array<t, 7> g;
 //                    if (mask2 == -1) {
 //                        g[0] = (t)(mask1 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (t)(mask1 - 1) / 8;
 //                    } else {
 //                        g[0] = (t)(mask2 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (t)(mask2 - 1) / 8;
 //                       
 //                    }
 //                    lattice.get(ix, iy, iz).setpopulations(g); // allocate the mask number
 //                }
 //            }
 //        }
 //    }
 //    virtual copygeometryscalar2masklattice3d<t1, descriptor, t2>* clone() const {
 //        return new copygeometryscalar2masklattice3d<t1, descriptor, t2>(*this);
 //    }
 //    virtual blockdomain::domaint appliesto() const {
 //        return blockdomain::bulkandenvelope;
 //    }
 //    void gettypeofmodification(std::vector<modif::modift>& modified) const {
 //        modified[0] = modif::staticvariables;
 //        modified[1] = modif::nothing;
 //    }
 //private:
 //    std::vector<plint> mask0;
 //};


//template<typename T1, template<typename U> class Descriptor, typename T2>
//class CopyGeometryScalar2maskLattice3D : public BoxProcessingFunctional3D_LS<T1, Descriptor, T2>
//{
//public:
//    CopyGeometryScalar2maskLattice3D(std::vector<plint> mask0_) : mask0(mask0_)
//    {}
//    virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor>& lattice, ScalarField3D<T2>& field) {
//        Dot3D offset = computeRelativeDisplacement(lattice, field);
//        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
//            plint iX1 = iX + offset.x;
//            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
//                plint iY1 = iY + offset.y;
//                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
//                    plint iZ1 = iZ + offset.z;
//                    bool flag = 0; T mask1 = field.get(iX1, iY1, iZ1); T mask2 = -1;
//                    for (size_t iM = 0; iM < mask0.size(); ++iM) {
//                        if (mask1 == mask0[iM]) {
//                            flag = 1;
//                            mask2 = mask0[iM];  // or another appropriate value
//                            break;
//                        }
//
//                        if (flag == 1) {
//                            break;
//                        }
//                    }
//
//                    Array<T, 7> g;
//                    if (flag == 0) {
//                        g[0] = (T)(mask1 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask1 - 1) / 8;
//                    }
//                    else {
//                        g[0] = (T)(mask2 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask2 - 1) / 8;
//                    }
//                    lattice.get(iX, iY, iZ).setPopulations(g); // allocate the mask number
//                }
//            }
//        }
//    }
//    virtual CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>* clone() const {
//        return new CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>(*this);
//    }
//    virtual BlockDomain::DomainT appliesTo() const {
//        return BlockDomain::bulkAndEnvelope;
//    }
//    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
//        modified[0] = modif::staticVariables;
//        modified[1] = modif::nothing;
//    }
//private:
//    std::vector<plint> mask0;
//};


template<typename T1, template<typename U> class Descriptor, typename T2>
class CopyGeometryScalar2maskLattice3D : public BoxProcessingFunctional3D_LS<T1, Descriptor, T2>
{
public:
    CopyGeometryScalar2maskLattice3D(std::vector<plint> mask0_) : mask0(mask0_)
    {}

    virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor>& lattice, ScalarField3D<T2>& field) {
        Dot3D offset = computeRelativeDisplacement(lattice, field);

        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            plint iX1 = iX + offset.x;
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint iY1 = iY + offset.y;
                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
                    plint iZ1 = iZ + offset.z;
                    bool flag = 0;
                    T mask1 = field.get(iX1, iY1, iZ1);
                    T mask2 = -1;

                    //pcout << "Checking position: (" << iX1 << ", " << iY1 << ", " << iZ1 << ")\n";
                    //pcout << "Current mask1 value: " << mask1 << "\n";

                    for (size_t iM = 0; iM < mask0.size(); ++iM) {
                        //pcout << "Comparing mask1 with mask0[" << iM << "] = " << mask0[iM] << "\n";

                        if (mask1 == mask0[iM]) {
                            //pcout << "Match found! Setting flag = 1 and mask2 = " << mask0[iM] << "\n";

                            flag = 1;
                            mask2 = mask0[iM];
                            break;
                        }

                        if (flag == 1) {
                            //pcout << "No match found. mask2 remains " << mask2 << "\n";

                            break;
                        }
                    }

            

                    Array<T, 7> g;
                    if (flag == 0) {
                        g[0] = (T)(mask1 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask1 - 1) / 8;
                        //pcout << "Flag is 0. Calculating g[] with mask1.\n";
                    }
                    else {
                        g[0] = (T)(mask2 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask2 - 1) / 8;
                        //pcout << "Flag is 1. Calculating g[] with mask2.\n";
                    }

                    lattice.get(iX, iY, iZ).setPopulations(g); // allocate the mask number
                    //pcout << "Setting populations with g[] values.\n";
                }
            }
        }
    }

    virtual CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>* clone() const {
        return new CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>(*this);
    }

    virtual BlockDomain::DomainT appliesTo() const {
        return BlockDomain::bulkAndEnvelope;
    }

    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::staticVariables;
        modified[1] = modif::nothing;
    }

private:
    std::vector<plint> mask0;
};


// template<typename T1, template<typename U> class Descriptor, typename T2>
// class CopyGeometryScalar2maskLattice3D : public BoxProcessingFunctional3D_LS<T1, Descriptor, T2>
// {
// public:
//     CopyGeometryScalar2maskLattice3D(std::vector<plint> mask0_) : mask0(mask0_)
//     {}

//     virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor>& lattice, ScalarField3D<T2>& field) {
//         Dot3D offset = computeRelativeDisplacement(lattice, field);

//         for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
//             plint iX1 = iX + offset.x;
//             for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
//                 plint iY1 = iY + offset.y;
//                 for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
//                     plint iZ1 = iZ + offset.z;
//                     bool flag = 0;
//                     T1 mask1 = field.get(iX1, iY1, iZ1);
//                     T1 mask2 = -1;

//                     pcout << "Checking position: (" << iX1 << ", " << iY1 << ", " << iZ1 << "), Current mask1 value: " << mask1 << std::endl;

//                     for (size_t iM = 0; iM < mask0.size(); ++iM) {
//                         pcout << "Comparing mask1 with mask0[" << iM << "] = " << mask0[iM] << std::endl;

//                         if (mask1 == mask0[iM]) {
//                             pcout << "Match found! Setting flag = 1 and mask2 = " << mask0[iM] << std::endl;
//                             flag = 1;
//                             mask2 = mask0[iM];
//                             break;
//                         }
//                     }

//                     if (flag == 0) {
//                         pcout << "No match found for position: (" << iX1 << ", " << iY1 << ", " << iZ1 << "), mask2 remains: " << mask2 << std::endl;
//                     }

//                     Array<T, 7> g; // Corrected the template use for Array
//                     if (flag == 0) {
//                         g[0] = (T)(mask1 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask1 - 1) / 8;
//                         pcout << "Flag is 0. Calculating g[] with mask1 at position (" << iX1 << ", " << iY1 << ", " << iZ1 << ")." << std::endl;
//                     }
//                     else {
//                         g[0] = (T)(mask2 - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask2 - 1) / 8; 
//                         pcout << "Flag is 1. Calculating g[] with mask2 at position (" << iX1 << ", " << iY1 << ", " << iZ1 << ")." << std::endl;
//                     }

//                     lattice.get(iX, iY, iZ).setPopulations(g); // Set the populations with the calculated values
//                     pcout << "Setting populations for position (" << iX << ", " << iY << ", " << iZ << ") with g[] values." << std::endl;
//                 }
//             }
//         }
//     }

//     virtual CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>* clone() const {
//         return new CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>(*this);
//     }

//     virtual BlockDomain::DomainT appliesTo() const {
//         return BlockDomain::bulkAndEnvelope;
//     }

//     void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
//         modified[0] = modif::staticVariables;
//         modified[1] = modif::nothing;
//     }

// private:
//     std::vector<plint> mask0;
// };


//template<typename T1, template<typename U> class Descriptor, typename T2>
//class CopyGeometryScalar2maskLattice3D : public BoxProcessingFunctional3D_LS<T1, Descriptor, T2>
//{
//public:
//    CopyGeometryScalar2maskLattice3D(std::vector<std::vector<plint>> mask0_) : mask0(mask0_)
//    {}
//    virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor>& lattice, ScalarField3D<T2>& field) {
//        Dot3D offset = computeRelativeDisplacement(lattice, field);
//        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
//            plint iX1 = iX + offset.x;
//            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
//                plint iY1 = iY + offset.y;
//                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
//                    plint iZ1 = iZ + offset.z;
//                    bool flag = 0;
//                    T mask1 = field.get(iX1, iY1, iZ1);
//                    T mask2 = -1;
//
//                    for (size_t iM = 0; iM < mask0.size(); ++iM) {
//                        for (size_t iN = 0; iN < mask0[iM].size(); ++iN) {
//                            if (mask1 == mask0[iM][iN]) {
//                                flag = 1;
//                                mask2 = mask0[iM][0];
//                                break;
//                            }
//                        }
//                        if (flag == 1) {
//                            break;
//                        }
//                    }
//
//                    Array<T, 7> g;
//                    if (flag == 0) {
//                        g[0] = (T)(mask1 - 1) / 4;
//                        g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask1 - 1) / 8;
//                    }
//                    else {
//                        g[0] = (T)(mask2 - 1) / 4;
//                        g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(mask2 - 1) / 8;
//                    }
//
//                    lattice.get(iX, iY, iZ).setPopulations(g);
//                }
//            }
//        }
//    }
//    virtual CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>* clone() const {
//        return new CopyGeometryScalar2maskLattice3D<T1, Descriptor, T2>(*this);
//    }
//    virtual BlockDomain::DomainT appliesTo() const {
//        return BlockDomain::bulkAndEnvelope;
//    }
//    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
//        modified[0] = modif::staticVariables;
//        modified[1] = modif::nothing;
//    }
//private:
//    std::vector<std::vector<plint>> mask0;
//};
//



template<typename T1, template<typename U> class Descriptor, typename T2>
class CopyGeometryScalar2distLattice3D : public BoxProcessingFunctional3D_LS<T1, Descriptor, T2>
{
public:
    CopyGeometryScalar2distLattice3D()
    {}
    virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor>& lattice, ScalarField3D<T2>& field) {
        Dot3D offset = computeRelativeDisplacement(lattice, field);
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            plint iX1 = iX + offset.x;
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint iY1 = iY + offset.y;
                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
                    plint iZ1 = iZ + offset.z;
                    plint dist = field.get(iX1, iY1, iZ1);
                    Array<T, 7> g;
                    g[0] = (T)(dist - 1) / 4; g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = (T)(dist - 1) / 8;
                    lattice.get(iX, iY, iZ).setPopulations(g);
                }
            }
        }
    }
    virtual CopyGeometryScalar2distLattice3D<T1, Descriptor, T2>* clone() const {
        return new CopyGeometryScalar2distLattice3D<T1, Descriptor, T2>(*this);
    }
    virtual BlockDomain::DomainT appliesTo() const {
        return BlockDomain::bulkAndEnvelope;
    }
    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::staticVariables;
        modified[1] = modif::nothing;
    }
private:
};



// The first code block defines two classes: CopyLattice2ScalarField3D and createDistanceDomain3D.

// CopyLattice2ScalarField3D is a class that copies the values from a lattice to a scalar field. The lattice and the scalar field are defined as template parameters T1 and T2, respectively, and the Descriptor template parameter refers to the lattice descriptor. The process function loops over the domain of the lattice and sets the value of the corresponding point in the scalar field to the rounded density value of the lattice. This function is used to copy the mask lattice to the geometry scalar field.

// createDistanceDomain3D is a class that creates a distance scalar field. The distance vector is defined as a std::vector<std::vector<std::vector<plint>>> and is passed to the constructor of the class. The process function loops over the domain of the scalar field and sets the value of the corresponding point to the value of the corresponding point in the distance vector. This function is used to create a scalar field that stores the distance from the surface of the geometry.

// Both of these classes are used in the implementation of lattice Boltzmann methods for simulating fluid flow. CopyLattice2ScalarField3D is used to copy the mask lattice to the geometry scalar field and createDistanceDomain3D is used to create a scalar field that stores the distance from the surface of the geometry. These scalar fields are used in various calculations, such as boundary conditions and source terms, that are important for accurately simulating fluid flow.

// Copy maskLattice to geometry field
template<typename T1, template<typename U> class Descriptor, typename T2>
class CopyLattice2ScalarField3D : public BoxProcessingFunctional3D_LS<T1, Descriptor, T2>
{
public:
    CopyLattice2ScalarField3D()
    {}
    virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor>& lattice, ScalarField3D<T2>& field) {
        Dot3D offset = computeRelativeDisplacement(lattice, field);
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            plint iX1 = iX + offset.x;
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint iY1 = iY + offset.y;
                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
                    plint iZ1 = iZ + offset.z;
                    field.get(iX1, iY1, iZ1) = util::roundToInt(lattice.get(iX, iY, iZ).computeDensity());
                }
            }
        }
    }
    virtual CopyLattice2ScalarField3D<T1, Descriptor, T2>* clone() const {
        return new CopyLattice2ScalarField3D<T1, Descriptor, T2>(*this);
    }
    virtual BlockDomain::DomainT appliesTo() const {
        return BlockDomain::bulkAndEnvelope;
    }
    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::nothing;
        modified[1] = modif::staticVariables;
    }
private:
    std::vector<std::vector<plint>> mask0;
};


// create a domain distance scalarfield3d
template<typename T1>
class createDistanceDomain3D : public BoxProcessingFunctional3D_S<T1>
{
public:
    createDistanceDomain3D(std::vector<std::vector<std::vector<plint>>> distVec_) : distVec(distVec_)
    {}
    virtual void process(Box3D domain, ScalarField3D<T1>& field) {
        Dot3D absoluteOffset = field.getLocation();
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            plint absX = iX + absoluteOffset.x;
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint absY = iY + absoluteOffset.y;
                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
                    plint absZ = iZ + absoluteOffset.z;
                    field.get(iX, iY, iZ) = distVec[absX][absY][absZ];
                }
            }
        }
    }
    virtual createDistanceDomain3D<T1>* clone() const {
        return new createDistanceDomain3D<T1>(*this);
    }
    virtual BlockDomain::DomainT appliesTo() const {
        return BlockDomain::bulkAndEnvelope;
    }
    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::staticVariables;
    }
private:
    std::vector<std::vector<std::vector<plint>>> distVec;
};


template<typename T1, template<typename U> class Descriptor, typename T2>
class stabilizeADElattice : public BoxProcessingFunctional3D_LS<T1,Descriptor,T2>
{
public:
    stabilizeADElattice(T c0_, std::vector<plint> pore_ ): c0(c0_), pore(pore_)
    
    {}
    virtual void process(Box3D domain, BlockLattice3D<T1,Descriptor>& lattice, ScalarField3D<T2> &field) {
        Dot3D offset = computeRelativeDisplacement(lattice, field);
        for (plint iX=domain.x0; iX<=domain.x1; ++iX) {
            plint iX1 = iX + offset.x;
            for (plint iY=domain.y0; iY<=domain.y1; ++iY) {
                plint iY1 = iY + offset.y;
                for (plint iZ=domain.z0; iZ<=domain.z1; ++iZ) {
                    plint iZ1 = iZ + offset.z;
                
                    T2 mask = field.get(iX1,iY1,iZ1);
                    bool chk = 0;
                    for (size_t iP=0; iP<pore.size(); ++iP) {
                        if ( mask==pore[iP] ) { chk = 1; break;}
                    }
                    
            
                    if (chk == 1) {
                        if (c0<thrd &&c0>-thrd) {c0 = 0;}
                        Array<T,7> g;
                        g[0]=(T) (c0-1)/4; g[1]=(T) (c0-1)/8; g[2]=(T) (c0-1)/8; g[3]=(T) (c0-1)/8; g[4]=(T) (c0-1)/8; g[5]=(T) (c0-1)/8; g[6]=(T) (c0-1)/8;
                        lattice.get(iX,iY,iZ).setPopulations(g);
                    }
                }

            }
           
            
        }
    }
    virtual stabilizeADElattice<T1,Descriptor,T2>* clone() const {
        return new stabilizeADElattice<T1,Descriptor,T2>(*this);
    }
    virtual BlockDomain::DomainT appliesTo() const {
        return BlockDomain::bulkAndEnvelope;
    }
    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::staticVariables;
        modified[1] = modif::nothing;
    }
private:
    T c0;
    std::vector<plint> pore;
    
};


// This code defines a class called createAgeDomain3D which creates a domain age scalar field. The class has a constructor that takes in a list of pore masks, a bounding box mask, and a solid mask. The process method of the class takes in a domain and a scalar field and sets the values of the scalar field based on the corresponding mask value. If the mask is equal to the solid or bounding box mask, it sets the scalar field value to -1. Otherwise, if the mask is in the list of pore masks, it sets the scalar field value to 0. Otherwise, 
// it sets the scalar field value to 1. This code is used to create a scalar field that represents the age of the material within the domain based on the different masks.


// create a domain distance scalarfield3d
template<typename T1>
class createAgeDomain3D : public BoxProcessingFunctional3D_S<T1>
{
public:
    createAgeDomain3D(std::vector<plint> pore_, plint bb_, plint solid_) : pore(pore_), bb(bb_), solid(solid_)
    {}
    virtual void process(Box3D domain, ScalarField3D<T1>& field) {
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
                    plint mask = field.get(iX, iY, iZ);
                    if (mask == solid || mask == bb) { field.get(iX, iY, iZ) = -1; }
                    else {
                        bool poreflag = 0;
                        for (size_t iP = 0; iP < pore.size(); ++iP) { if (mask == pore[iP]) { poreflag = 1; break; } }
                        if (poreflag == 1) { field.get(iX, iY, iZ) = 0; }
                        else { field.get(iX, iY, iZ) = 1; }
                    }
                }
            }
        }
    }
    virtual createAgeDomain3D<T1>* clone() const {
        return new createAgeDomain3D<T1>(*this);
    }
    virtual BlockDomain::DomainT appliesTo() const {
        return BlockDomain::bulkAndEnvelope;
    }
    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::staticVariables;
    }
private:
    std::vector<plint> pore;
    plint bb, solid;
};



/* ===============================================================================================================
   ========================================== REDUCTIVE DATA PROCESSORS ==========================================
   =============================================================================================================== */


// This code defines a template class MaskedBoxScalarCountFunctional3D, which is a subclass of ReductiveBoxProcessingFunctional3D_S. It takes a scalar field as input, and counts the number of voxels that have a specified mask value.

// The constructor takes an integer argument mask_, which specifies the mask value to be counted. During processing, the scalar field is iterated over within the given domain, and the mask value at each voxel is checked. If the mask value matches the specified value, the count is incremented.

// The class provides a method getCount(), which returns the count of voxels with the specified mask value.

// This code is useful for counting the number of voxels in a scalar field that meet a specific criterion, such as the number of voxels occupied by a particular phase or material.
template<typename T1>
class MaskedBoxScalarCountFunctional3D : public ReductiveBoxProcessingFunctional3D_S<T1>
{
public:
    MaskedBoxScalarCountFunctional3D(plint mask_) : countId(this->getStatistics().subscribeSum()), mask(mask_)
    {}
    virtual void process(Box3D domain, ScalarField3D<T1>& scalar) {
        BlockStatistics& statistics = this->getStatistics();
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
                    plint tmpMask = util::roundToInt(scalar.get(iX, iY, iZ));
                    if (tmpMask == mask) {
                        statistics.gatherSum(countId, (int)1);
                    }
                }
            }
        }
    }
    virtual MaskedBoxScalarCountFunctional3D<T1>* clone() const {
        return new MaskedBoxScalarCountFunctional3D<T1>(*this);
    }
    virtual void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::nothing;
    }
    plint getCount() const {
        plint doubleSum = this->getStatistics().getSum(countId);
        return (T)doubleSum;
    }
private:
    plint countId;
    plint mask;
};
// This code defines a function MaskedScalarCounts3D that takes in a 3D domain domain, a 3D multi-scalar field , and a scalar value mask. The function calculates the number of voxels in the field that have the same scalar value as the given mask, by applying a MaskedBoxScalarCountFunctional3D processing functional to the field.

// The MaskedBoxScalarCountFunctional3D processing functional defines a method to count the number of voxels in a given box that have a certain scalar value, by subscribing to the statistics gathering system of the lattice Boltzmann simulation library.

// This function is useful for analyzing the properties of the multi-scalar field, such as determining the number of voxels that belong to a certain material or phase.
template<typename T1>
plint MaskedScalarCounts3D(Box3D domain, MultiScalarField3D<T1>& field, plint mask) {
    MaskedBoxScalarCountFunctional3D<T1> functional = MaskedBoxScalarCountFunctional3D<T1>(mask);
    applyProcessingFunctional(functional, domain, field);
    return functional.getCount();
}


// This code defines a class BoxLatticeRMSEFunctional3D that calculates the root mean square error (RMSE) between two 3D block lattices lattice0 and lattice1. The lattices have different types T1 and T2 and different lattice descriptors Descriptor1 and Descriptor2.

// The process method takes a Box3D domain and iterates over each lattice point within that domain. It calculates the difference in density between corresponding lattice points in lattice0 and lattice1 and adds the squared difference to the sum of squared differences.

// The getCount method returns the computed RMSE value.

// This code could be used to compare two 3D block lattices and determine how different they are in terms of their density values.

template<typename T1, template<typename U1> class Descriptor1, typename T2, template<typename U2> class Descriptor2>
class BoxLatticeRMSEFunctional3D : public ReductiveBoxProcessingFunctional3D_LL<T1, Descriptor1, T2, Descriptor2>
{
public:
    BoxLatticeRMSEFunctional3D() : sumId(this->getStatistics().subscribeSum())
    {}
    virtual void process(Box3D domain, BlockLattice3D<T1, Descriptor1>& lattice0, BlockLattice3D<T2, Descriptor2>& lattice1) {
        BlockStatistics& statistics = this->getStatistics();
        Dot3D offset_01 = computeRelativeDisplacement(lattice0, lattice1);
        for (plint iX0 = domain.x0; iX0 <= domain.x1; ++iX0) {
            for (plint iY0 = domain.y0; iY0 <= domain.y1; ++iY0) {
                for (plint iZ0 = domain.z0; iZ0 <= domain.z1; ++iZ0) {
                    plint iX1 = iX0 + offset_01.x;
                    plint iY1 = iY0 + offset_01.y;
                    plint iZ1 = iZ0 + offset_01.z;
                    T deltaC = lattice0.get(iX0, iY0, iZ0).computeDensity() - lattice1.get(iX1, iY1, iZ1).computeDensity();
                    T RMSE = deltaC * deltaC;
                    statistics.gatherSum(sumId, RMSE);
                }
            }
        }
    }
    virtual BoxLatticeRMSEFunctional3D<T1, Descriptor1, T2, Descriptor2>* clone() const {
        return new BoxLatticeRMSEFunctional3D<T1, Descriptor1, T2, Descriptor2>(*this);
    }
    virtual void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::nothing;
        modified[1] = modif::nothing;
    }
    T getCount() const {
        double doubleSum = this->getStatistics().getSum(sumId);
        if (std::numeric_limits<T>::is_integer) {
            return (T)util::roundToInt(doubleSum);
        }
        return (T)doubleSum;
    }
private:
    plint sumId;
};

// The code defines a function named "computeRMSE3D" that calculates the root mean square error (RMSE) between two MultiBlockLattice3D objects (lattice0 and lattice1) over a given domain (domain). The RMSE is normalized by the pore volume (poreVolume).

// The function starts by instantiating an object of the class "BoxLatticeRMSEFunctional3D", which calculates the RMSE between two lattices in a given domain. This class is a subclass of "ReductiveBoxProcessingFunctional3D_LL", which is a general class for processing two lattices.

// Next, the "applyProcessingFunctional" function is called to apply the BoxLatticeRMSEFunctional3D object to the two input lattices and the given domain. The result of the computation is accumulated in a sum ID, which is later used to calculate the RMSE.

// Finally, the RMSE is computed using the accumulated sum ID and the pore volume, and the result is returned.

// Overall, this code is used to calculate the RMSE between two lattices over a given domain.

template<typename T1, template<typename U1> class Descriptor1, typename T2, template<typename U2> class Descriptor2>
T computeRMSE3D(Box3D domain, MultiBlockLattice3D<T1, Descriptor1>& lattice0, MultiBlockLattice3D<T2, Descriptor2>& lattice1, T poreVolume) {
    BoxLatticeRMSEFunctional3D<T1, Descriptor1, T2, Descriptor2> functional;
    applyProcessingFunctional(functional, domain, lattice0, lattice1);
    return std::sqrt(functional.getCount() / poreVolume);
}








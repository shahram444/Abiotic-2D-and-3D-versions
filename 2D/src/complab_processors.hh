#include "../defineKinetics.hh"
#include <random>
#include <iomanip>

using namespace plb;
typedef double T;

#define DESCRIPTOR descriptors::D2Q9Descriptor  // Cs2 = 1/3
#define BGK descriptors::AdvectionDiffusionD2Q5Descriptor // Cs2 = 1/3

/* ===============================================================================================================
   =============================================== DATA PROCESSORS ===============================================
   =============================================================================================================== */

// The run_kinetics class is a subclass of LatticeBoxProcessingFunctional2D. This class is responsible for
// calculating the reaction kinetics of the system.

template<typename T, template<typename U> class Descriptor>
class run_kinetics : public LatticeBoxProcessingFunctional2D<T, Descriptor>
{
public:
   // Constructor: initialize the members variables
   run_kinetics(plint nx_, plint subsNum_, T dt_, std::vector< std::vector<T>> vec2_Kc_kns_, plint solid_, plint bb_)
       : nx(nx_), subsNum(subsNum_), dt(dt_), vec2_Kc_kns(vec2_Kc_kns_), solid(solid_), bb(bb_), dCloc(subsNum_), maskLloc(2 * (subsNum_))
   {}

          virtual void process(Box2D domain, std::vector<BlockLattice2D<T, Descriptor>*> lattices) {
            Dot2D absoluteOffset = lattices[0]->getLocation();

            for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
                plint absX = iX + absoluteOffset.x;

                if (absX > 0 && absX < nx - 1) {
                    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                                Dot2D maskOffset = computeRelativeDisplacement(*lattices[0], *lattices[maskLloc]);
                                plint mask = util::roundToInt(lattices[maskLloc]->get(iX + maskOffset.x, iY + maskOffset.y).computeDensity());
                                if (mask != solid && mask != bb) {

                                    std::vector<Dot2D> vec_offset;
                                    for (plint iT = 0; iT < maskLloc; ++iT) {
                                        vec_offset.push_back(computeRelativeDisplacement(*lattices[0], *lattices[iT]));
                                    }


                                    std::vector<T> conc, subs_rate(subsNum, 0);


                                    for (plint iS = 0; iS < subsNum; ++iS) {
                                        plint iXs = iX + vec_offset[iS].x, iYs = iY + vec_offset[iS].y;
                                        T c0 = lattices[iS]->get(iXs, iYs).computeDensity();
                                        if (c0 < thrd) { c0 = 0; }
                                        conc.push_back(c0);
                                    }

                                    defineRxnKinetics(conc, subs_rate, mask);


                                   // update concentration
                                    for (plint iS=0; iS<subsNum; ++iS) {
                                        // forward-Euler method
                                        T dC = subs_rate[iS]*dt;

                                        if (dC > thrd || dC < -thrd) {
                                            Array<T,5> g;
                                            plint iXt = iX+vec_offset[iS].x, iYt = iY+vec_offset[iS].y;
                                            lattices[iS]->get(iXt,iYt).getPopulations(g);

                                            // Update the populations (D2Q5 weights: 1/3 and 1/6)
                                            g[0]+=(T) (dC)/3; g[1]+=(T) (dC)/6; g[2]+=(T) (dC)/6; g[3]+=(T) (dC)/6; g[4]+=(T) (dC)/6;
                                            lattices[iS]->get(iXt,iYt).setPopulations(g);
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


    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        for (plint iT = dCloc; iT < maskLloc; ++iT) {
            modified[iT] = modif::staticVariables;
        }

    }


    private:
    plint nx;
    plint subsNum;
    T dt;
    std::vector<std::vector<T>> vec2_Kc_kns;
    plint solid, bb;
    plint dCloc, maskLloc;



};


// Link geometry scalar numbers and maskLattice (2D version)
template<typename T1, template<typename U> class Descriptor, typename T2>
class CopyGeometryScalar2maskLattice2D : public BoxProcessingFunctional2D_LS<T1, Descriptor, T2>
{
public:
    CopyGeometryScalar2maskLattice2D(std::vector<plint> mask0_) : mask0(mask0_)
    {}

    virtual void process(Box2D domain, BlockLattice2D<T1, Descriptor>& lattice, ScalarField2D<T2>& field) {
        Dot2D offset = computeRelativeDisplacement(lattice, field);

        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            plint iX1 = iX + offset.x;
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint iY1 = iY + offset.y;
                bool flag = 0;
                T mask1 = field.get(iX1, iY1);
                T mask2 = -1;

                for (size_t iM = 0; iM < mask0.size(); ++iM) {
                    if (mask1 == mask0[iM]) {
                        flag = 1;
                        mask2 = mask0[iM];
                        break;
                    }

                    if (flag == 1) {
                        break;
                    }
                }


                Array<T, 5> g;
                if (flag == 0) {
                    g[0] = (T)(mask1 - 1) / 3; g[1] = g[2] = g[3] = g[4] = (T)(mask1 - 1) / 6;
                }
                else {
                    g[0] = (T)(mask2 - 1) / 3; g[1] = g[2] = g[3] = g[4] = (T)(mask2 - 1) / 6;
                }

                lattice.get(iX, iY).setPopulations(g); // allocate the mask number
            }
        }
    }

    virtual CopyGeometryScalar2maskLattice2D<T1, Descriptor, T2>* clone() const {
        return new CopyGeometryScalar2maskLattice2D<T1, Descriptor, T2>(*this);
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



template<typename T1, template<typename U> class Descriptor, typename T2>
class CopyGeometryScalar2distLattice2D : public BoxProcessingFunctional2D_LS<T1, Descriptor, T2>
{
public:
    CopyGeometryScalar2distLattice2D()
    {}
    virtual void process(Box2D domain, BlockLattice2D<T1, Descriptor>& lattice, ScalarField2D<T2>& field) {
        Dot2D offset = computeRelativeDisplacement(lattice, field);
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            plint iX1 = iX + offset.x;
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint iY1 = iY + offset.y;
                plint dist = field.get(iX1, iY1);
                Array<T, 5> g;
                g[0] = (T)(dist - 1) / 3; g[1] = g[2] = g[3] = g[4] = (T)(dist - 1) / 6;
                lattice.get(iX, iY).setPopulations(g);
            }
        }
    }
    virtual CopyGeometryScalar2distLattice2D<T1, Descriptor, T2>* clone() const {
        return new CopyGeometryScalar2distLattice2D<T1, Descriptor, T2>(*this);
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



// Copy maskLattice to geometry field
template<typename T1, template<typename U> class Descriptor, typename T2>
class CopyLattice2ScalarField2D : public BoxProcessingFunctional2D_LS<T1, Descriptor, T2>
{
public:
    CopyLattice2ScalarField2D()
    {}
    virtual void process(Box2D domain, BlockLattice2D<T1, Descriptor>& lattice, ScalarField2D<T2>& field) {
        Dot2D offset = computeRelativeDisplacement(lattice, field);
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            plint iX1 = iX + offset.x;
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint iY1 = iY + offset.y;
                field.get(iX1, iY1) = util::roundToInt(lattice.get(iX, iY).computeDensity());
            }
        }
    }
    virtual CopyLattice2ScalarField2D<T1, Descriptor, T2>* clone() const {
        return new CopyLattice2ScalarField2D<T1, Descriptor, T2>(*this);
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


// create a domain distance scalarfield2d
template<typename T1>
class createDistanceDomain2D : public BoxProcessingFunctional2D_S<T1>
{
public:
    createDistanceDomain2D(std::vector<std::vector<plint>> distVec_) : distVec(distVec_)
    {}
    virtual void process(Box2D domain, ScalarField2D<T1>& field) {
        Dot2D absoluteOffset = field.getLocation();
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            plint absX = iX + absoluteOffset.x;
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint absY = iY + absoluteOffset.y;
                field.get(iX, iY) = distVec[absX][absY];
            }
        }
    }
    virtual createDistanceDomain2D<T1>* clone() const {
        return new createDistanceDomain2D<T1>(*this);
    }
    virtual BlockDomain::DomainT appliesTo() const {
        return BlockDomain::bulkAndEnvelope;
    }
    void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::staticVariables;
    }
private:
    std::vector<std::vector<plint>> distVec;
};


template<typename T1, template<typename U> class Descriptor, typename T2>
class stabilizeADElattice : public BoxProcessingFunctional2D_LS<T1,Descriptor,T2>
{
public:
    stabilizeADElattice(T c0_, std::vector<plint> pore_ ): c0(c0_), pore(pore_)

    {}
    virtual void process(Box2D domain, BlockLattice2D<T1,Descriptor>& lattice, ScalarField2D<T2> &field) {
        Dot2D offset = computeRelativeDisplacement(lattice, field);
        for (plint iX=domain.x0; iX<=domain.x1; ++iX) {
            plint iX1 = iX + offset.x;
            for (plint iY=domain.y0; iY<=domain.y1; ++iY) {
                plint iY1 = iY + offset.y;

                T2 mask = field.get(iX1,iY1);
                bool chk = 0;
                for (size_t iP=0; iP<pore.size(); ++iP) {
                    if ( mask==pore[iP] ) { chk = 1; break;}
                }


                if (chk == 1) {
                    if (c0<thrd &&c0>-thrd) {c0 = 0;}
                    Array<T,5> g;
                    g[0]=(T) (c0-1)/3; g[1]=(T) (c0-1)/6; g[2]=(T) (c0-1)/6; g[3]=(T) (c0-1)/6; g[4]=(T) (c0-1)/6;
                    lattice.get(iX,iY).setPopulations(g);
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


// create a domain age scalarfield2d
template<typename T1>
class createAgeDomain2D : public BoxProcessingFunctional2D_S<T1>
{
public:
    createAgeDomain2D(std::vector<plint> pore_, plint bb_, plint solid_) : pore(pore_), bb(bb_), solid(solid_)
    {}
    virtual void process(Box2D domain, ScalarField2D<T1>& field) {
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint mask = field.get(iX, iY);
                if (mask == solid || mask == bb) { field.get(iX, iY) = -1; }
                else {
                    bool poreflag = 0;
                    for (size_t iP = 0; iP < pore.size(); ++iP) { if (mask == pore[iP]) { poreflag = 1; break; } }
                    if (poreflag == 1) { field.get(iX, iY) = 0; }
                    else { field.get(iX, iY) = 1; }
                }
            }
        }
    }
    virtual createAgeDomain2D<T1>* clone() const {
        return new createAgeDomain2D<T1>(*this);
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


template<typename T1>
class MaskedBoxScalarCountFunctional2D : public ReductiveBoxProcessingFunctional2D_S<T1>
{
public:
    MaskedBoxScalarCountFunctional2D(plint mask_) : countId(this->getStatistics().subscribeSum()), mask(mask_)
    {}
    virtual void process(Box2D domain, ScalarField2D<T1>& scalar) {
        BlockStatistics& statistics = this->getStatistics();
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                plint tmpMask = util::roundToInt(scalar.get(iX, iY));
                if (tmpMask == mask) {
                    statistics.gatherSum(countId, (int)1);
                }
            }
        }
    }
    virtual MaskedBoxScalarCountFunctional2D<T1>* clone() const {
        return new MaskedBoxScalarCountFunctional2D<T1>(*this);
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

template<typename T1>
plint MaskedScalarCounts2D(Box2D domain, MultiScalarField2D<T1>& field, plint mask) {
    MaskedBoxScalarCountFunctional2D<T1> functional = MaskedBoxScalarCountFunctional2D<T1>(mask);
    applyProcessingFunctional(functional, domain, field);
    return functional.getCount();
}


template<typename T1, template<typename U1> class Descriptor1, typename T2, template<typename U2> class Descriptor2>
class BoxLatticeRMSEFunctional2D : public ReductiveBoxProcessingFunctional2D_LL<T1, Descriptor1, T2, Descriptor2>
{
public:
    BoxLatticeRMSEFunctional2D() : sumId(this->getStatistics().subscribeSum())
    {}
    virtual void process(Box2D domain, BlockLattice2D<T1, Descriptor1>& lattice0, BlockLattice2D<T2, Descriptor2>& lattice1) {
        BlockStatistics& statistics = this->getStatistics();
        Dot2D offset_01 = computeRelativeDisplacement(lattice0, lattice1);
        for (plint iX0 = domain.x0; iX0 <= domain.x1; ++iX0) {
            for (plint iY0 = domain.y0; iY0 <= domain.y1; ++iY0) {
                plint iX1 = iX0 + offset_01.x;
                plint iY1 = iY0 + offset_01.y;
                T deltaC = lattice0.get(iX0, iY0).computeDensity() - lattice1.get(iX1, iY1).computeDensity();
                T RMSE = deltaC * deltaC;
                statistics.gatherSum(sumId, RMSE);
            }
        }
    }
    virtual BoxLatticeRMSEFunctional2D<T1, Descriptor1, T2, Descriptor2>* clone() const {
        return new BoxLatticeRMSEFunctional2D<T1, Descriptor1, T2, Descriptor2>(*this);
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

template<typename T1, template<typename U1> class Descriptor1, typename T2, template<typename U2> class Descriptor2>
T computeRMSE2D(Box2D domain, MultiBlockLattice2D<T1, Descriptor1>& lattice0, MultiBlockLattice2D<T2, Descriptor2>& lattice1, T poreVolume) {
    BoxLatticeRMSEFunctional2D<T1, Descriptor1, T2, Descriptor2> functional;
    applyProcessingFunctional(functional, domain, lattice0, lattice1);
    return std::sqrt(functional.getCount() / poreVolume);
}

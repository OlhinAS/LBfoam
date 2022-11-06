/*
 *  LBfoam: An open-source software package for the simulation of foaming
 *  using the Lattice Boltzmann Method
 *  Copyright (C) 2020 Mohammadmehdi Ataei
 *  m.ataei@mail.utoronto.ca
 *  This file is part of LBfoam.
 *
 *  LBfoam is free software: you can redistribute it and/or modify it under
 *  the terms of the GNU Affero General Public License as published by the
 *  Free Software Foundation version 3.
 *
 *  LBfoam is distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 *  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU Affero General Public License
 *  along with this Program. If not, see <http://www.gnu.org/licenses/>.
 *
 *  #############################################################################
 *
 *  Author:         Mohammadmehdi Ataei, 2020
 *
 *  #############################################################################
 *
 *  Parts of the LBfoam code that originate from Palabos are distributed
 *  under the terms of the AGPL 3.0 license with the following copyright
 *  notice:
 *
 *  This file is part of the Palabos library.
 *
 *  Copyright (C) 2011-2017 FlowKit Sarl
 *  Route d'Oron 2
 *  1010 Lausanne, Switzerland
 *  E-mail contact: contact@flowkit.com
 *
 *  The most recent release of Palabos can be downloaded at
 *  <http://www.palabos.org/>
 *
 *  The library Palabos is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU Affero General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  The library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License for more details.
 *
 *  You should have received a copy of the GNU Affero General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "fenv.h"
#include "palabos3D.h"
#include "palabos3D.hh"

using namespace plb;
using namespace lbfoam;
#define DESCRIPTOR descriptors::ForcedD3Q19Descriptor
#define ADESCRIPTOR descriptors::AdvectionDiffusionWithSourceD3Q7Descriptor
#define ADYNAMICS AdvectionDiffusionWithSourceBGKdynamics
#define PADDING 8

typedef double T;
typedef Array<T, 3> Velocity;

std::string outDir("./tmp/");

struct SimulationParameters {
    /*
     * Parameters set by the user.
     */

    // Geometric parameters.
    std::map<int, Array<T, 3>> nucleiCenters;
    int numRowsX, numRowsY;
    int numberOfBubbles;
//    T shift;
    T radius;
    T packingOffset;
    std::string distribution;

    T contactAngle;
    T gasIni_LB;
    T temperature;
    T p_ini;
    T R_s;
    T source_LB;

    plint maxIter;
    T cSmago;
    bool freezeLargestBubble;
    bool surfaceDiffusion;
    bool gravity;

    T bubbleVolumeRatio;
    T alpha, beta;

    plint statIter;  // Output parameters.
    plint outIter;

//    plint nx, ny, nz;
    T fluidPoolHeight_LB;
    Array<T, 3> gVector_LB;
    T g_LB;
    T rho_LB;
    T tau_LB;
    T sigma_LB;
    T tauD_LB;
    T omega;
    T adOmega;
    T kh_LB;
    T pi_LB;

    plint extraLayer = 0;           // Make the bounding box larger; for visualization purposes
    //   only. For the simulation, it is OK to have extraLayer=0.
    const plint blockSize = 20;     // Zero means: no sparse representation.
    const plint envelopeWidth = 1;  // For standard BGK dynamics.
    const plint extendedEnvelopeWidth = 2;  // Because the Guo off lattice boundary condition
//   needs 2-cell neighbor access.


    plint referenceDirection = 0;
    plint openingSortDirection = 0;

    plint referenceResolution = 150;

    T uAveLB = 0.2;
    bool poiseuilleInlet = true;

    bool performOutput = true;

    std::string meshFileName;
};

template<typename T>
struct Opening {
    bool inlet;
    Array<T, 3> center;
    T innerRadius;
};

std::vector<Opening<T> > openings;

void iniLattice(MultiBlockLattice3D<T, DESCRIPTOR> &lattice, VoxelizedDomain3D<T> &voxelizedDomain) {
    // Switch all remaining outer cells to no-dynamics, except the outer
    //   boundary layer, and keep the rest as BGKdynamics.
    defineDynamics(
            lattice, voxelizedDomain.getVoxelMatrix(), lattice.getBoundingBox(),
            new NoDynamics<T, DESCRIPTOR>, voxelFlag::outside);
    initializeAtEquilibrium(
            lattice, lattice.getBoundingBox(), (T) 1., Array<T, 3>((T) 0., (T) 0., (T) 0.));
    lattice.initialize();
}

// This function assigns proper boundary conditions to the openings of the surface geometry
//   of the aneurysm. Which opening is inlet and which is outlet is defined by the user in
//   the input XML file. For the inlet, there is a choice between a Poiseuille velocity
//   profile and a simple plug velocity profile. At the outlets a Neumann boundary
//   condition with constant pressure is prescribed.
void setOpenings(
        std::vector<BoundaryProfile3D<T, Velocity> *> &inletOutlets, TriangleBoundary3D<T> &boundary,
        T uLB, plint &openingSortDirection, bool &poiseuilleInlet) {
    for (pluint i = 0; i < openings.size(); ++i) {
        Opening<T> &opening = openings[i];
        opening.center =
                computeBaryCenter(boundary.getMesh(), boundary.getInletOutlet(openingSortDirection)[i]);
        opening.innerRadius = computeInnerRadius(
                boundary.getMesh(), boundary.getInletOutlet(openingSortDirection)[i]);

        if (opening.inlet) {
            if (poiseuilleInlet) {
                inletOutlets.push_back(new PoiseuilleProfile3D<T>(uLB));
            } else {
                inletOutlets.push_back(new VelocityPlugProfile3D<T>(uLB));
            }
        } else {
            inletOutlets.push_back(new DensityNeumannBoundaryProfile3D<T>);
        }
    }
}

SimulationParameters param;

void readUserDefinedSimulationParameters(std::string xmlInputFileName,
                                         SimulationParameters &param) {

    std::vector<std::string> openingType;
    XMLreader document(xmlInputFileName);
//    std::vector<std::string> openingType{"Inlet Outlet Outlet"};


    document["geometry"]["mesh"].read(param.meshFileName);
    document["geometry"]["openings"]["type"].read(openingType);

    document["Nucleation"]["radius"].read(param.radius);
//    document["Nucleation"]["shift"].read(param.shift);
    document["Nucleation"]["distribution"].read(param.distribution);
    document["Nucleation"]["numberOfBubbles"].read(param.numberOfBubbles);
    document["Nucleation"]["packingOffset"].read(param.packingOffset);

    document["fluid"]["rho_LB"].read(param.rho_LB);
    document["fluid"]["R_s"].read(param.R_s);
    document["fluid"]["p_ini"].read(param.p_ini);
    document["fluid"]["temperature"].read(param.temperature);
    document["fluid"]["tau_LB"].read(param.tau_LB);
    document["fluid"]["tauD_LB"].read(param.tauD_LB);
    document["fluid"]["sigma_LB"].read(param.sigma_LB);
    document["fluid"]["kh_LB"].read(param.kh_LB);
    document["fluid"]["gasIni_LB"].read(param.gasIni_LB);
    document["fluid"]["pi_LB"].read(param.pi_LB);
    document["fluid"]["contactAngle"].read(param.contactAngle);
    document["fluid"]["surfaceDiffusion"].read(param.surfaceDiffusion);
    document["fluid"]["source_LB"].read(param.source_LB);

    T pi = acos((T) -1);
    param.contactAngle *= pi / 180.0;

    document["numerics"]["maxIter"].read(param.maxIter);
    document["numerics"]["cSmago"].read(param.cSmago);
    document["numerics"]["freezeLargestBubble"].read(param.freezeLargestBubble);
    document["numerics"]["gravity"].read(param.gravity);
    document["numerics"]["g_LB"].read(param.g_LB);

    document["numerics"]["bubbleVolumeRatio"].read(param.bubbleVolumeRatio);
    document["numerics"]["alpha"].read(param.alpha);
    document["numerics"]["beta"].read(param.beta);

    document["output"]["statIter"].read(param.statIter);
    document["output"]["outIter"].read(param.outIter);

    openings.resize(openingType.size());
    for (pluint i = 0; i < openingType.size(); ++i) {
        std::string next_opening = util::tolower(openingType[i]);
        if (next_opening == "inlet") {
            openings[i].inlet = true;
        } else if (next_opening == "outlet") {
            openings[i].inlet = false;
        } else {
            plbIOError("Unknown opening type.");
        }
    }
}

template<typename T, template<typename U> class Descriptor>
class SourceTerm : public BoxProcessingFunctional3D_L<T, Descriptor> {
public:
    SourceTerm(T source_) : source(source_) {};

    virtual void process(Box3D domain, BlockLattice3D<T, Descriptor> &lattice) {
        for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
            for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
                    lattice.get(iX, iY, iZ)
                            .setExternalField(Descriptor<T>::ExternalField::scalarBeginsAt,
                                              Descriptor<T>::ExternalField::sizeOfScalar,
                                              &source);
                }
            }
        }
    };

    virtual SourceTerm<T, Descriptor> *clone() const {
        return new SourceTerm<T, Descriptor>(*this);
    };

    virtual void getTypeOfModification(
            std::vector<modif::ModifT> &modified) const {
        modified[0] = modif::staticVariables;
    };

private:
    T source;
};

void calculateDerivedSimulationParameters(SimulationParameters &param) {
    // Derived quantities.

    if (!param.gravity) {
        param.g_LB = 0.;
    }

    param.gVector_LB = Array<T, 3>((T) 0, (T) 0, (T) -param.g_LB);
    param.adOmega = 1.0 / param.tauD_LB;
    param.omega = 1.0 / param.tau_LB;

    if (param.distribution == "random") {
        //       std::random_device rd;  //Will be used to obtain a seed for the
        //       random number engine

        //       std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded
        //       with rd()

//        T lowX = param.shift;
        T lowX = 60.;
//        T highX = param.nx - param.shift;
        T highX = 90.;

        T lowY = 50.;
        T highY = 70.;

        T lowZ = 245.;
        T highZ = 285.;

        namespace pds = thinks::poisson_disk_sampling;

        // Input parameters.

        T diameter = 2 * param.radius + param.packingOffset;
        const auto min = std::array<T, 3>{{lowX, lowY, lowZ}};
        const auto max = std::array<T, 3>{{highX, highY, highZ}};

        // Samples returned as std::vector<std::array<T, 3>>.
        // Default seed and max attempts.
        auto samples = pds::PoissonDiskSampling<T>(diameter, min, max);

        auto rng = std::default_random_engine(param.numberOfBubbles);

        std::shuffle(samples.begin(), samples.end(), rng);

        int N = 0;
        for (auto sample: samples) {
            if (N == param.numberOfBubbles) {
                break;
            }

            Array<T, 3> center(sample[0], sample[1], sample[2]);

            param.nucleiCenters.insert(std::pair<int, Array<T, 3>>(N, center));

            N++;
        }
    }
}

void printSimulationParameters(SimulationParameters const &param) {
    pcout << "fluidPoolHeight_LB = " << param.fluidPoolHeight_LB << std::endl;

    pcout << "g_LB = (" << param.gVector_LB[0] << ", " << param.gVector_LB[1]
          << ", " << param.gVector_LB[2] << " )" << std::endl;
    pcout << "rho_LB = " << param.rho_LB << std::endl;
    pcout << "sigma_LB = " << param.sigma_LB << std::endl;
    pcout << "omega = " << param.omega << std::endl;
    pcout << "tau_LB = " << param.tau_LB << std::endl;
    pcout << std::endl;
    pcout << "contractAngle = " << param.contactAngle * 180.0 / acos((T) -1)
          << std::endl;

    pcout << "gas_ini = " << param.gasIni_LB << std::endl;
    pcout << "tau_LB = " << param.tau_LB << std::endl;
    pcout << "adOmega = " << param.adOmega << std::endl;
    pcout << "Kh_LB = " << param.kh_LB << std::endl;
    pcout << "maxIter = " << param.maxIter << std::endl;
    pcout << "cSmago = " << param.cSmago << std::endl;
    pcout << "source = " << param.source_LB << std::endl;

    pcout << "freezeLargestBubble = "
          << (param.freezeLargestBubble ? "true" : "false") << std::endl;
    pcout << "bubbleVolumeRatio = " << param.bubbleVolumeRatio << std::endl;
    pcout << "alpha = " << param.alpha << std::endl;
    pcout << "beta = " << param.beta << std::endl;

    pcout << "statIter = " << param.statIter << std::endl;
    pcout << "outIter = " << param.outIter << std::endl;

    pcout << "nx = " << param.nx << std::endl;
    pcout << "ny = " << param.ny << std::endl;
    pcout << "nz = " << param.nz << std::endl;
    pcout << "distribution = " << param.distribution << std::endl;
    pcout << "number of bubbles = " << param.numberOfBubbles << std::endl;
//    pcout << "shift = " << param.shift << std::endl;
}

bool insideSphere(T x, T y, T z) {
    bool isInside = false;

    Array<T, 3> pos(x, y, z);
    typename std::map<int, Array<T, 3>>::const_iterator it =
            param.nucleiCenters.begin();
    for (; it != param.nucleiCenters.end(); ++it) {
        Array<T, 3> center = it->second;
        T r = norm<T, 3>(pos - center);
        if (r <= param.radius) {
            isInside = true;
        }
    }

    return isInside;
}

bool insideFluidPool(T x, T y, T z) {
    if (z <= param.fluidPoolHeight_LB) {
        return true;
    }
    return false;
}

bool insideFluid(T x, T y, T z) {
    if (insideFluidPool(x, y, z)) {
        return true;
    }
    return false;
}

void nucleateBubbles(FreeSurfaceFields3D<T, DESCRIPTOR> &fields,
                     plint subDivision, Dynamics<T, DESCRIPTOR> *dynamics) {
    typename std::map<int, Array<T, 3>>::const_iterator it =
            param.nucleiCenters.begin();

    for (; it != param.nucleiCenters.end(); ++it) {
        Array<T, 3> center = it->second;

        analyticalPunchSphere(fields, center, param.radius, param.rho_LB,
                              param.rho_LB, subDivision, *dynamics);
    }
}

// Specifies the initial condition for the fluid (each cell is assigned the
// flag "fluid", "empty", or "wall").
int initialFluidFlags(plint iX, plint iY, plint iZ) {
    if (insideFluid(iX, iY, iZ)) {
        return freeSurfaceFlag3D::fluid;
    }
    return freeSurfaceFlag3D::empty;
}

void writeResults(FreeSurfaceFields3D<T, DESCRIPTOR> *fields,
                  MultiBlockLattice3D<T, ADESCRIPTOR> adLattice,
                  MultiScalarField3D<plint> *tagMatrix,
                  MultiScalarField3D<double> *disjoiningPressureField,
                  plint iT) {
    std::unique_ptr<MultiScalarField3D<T>> smoothVF(lbmSmoothen<T, DESCRIPTOR>(
            fields->volumeFraction, fields->volumeFraction.getBoundingBox()));

    std::vector<T> isoLevels;
    isoLevels.push_back(0.5);

    typedef TriangleSet<T>::Triangle Triangle;
    std::vector<Triangle> triangles;
    isoSurfaceMarchingCube(triangles, *smoothVF, isoLevels,
                           smoothVF->getBoundingBox().enlarge(-2));
    {
        TriangleSet<T> triangleSet(triangles);
        triangleSet.scale(1.);
        triangleSet.writeBinarySTL(
                createFileName(outDir + "smoothedInterface_", iT, PADDING) + ".stl");
    }
    triangles.clear();
    isoSurfaceMarchingCube(triangles, fields->volumeFraction, isoLevels,
                           fields->volumeFraction.getBoundingBox().enlarge(-2));
    {
        TriangleSet<T> triangleSet(triangles);
        triangleSet.scale(1.);
        triangleSet.writeBinarySTL(
                createFileName(outDir + "interface_", iT, PADDING) + ".stl");
    }

    // T coef = 1.0 / 3.0;
    VtkImageOutput3D<T> vtkOut(
            createFileName(outDir + "volumeData_", iT, PADDING));
    std::unique_ptr<MultiTensorField3D<T, 3>> v = computeVelocity(fields->lattice);
    //    std::auto_ptr<MultiTensorField3D<T, 3> > adv =
    //    computeVelocity(adLattice);
    std::unique_ptr<MultiScalarField3D<T>> rho = computeDensity(fields->lattice);
    //    std::auto_ptr<MultiScalarField3D<T> > outdens =
    //            computeAbsoluteValue(fields->outsideDensity);
    std::unique_ptr<MultiScalarField3D<T>> adrho = computeDensity(adLattice);
    vtkOut.writeData<3, float>(*v, "velocity");
    //    vtkOut.writeData<3, float>(*adv, "adVelocity");
    vtkOut.writeData<float>(*rho, "pressure", 1);
    vtkOut.writeData<float>(*adrho, "adDensity", 1);
    //    vtkOut.writeData<float>(*outdens, "outsideDensity", 1);
    vtkOut.writeData<float>(fields->volumeFraction, "volumeFraction", 1.0);
    //    vtkOut.writeData<float>(*smoothVF, "smoothedVolumeFraction", 1.0);
    // vtkOut.writeData<float>(fields->outsideDensity, "outsidePressure");
    //    vtkOut.writeData<float>(*copyConvert<plint, T>(*tagMatrix),
    //    "bubbleTags",
    //                            1.0);
    //    vtkOut.writeData<float>(*copyConvert<double,
    //    T>(*disjoiningPressureField), "disjoiningPressure",
    //                            1.0);
}

bool entrapBubbles = false;
plint numRememberedVolumes = 1;

int main(int argc, char **argv) {
    plbInit(&argc, &argv);

    std::cout.precision(10);
    std::scientific(std::cout);

    // Command-line arguments

    if (argc != 2) {
        pcout << "Usage: " << argv[0] << " xml-input-file-name" << std::endl;
        exit(1);
    }

    std::string xmlInputFileName;
    xmlInputFileName = std::string(argv[1]);

    // Set the simulation parameters.

    readUserDefinedSimulationParameters(xmlInputFileName, param);
    calculateDerivedSimulationParameters(param);
    printSimulationParameters(param);

    plint margin = 3;       // Extra margin of allocated cells around the obstacle.
    plint borderWidth = 1;  // Because the Guo boundary condition acts in a one-cell layer.
    // Requirement: margin>=borderWidth.

    // The resolution is doubled at each coordinate direction with the increase of the
    //   resolution level by one. The parameter ``referenceResolution'' is by definition
    //   the resolution at grid refinement level 0.
//    plint resolution = param.referenceResolution * util::twoToThePower(level);
    plint resolution = param.referenceResolution;

    // The next few lines of code are typical. They transform the surface geometry of the
    //   aneurysm given by the user to more efficient data structures that are internally
    //   used by palabos. The TriangleBoundary3D structure will be later used to assign
    //   proper boundary conditions.

    TriangleSet<T> *triangleSet = new TriangleSet<T>(param.meshFileName, DBL);

    DEFscaledMesh<T> *defMesh =
            new DEFscaledMesh<T>(*triangleSet, resolution, param.referenceDirection, margin, param.extraLayer);
    TriangleBoundary3D<T> boundary(*defMesh);
    delete defMesh;
    boundary.getMesh().inflate();


    // Next the inlets and outlets are identified (according to what the user has specified)
    //   in the input XML file, and proper boundary conditions are assigned.
    std::vector<BoundaryProfile3D<T, Velocity> *> inletOutlets;
    setOpenings(inletOutlets, boundary, param.uAveLB, param.openingSortDirection, param.poiseuilleInlet);
    Array<T, 3> inletCenter(0.0, 0.0, 0.0);
    for (pluint i = 0; i < openings.size(); ++i) {
        if (openings[i].inlet) {
            pcout << "Inner radius of inlet " << i << " : " << openings[i].innerRadius
                  << " lattice nodes" << std::endl;
            inletCenter = openings[i].center;
        }
    }
    T inletZpos = util::roundToInt(inletCenter[2]) + 1;
    BoundaryProfiles3D<T, Velocity> profiles;
    profiles.defineInletOutletTags(boundary, param.openingSortDirection);
    profiles.setInletOutlet(inletOutlets);

    // The aneurysm simulation is an interior (as opposed to exterior) flow problem. For
    //   this reason, the lattice nodes that lay inside the computational domain must
    //   be identified and distinguished from the ones that lay outside of it. This is
    //   handled by the following voxelization process.
    const int flowType = voxelFlag::inside;
    VoxelizedDomain3D<T> voxelizedDomain(
            boundary, flowType, param.extraLayer, borderWidth, param.extendedEnvelopeWidth, param.blockSize);
    if (param.performOutput) {
        pcout << getMultiBlockInfo(voxelizedDomain.getVoxelMatrix()) << std::endl;
    }

    MultiScalarField3D<int> flagMatrix((MultiBlock3D &) voxelizedDomain.getVoxelMatrix());
    setToConstant(
            flagMatrix, voxelizedDomain.getVoxelMatrix(), voxelFlag::inside,
            flagMatrix.getBoundingBox(), 1);
    setToConstant(
            flagMatrix, voxelizedDomain.getVoxelMatrix(), voxelFlag::innerBorder,
            flagMatrix.getBoundingBox(), 1);
    pcout << "Number of fluid cells: " << computeSum(flagMatrix) << std::endl;

    {
        VtkImageOutput3D<T> vtkOut("./tmp/voxels_full_domain");
        vtkOut.writeData<float>(*copyConvert<int, T>(voxelizedDomain.getVoxelMatrix(),
                                                     voxelizedDomain.getVoxelMatrix().getBoundingBox()), "voxel", 1.0);
    }

//    SparseBlockStructure3D blockStructure(
//            createRegularDistribution3D(param.nx, param.ny, param.nz));
    SparseBlockStructure3D blockStructure(voxelizedDomain.getVoxelMatrix().getBoundingBox());

    Dynamics<T, DESCRIPTOR> *dynamics =
            new BGKdynamics<T, DESCRIPTOR>(param.omega);
    Dynamics<T, ADESCRIPTOR> *adynamics =
            new ADYNAMICS<T, ADESCRIPTOR>(param.adOmega);
    Dynamics<T, ADESCRIPTOR> *emptyDynamics = new NoDynamics<T, ADESCRIPTOR>();

    FreeSurfaceFields3D<T, DESCRIPTOR> fields(
            blockStructure, dynamics->clone(), param.rho_LB, param.sigma_LB,
            param.contactAngle, param.gVector_LB);
//    MultiBlockLattice3D<T, ADESCRIPTOR> adLattice(param.nx, param.ny, param.nz,
//                                                  adynamics->clone());

    // Temperature lattice.
    MultiBlockLattice3D<T, ADESCRIPTOR> adLattice =
            *new MultiBlockLattice3D<T, ADESCRIPTOR>(
                    (MultiBlock3D &)voxelizedDomain.getVoxelMatrix());
    defineDynamics(adLattice, adLattice.getBoundingBox(), adynamics);
    defineDynamics(adLattice, voxelizedDomain.getVoxelMatrix(), adLattice.getBoundingBox(), emptyDynamics, voxelFlag::inside);
    adLattice.toggleInternalStatistics(false);
    adLattice.periodicity().toggleAll(false);

    Array<T, 3> u0((T) 0, (T) 0, (T) 0);

    // The next piece of code is put for efficiency reasons at communications in parallel runs.
    //   The efficiency advantage comes essentially because the density and velocity are
    //   written in different fields.
    std::vector<MultiBlock3D *> rhoBarJarg;
    plint numScalars = 4;
    MultiNTensorField3D<T> *rhoBarJfield =
            generateMultiNTensorField3D<T>(fields.lattice, param.extendedEnvelopeWidth, numScalars);
    rhoBarJfield->toggleInternalStatistics(false);
    rhoBarJarg.push_back(rhoBarJfield);
    plint processorLevel = 0;
    integrateProcessingFunctional(
            new PackedRhoBarJfunctional3D<T, DESCRIPTOR>(), fields.lattice.getBoundingBox(), fields.lattice,
            *rhoBarJfield, processorLevel);

    // The Guo off lattice boundary condition is set up.
    GuoOffLatticeModel3D<T, DESCRIPTOR> *model = new GuoOffLatticeModel3D<T, DESCRIPTOR>(
            new TriangleFlowShape3D<T, Array<T, 3> >(voxelizedDomain.getBoundary(), profiles), flowType,
            true);
    model->setVelIsJ(
            false);  // When the incompressible BGK model is used, velocity equals momentum.
    model->selectUseRegularizedModel(true);
    model->selectComputeStat(false);
    OffLatticeBoundaryCondition3D<T, DESCRIPTOR, Velocity> boundaryCondition(
            model, voxelizedDomain, fields.lattice);
    boundaryCondition.insert(rhoBarJarg);


    // The boundary condition algorithm for the object.
    //TODO: Get out with T_wall values
    T T_wall = 1.0;
    BoundaryProfiles3D<T, Array<T, 2> > temperatureProfiles;
    temperatureProfiles.setWallProfile(new ScalarDirichletProfile3D<T>(T_wall));
    GuoAdvDiffOffLatticeModel3D<T, ADESCRIPTOR> *advDiffOffLatticeModel =
            new GuoAdvDiffOffLatticeModel3D<T, ADESCRIPTOR>(
                    new TriangleFlowShape3D<T, Array<T, 2> >(
                            voxelizedDomain.getBoundary(), temperatureProfiles),
                    flowType);
    advDiffOffLatticeModel->selectSecondOrder(true);
    OffLatticeBoundaryCondition3D<T, ADESCRIPTOR, Array<T, 2> >
            *temperatureBoundaryCondition =
            new OffLatticeBoundaryCondition3D<T, ADESCRIPTOR, Array<T, 2> >(
                    advDiffOffLatticeModel, voxelizedDomain, adLattice);
    temperatureBoundaryCondition->insert();



    // Initialization
    initializeAtEquilibrium(adLattice, adLattice.getBoundingBox(),
                            param.gasIni_LB, u0);
    adLattice.initialize();

    pcout << "Setting up initial condition." << std::endl;

    analyticalIniVolumeFraction(fields.volumeFraction, fields.flag, insideFluid,
                                32);

    nucleateBubbles(fields, 2, dynamics->clone());

//    Box3D bottom(0, param.nx - 1, 0, param.ny - 1, 0, 0);
//    Box3D top(0, param.nx - 1, 0, param.ny - 1, param.nz - 1, param.nz - 1);
//    Box3D lateral1(0, 0, 0, param.ny - 1, 0, param.nz - 1);
//    Box3D lateral2(param.nx - 1, param.nx - 1, 0, param.ny - 1, 0, param.nz - 1);
//    Box3D lateral3(0, param.nx - 1, 0, 0, 0, param.nz - 1);
//    Box3D lateral4(0, param.nx - 1, param.ny - 1, param.ny - 1, 0, param.nz - 1);
//    setToConstant(fields.flag, bottom, (int) freeSurfaceFlag3D::wall);
//    setToConstant(fields.flag, top, (int) freeSurfaceFlag3D::wall);
//
//    setToConstant(fields.flag, lateral1, (int) freeSurfaceFlag3D::wall);
//    setToConstant(fields.flag, lateral2, (int) freeSurfaceFlag3D::wall);
//    setToConstant(fields.flag, lateral3, (int) freeSurfaceFlag3D::wall);
//    setToConstant(fields.flag, lateral4, (int) freeSurfaceFlag3D::wall);

    fields.periodicityToggleAll(false);
//    adLattice.periodicity().toggleAll(false);
    fields.partiallyDefaultInitialize();

    plint iniIter = 0;

    BubbleTracking3D bubbleTracking(fields.flag);
    BubbleGrowth3D<T> bubbleGrowth(fields.flag);

    std::string fname = outDir + "bubbles.log";
    FILE *fp = fopen(fname.c_str(), "w");

    pcout << std::endl;

    // MultiScalarField3D<T> newvof = fields.volumeFraction;
    MultiScalarField3D<T> oldvof = fields.volumeFraction;

    integrateProcessingFunctional(new SourceTerm<T, ADESCRIPTOR>(param.source_LB),
                                  adLattice.getBoundingBox(), adLattice);
    // Main iteration loop.
    for (plint iT = iniIter; iT < param.maxIter; iT++) {
        if (iT % param.statIter == 0 || iT == param.maxIter - 1) {
            pcout << "At iteration " << iT << std::endl;
            T avE = computeAverageEnergy(fields.lattice);
            pcout << "Average kinetic energy: " << avE << std::endl;
            plint numIntCells = fields.lattice.getInternalStatistics().getIntSum(0);
            pcout << "Number of interface cells: " << numIntCells << std::endl;
            if (iT != iniIter) {
                pcout << "Time spent for each iteration: "
                      << global::timer("iteration").getTime() / (T) param.statIter
                      << std::endl;
                global::timer("iteration").reset();
            }
            pcout << std::endl;
        }

        if (iT % param.outIter == 0 || iT == param.maxIter - 1) {
            pcout << "Writing results at iteration " << iT << std::endl;
            global::timer("images").restart();
            writeResults(&fields, adLattice, bubbleTracking.getTagMatrix(),
                         bubbleTracking.setDisjoiningPressureFieldField(), iT);
            global::timer("images").stop();
            pcout << "Time spent for writing results: "
                  << global::timer("images").getTime() << std::endl;
            pcout << std::endl;
        }

        global::timer("iteration").start();
        T bubbleVolumeRatio = iT == 0 ? 1.0 : param.bubbleVolumeRatio;
        bool incompressibleModel = fields.dynamics->velIsJ();

        bubbleTracking.execute<T, DESCRIPTOR, ADESCRIPTOR>(
                fields.volumeFraction, fields.flag, fields.normal, fields.rhoBar,
                fields.mass, fields.j, adLattice, oldvof, false, param.pi_LB);

        oldvof = fields.volumeFraction;

        bubbleGrowth.transition(bubbleTracking, iT, param.temperature, param.R_s,
                                param.p_ini, 1., param.rho_LB, bubbleVolumeRatio,
                                entrapBubbles, numRememberedVolumes);
        bubbleGrowth.updateBubbleGrowth(fields.outsideDensity, param.rho_LB,
                                        param.alpha, param.beta, (T) 1.);

        // Free surface
        fields.lattice.executeInternalProcessors();
        fields.lattice.evaluateStatistics();
        fields.lattice.incrementTime();

        if (iT == 0 && param.freezeLargestBubble) {
            bubbleGrowth.freezeLargestBubble();
        }

        // order matters
        std::vector<MultiBlock3D *> couplingBlocks;
        couplingBlocks.push_back(&adLattice);
        couplingBlocks.push_back(&fields.lattice);
        couplingBlocks.push_back(&fields.flag);
        couplingBlocks.push_back(&fields.j);
        couplingBlocks.push_back(&fields.outsideDensity);
        couplingBlocks.push_back(bubbleGrowth.getOldTagMatrix());

        adLattice.collideAndStream();

        applyProcessingFunctional(
                new GrowthCoupling3D<T, ADESCRIPTOR, DESCRIPTOR>(
                        adynamics->clone(), emptyDynamics->clone(), param.kh_LB,
                        bubbleGrowth.getBubbles(), param.surfaceDiffusion),
                adLattice.getBoundingBox(), couplingBlocks);


        global::timer("iteration").stop();

        if (iT % param.statIter == 0 || iT == param.maxIter - 1) {
            bubbleGrowth.timeHistoryLog(outDir + "bubbleTimeHistory.log");
            bubbleGrowth.fullBubbleLog(outDir + "FullBubbleRecord3D.log");

            // We do not log frozen bubbles.
            std::map<plint, lbfoam::BubbleInfo3D>::const_iterator it =
                    bubbleGrowth.getBubbles().begin();
            T totalBubbleVolume = T();
            T currentDensity = T();
            T totalDisjoining = T();
            plint numBubbles = 0;
            for (; it != bubbleGrowth.getBubbles().end(); ++it) {
                if (it->second.isFrozen()) {
                    pcout << "Bubble with this ID is frozen: " << it->first << std::endl;
                    continue;
                }
                numBubbles++;
                T v = it->second.getVolume();
                T d = it->second.getCurrentDensity();
                T j = it->second.setDisjoiningPressureField();

                totalBubbleVolume += v;
                currentDensity += d;
                totalDisjoining += j;
            }
            pcout << "At iteration " << iT << ", the number of bubbles is "
                  << numBubbles << std::endl;
            pcout << "The total volume of bubbles is: " << totalBubbleVolume
                  << std::endl;
            pcout << "The total density is: " << currentDensity << std::endl;
            pcout << "The total disjoining pressure is: " << totalDisjoining
                  << std::endl;

            pcout << std::endl;
            fflush(fp);
        }
    }

    fclose(fp);
    delete dynamics;
    delete emptyDynamics;
    delete adynamics;

    exit(0);
}

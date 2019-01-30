#if !defined(_MFEMOPT_MFEMEXTRA_HPP)
#define _MFEMOPT_MFEMEXTRA_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemopt/reducedfunctional.hpp>
#include <mfem/fem/pfespace.hpp>
#include <mfem/mesh/pmesh.hpp>
#include <mfem/fem/coefficient.hpp>
#include <mfem/linalg/petsc.hpp>

namespace mfemopt
{

mfem::ParMesh* ParMeshTest(MPI_Comm,mfem::Mesh&);

void MeshGetElementsTagged(mfem::Mesh*,const mfem::Array<int>&,mfem::Array<bool>&);
void MeshGetElementsTagged(mfem::Mesh*,bool(*)(const mfem::Vector&),mfem::Array<bool>&);

void FiniteElementSpaceGetRangeAndDeriv(mfem::FiniteElementSpace&,int*,int*);
void ParFiniteElementSpaceGetRangeAndDeriv(mfem::ParFiniteElementSpace&,int*,int*);

class ComponentCoefficient : public mfem::Coefficient
{
private:
   mfem::VectorCoefficient *VQ;
   int c;
   mfem::Vector w;

public:
   ComponentCoefficient(mfem::VectorCoefficient&,int);
   virtual double Eval(mfem::ElementTransformation&,
                       const mfem::IntegrationPoint&);
};

class DiagonalMatrixCoefficient : public mfem::MatrixCoefficient
{
private:
   mfem::VectorCoefficient *VQ;
   mfem::Vector w;
   bool own;

public:
   DiagonalMatrixCoefficient(mfem::VectorCoefficient*,bool=false);

   virtual void Eval(mfem::DenseMatrix&,mfem::ElementTransformation&,
                     const mfem::IntegrationPoint&);
   virtual ~DiagonalMatrixCoefficient();
};

class PetscNonlinearSolverOpt : public mfem::PetscNonlinearSolver
{
public:
   PetscNonlinearSolverOpt(MPI_Comm,ReducedFunctional&,
                           const std::string& = std::string(),
                           bool = true);
};

}
#endif

#endif

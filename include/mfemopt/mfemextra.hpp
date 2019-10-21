#if !defined(_MFEMOPT_MFEMEXTRA_HPP)
#define _MFEMOPT_MFEMEXTRA_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfemopt/reducedfunctional.hpp>
#include <mfemopt/datareplicator.hpp>
#include <mfem/fem/pfespace.hpp>
#include <mfem/mesh/pmesh.hpp>
#include <mfem/fem/coefficient.hpp>
#include <mfem/linalg/petsc.hpp>
#include <petscsftypes.h>

namespace mfemopt
{

mfem::ParMesh* ParMeshTest(MPI_Comm,mfem::Mesh&);

void ParMeshPrint(mfem::ParMesh& mesh, const char* filename);

// Cannot inherit from ParMesh, since the default constructor is protected
class ReplicatedParMesh
{
private:
   DataReplicator drep;
   mfem::ParMesh *parent_mesh;
   mfem::ParMesh *child_mesh;

public:
   ReplicatedParMesh(MPI_Comm,mfem::Mesh&,int,bool=true,int** =NULL);
   inline mfem::ParMesh* GetChild() { return child_mesh; }
   inline mfem::ParMesh* GetParent() { return parent_mesh; }
   inline int GetColor() { return drep.GetColor(); }
   inline MPI_Comm GetRedComm() { return drep.GetRedComm(); }
   bool IsMaster() { return drep.GetColor() ? false : true; }
   DataReplicator& GetReplicator() { return drep; }
   virtual ~ReplicatedParMesh();
};

class ReplicatedParFiniteElementSpace
{
private:
   mfem::ParFiniteElementSpace *cfes;
   mfem::ParFiniteElementSpace *pfes;

   PetscSF red_V_sf;
   PetscSF red_T_sf;

public:
   ReplicatedParFiniteElementSpace(ReplicatedParMesh*,const mfem::FiniteElementCollection*,
                                   int=1,int=mfem::Ordering::byNODES);
   void Broadcast(const mfem::Vector&,mfem::Vector&);
   void Reduce(const mfem::Vector&,mfem::Vector&,MPI_Op = MPI_SUM);
   void TBroadcast(const mfem::Vector&,mfem::Vector&);
   void TReduce(const mfem::Vector&,mfem::Vector&,MPI_Op = MPI_SUM);
   inline mfem::ParFiniteElementSpace* GetParent() { return pfes; }
   inline mfem::ParFiniteElementSpace* GetChild() { return cfes; }
   virtual ~ReplicatedParFiniteElementSpace();
};

void MeshGetElementsTagged(mfem::Mesh*,const mfem::Array<int>&,mfem::Array<bool>&);
void MeshGetElementsTagged(mfem::Mesh*,bool(*)(const mfem::Vector&),mfem::Array<bool>&,bool = false);

void FiniteElementSpaceGetRangeAndDeriv(mfem::FiniteElementSpace&,int*,int*);
void ParFiniteElementSpaceGetRangeAndDeriv(mfem::ParFiniteElementSpace&,int*,int*);

class FunctionOfCoefficient : public mfem::Coefficient
{
private:
   double (*f)(double);
   mfem::Coefficient *g;

public:
   // Result is f(g(x))
   FunctionOfCoefficient(double (*_f)(double), mfem::Coefficient &_g): f(_f), g(&_g) { }

   virtual double Eval(mfem::ElementTransformation&,const mfem::IntegrationPoint&);
};

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

}
#endif

#endif

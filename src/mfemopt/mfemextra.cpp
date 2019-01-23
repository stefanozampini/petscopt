#include <mfemopt/mfemextra.hpp>

static void __mfemopt_snes_obj(mfem::Operator*,const mfem::Vector&,double*);
static void __mfemopt_snes_postcheck(mfem::Operator*,const mfem::Vector&,mfem::Vector&,mfem::Vector&,bool&,bool&);

namespace mfemopt
{
using namespace mfem;

ParMesh* ParMeshTest(MPI_Comm comm, Mesh &mesh)
{
   int size;
   MPI_Comm_size(comm,&size);
   int nel = mesh.GetNE();
   int *test_partitioning = new int[nel];
   for (int i = 0; i < nel; i++) test_partitioning[i] = (int)((1.0*i*size)/(1.0*nel));

   ParMesh *pmesh = new ParMesh(comm,mesh,test_partitioning);
   delete[] test_partitioning;
   return pmesh;
}

PetscNonlinearSolverOpt::PetscNonlinearSolverOpt(MPI_Comm comm, ReducedFunctional &rf,
                         const std::string &prefix, bool obj) : PetscNonlinearSolver(comm,rf,prefix)
{
   if (obj)
   {
      SetObjective(__mfemopt_snes_obj);
   }
   SetPostCheck(__mfemopt_snes_postcheck);
}

}

void __mfemopt_snes_obj(mfem::Operator *op, const mfem::Vector& u, double *f)
{
   mfemopt::ReducedFunctional *rf = dynamic_cast<mfemopt::ReducedFunctional*>(op);
   MFEM_VERIFY(rf,"Not a mfemopt::ReducedFunctional operator");
   rf->ComputeObjective(u,f);
}

void __mfemopt_snes_postcheck(mfem::Operator *op, const mfem::Vector& X, mfem::Vector& Y, mfem::Vector &W, bool& cy, bool& cw)
{
   mfemopt::ReducedFunctional *rf = dynamic_cast<mfemopt::ReducedFunctional*>(op);
   MFEM_VERIFY(rf,"Not a mfemopt::ReducedFunctional operator");
   cy = false;
   cw = false;
   rf->PostCheck(X,Y,W,cy,cw);
}

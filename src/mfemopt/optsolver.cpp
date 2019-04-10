#include <mfemopt/optsolver.hpp>

static void __mfemopt_snes_obj_rf(mfem::Operator*,const mfem::Vector&,double*);
static void __mfemopt_snes_update_rf(mfem::Operator*,int,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&);
static void __mfemopt_snes_postcheck_rf(mfem::Operator*,const mfem::Vector&,mfem::Vector&,mfem::Vector&,bool&,bool&);

namespace mfemopt
{
using namespace mfem;

/*
   In principle, if a BCHandler is present,
   essential BC will not be properly imposed if the first step is not a full step (lambda = 1.0)
   This should be fixed in the MFEM classes
*/
PetscNonlinearSolverOpt::PetscNonlinearSolverOpt(MPI_Comm comm, ReducedFunctional &rf,
                         const std::string &prefix, bool obj) : PetscNonlinearSolver(comm,rf,prefix)
{
   if (obj) /* Use objective in line-search based methods */
   {
      SetObjective(__mfemopt_snes_obj_rf);
   }
   /* method to be called BEFORE sampling the Jacobian */
   SetUpdate(__mfemopt_snes_update_rf);
   /* method to be called AFTER successfull line-search */
   SetPostCheck(__mfemopt_snes_postcheck_rf);
}

}

void __mfemopt_snes_obj_rf(mfem::Operator *op, const mfem::Vector& u, double *f)
{
   mfemopt::ReducedFunctional *rf = dynamic_cast<mfemopt::ReducedFunctional*>(op);
   MFEM_VERIFY(rf,"Not a mfemopt::ReducedFunctional operator");
   rf->ComputeObjective(u,f);
}

void __mfemopt_snes_update_rf(mfem::Operator *op, int it, const mfem::Vector& X, const mfem::Vector& Y, const mfem::Vector &W, const mfem::Vector &Z)
{
   mfemopt::ReducedFunctional *rf = dynamic_cast<mfemopt::ReducedFunctional*>(op);
   MFEM_VERIFY(rf,"Not a mfemopt::ReducedFunctional operator");
   rf->Update(it,X,Y,W,Z);
}

void __mfemopt_snes_postcheck_rf(mfem::Operator *op, const mfem::Vector& X, mfem::Vector& Y, mfem::Vector &W, bool& cy, bool& cw)
{
   mfemopt::ReducedFunctional *rf = dynamic_cast<mfemopt::ReducedFunctional*>(op);
   MFEM_VERIFY(rf,"Not a mfemopt::ReducedFunctional operator");
   cy = false;
   cw = false;
   rf->PostCheck(X,Y,W,cy,cw);
}

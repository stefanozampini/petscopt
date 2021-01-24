#include <petscsnes.h>
#include <petscopt/petscopt_types.h>
#include <petscopt/ksp.h>
#include <mfemopt/optsolver.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>

static void __mfemopt_snes_obj_rf(mfem::Operator*,const mfem::Vector&,double*);
static void __mfemopt_snes_update_rf(mfem::Operator*,int,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&);
static void __mfemopt_snes_postcheck_rf(mfem::Operator*,const mfem::Vector&,mfem::Vector&,mfem::Vector&,bool&,bool&);
static PetscErrorCode __mfemopt_hilbert_riesz(Vec,Vec,void*);
static PetscErrorCode __mfemopt_hilbert_inner(Vec,Vec,double*,void*);

namespace mfemopt
{
using namespace mfem;

/*
   In principle, if a BCHandler is present,
   essential BC will not be properly imposed if the first step is not a full step (lambda = 1.0)
   How to handle it properly?
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

   /* Use special KSP */
   HilbertReducedFunctional *hf = const_cast<HilbertReducedFunctional *>
                          (dynamic_cast<const HilbertReducedFunctional *>(&rf));
   if (hf)
   {
      KSP ksp;
      PetscErrorCode ierr;

      ierr = SNESGetKSP(*this,&ksp);PCHKERRQ(*this,ierr);
      ierr = KSPSetType(ksp,KSPHILBERTCG);PCHKERRQ(*this,ierr);
      ierr = KSPHilbertCGSetMaps(ksp,__mfemopt_hilbert_inner,__mfemopt_hilbert_riesz,&rf);PCHKERRQ(*this,ierr);
   }
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

#include <petsc/private/petscimpl.h>
#include <mfemopt/private/utils.hpp>

PetscErrorCode __mfemopt_hilbert_riesz(Vec x, Vec y, void* op)
{
   PetscErrorCode ierr;
   mfemopt::HilbertReducedFunctional *rf = mfemopt::mi_void_safe_cast<mfemopt::HilbertReducedFunctional>(op);

   PetscFunctionBeginUser;
   if (!rf) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_LIB,"Not a mfemopt::HilbertReducedFunctional operator");
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   rf->Riesz(xx,yy);
   ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

PetscErrorCode __mfemopt_hilbert_inner(Vec x, Vec y, double *f, void* op)
{
   mfemopt::HilbertReducedFunctional *rf = mfemopt::mi_void_safe_cast<mfemopt::HilbertReducedFunctional>(op);

   PetscFunctionBeginUser;
   if (!rf) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_LIB,"Not a mfemopt::HilbertReducedFunctional operator");
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   rf->Inner(xx,yy,f);
   PetscFunctionReturn(0);
}

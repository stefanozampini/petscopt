#include <mfemopt/monitor.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <petscsnes.h>
#include <petscdm.h>

namespace mfemopt
{

using namespace mfem;

void NewtonMonitor::MonitorSolver(PetscSolver *solver)
{
   PetscNonlinearSolver *newton = dynamic_cast<PetscNonlinearSolver *>(solver);
   MFEM_VERIFY(newton,"Not a nonlinear solver");

   SNES snes = (SNES)(*newton);

   DM dm;
   KSP ksp;
   PetscErrorCode ierr;
   SNESLineSearch ls;
   Vec X,dX,G,pG;
   PetscErrorCode (*snesobj)(SNES,Vec,PetscReal*,void*);
   PetscReal lambda,normg;
   PetscScalar inn;
   PetscInt it,lit;

   ierr = SNESGetDM(snes,&dm); PCHKERRQ(snes,ierr);
   ierr = DMGetNamedGlobalVector(dm,"mfemopt_prev_gradient",&pG); PCHKERRQ(snes,ierr);
   ierr = SNESGetKSP(snes,&ksp); PCHKERRQ(snes,ierr);
   ierr = KSPGetIterationNumber(ksp,&lit); PCHKERRQ(snes,ierr);
   ierr = SNESGetSolution(snes,&X); PCHKERRQ(snes,ierr);
   ierr = SNESGetSolutionUpdate(snes,&dX); PCHKERRQ(snes,ierr);
   ierr = SNESGetObjective(snes,&snesobj,NULL); PCHKERRQ(snes,ierr);
   ierr = SNESGetIterationNumber(snes,&it); PCHKERRQ(snes,ierr);
   ierr = SNESGetLineSearch(snes,&ls); PCHKERRQ(snes,ierr);
   ierr = SNESLineSearchGetLambda(ls,&lambda); PCHKERRQ(snes,ierr);
   ierr = VecDot(pG,dX,&inn); PCHKERRQ(snes,ierr);
   ierr = VecNorm(pG,NORM_2,&normg); PCHKERRQ(snes,ierr);
   if (!it) {
     PetscInt dofs;

     ierr = VecGetSize(X,&dofs); PCHKERRQ(snes,ierr);
     ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Number of dofs %D\n",dofs); PCHKERRQ(snes,ierr);
     ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"it\tlit\tenergy\t\t(g,du)\t\t||g||_l2\tstep\n"); PCHKERRQ(snes,ierr);
     lambda = 0.0;
   }
   if (snesobj)
   {
      PetscReal f;

      ierr = SNESComputeObjective(snes,X,&f); PCHKERRQ(snes,ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"%D\t%D\t%1.6e\t",it,lit,(double)f); PCHKERRQ(snes,ierr);
   }
   else
   {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"%D\t%D\t------------\t",it,lit); PCHKERRQ(snes,ierr);
   }
   ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"%1.6e\t%1.6e\t%1.6e\n",-(double)PetscRealPart(inn),(double)normg,(double)lambda); PCHKERRQ(snes,ierr);
   ierr = SNESGetFunction(snes,&G,NULL,NULL); PCHKERRQ(snes,ierr);
   ierr = VecCopy(G,pG); PCHKERRQ(snes,ierr);
   ierr = DMRestoreNamedGlobalVector(dm,"mfemopt_prev_gradient",&pG); PCHKERRQ(snes,ierr);
}

}

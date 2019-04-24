#include <mfemopt/monitor.hpp>
#include <mfemopt/optsolver.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <petsctao.h>
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
   /* Comment the next two lines to check against Georg's code */
   ierr = SNESGetFunction(snes,&G,NULL,NULL); PCHKERRQ(snes,ierr);
   ierr = VecCopy(G,pG); PCHKERRQ(snes,ierr);
   ierr = VecNorm(pG,NORM_2,&normg); PCHKERRQ(snes,ierr);
   if (snesobj)
   {
      PetscReal f;

      ierr = SNESComputeObjective(snes,X,&f); PCHKERRQ(snes,ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"it=%D\tlit=%D\tobj=%1.6e\t",it,lit,(double)f); PCHKERRQ(snes,ierr);
   }
   else
   {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"it=%D\tlit=%D\t------------\t",it,lit); PCHKERRQ(snes,ierr);
   }
   ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"(g,du)=%1.6e\t||g||=%1.6e\tstep=%1.6e\n",-(double)PetscRealPart(inn),(double)normg,(double)lambda); PCHKERRQ(snes,ierr);
   ierr = SNESGetFunction(snes,&G,NULL,NULL); PCHKERRQ(snes,ierr);
   ierr = VecCopy(G,pG); PCHKERRQ(snes,ierr);
   ierr = DMRestoreNamedGlobalVector(dm,"mfemopt_prev_gradient",&pG); PCHKERRQ(snes,ierr);
}

void OptimizationMonitor::MonitorSolver(PetscSolver *solver)
{
   PetscOptimizationSolver *opt = dynamic_cast<PetscOptimizationSolver *>(solver);
   MFEM_VERIFY(opt,"Not a optimization solver");

   Tao tao = (Tao)(*opt);

   KSP ksp;
   PetscErrorCode ierr;
   TaoLineSearch ls;
   Vec X,dX = NULL,G,pG;
   PetscReal lambda = 1.0,normg,f;
   PetscScalar inn = 0.0;
   PetscInt it,lit = 0;

   ierr = TaoGetSolutionVector(tao,&X); PCHKERRQ(tao,ierr);
   ierr = TaoGetGradientVector(tao,&G); PCHKERRQ(tao,ierr);
   ierr = PetscObjectQuery((PetscObject)tao,"mfemopt_prev_gradient",(PetscObject*)&pG); PCHKERRQ(tao,ierr);
   if (!pG) {
     ierr = VecDuplicate(G,&pG); PCHKERRQ(G,ierr);
     ierr = PetscObjectCompose((PetscObject)tao,"mfemopt_prev_gradient",(PetscObject)pG); PCHKERRQ(tao,ierr);
     ierr = PetscObjectDereference((PetscObject)pG); PCHKERRQ(tao,ierr);
   }
   ierr = TaoGetKSP(tao,&ksp); PCHKERRQ(tao,ierr);
   if (ksp)
   {
      ierr = KSPGetIterationNumber(ksp,&lit); PCHKERRQ(tao,ierr);
   }
   ierr = TaoGetIterationNumber(tao,&it); PCHKERRQ(tao,ierr);
   ierr = TaoGetLineSearch(tao,&ls); PCHKERRQ(tao,ierr);
   if (ls)
   {
      ierr = TaoLineSearchGetStepLength(ls,&lambda); PCHKERRQ(ls,ierr);
      ierr = TaoLineSearchGetStepDirection(ls,&dX); PCHKERRQ(ls,ierr);
   }
   if (dX)
   {
      ierr = VecDot(pG,dX,&inn); PCHKERRQ(tao,ierr);
      inn *= -1.0; /* Tao update is Xnew = Xold + lambda * dX, SNES update is Xnew = Xold - lambda * dX */
   }
   /* Comment the next two line to check against Georg's code */
   ierr = VecCopy(G,pG); PCHKERRQ(tao,ierr);
   ierr = VecNorm(pG,NORM_2,&normg); PCHKERRQ(tao,ierr);

   ierr = TaoComputeObjective(tao,X,&f); PCHKERRQ(tao,ierr);
   ierr = PetscPrintf(PetscObjectComm((PetscObject)tao),"it=%D\tlit=%D\tobj=%1.6e\t",it,lit,(double)f); PCHKERRQ(tao,ierr);
   ierr = PetscPrintf(PetscObjectComm((PetscObject)tao),"(g,du)=%1.6e\t||g||=%1.6e\tstep=%1.6e\n",-(double)PetscRealPart(inn),(double)normg,(double)lambda); PCHKERRQ(tao,ierr);
   ierr = VecCopy(G,pG); PCHKERRQ(tao,ierr);
}

}

/* This code can be eventually be moved to MFEM */
#include <mfemopt/optsolver.hpp>
#include <mfemopt/pdoperator.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <petsctao.h>
#include <petsc/private/petscimpl.h>

static PetscErrorCode ierr;

static PetscErrorCode __mfem_tao_monitor(Tao,void*);
static PetscErrorCode __mfem_tao_obj(Tao,Vec,PetscReal*,void*);
static PetscErrorCode __mfem_tao_grad(Tao,Vec,Vec,void*);
static PetscErrorCode __mfem_tao_hessian(Tao,Vec,Mat,Mat,void*);
static PetscErrorCode __mfem_tao_objgrad(Tao,Vec,PetscReal*,Vec,void*);
static PetscErrorCode __mfem_tao_update(Tao,PetscInt,void*);

typedef struct
{
   mfemopt::ReducedFunctional *objective;
} __mfem_tao_ctx;


// duplicated from MFEM
static PetscErrorCode __mfem_monitor_ctx_destroy(void **);
typedef struct
{
   mfem::PetscSolver        *solver;  // The solver object
   mfem::PetscSolverMonitor *monitor; // The user-defined monitor class
} __mfem_monitor_ctx;

namespace mfemopt
{

using namespace mfem;

PetscOptimizationSolver::PetscOptimizationSolver(MPI_Comm comm, const std::string &prefix)
{
   private_ctx = NULL;
   CreatePrivateContext();
   ierr = TaoCreate(comm,&tao); CCHKERRQ(comm,ierr);
   ierr = TaoSetType(tao,TAOCG); PCHKERRQ(tao,ierr);
   ierr = TaoSetOptionsPrefix(tao, prefix.c_str()); PCHKERRQ(tao, ierr);
}

PetscOptimizationSolver::PetscOptimizationSolver(MPI_Comm comm, ReducedFunctional& f, const std::string &prefix)
{
   private_ctx = NULL;
   CreatePrivateContext();
   ierr = TaoCreate(comm,&tao); CCHKERRQ(comm,ierr);
   ierr = TaoSetType(tao,TAOCG); PCHKERRQ(tao,ierr);
   ierr = TaoSetOptionsPrefix(tao, prefix.c_str()); PCHKERRQ(tao, ierr);
   Init(f);
}

void PetscOptimizationSolver::CreatePrivateContext()
{
   ierr = PetscFree(private_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);

   __mfem_tao_ctx *tao_ctx;
   ierr = PetscNew(&tao_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
   private_ctx = tao_ctx;
}

void PetscOptimizationSolver::Init(ReducedFunctional& f)
{
   // base class
   objective = &f;

   // private context
   __mfem_tao_ctx *tao_ctx = (__mfem_tao_ctx*)private_ctx;
   tao_ctx->objective = objective;

   ierr = TaoSetObjectiveRoutine(tao,__mfem_tao_obj,tao_ctx); PCHKERRQ(tao,ierr);
   ierr = TaoSetGradientRoutine(tao,__mfem_tao_grad,tao_ctx); PCHKERRQ(tao,ierr);
   ierr = TaoSetObjectiveAndGradientRoutine(tao,__mfem_tao_objgrad,tao_ctx); PCHKERRQ(tao,ierr);

   Mat H;
   ierr = MatCreate(PetscObjectComm((PetscObject)tao),&H); PCHKERRQ(tao,ierr);
   ierr = TaoSetHessianRoutine(tao,H,H,__mfem_tao_hessian,tao_ctx); PCHKERRQ(tao,ierr);
   ierr = MatDestroy(&H); PCHKERRQ(tao,ierr);

   ierr = TaoSetUpdate(tao,__mfem_tao_update,tao_ctx); PCHKERRQ(tao,ierr);

   ierr = TaoSetFromOptions(tao); PCHKERRQ(tao,ierr);

   Vector l,h;
   objective->GetBounds(l,h);
   PetscParVector pL(PetscObjectComm((PetscObject)tao),l,true);
   PetscParVector pH(PetscObjectComm((PetscObject)tao),h,true);
   ierr = TaoSetVariableBounds(tao,pL,pH); PCHKERRQ(tao,ierr);

}

// duplicated from PetscSolver::SetMonitor
void PetscOptimizationSolver::SetMonitor(PetscSolverMonitor *ctx)
{
   __mfem_monitor_ctx *monctx;
   ierr = PetscNew(&monctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
   monctx->solver = this;
   monctx->monitor = ctx;
   ierr = TaoSetMonitor(tao,__mfem_tao_monitor,monctx,__mfem_monitor_ctx_destroy); PCHKERRQ(tao,ierr);
}

void PetscOptimizationSolver::Solve(Vector& sol)
{
   if (!iterative_mode) objective->ComputeGuess(sol); /* XXX remove ? */
   objective->Update(sol); /* XXX rename ? */

   PetscParVector X(PetscObjectComm((PetscObject)tao),sol);
   X.PlaceArray(sol.GetData());

   ierr = TaoSetInitialVector(tao,X); PCHKERRQ(tao,ierr);
   ierr = TaoSolve(tao); PCHKERRQ(tao,ierr);

   X.ResetArray();
}

PetscOptimizationSolver::~PetscOptimizationSolver()
{
   MPI_Comm comm = PetscObjectComm((PetscObject)tao);
   ierr = TaoDestroy(&tao); CCHKERRQ(comm,ierr);
   ierr = PetscFree(private_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

} //namespace mfemopt

static PetscErrorCode __mfem_tao_obj(Tao tao, Vec x, PetscReal* f, void* ctx)
{
   __mfem_tao_ctx *tao_ctx = (__mfem_tao_ctx*)ctx;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   tao_ctx->objective->ComputeObjective(xx,f);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_tao_grad(Tao tao, Vec x, Vec g, void* ctx)
{
   __mfem_tao_ctx *tao_ctx = (__mfem_tao_ctx*)ctx;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector gg(g,true);
   tao_ctx->objective->ComputeGradient(xx,gg);
   ierr = PetscObjectStateIncrease((PetscObject)g);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_tao_objgrad(Tao tao, Vec x, PetscReal *f, Vec g, void* ctx)
{
   __mfem_tao_ctx *tao_ctx = (__mfem_tao_ctx*)ctx;
   double lf = 0.0;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector gg(g,true);
   tao_ctx->objective->ComputeObjectiveAndGradient(xx,&lf,gg);
   *f = lf;
   ierr = PetscObjectStateIncrease((PetscObject)g);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_tao_hessian(Tao tao, Vec x, Mat H, Mat Hpre, void* ctx)
{
   __mfem_tao_ctx *tao_ctx = (__mfem_tao_ctx*)ctx;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::Operator& HH = tao_ctx->objective->GetHessian(xx);
   mfem::PetscParMatrix *pHH = const_cast<mfem::PetscParMatrix *>
                               (dynamic_cast<const mfem::PetscParMatrix *>(&HH));
   bool delete_mat = false;
   if (!pHH)
   {
      pHH = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)tao),&HH);
      delete_mat = true;
   }

   // Avoid unneeded copy of the matrix by hacking
   Mat B = pHH->ReleaseMat(false);
   ierr = MatHeaderReplace(H,&B);CHKERRQ(ierr);
   if (delete_mat) delete pHH;
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_tao_monitor(Tao tao,void* ctx)
{
   __mfem_monitor_ctx *monctx = (__mfem_monitor_ctx*)ctx;

   PetscFunctionBeginUser;
   if (!monctx)
   {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing monitor context");
   }

   mfem::PetscSolver *solver = (mfem::PetscSolver*)(monctx->solver);
   mfem::PetscSolverMonitor *user_monitor = (mfem::PetscSolverMonitor *)(
                                               monctx->monitor);
   if (user_monitor->mon_sol)
   {
      Vec M;
      ierr = TaoGetSolutionVector(tao,&M);CHKERRQ(ierr);
      mfem::PetscParVector m(M,true);

      PetscInt  it;
      PetscReal res;
      ierr = TaoGetSolutionStatus(tao,&it,NULL,&res,NULL,NULL,NULL); CHKERRQ(ierr);
      user_monitor->MonitorSolution(it,res,m);
   }
   user_monitor->MonitorSolver(solver);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_tao_update(Tao tao, PetscInt it, void* ctx)
{
   Vec F,X,dX,pX;
   __mfem_tao_ctx *tao_ctx = (__mfem_tao_ctx*)ctx;

   PetscFunctionBeginUser;
   ierr = TaoGetSolutionVector(tao,&X); CHKERRQ(ierr);
   ierr = TaoGetGradientVector(tao,&F); CHKERRQ(ierr);
   if (!it)
   {
      ierr = VecDuplicate(X,&pX); CHKERRQ(ierr);
      ierr = VecDuplicate(X,&dX); CHKERRQ(ierr);
      ierr = VecSet(pX,0.0);CHKERRQ(ierr);
      ierr = VecSet(dX,0.0);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)tao,"_mfemopt_tao_xp",(PetscObject)pX); CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)tao,"_mfemopt_tao_dx",(PetscObject)dX); CHKERRQ(ierr);
      ierr = VecDestroy(&pX);CHKERRQ(ierr);
      ierr = VecDestroy(&dX);CHKERRQ(ierr);
   }
   ierr = PetscObjectQuery((PetscObject)tao,"_mfemopt_tao_xp",(PetscObject*)&pX); CHKERRQ(ierr);
   ierr = PetscObjectQuery((PetscObject)tao,"_mfemopt_tao_dx",(PetscObject*)&dX); CHKERRQ(ierr);
   if (!pX) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER,
            "Missing previous solution");

   if (it)
   {
     TaoLineSearch taols;

     ierr = TaoGetLineSearch(tao,&taols); CHKERRQ(ierr);
     if (taols)
     {
        Vec dXt;
        /* Tao update is Xnew = Xold + lambda * dX, SNES update is Xnew = Xold - lambda * dX */
        ierr = TaoLineSearchGetStepDirection(taols,&dXt); CHKERRQ(ierr);
        ierr = VecCopy(dXt,dX); CHKERRQ(ierr);
        ierr = VecScale(dX,-1.0); CHKERRQ(ierr);
     }
     else
     {
        ierr = VecWAXPY(dX,-1.0,X,pX); CHKERRQ(ierr);
     }
   }
   mfem::PetscParVector f(F,true);
   mfem::PetscParVector x(X,true);
   mfem::PetscParVector dx(dX,true);
   mfem::PetscParVector px(pX,true);
   tao_ctx->objective->Update(it,f,x,dx,px);

   /* Store previous solution */
   ierr = VecCopy(X,pX); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_monitor_ctx_destroy(void **ctx)
{
   PetscErrorCode  ierr;

   PetscFunctionBeginUser;
   ierr = PetscFree(*ctx); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

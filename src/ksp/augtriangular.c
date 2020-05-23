#include <petscopt/private/augmentedtsimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscdmcomposite.h>

static PetscErrorCode KSPDestroy_AugTriangular(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetApplicationContext(ksp,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_AugTriangular(KSP aksp)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  KSP            ksp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetApplicationContext(aksp,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_AugTriangular(KSP aksp, PetscViewer viewer)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  KSP            ksp;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Augmented KSP, model KSP\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  }
  ierr = KSPGetApplicationContext(aksp,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPView(ksp,viewer);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_AugTriangular_Private(KSP aksp, Vec x, Vec y, PetscBool trans)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  Mat            A, Asub;
  Mat            P, Psub;
  DM             dm;
  PetscInt       i;
  PetscBool      reuse;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetApplicationContext(aksp,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (!trans && actx->adjoint) SETERRQ(PetscObjectComm((PetscObject)aksp),PETSC_ERR_SUP,"KSPSolve + adjoint not implemented");
  if (trans && !actx->adjoint) SETERRQ(PetscObjectComm((PetscObject)aksp),PETSC_ERR_SUP,"KSPSolveTranspose + !adjoint not implemented");
  ierr = KSPGetOperators(aksp,&A,&P);CHKERRQ(ierr);
  ierr = KSPGetPC(aksp,&pc);CHKERRQ(ierr);
  ierr = PCGetReusePreconditioner(pc,&reuse);CHKERRQ(ierr);
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,x,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,y,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = MatNestGetSubMat(A,0,0,&Asub);CHKERRQ(ierr);
  ierr = MatNestGetSubMat(P,0,0,&Psub);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,Asub,Psub);CHKERRQ(ierr);
  ierr = KSPSetReusePreconditioner(ksp,reuse);CHKERRQ(ierr);
  if (!trans) {
    ierr = KSPSolve(ksp,actx->F[0],actx->U[0]);CHKERRQ(ierr);
  } else {
    ierr = KSPSolveTranspose(ksp,actx->F[0],actx->U[0]);CHKERRQ(ierr);
  }
  ierr = KSPGetIterationNumber(ksp,&aksp->its);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp,&aksp->reason);CHKERRQ(ierr);
  if (aksp->reason <=0) {
    ierr = DMCompositeRestoreAccessArray(dm,x,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccessArray(dm,y,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
    ierr = VecLockReadPop(x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (i=0;i<actx->nqts;i++) {
    Vec d;
    DM  qdm;

    if (!trans) {
      ierr = MatNestGetSubMat(A,i+1,0,&Asub);CHKERRQ(ierr);
    } else {
      ierr = MatNestGetSubMat(A,0,i+1,&Asub);CHKERRQ(ierr);
    }
    if (Asub) {
      if (!trans) {
        ierr = MatMult(Asub,actx->U[0],actx->U[i+1]);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(Asub,actx->U[0],actx->U[i+1]);CHKERRQ(ierr);
      }
      ierr = VecAYPX(actx->U[i+1],-1.,actx->F[i+1]);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(actx->F[i+1],actx->U[i+1]);CHKERRQ(ierr);
    }
    ierr = MatNestGetSubMat(A,i+1,i+1,&Asub);CHKERRQ(ierr);
    ierr = TSGetDM(actx->qts[i],&qdm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(qdm,&d);CHKERRQ(ierr);
    ierr = MatGetDiagonal(Asub,d);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(actx->U[i+1],actx->U[i+1],d);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(qdm,&d);CHKERRQ(ierr);
  }
  ierr = DMCompositeRestoreAccessArray(dm,x,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,y,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_AugTriangular(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSolve_AugTriangular_Private(ksp,ksp->vec_rhs,ksp->vec_sol,ksp->transpose_solve);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode KSPCreate_AugTriangular(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);

  ksp->data                = NULL;
  ksp->ops->solve          = KSPSolve_AugTriangular;
  ksp->ops->view           = KSPView_AugTriangular;
  ksp->ops->setup          = KSPSetUp_AugTriangular;
  ksp->ops->reset          = NULL;
  ksp->ops->destroy        = KSPDestroy_AugTriangular;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}

#include <petscopt/augmentedts.h>
#include <petscopt/petscopt.h>
#include <petscopt/adjointts.h>
#include <petscopt/tlmts.h>
#include <petscopt/tsutils.h>
#include <petscopt/private/augmentedtsimpl.h>
#include <petsc/private/tsimpl.h>
#include <petscdmcomposite.h>
#include <petscdmshell.h>

#include <petsc/private/dmimpl.h>
static PetscErrorCode DMGetLocalToGlobalMapping_Dummy(DM dm)
{
  Vec            g;
  IS             is;
  PetscInt       n,st;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingDestroy(&dm->ltogmap);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&g);CHKERRQ(ierr);
  ierr = VecGetLocalSize(g,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(g,&st,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)dm),n,st,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&dm->ltogmap);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,13,0)
/* the *dd part of the code is wrong */
static PetscErrorCode MatMissingDiagonal_Nest(Mat mat,PetscBool *missing,PetscInt *dd)
{
  Mat            **nest;
  IS             *rows;
  PetscInt       nr,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  ierr = MatNestGetSubMats(mat,&nr,NULL,&nest);CHKERRQ(ierr);
  ierr = PetscMalloc1(nr,&rows);CHKERRQ(ierr);
  ierr = MatNestGetISs(mat,rows,NULL);CHKERRQ(ierr);
  for (i = 0; i < nr && !(*missing); i++) {
    PetscInt ndd;

    if (nest[i][i]) {
      ierr = MatMissingDiagonal(nest[i][i],missing,&ndd);CHKERRQ(ierr);
    } else {
      *missing = PETSC_TRUE;
      ndd = 0;
    }
    if (*missing && dd) {
      const PetscInt *idxs;
      PetscInt       n;

      ierr = ISGetIndices(rows[i],&idxs);CHKERRQ(ierr);
      ierr = ISGetLocalSize(rows[i],&n);CHKERRQ(ierr);
      *dd  = n ? idxs[ndd] : 0;
      ierr = ISRestoreIndices(rows[i],&idxs);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#include <petsc/private/snesimpl.h>

static PetscErrorCode SNESReset_Augmented(SNES asnes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSetApplicationContext(asnes,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_Augmented(PetscOptionItems *PetscOptionsObject,SNES asnes)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(asnes,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    ierr = TSGetSNES(actx->qts[i],&snes);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_Augmented(SNES asnes)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(asnes,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESSetUp(snes);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    ierr = TSGetSNES(actx->qts[i],&snes);CHKERRQ(ierr);
    ierr = SNESSetUp(snes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_Augmented(SNES asnes, PetscViewer viewer)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  PetscBool      isascii;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Augmented SNES, model SNES\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  }
  ierr = SNESGetApplicationContext(asnes,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESView(snes,viewer);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    if (isascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Augmented SNES, dependent TS %D\n",i);CHKERRQ(ierr);
    }
    ierr = TSGetSNES(actx->qts[i],&snes);CHKERRQ(ierr);
    ierr = SNESView(snes,viewer);CHKERRQ(ierr);
  }
  if (isascii) {
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFunction_Segregated(SNES snes, Vec X, Vec B, void *ctx)
{
  TS             ats = (TS)(ctx);
  TSAugCtx       *actx;
  SNES           asnes;
  DM             dm;
  Vec            aU, aF, U, F;
  PetscInt       z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetSNES(ats,&asnes);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  z    = actx->activets;
  if (z < 0 || z > actx->nqts) SETERRQ2(PetscObjectComm((PetscObject)ats),PETSC_ERR_PLIB,"Invalid component %D (should be in [0 ,%D]) ",z,actx->nqts);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&aF);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&aU);CHKERRQ(ierr);
  if (z != 0) { ierr = VecCopy(asnes->vec_sol,aU);CHKERRQ(ierr); } /* assumes the solution of the model has been already computed */
  ierr = DMCompositeGetAccessArray(dm,aU,1,&z,&U);CHKERRQ(ierr);
  ierr = VecCopy(X,U);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,aU,1,&z,&U);CHKERRQ(ierr);
  ierr = SNESComputeFunction(asnes,aU,aF);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,aF,1,&z,&F);CHKERRQ(ierr);
  ierr = VecCopy(F,B);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,aF,1,&z,&F);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&aF);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&aU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESJacobian_Segregated(SNES snes, Vec X, Mat A, Mat B, void *ctx)
{
  TS             ats = (TS)(ctx);
  TSAugCtx       *actx;
  SNES           asnes;
  DM             dm;
  Mat            aA, aB, sA, sB;
  Vec            aU, U;
  PetscInt       z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetSNES(ats,&asnes);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  z    = actx->activets;
  if (z < 0 || z > actx->nqts) SETERRQ2(PetscObjectComm((PetscObject)ats),PETSC_ERR_PLIB,"Invalid component %D (should be in [0 ,%D]) ",z,actx->nqts);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&aU);CHKERRQ(ierr);
  if (z != 0) { ierr = VecCopy(asnes->vec_sol,aU);CHKERRQ(ierr); } /* assumes the solution of the model has been already computed */
  ierr = DMCompositeGetAccessArray(dm,aU,1,&z,&U);CHKERRQ(ierr);
  ierr = VecCopy(X,U);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,aU,1,&z,&U);CHKERRQ(ierr);
  ierr = SNESGetJacobian(asnes,&aA,&aB,NULL,NULL);CHKERRQ(ierr);
  ierr = SNESComputeJacobian(asnes,aU,aA,aB);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&aU);CHKERRQ(ierr);
  ierr = MatNestGetSubMat(aA,z,z,&sA);CHKERRQ(ierr);
  ierr = MatNestGetSubMat(aB,z,z,&sB);CHKERRQ(ierr);
  ierr = MatCopy(sA,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatCopy(sB,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_Augmented(SNES asnes)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  Vec            F,U,aF,aU;
  PetscErrorCode (*of)(SNES,Vec,Vec,void*);
  PetscErrorCode (*oJ)(SNES,Vec,Mat,Mat,void*);
  void           *octx,*oJctx;
  DM             dm;
  PetscInt       i,z=0,ol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(asnes,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugmentedTS(ats);
  aU   = asnes->vec_sol;
  aF   = asnes->vec_func;
  ierr = SNESComputeFunction(asnes,aU,aF);CHKERRQ(ierr);

  asnes->iter = 0;

  ierr = VecNorm(aF,NORM_2,&asnes->norm);CHKERRQ(ierr);
  SNESCheckFunctionNorm(asnes,asnes->norm);
  ierr = PetscObjectSAWsTakeAccess((PetscObject)asnes);CHKERRQ(ierr);
  ierr = PetscObjectSAWsGrantAccess((PetscObject)asnes);CHKERRQ(ierr);
  ierr = SNESLogConvergenceHistory(asnes,asnes->norm,0);CHKERRQ(ierr);
  ierr = SNESMonitor(asnes,asnes->iter,asnes->norm);CHKERRQ(ierr);

  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,aF,1,&z,&F);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,aU,1,&z,&U);CHKERRQ(ierr);
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,NULL,&of,&octx);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,NULL,NULL,&oJ,&oJctx);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,NULL,SNESFunction_Segregated,ats);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,NULL,NULL,SNESJacobian_Segregated,ats);CHKERRQ(ierr);
  ol   = ((PetscObject)snes)->tablevel;
  ((PetscObject)snes)->tablevel = ((PetscObject)asnes)->tablevel + 1;
  actx->activets = 0;
  ierr = SNESSetInitialFunction(snes,F);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,U);CHKERRQ(ierr);
  actx->activets = -1;
  ((PetscObject)snes)->tablevel = ol; 
  ierr = SNESSetFunction(snes,NULL,of,octx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,NULL,NULL,oJ,oJctx);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,aU,1,&z,&U);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,aF,1,&z,&F);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&asnes->reason);CHKERRQ(ierr);
  if (asnes->reason<=0) PetscFunctionReturn(0);

  asnes->iter++;

  for (i=0;i<actx->nqts;i++) {
    z = i+1;

    ierr = SNESComputeFunction(asnes,aU,aF);CHKERRQ(ierr); 
    ierr = VecNorm(aF,NORM_2,&asnes->norm);CHKERRQ(ierr);
    SNESCheckFunctionNorm(asnes,asnes->norm);
    ierr = PetscObjectSAWsTakeAccess((PetscObject)asnes);CHKERRQ(ierr);
    ierr = PetscObjectSAWsGrantAccess((PetscObject)asnes);CHKERRQ(ierr);
    ierr = SNESLogConvergenceHistory(asnes,asnes->norm,asnes->iter);CHKERRQ(ierr);
    ierr = SNESMonitor(asnes,asnes->iter,asnes->norm);CHKERRQ(ierr);

    ierr = DMCompositeGetAccessArray(dm,aU,1,&z,&U);CHKERRQ(ierr);
    ierr = DMCompositeGetAccessArray(dm,aF,1,&z,&F);CHKERRQ(ierr);
    ierr = TSGetSNES(actx->qts[i],&snes);CHKERRQ(ierr);
    ierr = SNESGetFunction(snes,NULL,&of,&octx);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,NULL,NULL,&oJ,&oJctx);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,NULL,SNESFunction_Segregated,ats);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,NULL,NULL,SNESJacobian_Segregated,ats);CHKERRQ(ierr);
    ol   = ((PetscObject)snes)->tablevel;
    ((PetscObject)snes)->tablevel = ((PetscObject)asnes)->tablevel + 1;
    actx->activets = z;
    ierr = SNESSetInitialFunction(snes,F);CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,U);CHKERRQ(ierr);
    actx->activets = -1;
    ((PetscObject)snes)->tablevel = ol; 
    ierr = SNESSetFunction(snes,NULL,of,octx);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,NULL,NULL,oJ,oJctx);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccessArray(dm,aF,1,&z,&F);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccessArray(dm,aU,1,&z,&U);CHKERRQ(ierr);
    if (snes->reason < 0) { asnes->reason = SNES_DIVERGED_INNER; break; }
    asnes->iter++;
  }
  ierr = SNESComputeFunction(asnes,aU,aF);CHKERRQ(ierr); 
  ierr = VecNorm(aF,NORM_2,&asnes->norm);CHKERRQ(ierr);
  SNESCheckFunctionNorm(asnes,asnes->norm);
  ierr = PetscObjectSAWsTakeAccess((PetscObject)asnes);CHKERRQ(ierr);
  ierr = PetscObjectSAWsGrantAccess((PetscObject)asnes);CHKERRQ(ierr);
  ierr = SNESLogConvergenceHistory(asnes,asnes->norm,asnes->iter);CHKERRQ(ierr);
  ierr = SNESMonitor(asnes,asnes->iter,asnes->norm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESCreate_Augmented(SNES snes)
{
  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_Augmented;
  snes->ops->solve          = SNESSolve_Augmented;
  snes->ops->setfromoptions = SNESSetFromOptions_Augmented;
  snes->ops->view           = SNESView_Augmented;
  snes->ops->reset          = SNESReset_Augmented;

  snes->npcside = PC_RIGHT;
  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;
  snes->alwayscomputesfinalresidual = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#include <petsc/private/kspimpl.h>

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

PetscErrorCode KSPCreate_AugTriangular(KSP ksp)
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

PetscErrorCode AdjointTSGetModelTS_Aug(TS ats, TS *fwdts)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = AdjointTSGetModelTS(actx->model,fwdts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AdjointTSGetTLMTSAndFOATS_Aug(TS ats, TS *tlmts, TS *foats)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = AdjointTSGetTLMTSAndFOATS(actx->model,tlmts,foats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AdjointTSGetDirectionVec_Aug(TS ats, Vec *d)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = AdjointTSGetDirectionVec(actx->model,d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* input vectors are defined on the model only */
/* forcing term, by convention, assumes the implicit interface */
PetscErrorCode AdjointTSComputeForcing_Aug(TS ats, PetscReal time, Vec U, Vec Udot, Vec FOAL, Vec FOALdot, Vec lU, Vec lUdot, PetscBool* hasf, Vec F)
{
  TSAugCtx       *actx;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,F,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  ierr = AdjointTSComputeForcing(actx->model,time,U,Udot,FOAL,FOALdot,lU,lUdot,hasf,actx->F[0]);CHKERRQ(ierr);
  if (*hasf) {
    PetscInt i;

    for (i=0;i<actx->nqts;i++) {
      ierr = VecSet(actx->F[i+1],0.0);CHKERRQ(ierr);
    }
  }
  ierr = DMCompositeRestoreAccessArray(dm,F,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* U,FOA,TLM on the state only, aL on state and quadrature */
PetscErrorCode AdjointTSComputeQuadrature_Aug(TS ats, PetscReal time, Vec U, Vec Udot, Vec aL, Vec FOAL, Vec FOALdot, Vec TLMU, Vec TLMUdot, PetscBool *has, Vec F)
{
  TSAugCtx       *actx;
  DM             dm,qdm;
  Vec            dummy,Q,L;
  PetscInt       s = 0, q = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (actx->nqts != 1) SETERRQ1(PetscObjectComm((PetscObject)ats),PETSC_ERR_PLIB,"Unexpected number of quadratures %D != 1",actx->nqts);
  if (!actx->updatestates[0]) SETERRQ(PetscObjectComm((PetscObject)ats),PETSC_ERR_PLIB,"Missing update function");
  *has = PETSC_TRUE;
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,aL,1,&s,&L);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,F,1,&q,&Q);CHKERRQ(ierr);
  ierr = (*actx->updatestates[0])(actx->qts[0],L,NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_U",(PetscObject)U);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_Udot",(PetscObject)Udot);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_FOAL",(PetscObject)FOAL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_FOALdot",(PetscObject)FOALdot);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_TLMU",(PetscObject)TLMU);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_TLMUdot",(PetscObject)TLMUdot);CHKERRQ(ierr);
  ierr = TSGetDM(actx->qts[0],&qdm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(qdm,&dummy);CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(actx->qts[0],time,dummy,Q);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(qdm,&dummy);CHKERRQ(ierr);
  ierr = (*actx->updatestates[0])(actx->qts[0],NULL,NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_U",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_Udot",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_FOAL",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_FOALdot",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_TLMU",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)L,"_ts_adjoint_discrete_TLMUdot",NULL);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,aL,1,&s,&L);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,F,1,&q,&Q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TLMTSGetModelTS_Aug(TS ats, TS *fwdts)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TLMTSGetModelTS(actx->model,fwdts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* U and Udot is on model only */
PetscErrorCode TLMTSComputeForcing_Aug(TS ats, PetscReal time, Vec U, Vec Udot, PetscBool* hasf, Vec F)
{
  TSAugCtx       *actx;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,F,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  ierr = TLMTSComputeForcing(actx->model,time,U,Udot,hasf,actx->F[0]);CHKERRQ(ierr);
  if (*hasf) {
    PetscInt i;

    for (i=0;i<actx->nqts;i++) {
      ierr = VecSet(actx->F[i+1],0.0);CHKERRQ(ierr);
    }
  }
  ierr = DMCompositeRestoreAccessArray(dm,F,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSSetUp(TS ats)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSSetUp(actx->model);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    ierr = TSSetUp(actx->qts[i]);CHKERRQ(ierr);
  }
  if (actx->setup) { ierr = (*actx->setup)(ats);CHKERRQ(ierr); }
  ats->ops->step = actx->model->ops->step;
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSTrajectorySetUp(TSTrajectory tj,TS ats)
{
  TSAugCtx       *actx;
  TSTrajectory   mtj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetTrajectory(actx->model,&mtj);CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(mtj,actx->model);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSTrajectorySet(TSTrajectory tj,TS ats,PetscInt stepnum,PetscReal time,Vec X)
{
  TSAugCtx         *actx;
  TSTrajectory     mtj;
  DM               dm;
  PetscInt         z=0;
  PetscObjectState st;
  PetscBool        flg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = TSGetTrajectory(actx->model,&mtj);CHKERRQ(ierr);
  ierr = TSTrajectoryGetSolutionOnly(mtj,&flg);CHKERRQ(ierr);
  if (!flg) {
    Vec      *Y,*aY;
    PetscInt i,ns,ans;

    ierr = TSGetStages(ats,&ans,&aY);CHKERRQ(ierr);
    ierr = TSGetStages(actx->model,&ns,&Y);CHKERRQ(ierr);
    if (ns != ans) SETERRQ2(PetscObjectComm((PetscObject)ats),PETSC_ERR_SUP,"Mismatch stages %D != %D\n",ans,ns);
    for (i=0;i<ns;i++) {

      ierr = PetscObjectStateGet((PetscObject)aY[i],&st);CHKERRQ(ierr);
      ierr = DMCompositeGetAccessArray(dm,aY[i],1,&z,actx->U);CHKERRQ(ierr);
      ierr = VecCopy(actx->U[0],Y[i]);CHKERRQ(ierr);
      ierr = DMCompositeRestoreAccessArray(dm,aY[i],1,&z,actx->U);CHKERRQ(ierr);
      ierr = PetscObjectStateSet((PetscObject)aY[i],st);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectStateGet((PetscObject)X,&st);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,X,1,&z,actx->U);CHKERRQ(ierr);
  ierr = TSTrajectorySet(mtj,actx->model,stepnum,time,actx->U[0]);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,X,1,&z,actx->U);CHKERRQ(ierr);
  ierr = PetscObjectStateSet((PetscObject)X,st);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSUpdateSolution(TS ats)
{
  TSAugCtx       *actx;
  DM             dm;
  Vec            U,mU;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = TSGetSolution(ats,&U);CHKERRQ(ierr);
  ierr = VecLockReadPush(U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
  ierr = VecCopy(actx->U[0],mU);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    ierr = TSGetSolution(actx->qts[i],&mU);CHKERRQ(ierr);
    ierr = VecCopy(actx->U[i+1],mU);CHKERRQ(ierr);
  }
  ierr = DMCompositeRestoreAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = VecLockReadPop(U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AugmentedTSUpdateModelSolution(TS ats)
{
  TSAugCtx         *actx;
  DM               dm;
  Vec              U,mU;
  PetscInt         z=0;
  PetscObjectState st;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = TSGetSolution(ats,&U);CHKERRQ(ierr);
  ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)U,&st);CHKERRQ(ierr);
  ierr = VecLockReadPush(U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U,1,&z,actx->U);CHKERRQ(ierr);
  ierr = VecCopy(actx->U[0],mU);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,U,1,&z,actx->U);CHKERRQ(ierr);
  ierr = VecLockReadPop(U);CHKERRQ(ierr);
  ierr = PetscObjectStateSet((PetscObject)U,st);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSUpdateModelTS(TS ats)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  actx->model->ptime               = ats->ptime;
  actx->model->time_step           = ats->time_step;
  actx->model->ptime_prev          = ats->ptime_prev;
  actx->model->ptime_prev_rollback = ats->ptime_prev_rollback;
  actx->model->steps               = ats->steps;
  actx->model->steprollback        = ats->steprollback;
  actx->model->steprestart         = ats->steprestart;
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSPostStep(TS ats)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = AugmentedTSUpdateModelTS(ats);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (actx->model->poststep) {
    Vec              mU;
    PetscObjectState sprev,spost;

    ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
    ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)mU,&sprev);CHKERRQ(ierr);
    PetscStackCallStandard((*actx->model->poststep),(actx->model));
    ierr = PetscObjectStateGet((PetscObject)mU,&spost);CHKERRQ(ierr);
    if (sprev != spost) {
      DM       dm;
      Vec      U;
      PetscInt z=0;

      ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
      ierr = TSGetSolution(ats,&U);CHKERRQ(ierr);
      ierr = DMCompositeGetAccessArray(dm,U,1,&z,actx->U);CHKERRQ(ierr);
      ierr = VecCopy(mU,actx->U[0]);CHKERRQ(ierr);
      ierr = DMCompositeRestoreAccessArray(dm,U,1,&z,actx->U);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSPreStep(TS ats)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = AugmentedTSUpdateModelTS(ats);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (actx->model->prestep) {
    Vec               mU;
    PetscObjectState  sprev,spost;
    TSConvergedReason reason;

    ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
    ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)mU,&sprev);CHKERRQ(ierr);
    PetscStackCallStandard((*actx->model->prestep),(actx->model));
    ierr = PetscObjectStateGet((PetscObject)mU,&spost);CHKERRQ(ierr);
    if (sprev != spost) {
      DM       dm;
      Vec      U;
      PetscInt z=0;

      ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
      ierr = TSGetSolution(ats,&U);CHKERRQ(ierr);
      ierr = DMCompositeGetAccessArray(dm,U,1,&z,actx->U);CHKERRQ(ierr);
      ierr = VecCopy(mU,actx->U[0]);CHKERRQ(ierr);
      ierr = DMCompositeRestoreAccessArray(dm,U,1,&z,actx->U);CHKERRQ(ierr);
    }
    ierr = TSGetConvergedReason(actx->model,&reason);CHKERRQ(ierr);
    ierr = TSSetConvergedReason(ats,reason);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSPostEvaluate(TS ats)
{
  TSAugCtx*      actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (actx->model->postevaluate) {
    Vec               mU;
    PetscObjectState  sprev,spost;
    TSConvergedReason reason;

    ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
    ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)mU,&sprev);CHKERRQ(ierr);
    PetscStackCallStandard((*actx->model->postevaluate),(actx->model));
    ierr = PetscObjectStateGet((PetscObject)mU,&spost);CHKERRQ(ierr);
    if (sprev != spost) {
      ierr = TSRestartStep(ats);CHKERRQ(ierr);
    }
    ierr = TSGetConvergedReason(actx->model,&reason);CHKERRQ(ierr);
    ierr = TSSetConvergedReason(ats,reason);CHKERRQ(ierr);
  }
  ierr = AugmentedTSUpdateModelTS(ats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSEventHandler(TS ats,PetscReal t,Vec U,PetscScalar fvalue[],void* ctx)
{
  TSAugCtx*      actx;
  TSEvent        mevent;
  Vec            mU;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  mevent = actx->model->event;
  if (!mevent || !mevent->eventhandler) PetscFunctionReturn(0);
  ierr = AugmentedTSUpdateModelTS(ats);CHKERRQ(ierr);
  ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
  ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
  ierr = (*mevent->eventhandler)(actx->model,t,mU,mevent->fvalue,mevent->ctx);CHKERRQ(ierr);
  for (i=0;i<mevent->nevents;i++) fvalue[i] = mevent->fvalue[i];
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSPostEvent(TS ats, PetscInt nevents_zero, PetscInt events_zero[], PetscReal t, Vec U, PetscBool forwardsolve, void* ctx)
{
  TSAugCtx*      actx;
  TSEvent        mevent;
  Vec            mU;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  mevent = actx->model->event;
  if (!mevent || !mevent->eventhandler) PetscFunctionReturn(0);
  /* No need to update the model TS nor its solution
     since this is always called after an eventhandler */
  ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
  ierr = (*mevent->postevent)(actx->model,nevents_zero,events_zero,t,mU,forwardsolve,mevent->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSDestroy_Private(void *ptr)
{
  TSAugCtx*      aug = (TSAugCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = TSDestroy(&aug->model);CHKERRQ(ierr);
  for (i=0;i<aug->nqts;i++) {
    ierr = TSDestroy(&aug->qts[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(aug->qts,aug->updatestates,aug->jaccoupling);CHKERRQ(ierr);
  ierr = PetscFree3(aug->U,aug->Udot,aug->F);CHKERRQ(ierr);
  ierr = PetscFree(aug);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSOptionsHandler(PetscOptionItems *PetscOptionsObject,PetscObject obj,PETSC_UNUSED void *ctx)
{
  TS             ats = (TS)obj;
  TSAugCtx       *actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Augmented TS options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSRHSFunction(TS ats, PetscReal time, Vec U, Vec F, void *ctx)
{
  PetscErrorCode ierr;
  TSAugCtx       *actx;
  DM             dm;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,F,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  if (actx->activets <=0) {
    ierr = TSComputeRHSFunction(actx->model,time,actx->U[0],actx->F[0]);CHKERRQ(ierr);
  }
  for (i=0;i<actx->nqts;i++) {
    if (actx->activets > -1 && actx->activets != i+1) continue;
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],actx->U[0],NULL);CHKERRQ(ierr); }
    ierr = TSComputeRHSFunction(actx->qts[i],time,actx->U[i+1],actx->F[i+1]);CHKERRQ(ierr);
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],NULL,NULL);CHKERRQ(ierr); }
  }
  ierr = DMCompositeRestoreAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,F,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* XXX update mat status? */
static PetscErrorCode AugmentedTSRHSJacobian(TS ats, PetscReal time, Vec U, Mat A, Mat P, void *ctx)
{
  PetscErrorCode ierr;
  TSAugCtx       *actx;
  DM             dm;
  Mat            Asub,Psub;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  if (actx->activets <=0) {
    ierr = MatNestGetSubMat(A,0,0,&Asub);CHKERRQ(ierr);
    ierr = MatNestGetSubMat(P,0,0,&Psub);CHKERRQ(ierr);
    ierr = TSComputeRHSJacobian(actx->model,time,actx->U[0],Asub,Psub);CHKERRQ(ierr);
  }
  for (i=0;i<actx->nqts;i++) {
    /* always recompute */
    /* if (actx->activets > -1 && actx->activets != i+1) continue; */
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],actx->U[0],NULL);CHKERRQ(ierr); }
    if (actx->jaccoupling[i] && actx->activets < 0) {
      PetscInt r = actx->adjoint ? 0 : i+1;
      PetscInt c = actx->adjoint ? i+1 : 0;

      ierr = MatNestGetSubMat(A,r,c,&Asub);CHKERRQ(ierr);
      ierr = MatNestGetSubMat(P,r,c,&Psub);CHKERRQ(ierr);
      ierr = (*actx->jaccoupling[i])(actx->qts[i],time,actx->U[i+1],NULL,0.0,Asub,Psub,actx);CHKERRQ(ierr);
    }
    ierr = MatNestGetSubMat(A,i+1,i+1,&Asub);CHKERRQ(ierr);
    ierr = MatNestGetSubMat(P,i+1,i+1,&Psub);CHKERRQ(ierr);
    ierr = TSComputeRHSJacobian(actx->qts[i],time,actx->U[i+1],Asub,Psub);CHKERRQ(ierr);
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],NULL,NULL);CHKERRQ(ierr); }
  }
  ierr = DMCompositeRestoreAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSIFunction(TS ats, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  PetscErrorCode ierr;
  TSAugCtx       *actx;
  DM             dm;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U   ,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,Udot,actx->nqts + 1,NULL,actx->Udot);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,F   ,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  if (actx->activets <=0) {
    ierr = TSComputeIFunction(actx->model,time,actx->U[0],actx->Udot[0],actx->F[0],PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=0;i<actx->nqts;i++) {
    if (actx->activets > -1 && actx->activets != i+1) continue;
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],actx->U[0],actx->Udot[0]);CHKERRQ(ierr); }
    ierr = TSComputeIFunction(actx->qts[i],time,actx->U[i+1],actx->Udot[i+1],actx->F[i+1],PETSC_FALSE);CHKERRQ(ierr);
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],NULL,NULL);CHKERRQ(ierr); }
  }
  ierr = DMCompositeRestoreAccessArray(dm,U   ,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,Udot,actx->nqts + 1,NULL,actx->Udot);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,F   ,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSIJacobian(TS ats, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat P, void *ctx)
{
  PetscErrorCode ierr;
  TSAugCtx       *actx;
  DM             dm;
  Mat            Asub,Psub;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U   ,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,Udot,actx->nqts + 1,NULL,actx->Udot);CHKERRQ(ierr);
  if (actx->activets <=0) {
    ierr = MatNestGetSubMat(A,0,0,&Asub);CHKERRQ(ierr);
    ierr = MatNestGetSubMat(P,0,0,&Psub);CHKERRQ(ierr);
    ierr = TSComputeIJacobian(actx->model,time,actx->U[0],actx->Udot[0],shift,Asub,Psub,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=0;i<actx->nqts;i++) {
    /* always recompute */
    /* if (actx->activets > -1 && actx->activets != i+1) continue; */
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],actx->U[0],actx->Udot[0]);CHKERRQ(ierr); }
    if (actx->jaccoupling[i] && actx->activets < 0) {
      PetscInt r = actx->adjoint ? 0 : i+1;
      PetscInt c = actx->adjoint ? i+1 : 0;

      ierr = MatNestGetSubMat(A,r,c,&Asub);CHKERRQ(ierr);
      ierr = MatNestGetSubMat(P,r,c,&Psub);CHKERRQ(ierr);
      ierr = (*actx->jaccoupling[i])(actx->qts[i],time,actx->U[i+1],actx->Udot[i+1],shift,Asub,Psub,actx);CHKERRQ(ierr);
    }
    ierr = MatNestGetSubMat(A,i+1,i+1,&Asub);CHKERRQ(ierr);
    ierr = MatNestGetSubMat(P,i+1,i+1,&Psub);CHKERRQ(ierr);
    ierr = TSComputeIJacobian(actx->qts[i],time,actx->U[i+1],actx->Udot[i+1],shift,Asub,Psub,PETSC_FALSE);CHKERRQ(ierr);
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],NULL,NULL);CHKERRQ(ierr); }
  }
  ierr = DMCompositeRestoreAccessArray(dm,U   ,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,Udot,actx->nqts + 1,NULL,actx->Udot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSMonitor(TS ats, PetscInt s, PetscReal t, Vec U, void* ctx)
{
  TSAugCtx       *actx;
  PetscErrorCode ierr;
  Vec            mU;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
  ierr = TSMonitor(actx->model,s,t,mU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AugmentedTSInitialize(TS ats)
{
  TSAugCtx               *actx;
  DM                     dm;
  Vec                    U,X;
  PetscReal              t0,dt,tf;
  PetscInt               i,st;
  PetscErrorCode         ierr;
  TSExactFinalTimeOption eftopt;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetMaxTime(actx->model,&tf);CHKERRQ(ierr);
  ierr = TSGetMaxSteps(actx->model,&st);CHKERRQ(ierr);
  ierr = TSGetTime(actx->model,&t0);CHKERRQ(ierr);
  ierr = TSGetTimeStep(actx->model,&dt);CHKERRQ(ierr);
  ierr = TSGetExactFinalTime(actx->model,&eftopt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ats,tf);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ats,st);CHKERRQ(ierr);
  ierr = TSSetTime(ats,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ats,dt);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ats,eftopt);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ats,0);CHKERRQ(ierr);

  /* init solution vector */
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = TSGetSolution(ats,&U);CHKERRQ(ierr);
  if (!U) {
    ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
    ierr = TSSetSolution(ats,U);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)U);CHKERRQ(ierr);
  }
  ierr = DMCompositeGetAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = TSGetSolution(actx->model,&X);CHKERRQ(ierr);
  ierr = VecCopy(X,actx->U[0]);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    ierr = TSGetSolution(actx->qts[i],&X);CHKERRQ(ierr);
    ierr = VecCopy(X,actx->U[i+1]);CHKERRQ(ierr);
  }
  ierr = DMCompositeRestoreAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);

  /* wrap TSTrajectory */
  ierr = TSTrajectoryDestroy(&ats->trajectory);CHKERRQ(ierr);
  if (actx->model->trajectory) {
    PetscBool flg;

    ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ats),&ats->trajectory);CHKERRQ(ierr);
    ierr = TSTrajectorySetType(ats->trajectory,ats,((PetscObject)actx->model->trajectory)->type_name);CHKERRQ(ierr);
    ierr = TSTrajectoryGetSolutionOnly(actx->model->trajectory,&flg);CHKERRQ(ierr);
    ierr = TSTrajectorySetSolutionOnly(ats->trajectory,flg);CHKERRQ(ierr);
    ats->trajectory->adjoint_solve_mode = PETSC_FALSE;
    ats->trajectory->ops->set = AugmentedTSTrajectorySet;
    ats->trajectory->ops->setup = AugmentedTSTrajectorySetUp;
  }

  /* wrap the event handler */
  ierr = TSEventDestroy(&ats->event);CHKERRQ(ierr);
  if (actx->model->event) {
    TSEvent mevent = actx->model->event;

    ierr = TSSetEventHandler(ats,mevent->nevents,mevent->direction,mevent->terminate,AugmentedTSEventHandler,AugmentedTSPostEvent,NULL);CHKERRQ(ierr);
  }

  /* */
  actx->activets = -1;
  if (actx->model->snes) {
    SNES     snes;
    PetscInt lag;

    ierr = SNESGetLagPreconditioner(actx->model->snes,&lag);CHKERRQ(ierr);
    if (lag < 0) lag = -2;
    ierr = TSGetSNES(ats,&snes);CHKERRQ(ierr);
    ierr = SNESSetLagPreconditioner(snes,lag);CHKERRQ(ierr);
  }

  /* compose callbacks for discrete adjoint */
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSComputeForcing_C",AdjointTSComputeForcing_Aug);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSGetModelTS_C",AdjointTSGetModelTS_Aug);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSGetTLMTSAndFOATS_C",AdjointTSGetTLMTSAndFOATS_Aug);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSGetDirectionVec_C",AdjointTSGetDirectionVec_Aug);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSComputeQuadrature_C",AdjointTSComputeQuadrature_Aug);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"TLMTSComputeForcing_C",TLMTSComputeForcing_Aug);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"TLMTSGetModelTS_C",TLMTSGetModelTS_Aug);CHKERRQ(ierr);

  if (actx->model->adapt) {
    PetscBool flg;

    ierr = PetscObjectTypeCompare((PetscObject)actx->model->adapt,TSADAPTHISTORY,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscObjectReference((PetscObject)actx->model->adapt);CHKERRQ(ierr);
      ierr = TSAdaptDestroy(&ats->adapt);CHKERRQ(ierr);
      ats->adapt = actx->model->adapt;
    }
  }

  /* mimick initialization in TSSolve for model TS */
  actx->model->ksp_its           = 0;
  actx->model->snes_its          = 0;
  actx->model->num_snes_failures = 0;
  actx->model->reject            = 0;
  actx->model->steprestart       = PETSC_TRUE;
  actx->model->steprollback      = PETSC_FALSE;
  actx->model->rhsjacobian.time  = PETSC_MIN_REAL;
  for (i=0;i<actx->nqts;i++) {
    actx->qts[i]->rhsjacobian.time = PETSC_MIN_REAL;
  }
  ierr = TSSetUp(actx->model);CHKERRQ(ierr);
  ierr = TSEventInitialize(actx->model->event,actx->model,t0,actx->model->vec_sol);CHKERRQ(ierr);

  if (actx->model->max_snes_failures != 1) { /* one is the default */
    ierr = TSSetMaxSNESFailures(ats,actx->model->max_snes_failures);CHKERRQ(ierr);
  } else { /* we always attempt to recovery */
    ierr = TSSetMaxSNESFailures(ats,-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AugmentedTSFinalize(TS ats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugmentedTS(ats);
  ierr = AugmentedTSUpdateModelTS(ats);CHKERRQ(ierr);
  ierr = AugmentedTSUpdateSolution(ats);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&ats->trajectory);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSComputeForcing_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSGetModelTS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSGetTLMTSAndFOATS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSGetDirectionVec_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"AdjointTSComputeQuadrature_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"TLMTSComputeForcing_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ats,"TLMTSGetModelTS_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSCreateAugmentedTS(TS ts, PetscInt n, TS qts[], PetscBool qactive[], PetscErrorCode (*updatestates[])(TS,Vec,Vec), TSIJacobian jaccoupling[], Mat Acoupling[], Mat Bcoupling[], PetscBool adjoint, TS* ats)
{
  Mat            A,B;
  DM             dm,adm;
  Vec            vatol,vrtol,va,vr;
  PetscContainer container;
  TSAugCtx       *aug_ctx;
  TSIFunction    ifunc;
  TSRHSFunction  rhsfunc;
  TSI2Function   i2func;
  TSProblemType  prtype;
  const char     *prefix;
  PetscReal      atol,rtol;
  PetscBool      linear;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,n,2);
  if (n) PetscValidPointer(qts,3);
  for (i=0;i<n;i++) {
    PetscValidHeaderSpecific(qts[i],TS_CLASSID,3);
  }
  if (qactive) {
    for (i=0;i<n;i++) {
      PetscValidLogicalCollectiveBool(ts,qactive[i],4);
    }
  }
  if (Acoupling) {
    for (i=0;i<n;i++) {
      if (!Acoupling[i]) continue;
      PetscValidHeaderSpecific(Acoupling[i],MAT_CLASSID,7);
    }
  }
  if (Bcoupling) {
    for (i=0;i<n;i++) {
      if (!Bcoupling[i]) continue;
      PetscValidHeaderSpecific(Bcoupling[i],MAT_CLASSID,8);
    }
  }
  PetscValidLogicalCollectiveBool(ts,adjoint,9);
  PetscValidPointer(ats,10);
  ierr = TSGetI2Function(ts,NULL,&i2func,NULL);CHKERRQ(ierr);
  if (i2func) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Second order DAEs are not supported");
  ierr = TSCreateWithTS(ts,ats);CHKERRQ(ierr);
  if (ts->adapt) {
    TSAdaptType adtype;
    TSAdapt     adapt;

    ierr = TSAdaptGetType(ts->adapt,&adtype);CHKERRQ(ierr);
    ierr = TSGetAdapt(*ats,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType(adapt,adtype);CHKERRQ(ierr);
  }

  ierr = DMCreate(PetscObjectComm((PetscObject)*ats),&adm);CHKERRQ(ierr);
  ierr = DMSetType(adm,DMCOMPOSITE);CHKERRQ(ierr);
  ierr = TSSetDM(*ats,adm);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)adm);CHKERRQ(ierr);

  ierr = PetscNew(&aug_ctx);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*ats,(void *)aug_ctx);CHKERRQ(ierr);
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(*ats)),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,aug_ctx);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,AugmentedTSDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*ats),"_ts_aug_ctx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  aug_ctx->model = ts;
  aug_ctx->setup = (*ats)->ops->setup;
  (*ats)->ops->setup = AugmentedTSSetUp;

  /* Augmented TS prefix: i.e. options called as -augmented_ts_monitor or -augmented_modelprefix_ts_monitor */
  ierr = TSGetOptionsPrefix(aug_ctx->model,&prefix);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(*ats,"augmented_");CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*ats,prefix);CHKERRQ(ierr);

  ierr = TSGetDM(aug_ctx->model,&dm);CHKERRQ(ierr);
  /* XXX HACK */
  {
    PetscBool isshell;
    ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
    if (isshell && !dm->ops->getlocaltoglobalmapping) {
      Vec      g,l;
      PetscInt n;

      ierr = DMGetGlobalVector(dm,&g);CHKERRQ(ierr);
      ierr = VecGetLocalSize(g,&n);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,n,&l);CHKERRQ(ierr);
      ierr = DMShellSetLocalVector(dm,l);CHKERRQ(ierr);
      ierr = VecDestroy(&l);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&g);CHKERRQ(ierr);
      dm->ops->getlocaltoglobalmapping = DMGetLocalToGlobalMapping_Dummy;
    }
  }
  ierr = DMCompositeAddDM(adm,dm);CHKERRQ(ierr);

  ierr = TSGetProblemType(aug_ctx->model,&prtype);CHKERRQ(ierr);
  linear = (prtype == TS_LINEAR ? PETSC_TRUE : PETSC_FALSE);

  aug_ctx->nqts = n;
  ierr = PetscMalloc3(aug_ctx->nqts,&aug_ctx->qts,aug_ctx->nqts,&aug_ctx->updatestates,aug_ctx->nqts,&aug_ctx->jaccoupling);CHKERRQ(ierr);
  for (i=0;i<aug_ctx->nqts;i++) {
    PetscBool has = PETSC_FALSE;

    if (Acoupling && Acoupling[i] && Bcoupling && Bcoupling[i]) has = PETSC_TRUE;
    ierr = PetscObjectReference((PetscObject)qts[i]);CHKERRQ(ierr);
    aug_ctx->qts[i] = qts[i];
    ierr = TSGetDM(aug_ctx->qts[i],&dm);CHKERRQ(ierr);
    ierr = DMCompositeAddDM(adm,dm);CHKERRQ(ierr);
    aug_ctx->updatestates[i] = updatestates ? updatestates[i] : NULL;
    aug_ctx->jaccoupling[i]  = (jaccoupling && has) ? jaccoupling[i] : NULL;
    ierr = TSGetProblemType(aug_ctx->qts[i],&prtype);CHKERRQ(ierr);
    linear = (PetscBool) (linear && (prtype == TS_LINEAR ? PETSC_TRUE : PETSC_FALSE));
  }
  if (linear) {
    ierr = TSSetProblemType(*ats,TS_LINEAR);CHKERRQ(ierr);
  }
  ierr = DMSetMatType(adm,MATNEST);CHKERRQ(ierr);
  ierr = DMSetUp(adm);CHKERRQ(ierr);

  ierr = PetscMalloc3(aug_ctx->nqts+1,&aug_ctx->U,aug_ctx->nqts+1,&aug_ctx->Udot,aug_ctx->nqts+1,&aug_ctx->F);CHKERRQ(ierr);
  ierr = TSGetTolerances(aug_ctx->model,&atol,&va,&rtol,&vr);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(adm,&vatol);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(adm,&vrtol);CHKERRQ(ierr);

  /* not contributing to the wlte */
  ierr = VecSet(vatol,-1.0);CHKERRQ(ierr);
  ierr = VecSet(vrtol,-1.0);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(adm,vatol,aug_ctx->nqts + 1,NULL,aug_ctx->U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(adm,vrtol,aug_ctx->nqts + 1,NULL,aug_ctx->F);CHKERRQ(ierr);
  if (va) { ierr = VecCopy(va,aug_ctx->U[0]);CHKERRQ(ierr); }
  else    { ierr = VecSet(aug_ctx->U[0],atol);CHKERRQ(ierr); }
  if (vr) { ierr = VecCopy(vr,aug_ctx->F[0]);CHKERRQ(ierr); }
  else    { ierr = VecSet(aug_ctx->F[0],rtol);CHKERRQ(ierr); }
  for (i=0;i<aug_ctx->nqts;i++) {
    PetscReal at,rt;
    PetscBool active = qactive ? qactive[i] : PETSC_FALSE;

    if (!active) continue;
    ierr = TSGetTolerances(aug_ctx->qts[i],&at,&va,&rt,&vr);CHKERRQ(ierr);
    if (va) { ierr = VecCopy(va,aug_ctx->U[i+1]);CHKERRQ(ierr); }
    else    { ierr = VecSet(aug_ctx->U[i+1],at);CHKERRQ(ierr); }
    if (vr) { ierr = VecCopy(vr,aug_ctx->F[i+1]);CHKERRQ(ierr); }
    else    { ierr = VecSet(aug_ctx->F[i+1],rt);CHKERRQ(ierr); }
    atol = PetscMin(atol,at);CHKERRQ(ierr);
    rtol = PetscMin(rtol,rt);CHKERRQ(ierr);
  }
  ierr = DMCompositeRestoreAccessArray(adm,vatol,aug_ctx->nqts + 1,NULL,aug_ctx->U);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(adm,vrtol,aug_ctx->nqts + 1,NULL,aug_ctx->F);CHKERRQ(ierr);
  ierr = TSSetTolerances(*ats,atol,vatol,rtol,vrtol);CHKERRQ(ierr);
  ierr = VecDestroy(&vatol);CHKERRQ(ierr);
  ierr = VecDestroy(&vrtol);CHKERRQ(ierr);

  /* setup callbacks */
  aug_ctx->adjoint = adjoint;
  ierr = TSGetIFunction(aug_ctx->model,NULL,&ifunc,NULL);CHKERRQ(ierr);
  ierr = TSGetRHSFunction(aug_ctx->model,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (ifunc) {
    Mat subA,subB;

    ierr = TSGetIJacobian(aug_ctx->model,&subA,&subB,NULL,NULL);CHKERRQ(ierr);
    ierr = DMCreateMatrix(adm,&B);CHKERRQ(ierr);
    if (subA != subB) {
      ierr = DMCreateMatrix(adm,&A);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
      A = B;
    }
    ierr = MatNestSetSubMat(A,0,0,subA);CHKERRQ(ierr);
    ierr = MatNestSetSubMat(B,0,0,subB);CHKERRQ(ierr);
    for (i=0;i<aug_ctx->nqts;i++) {
      PetscInt r = aug_ctx->adjoint ? 0 : i+1;
      PetscInt c = aug_ctx->adjoint ? i+1 : 0;

      ierr = TSGetIJacobian(aug_ctx->qts[i],&subA,&subB,NULL,NULL);CHKERRQ(ierr);
      if (A == B && subA != subB) SETERRQ(PetscObjectComm((PetscObject)*ats),PETSC_ERR_PLIB,"Incompatible IJacobian matrices");
      ierr = MatNestSetSubMat(A,i+1,i+1,subA);CHKERRQ(ierr);
      ierr = MatNestSetSubMat(B,i+1,i+1,subB);CHKERRQ(ierr);
      subA = Acoupling ? Acoupling[i] : NULL;
      subB = Bcoupling ? Bcoupling[i] : NULL;
      if (A == B && subA != subB) SETERRQ(PetscObjectComm((PetscObject)*ats),PETSC_ERR_PLIB,"Incompatible IJacobian matrices");
      if (!aug_ctx->jaccoupling[i]) { /* scale the matrices once */
        if (subA) { ierr = MatScale(subA,-1.0);CHKERRQ(ierr); }
        if (subB && subB != subA) { ierr = MatScale(subB,-1.0);CHKERRQ(ierr); }
      }
      if (subA) { ierr = MatNestSetSubMat(A,r,c,subA);CHKERRQ(ierr); }
      if (subB) { ierr = MatNestSetSubMat(B,r,c,subB);CHKERRQ(ierr); }
    }
    ierr = TSSetIFunction(*ats,NULL,AugmentedTSIFunction,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(*ats,A,B,AugmentedTSIJacobian,NULL);CHKERRQ(ierr);
  } else {
    Mat subA,subB;

    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");

    ierr = TSGetRHSMats_Private(aug_ctx->model,&subA,&subB);CHKERRQ(ierr);
    ierr = DMCreateMatrix(adm,&B);CHKERRQ(ierr);
    if (subA != subB) {
      ierr = DMCreateMatrix(adm,&A);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
      A = B;
    }
    ierr = MatNestSetSubMat(A,0,0,subA);CHKERRQ(ierr);
    ierr = MatNestSetSubMat(B,0,0,subB);CHKERRQ(ierr);

    for (i=0;i<aug_ctx->nqts;i++) {
      PetscInt r = aug_ctx->adjoint ? 0 : i+1;
      PetscInt c = aug_ctx->adjoint ? i+1 : 0;

      ierr = TSGetRHSMats_Private(aug_ctx->qts[i],&subA,&subB);CHKERRQ(ierr);
      if (A == B && subA != subB) SETERRQ(PetscObjectComm((PetscObject)*ats),PETSC_ERR_PLIB,"Incompatible RHSJacobian matrices");
      /* this check is needed because we can use an implicit method, and the submatrices will get shifted differently */
      if (A != B && subA == subB) SETERRQ(PetscObjectComm((PetscObject)*ats),PETSC_ERR_PLIB,"Incompatible RHSJacobian matrices");
      ierr = MatNestSetSubMat(A,i+1,i+1,subA);CHKERRQ(ierr);
      ierr = MatNestSetSubMat(B,i+1,i+1,subB);CHKERRQ(ierr);
      subA = Acoupling ? Acoupling[i] : NULL;
      subB = Bcoupling ? Bcoupling[i] : NULL;
      if (A == B && subA != subB) SETERRQ(PetscObjectComm((PetscObject)*ats),PETSC_ERR_PLIB,"Incompatible RHSJacobian matrices");
      /* this check is needed because we can use an implicit method, and the submatrices will get shifted differently */
      if (A != B && (subA && subA == subB)) SETERRQ(PetscObjectComm((PetscObject)*ats),PETSC_ERR_PLIB,"Incompatible RHSJacobian matrices");
      if (subA) { ierr = MatNestSetSubMat(A,r,c,subA);CHKERRQ(ierr); }
      if (subB) { ierr = MatNestSetSubMat(B,r,c,subB);CHKERRQ(ierr); }
    }

    ierr = TSSetRHSFunction(*ats,NULL,AugmentedTSRHSFunction,NULL);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(*ats,A,B,AugmentedTSRHSJacobian,NULL);CHKERRQ(ierr);
  }
#if PETSC_VERSION_LT(3,13,0)
  ierr = MatSetOperation(A,MATOP_MISSING_DIAGONAL,(void (*)(void))MatMissingDiagonal_Nest);CHKERRQ(ierr);
  ierr = MatSetOperation(B,MATOP_MISSING_DIAGONAL,(void (*)(void))MatMissingDiagonal_Nest);CHKERRQ(ierr);
#endif
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Setup KSP for augmented system */
  if (aug_ctx->model->snes) {
    TSProblemType ptype;
    SNES          snes;

    ierr = TSGetProblemType(*ats,&ptype);CHKERRQ(ierr);
    ierr = TSGetSNES(*ats,&snes);CHKERRQ(ierr);
    if (ptype != TS_LINEAR) {
      ierr = SNESSetType(snes,SNESAUGMENTED);CHKERRQ(ierr);
      ierr = SNESSetApplicationContext(snes,*ats);CHKERRQ(ierr);
    } else {
      KSP ksp;

      ierr = SNESSetType(snes,aug_ctx->adjoint ? SNESKSPTRANSPOSEONLY : SNESKSPONLY);CHKERRQ(ierr);
      ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPAUGTRIANGULAR);CHKERRQ(ierr);
      ierr = KSPSetApplicationContext(ksp,*ats);CHKERRQ(ierr);
    }
  }

  /* handle specific options */
  ierr = PetscObjectAddOptionsHandler((PetscObject)(*ats),AugmentedTSOptionsHandler,NULL,NULL);CHKERRQ(ierr);

  ierr = TSSetPreStep(*ats,AugmentedTSPreStep);CHKERRQ(ierr);
  ierr = TSSetPostEvaluate(*ats,AugmentedTSPostEvaluate);CHKERRQ(ierr);
  ierr = TSSetPostStep(*ats,AugmentedTSPostStep);CHKERRQ(ierr);
  ierr = TSMonitorSet(*ats,AugmentedTSMonitor,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

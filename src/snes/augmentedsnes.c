#include <petscopt/private/augmentedtsimpl.h>
#include <petscdmcomposite.h>
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

PETSC_EXTERN PetscErrorCode SNESCreate_Augmented(SNES snes)
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

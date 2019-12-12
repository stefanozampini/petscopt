#include <petscopt/augmentedts.h>
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

static PetscErrorCode PCSetUp_AugTriangular(PC pc)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  KSP            ksp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_AugTriangular(PC pc, Vec x, Vec y)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  KSP            ksp;
  Mat            A;
  DM             dm;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (actx->adjoint) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not implemented");
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,x,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,y,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,actx->F[0],actx->U[0]);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&A,NULL);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    Mat Asub;
    Vec d;
    DM  qdm;

    /* can be more memory efficient */
    ierr = MatNestGetSubMat(A,i+1,0,&Asub);CHKERRQ(ierr);
    if (Asub) {
      ierr = MatMult(Asub,actx->U[0],actx->U[i+1]);CHKERRQ(ierr);
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

static PetscErrorCode PCApplyTranspose_AugTriangular(PC pc, Vec x, Vec y)
{
  TS             ats;
  TSAugCtx       *actx;
  SNES           snes;
  KSP            ksp;
  Mat            A;
  DM             dm;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&ats);CHKERRQ(ierr);
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (!actx->adjoint) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not implemented");
  ierr = TSGetSNES(actx->model,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,x,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,y,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(ksp,actx->F[0],actx->U[0]);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&A,NULL);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    Mat Asub;
    Vec d;
    DM  qdm;

    /* can be more memory efficient */
    ierr = MatNestGetSubMat(A,0,i+1,&Asub);CHKERRQ(ierr);
    if (Asub) {
      ierr = MatMultTranspose(Asub,actx->U[0],actx->U[i+1]);CHKERRQ(ierr);
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

static PetscErrorCode AugmentedTSTrajectorySet(TSTrajectory tj,TS ats,PetscInt stepnum,PetscReal time,Vec X)
{
  TSAugCtx         *actx;
  TSTrajectory     mtj;
  DM               dm;
  PetscInt         z=0;
  PetscObjectState st;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)X,&st);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,X,1,&z,actx->U);CHKERRQ(ierr);
  ierr = TSGetTrajectory(actx->model,&mtj);CHKERRQ(ierr);
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
  PetscCheckAugumentedTS(ats);
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
  PetscCheckAugumentedTS(ats);
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
  PetscCheckAugumentedTS(ats);
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
  PetscCheckAugumentedTS(ats);
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
  PetscCheckAugumentedTS(ats);
  ierr = AugmentedTSUpdateModelTS(ats);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (actx->model->prestep) {
    Vec              mU;
    PetscObjectState sprev,spost;

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
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSPostEvaluate(TS ats)
{
  TSAugCtx*      actx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  if (actx->model->postevaluate) {
    Vec              mU;
    PetscObjectState sprev,spost;

    ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
    ierr = TSGetSolution(actx->model,&mU);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)mU,&sprev);CHKERRQ(ierr);
    PetscStackCallStandard((*actx->model->postevaluate),(actx->model));
    ierr = PetscObjectStateGet((PetscObject)mU,&spost);CHKERRQ(ierr);
    if (sprev != spost) {
      ierr = TSRestartStep(ats);CHKERRQ(ierr);
    }
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
  PetscCheckAugumentedTS(ats);
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
  PetscCheckAugumentedTS(ats);
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
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSSetFromOptions(actx->model);CHKERRQ(ierr);
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
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,F,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(actx->model,time,actx->U[0],actx->F[0]);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],actx->U[0],NULL);CHKERRQ(ierr); }
    ierr = TSComputeRHSFunction(actx->qts[i],time,actx->U[i+1],actx->F[i+1]);CHKERRQ(ierr);
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],NULL,NULL);CHKERRQ(ierr); }
  }
  ierr = DMCompositeRestoreAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,F,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AugmentedTSRHSJacobian(TS ats, PetscReal time, Vec U, Mat A, Mat P, void *ctx)
{
  PetscErrorCode ierr;
  TSAugCtx       *actx;
  DM             dm;
  Mat            Asub,Psub;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = MatNestGetSubMat(A,0,0,&Asub);CHKERRQ(ierr);
  ierr = MatNestGetSubMat(P,0,0,&Psub);CHKERRQ(ierr);
  ierr = TSComputeRHSJacobian(actx->model,time,actx->U[0],Asub,Psub);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],actx->U[0],NULL);CHKERRQ(ierr); }
    if (actx->jaccoupling[i]) {
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
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U   ,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,Udot,actx->nqts + 1,NULL,actx->Udot);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,F   ,actx->nqts + 1,NULL,actx->F);CHKERRQ(ierr);
  ierr = TSComputeIFunction(actx->model,time,actx->U[0],actx->Udot[0],actx->F[0],PETSC_FALSE);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
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
  PetscCheckAugumentedTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&actx);CHKERRQ(ierr);
  ierr = TSGetDM(ats,&dm);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,U   ,actx->nqts + 1,NULL,actx->U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,Udot,actx->nqts + 1,NULL,actx->Udot);CHKERRQ(ierr);
  ierr = MatNestGetSubMat(A,0,0,&Asub);CHKERRQ(ierr);
  ierr = MatNestGetSubMat(P,0,0,&Psub);CHKERRQ(ierr);
  ierr = TSComputeIJacobian(actx->model,time,actx->U[0],actx->Udot[0],shift,Asub,Psub,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0;i<actx->nqts;i++) {
    if (actx->updatestates[i]) { ierr = (*actx->updatestates[i])(actx->qts[i],actx->U[0],actx->Udot[0]);CHKERRQ(ierr); }
    if (actx->jaccoupling[i]) {
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
  PetscCheckAugumentedTS(ats);
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
    ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ats),&ats->trajectory);CHKERRQ(ierr);
    ierr = TSTrajectorySetType(ats->trajectory,ats,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
    ierr = TSTrajectorySetSolutionOnly(ats->trajectory,PETSC_TRUE);CHKERRQ(ierr);
    ats->trajectory->adjoint_solve_mode = PETSC_FALSE;
    ats->trajectory->ops->set = AugmentedTSTrajectorySet;
  }

  /* wrap the event handler */
  ierr = TSEventDestroy(&ats->event);CHKERRQ(ierr);
  if (actx->model->event) {
    TSEvent mevent = actx->model->event;

    ierr = TSSetEventHandler(ats,mevent->nevents,mevent->direction,mevent->terminate,AugmentedTSEventHandler,AugmentedTSPostEvent,NULL);CHKERRQ(ierr);
  }

  /* mimick initialization in TSSolve for model TS */
  actx->model->ksp_its           = 0;
  actx->model->snes_its          = 0;
  actx->model->num_snes_failures = 0;
  actx->model->reject            = 0;
  actx->model->steprestart       = PETSC_TRUE;
  actx->model->steprollback      = PETSC_FALSE;
  ierr = TSSetUp(actx->model);CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(actx->model->trajectory,actx->model);CHKERRQ(ierr);
  ierr = TSEventInitialize(actx->model->event,actx->model,t0,actx->model->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AugmentedTSFinalize(TS ats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAugumentedTS(ats);
  ierr = AugmentedTSUpdateModelTS(ats);CHKERRQ(ierr);
  ierr = AugmentedTSUpdateSolution(ats);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&ats->trajectory);CHKERRQ(ierr);
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
  /* XXX */
  if (ts->adapt) {
    ierr = TSAdaptDestroy(&((*ats)->adapt));CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)ts->adapt);CHKERRQ(ierr);
    (*ats)->adapt = ts->adapt;
    {  /* HACK (these stores a vector internally!!) */
      PetscBool needreset;
      ierr = PetscObjectTypeCompareAny((PetscObject)ts->adapt,&needreset,TSADAPTBASIC,TSADAPTDSP,"");CHKERRQ(ierr);
      if (needreset) { ierr = TSAdaptReset(ts->adapt);CHKERRQ(ierr); }
    }
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

    ierr = PetscObjectReference((PetscObject)qts[i]);CHKERRQ(ierr);
    aug_ctx->qts[i] = qts[i];
    ierr = TSGetDM(aug_ctx->qts[i],&dm);CHKERRQ(ierr);
    ierr = DMCompositeAddDM(adm,dm);CHKERRQ(ierr);
    aug_ctx->updatestates[i] = updatestates ? updatestates[i] : NULL;
    aug_ctx->jaccoupling[i]  = jaccoupling  ? jaccoupling[i]  : NULL;
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
  ierr = DMGetGlobalVector(adm,&vatol);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(adm,&vrtol);CHKERRQ(ierr);
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
  ierr = DMRestoreGlobalVector(adm,&vatol);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(adm,&vrtol);CHKERRQ(ierr);

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

  /* Setup PC for augmented system */
  if (aug_ctx->model->snes) {
    SNES snes;
    KSP  ksp;
    PC   pc;

    ierr = TSGetSNES(*ats,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
    ierr = PCShellSetContext(pc,*ats);CHKERRQ(ierr);
    ierr = PCShellSetSetUp(pc,PCSetUp_AugTriangular);CHKERRQ(ierr);
    ierr = PCShellSetApply(pc,PCApply_AugTriangular);CHKERRQ(ierr);
    ierr = PCShellSetApplyTranspose(pc,PCApplyTranspose_AugTriangular);CHKERRQ(ierr);
  }

  /* handle specific options */
  ierr = PetscObjectAddOptionsHandler((PetscObject)(*ats),AugmentedTSOptionsHandler,NULL,NULL);CHKERRQ(ierr);

  ierr = TSSetPreStep(*ats,AugmentedTSPreStep);CHKERRQ(ierr);
  ierr = TSSetPostEvaluate(*ats,AugmentedTSPostEvaluate);CHKERRQ(ierr);
  ierr = TSSetPostStep(*ats,AugmentedTSPostStep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

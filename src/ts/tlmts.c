#include <petscopt/tsutils.h>
#include <petscopt/private/tlmtsimpl.h>
#include <petscopt/private/tsoptimpl.h>
#include <petscopt/private/tssplitjacimpl.h>
#include <petscopt/private/discretetsimpl.h>
#include <petscopt/private/tspdeconstrainedutilsimpl.h>
#include <petsc/private/tsimpl.h>
#include <petscdm.h>

static PetscErrorCode TLMTSDestroy_Private(void *ptr)
{
  TLMTS_Ctx*     tlm = (TLMTS_Ctx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&tlm->design);CHKERRQ(ierr);
  ierr = VecDestroy(&tlm->mdelta);CHKERRQ(ierr);
  ierr = VecDestroy(&tlm->workrhs);CHKERRQ(ierr);
  ierr = TSDestroy(&tlm->model);CHKERRQ(ierr);
  ierr = PetscFree(tlm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TLMTS can be called by AdjointTS, this is a shortcut */
static PetscErrorCode TLMTSComputeSplitJacobians(TS ts, PetscReal time, Vec U, Vec Udot, Mat A, Mat pA, Mat B, Mat pB)
{
  TLMTS_Ctx      *tlm_ctx;
  Mat            J_U = NULL,J_Udot = NULL,pJ_U = NULL,pJ_Udot = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTLMTS(ts);
  if (A == B) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"A and B must be different matrices");
  if (pA == pB) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"pA and pB must be different matrices");
  ierr = TSGetApplicationContext(ts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSUpdateSplitJacobiansFromHistory_Private(tlm_ctx->model,time);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(tlm_ctx->model,&J_U,&pJ_U,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
  if (A) { ierr = MatCopy(J_U,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  if (pA && pA != A) { ierr = MatCopy(pJ_U,pA,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  if (B) { ierr = MatCopy(J_Udot,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  if (pB && pB != B) { ierr = MatCopy(pJ_Udot,pB,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

PetscErrorCode TLMTSComputeForcing(TS lts, PetscReal time, Vec U, Vec Udot, PetscBool *hasf, Vec F)
{
  PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,PetscBool*,Vec);
  TLMTS_Ctx      *tlm_ctx;
  TSOpt          tsopt;
  PetscBool      has,hasnc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(lts,time,2);
  if (U) PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  if (Udot) PetscValidHeaderSpecific(Udot,VEC_CLASSID,4);
  PetscValidPointer(hasf,5);
  PetscValidHeaderSpecific(F,VEC_CLASSID,6);
  ierr = PetscObjectQueryFunction((PetscObject)lts,"TLMTSComputeForcing_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(lts,time,U,Udot,hasf,F);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscCheckTLMTS(lts);
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSGetTSOpt(tlm_ctx->model,&tsopt);CHKERRQ(ierr);
  ierr = TSOptHasGradientDAE(tsopt,&has,&hasnc);CHKERRQ(ierr);
  if (has) {
    if (hasnc) { /* non constant dependence */
      Mat F_m;

      ierr = TSOptEvalGradientDAE(tsopt,time,U,Udot,tlm_ctx->design,&F_m,NULL);CHKERRQ(ierr);
      ierr = MatMult(F_m,tlm_ctx->mdelta,F);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(tlm_ctx->workrhs,F);CHKERRQ(ierr);
    }
  }
  *hasf = has;
  PetscFunctionReturn(0);
}

/* The TLM DAE is J_Udot * U_dot + J_U * U + f = 0, with f = dH/dm * deltam */
static PetscErrorCode TLMTSIFunctionLinear(TS lts, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  Mat            J_U = NULL, J_Udot = NULL;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTLMTS(lts);
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TLMTSComputeForcing(lts,time,NULL,NULL,&has,tlm_ctx->workrhs);CHKERRQ(ierr);
  ierr = TSUpdateSplitJacobiansFromHistory_Private(tlm_ctx->model,time);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(tlm_ctx->model,&J_U,NULL,&J_Udot,NULL);CHKERRQ(ierr);
  ierr = MatMult(J_U,U,F);CHKERRQ(ierr);
  ierr = MatMultAdd(J_Udot,Udot,F,F);CHKERRQ(ierr);
  if (has) {
    ierr = VecAXPY(F,1.0,tlm_ctx->workrhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSIJacobian(TS lts, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  TS             model;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTLMTS(lts);
  ierr  = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  model = tlm_ctx->model;
  if (tlm_ctx->discrete) {
    ierr = TSComputeIJacobian(model,time,U,Udot,shift,A,B,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    if (tlm_ctx->userijac) {
      Vec W[2];

      ierr = TSTrajectoryGetUpdatedHistoryVecs(model->trajectory,model,time,&W[0],&W[1]);CHKERRQ(ierr);
      ierr = TSComputeIJacobian(model,time,W[0],W[1],shift,A,B,PETSC_FALSE);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(model->trajectory,&W[0],&W[1]);CHKERRQ(ierr);
    } else {
      ierr = TSComputeIJacobianWithSplits_Private(model,time,NULL,NULL,shift,A,B,ctx);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* The TLM DAE is U_dot = J_U * U - f, with f = dH/dm * deltam */
static PetscErrorCode TLMTSRHSFunctionLinear(TS lts, PetscReal time, Vec U, Vec F, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTLMTS(lts);
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TLMTSComputeForcing(lts,time,NULL,NULL,&has,tlm_ctx->workrhs);CHKERRQ(ierr);
  /* force recomputation of RHS Jacobian */
  lts->rhsjacobian.time = PETSC_MIN_REAL;
  ierr = TSComputeRHSJacobian(lts,time,U,lts->Arhs,lts->Brhs);CHKERRQ(ierr);
  ierr = MatMult(lts->Arhs,U,F);CHKERRQ(ierr);
  if (has) {
    ierr = VecAXPY(F,-1.0,tlm_ctx->workrhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSRHSJacobian(TS lts, PetscReal time, Vec U, Mat A, Mat P, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  TSProblemType  type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTLMTS(lts);
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSGetProblemType(tlm_ctx->model,&type);CHKERRQ(ierr);
  if (type > TS_LINEAR && !tlm_ctx->discrete) {
    ierr = TSTrajectoryGetUpdatedHistoryVecs(tlm_ctx->model->trajectory,tlm_ctx->model,time,&U,NULL);CHKERRQ(ierr);
  }
  /* force recomputation of RHS Jacobian: this is needed because this function can be called from within an adjoint solver */
  if (lts->rhsjacobian.time == PETSC_MIN_REAL) tlm_ctx->model->rhsjacobian.time = PETSC_MIN_REAL;
  ierr = TSComputeRHSJacobian(tlm_ctx->model,time,U,A,P);CHKERRQ(ierr);
  if (type > TS_LINEAR && !tlm_ctx->discrete) {
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlm_ctx->model->trajectory,&U,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSOptionsHandler(PetscOptionItems *PetscOptionsObject,PetscObject obj,PETSC_UNUSED void *ctx)
{
  TS             lts = (TS)obj;
  TLMTS_Ctx      *tlm_ctx;
  PetscContainer container;
  PetscBool      jcon,rksp,flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTLMTS(lts);
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"TLMTS options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-userijacobian","Use the user-provided IJacobian routine, instead of the splits, to compute the Jacobian",NULL,tlm_ctx->userijac,&tlm_ctx->userijac,NULL);CHKERRQ(ierr);
  jcon = PETSC_FALSE;
  ierr = PetscOptionsBool("-constjacobians","Whether or not the DAE Jacobians are constant",NULL,jcon,&jcon,NULL);CHKERRQ(ierr);
  rksp = PETSC_FALSE;
  ierr = PetscOptionsBool("-reuseksp","Reuse the KSP solver from the nonlinear model",NULL,rksp,&rksp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-discrete","Use discrete tangent linear models (not available for all methods)",NULL,tlm_ctx->discrete,&tlm_ctx->discrete,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = TLMTSSetUpStep(lts);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)tlm_ctx->model,"_ts_splitJac",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    TSSplitJacobians *splitJ;

    ierr = PetscContainerGetPointer(container,(void**)&splitJ);CHKERRQ(ierr);
    splitJ->jacconsts = jcon;
  }
  if (jcon) {
    TSRHSFunction rhsfunc;

    ierr = TSGetRHSFunction(lts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
    if (rhsfunc) { /* just to make sure we have a correct Jacobian */
      DM  dm;
      Mat A,B;
      Vec U;

      ierr = TSGetRHSMats_Private(tlm_ctx->model,&A,&B);CHKERRQ(ierr);
      ierr = TSGetDM(tlm_ctx->model,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobian(tlm_ctx->model,PETSC_MIN_REAL,U,A,B);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(lts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    }
  }
  if (rksp) { /* reuse the same KSP */
    SNES snes;
    KSP  ksp;

    ierr = TSGetSNES(tlm_ctx->model,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = TSGetSNES(lts,&snes);CHKERRQ(ierr);
    ierr = SNESSetKSP(snes,ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TLMTSIsDiscrete(TS lts, PetscBool *flg)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidPointer(flg,2);
  ierr = TLMTSSetUpStep(lts);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  *flg = tlm->discrete;
  PetscFunctionReturn(0);
}

/*@C
   TLMTSGetRHSVec - Gets the vector used to compute the forcing term for the Tangent Linear Model.

   Synopsis:
   #include <petsc/private/tstlmtsimpl.h>

   Logically Collective on TS

   Input Parameters:
.  lts - the TS context obtained from TSCreateTLMTS()

   Output Parameters:
.  rhs - the rhs work vector

   Notes: this should not be modified unless you know what you are doing.

   Level: developer

.seealso: TSCreateTLMTS(), TLMTSGetModelTS(), TLMTSSetPerturbationVec(), TLMTSSetDesignVec()
@*/
PetscErrorCode TLMTSGetRHSVec(TS lts, Vec *rhs)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscCheckTLMTS(lts);
  PetscValidPointer(rhs,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  if (!tlm->workrhs) {
    Vec U;

    ierr = TSGetSolution(lts,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&tlm->workrhs);CHKERRQ(ierr);
  }
  *rhs = tlm->workrhs;
  PetscFunctionReturn(0);
}

/*@C
   TLMTSSetPerturbationVec - Sets the vector that stores the current perturbation of the model parameters.

   Synopsis:
   #include <petsc/private/tstlmtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  lts    - the TS context obtained from TSCreateTLMTS()
-  mdelta - the vector that stores the perturbartion

   Notes: Pass NULL if you want to destroy the perturbation vector stored inside the TLMTS.

   Level: developer

.seealso: TSCreateTLMTS(), TLMTSGetModelTS(), TLMTSSetDesignVec(), TLMTSGetDesignVec(), TLMTSGetRHSVec()
@*/
PetscErrorCode TLMTSSetPerturbationVec(TS lts, Vec mdelta)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscCheckTLMTS(lts);
  if (mdelta) PetscValidHeaderSpecific(mdelta,VEC_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  if (mdelta) {
    ierr = PetscObjectReference((PetscObject)mdelta);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&tlm->mdelta);CHKERRQ(ierr);
  tlm->mdelta = mdelta;
  PetscFunctionReturn(0);
}

/*@C
   TLMTSSetDesignVec - Sets the vector that stores the current design.

   Synopsis:
   #include <petsc/private/tstlmtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  lts    - the TS context obtained from TSCreateTLMTS()
-  design - the vector that stores the current values of the parameters

   Level: developer

.seealso: TSCreateTLMTS(), TLMTSGetModelTS(), TLMTSSetPerturbationVec(), TLMTSGetDesignVec(), TLMTSGetRHSVec()
@*/
PetscErrorCode TLMTSSetDesignVec(TS lts, Vec design)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscCheckTLMTS(lts);
  PetscValidHeaderSpecific(design,VEC_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)design);CHKERRQ(ierr);
  ierr = VecDestroy(&tlm->design);CHKERRQ(ierr);
  tlm->design = design;
  PetscFunctionReturn(0);
}

/*@C
   TLMTSGetDesignVec - Gets the vector for the current design set with TLMTSSetDesignVec().

   Synopsis:
   #include <petsc/private/tstlmtsimpl.h>

   Logically Collective on TS

   Input Parameters:
.  lts - the TS context obtained from TSCreateTLMTS()

   Output Parameters:
.  design - the vector that stores the current values of the parameters

   Level: developer

.seealso: TSCreateTLMTS(), TLMTSGetModelTS(), TLMTSSetPerturbationVec(), TLMTSSetDesignVec(), TLMTSGetRHSVec()
@*/
PetscErrorCode TLMTSGetDesignVec(TS lts, Vec* design)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscCheckTLMTS(lts);
  PetscValidPointer(design,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  *design = tlm->design;
  PetscFunctionReturn(0);
}

/*@C
   TLMTSGetModelTS - Gets the model DAE used to create the TLMTS.

   Synopsis:
   #include <petsc/private/tstlmtsimpl.h>

   Logically Collective on TS

   Input Parameters:
.  lts - the TS context obtained from TSCreateTLMTS()

   Output Parameters:
.  ts - the TS context used to create the TLMTS.

   Level: developer

.seealso: TSCreateTLMTS(), TSSetGradientDAE(), TSSetGradientIC()
@*/
PetscErrorCode TLMTSGetModelTS(TS lts, TS* ts)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode (*f)(TS,TS*);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidPointer(ts,2);
  ierr = PetscObjectQueryFunction((PetscObject)lts,"TLMTSGetModelTS_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(lts,ts);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscCheckTLMTS(lts);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  *ts  = tlm->model;
  PetscFunctionReturn(0);
}

/*@C
   TLMTSComputeInitialConditions - Computes the initial conditions for the Tangent Linear Model TS.

   Synopsis:
   #include <petsc/private/tstlmtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  lts - the TS context obtained from TSCreateTLMTS()
.  t0  - the initial time
-  x0  - the initial vector where to linearize

   Notes: Initial conditions are computed as

$     Y(t0) = -G^-1(x0,m) * G_m(x0,m) * deltam

   where deltam is the vector provided with TLMTSSetPerturbationVec(), and m the design vector set with TLMTSSetDesignVec().
   G(x0,m) = 0 describes the model initial conditions in implicit form; G_x and G_m should be provided with TSSetGradientIC().

   Level: developer

.seealso: TSCreateTLMTS(), TLMTSSetPerturbationVec(), TLMTSSetDesignVec(), TSSetGradientIC()
@*/
PetscErrorCode TLMTSComputeInitialConditions(TS lts, PetscReal t0, Vec x0)
{
  Vec            eta;
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  TSOpt          tsopt;
  TSRHSFunction  rhsfunc;
  PetscBool      has,hasnc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscCheckTLMTS(lts);
  PetscValidLogicalCollectiveReal(lts,t0,2);
  PetscValidHeaderSpecific(x0,VEC_CLASSID,3);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  if (!tlm->design) SETERRQ(PetscObjectComm((PetscObject)lts),PETSC_ERR_USER,"Missing design vector. Maybe you forgot to call TLMTSSetDesignVec()");
  if (!tlm->mdelta) SETERRQ(PetscObjectComm((PetscObject)lts),PETSC_ERR_USER,"Missing design vector. Maybe you forgot to call TLMTSSetPerturbationVec()");
  ierr = TSGetSolution(lts,&eta);CHKERRQ(ierr);
  if (!eta) {
    Vec U;

    ierr = TSGetSolution(tlm->model,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&eta);CHKERRQ(ierr);
    ierr = TSSetSolution(lts,eta);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)eta);CHKERRQ(ierr);
  }
  ierr = VecLockReadPush(x0);CHKERRQ(ierr);
  ierr = TSGetTSOpt(tlm->model,&tsopt);CHKERRQ(ierr);
  ierr = TSOptHasGradientIC(tsopt,&has);CHKERRQ(ierr);
  if (!has) {
    /* For propagator computations, the linear dependence on the initial conditions is attached to the TLMTS if the model TS does not have any set */
    ierr = TSLinearizedICApply_Private(lts,t0,x0,tlm->design,tlm->mdelta,eta,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = TSLinearizedICApply_Private(tlm->model,t0,x0,tlm->design,tlm->mdelta,eta,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = VecScale(eta,-1.0);CHKERRQ(ierr);

  /* initialize tlm->workrhs if needed */
  ierr = TLMTSGetRHSVec(lts,&tlm->workrhs);CHKERRQ(ierr);
  ierr = TSOptHasGradientDAE(tsopt,&has,&hasnc);CHKERRQ(ierr);
  if (has && !hasnc) { /* constant dependence */
    Mat F_m;

    ierr = TSOptEvalGradientDAE(tsopt,t0,NULL,NULL,NULL,&F_m,NULL);CHKERRQ(ierr);
    ierr = MatMult(F_m,tlm->mdelta,tlm->workrhs);CHKERRQ(ierr);
  }

  /* XXX make sure we have a correct Jacobian */
  ierr = TSGetRHSFunction(lts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (rhsfunc) {
    TSRHSJacobian rhsjacfunc;

    ierr = TSGetRHSJacobian(lts,NULL,NULL,&rhsjacfunc,NULL);CHKERRQ(ierr);
    if (rhsjacfunc == TSComputeRHSJacobianConstant) {
      DM  dm;
      Mat A,B;
      Vec U;

      ierr = TSGetRHSMats_Private(tlm->model,&A,&B);CHKERRQ(ierr);
      ierr = TSGetDM(tlm->model,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobian(tlm->model,PETSC_MIN_REAL,U,A,B);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&U);CHKERRQ(ierr);
    }
  }

  ierr = VecLockReadPop(x0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TLMTSSetUpStep(TS lts)
{
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscCheckTLMTS(lts);
  ierr = TSGetApplicationContext(lts,(void*)&tlm);CHKERRQ(ierr);
  if (!tlm->cstep) tlm->cstep = lts->ops->step;
  if (tlm->discrete) {
    SNES      snes;
    KSP       ksp;
    PetscBool rk,theta,cn;

    ierr = PetscObjectTypeCompare((PetscObject)lts,TSRK,&rk);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)lts,TSTHETA,&theta);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)lts,TSCN,&cn);CHKERRQ(ierr);
    if (rk)               lts->ops->step = TSStep_TLM_RK;
    else if (theta || cn) lts->ops->step = TSStep_TLM_Theta;
    else {
      TSType tstype;

      ierr = TSGetType(lts,&tstype);CHKERRQ(ierr);
      SETERRQ1(PetscObjectComm((PetscObject)lts),PETSC_ERR_SUP,"Discrete TLM not available for type %s\n",tstype);
    }
    /* reuse KSP */
    ierr = TSGetSNES(tlm->model,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = TSGetSNES(lts,&snes);CHKERRQ(ierr);
    ierr = SNESSetKSP(snes,ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSSetUp(TS lts)
{
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TLMTSSetUpStep(lts);CHKERRQ(ierr);
  PetscCheckTLMTS(lts);
  ierr = TSGetApplicationContext(lts,(void*)&tlm);CHKERRQ(ierr);
  if (tlm->setup) { ierr = (*tlm->setup)(lts);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*@C
   TSCreateTLMTS - Creates a TS object that can be used to solve the Tangent Linear Model of a parameter dependent DAE.

   Synopsis:
   #include <petsc/private/tstlmtsimpl.h>

   Logically Collective on TS

   Input Parameters:
.  ts - the model TS context obtained from TSCreate()

   Output Parameters:
.  lts - the new TS context for the Tangent Linear Model DAE

   Options Database Keys:
+  -tlm_userijacobian <0> - use the user-callback to compute the IJacobian. Defaults to TSComputeIJacobianWithSplits_Private()
.  -tlm_constjacobians <0> - if the Jacobians are constant
-  -tlm_reuseksp <0> - if the TLMTS should reuse the same KSP object used to solve the model DAE

   Notes: Given the parameter dependent DAE in implicit form F(t,x,xdot;m) = 0, with initial conditions expressed in implicit form G(x_0,m) = 0,
          the TLMTS solves the linear DAE F_xdot Y_dot + F_x Y + F_m * deltam = 0, with initial conditions taken as Y_0 = -G^-1(x_0,m) G_m deltam,
          where deltam is a perturbation of the parameters passed via TLMTSSetPerturbationVec().

          Both the IFunction/IJacobian and the RHSFunction/RHSJacobian interfaces are supported. The Jacobians needed to perform the TLM solve are automatically constructed. Alternatively, the user can compose the "TSComputeSplitJacobians_C" function in the model TS to compute them.

          The forcing term can be customized by calling TSSetGradientDAE() on the model TS.

          Parameter dependent initial conditions should be passed with TSSetGradientIC() on the model TS.

          Initial conditions can be computed with TLMTSComputeInitialConditions().

          The TLMTS inherits the prefix of the model TS. E.g. if the model TS has prefix "burgers", the options prefix for the TLMTS is -tlm_burgers_. The user needs to call TSSetFromOptions() on the TLMTS to trigger its customization.

   Level: developer

.seealso: TSCreateTLMTS(), TLMTSSetPerturbationVec(), TLMTSSetDesignVec(), TSSetGradientDAE(), TSSetGradientIC()
@*/
PetscErrorCode TSCreateTLMTS(TS ts, TS* lts)
{
  SNES             snes;
  KSP              ksp;
  Mat              A,B;
  Vec              vatol,vrtol;
  PetscContainer   container;
  TLMTS_Ctx        *tlm_ctx;
  TSIFunction      ifunc;
  TSRHSFunction    rhsfunc;
  TSI2Function     i2func;
  KSPType          ksptype;
  const char       *prefix;
  PetscReal        atol,rtol,dtol;
  PetscInt         maxits;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(lts,2);
  ierr = TSGetI2Function(ts,NULL,&i2func,NULL);CHKERRQ(ierr);
  if (i2func) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Second order DAEs are not supported");
  ierr = TSCreateWithTS(ts,lts);CHKERRQ(ierr);
  ierr = TSGetTolerances(ts,&atol,&vatol,&rtol,&vrtol);CHKERRQ(ierr);
  ierr = TSSetTolerances(*lts,atol,vatol,rtol,vrtol);CHKERRQ(ierr);
  if (ts->adapt) {
    ierr = TSAdaptCreate(PetscObjectComm((PetscObject)*lts),&(*lts)->adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType((*lts)->adapt,((PetscObject)ts->adapt)->type_name);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunction((PetscObject)(*lts),"TSComputeSplitJacobians_C",TLMTSComputeSplitJacobians);CHKERRQ(ierr);

  ierr = PetscNew(&tlm_ctx);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*lts,(void *)tlm_ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  tlm_ctx->model = ts;
  tlm_ctx->setup = (*lts)->ops->setup;
  (*lts)->ops->setup = TLMTSSetUp;

  /* wrap application context in a container, so that it will be destroyed when calling TSDestroy on lts */
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(*lts)),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,tlm_ctx);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,TLMTSDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*lts),"_ts_tlm_ctx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  /* TLMTS prefix: i.e. options called as -tlm_ts_monitor or -tlm_modelprefix_ts_monitor */
  ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(*lts,"tlm_");CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*lts,prefix);CHKERRQ(ierr);

  /* setup callbacks for the tangent linear model DAE: we reuse the same jacobian matrices of the forward model */
  ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
  ierr = TSGetRHSFunction(ts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (ifunc) {
    TSSplitJacobians *splitJ;

    ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
    ierr = TSSetIFunction(*lts,NULL,TLMTSIFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(*lts,A,B,TLMTSIJacobian,NULL);CHKERRQ(ierr);
    /* setup _ts_splitJac container */
    ierr = TSGetSplitJacobians(ts,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    /* caching to prevent from recomputation of Jacobians */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&container);CHKERRQ(ierr);
    if (!container) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing _ts_splitJac container");
    /* we can setup an AdjointTS from a TLMTS -> propagate splitJac to save memory */
    ierr = PetscObjectCompose((PetscObject)(*lts),"_ts_splitJac",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(container,(void**)&splitJ);CHKERRQ(ierr);

    splitJ->splitdone = PETSC_FALSE;
  } else {
    TSRHSJacobian rhsjacfunc;

    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");
    ierr = TSSetRHSFunction(*lts,NULL,TLMTSRHSFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobian(ts,NULL,NULL,&rhsjacfunc,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSMats_Private(ts,&A,&B);CHKERRQ(ierr);
    if (rhsjacfunc == TSComputeRHSJacobianConstant) {
      ierr = TSSetRHSJacobian(*lts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(*lts,A,B,TLMTSRHSJacobian,NULL);CHKERRQ(ierr);
    }
  }

  /* tangent linear model DAE is linear */
  ierr = TSSetProblemType(*lts,TS_LINEAR);CHKERRQ(ierr);

  /* tangent linear model linear solver -> propagate KSP info of the forward model but use a different object */
  if (ts->snes) {
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
    ierr = KSPGetTolerances(ksp,&rtol,&atol,&dtol,&maxits);CHKERRQ(ierr);
    ierr = TSGetSNES(*lts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    if (ksptype) { ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr); }
    ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxits);CHKERRQ(ierr);
  }

  /* handle specific TLMTS options */
  ierr = PetscObjectAddOptionsHandler((PetscObject)(*lts),TLMTSOptionsHandler,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

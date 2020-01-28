#include <petscopt/tlmts.h>
#include <petscopt/tsutils.h>
#include <petscopt/private/augmentedtsimpl.h>
#include <petscopt/private/adjointtsimpl.h>
#include <petscopt/private/discretetsimpl.h>
#include <petscopt/private/tsobjimpl.h>
#include <petscopt/private/tsoptimpl.h>
#include <petscopt/private/tspdeconstrainedutilsimpl.h>
#include <petscopt/private/tssplitjacimpl.h>
#include <petsc/private/tsimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscdm.h>

/* ------------------ Routines for adjoints of DAE, namespaced with AdjointTS ----------------------- */
/*
  This code is very much inspired to the papers
   [1] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: algorithms and software, JCAM 149, 2002.
   [2] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: the adjoint DAE system and its numerical solution, SISC 24, 2003.
   [3] Ozyurt, Barton. Cheap second order directional derivatives of stiff ODE embedded functionals, SISC 26, 2005.
  Do we need to implement the augmented formulation (25) in [2] for implicit problems ?
   - Initial conditions for the adjoint variable are fine as they are now for the cases:
     - integrand terms : all but index-2 DAEs
     - g(x,T,p)        : all but index-2 DAEs
   TODO: register citations
*/
typedef struct {
  TS  adjts;
  Vec work1;
  Vec work2;
} AdjEvalQuadCtx;

static PetscErrorCode EvalQuadIntegrand_ADJ(Vec L, Vec Ldot, PetscReal t, Vec F, void* ctx)
{
  Mat            adjF_m;
  AdjEvalQuadCtx *q = (AdjEvalQuadCtx*)ctx;
  AdjointCtx     *adj_ctx;
  TSOpt          tsopt;
  TSObj          tsobj;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(q->adjts);
  ierr  = TSGetApplicationContext(q->adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt  = adj_ctx->tf - t + adj_ctx->t0;
  ierr  = TSGetTSOpt(adj_ctx->fwdts,&tsopt);CHKERRQ(ierr);
  tsobj = adj_ctx->tsobj;
  ierr  = VecSet(F,0.0);CHKERRQ(ierr);
  if (adj_ctx->discrete) {
    Vec       U,Udot;
    PetscBool has;

    ierr = PetscObjectQuery((PetscObject)L,"_ts_adjoint_discrete_U",(PetscObject*)&U);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)L,"_ts_adjoint_discrete_Udot",(PetscObject*)&Udot);CHKERRQ(ierr);
    if (!U) SETERRQ(PetscObjectComm((PetscObject)L),PETSC_ERR_PLIB,"Missing sampling vector");
    ierr = TSOptHasGradientDAE(tsopt,&has,NULL);CHKERRQ(ierr);
    if (has) {
      ierr = TSOptEvalGradientDAE(tsopt,fwdt,U,Udot,adj_ctx->design,NULL,&adjF_m);CHKERRQ(ierr);
      ierr = MatMult(adjF_m,L,F);CHKERRQ(ierr);
    }
    if (!adj_ctx->direction) {
      ierr = TSObjEval_M(tsobj,U,adj_ctx->design,fwdt,q->work1,&has,q->work2);CHKERRQ(ierr);
      if (has) {
        ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr);
      }
    } else { /* second order adjoint */
      Vec       TLMU,TLMUdot;
      Vec       FOAL;
      PetscBool flg;

      ierr = PetscObjectQuery((PetscObject)L,"_ts_adjoint_discrete_TLMU",(PetscObject*)&TLMU);CHKERRQ(ierr);
      if (!TLMU) SETERRQ(PetscObjectComm((PetscObject)L),PETSC_ERR_PLIB,"Missing TLM sampling vector");
      ierr = TSObjEval_MU(tsobj,U,adj_ctx->design,fwdt,TLMU,q->work1,&flg,q->work2);CHKERRQ(ierr);
      if (flg) { ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr); }
      ierr = TSObjEval_MM(tsobj,U,adj_ctx->design,fwdt,adj_ctx->direction,q->work1,&flg,q->work2);CHKERRQ(ierr);
      if (flg) { ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr); }
      ierr = PetscObjectQuery((PetscObject)L,"_ts_adjoint_discrete_FOAL",(PetscObject*)&FOAL);CHKERRQ(ierr);
      if (FOAL) { /* Full hessian */
        TSIFunction ifunc;
        PetscBool   Hhas[3][3];

        ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
        ierr = TSGetIFunction(adj_ctx->fwdts,NULL,&ifunc,NULL);CHKERRQ(ierr);
        if (!ifunc) Hhas[2][1] = PETSC_FALSE;
        if (Hhas[2][2]) { /* (L^T \otimes I_M) H_MM direction */
          ierr = TSOptEvalHessianDAE(tsopt,2,2,fwdt,U,Udot,adj_ctx->design,FOAL,adj_ctx->direction,q->work1);CHKERRQ(ierr);
          ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
        }
        if (Hhas[2][0]) { /* (L^T \otimes I_M) H_MX \eta, \eta the TLM solution */
          ierr = TSOptEvalHessianDAE(tsopt,2,0,fwdt,U,Udot,adj_ctx->design,FOAL,TLMU,q->work1);CHKERRQ(ierr);
          ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
        }
        /* this is evaluated only via the IFunction interface */
        if (Hhas[2][1] && Udot) { /* (L^T \otimes I_M) H_MXdot \etadot */
          ierr = PetscObjectQuery((PetscObject)L,"_ts_adjoint_discrete_TLMUdot",(PetscObject*)&TLMUdot);CHKERRQ(ierr);
          if (!TLMUdot) SETERRQ(PetscObjectComm((PetscObject)L),PETSC_ERR_PLIB,"Missing TLM sampling vector");
          ierr = TSOptEvalHessianDAE(tsopt,2,1,t,U,Udot,adj_ctx->design,FOAL,TLMUdot,q->work1);CHKERRQ(ierr);
          ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
        }
      }
    }
  } else {
    PetscBool has;

    ierr = TSOptHasGradientDAE(tsopt,&has,NULL);CHKERRQ(ierr);
    if (has) {
      ierr = TSOptEvalGradientDAE(tsopt,fwdt,NULL,NULL,adj_ctx->design,NULL,&adjF_m);CHKERRQ(ierr);
      ierr = MatMult(adjF_m,L,F);CHKERRQ(ierr);
    }
    ierr = TSObjHasObjectiveIntegrand(tsobj,NULL,NULL,&has,NULL,NULL,NULL);CHKERRQ(ierr);
    if (has && !adj_ctx->direction) {
      Vec U;

      ierr = TSTrajectoryGetUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,&U,NULL);CHKERRQ(ierr);
      ierr = TSObjEval_M(tsobj,U,adj_ctx->design,fwdt,q->work1,&has,q->work2);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,&U,NULL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSDestroy_Private(void *ptr)
{
  AdjointCtx*    adj = (AdjointCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&adj->design);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->workinit);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->quadvec);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->wquad);CHKERRQ(ierr);
  ierr = TSDestroy(&adj->fwdts);CHKERRQ(ierr);
  ierr = TSDestroy(&adj->tlmts);CHKERRQ(ierr);
  ierr = TSDestroy(&adj->foats);CHKERRQ(ierr);
  ierr = PetscFree(adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* use U to sample the Jacobian for discrete adjoints */
static PetscErrorCode AdjointTSRHSJacobian(TS adjts, PetscReal time, Vec U, Mat A, Mat P, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      ft;
  TSProblemType  type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  ierr = TSGetProblemType(adj_ctx->fwdts,&type);CHKERRQ(ierr);
  ft   = adj_ctx->tf - time + adj_ctx->t0;
  /* force recomputation of RHS Jacobian XXX CHECK WITH RK FOR CACHING */
  if (adjts->rhsjacobian.time == PETSC_MIN_REAL) adj_ctx->fwdts->rhsjacobian.time = PETSC_MIN_REAL;
  if (type > TS_LINEAR && !adj_ctx->discrete) {
    ierr = TSTrajectoryGetUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,ft,&U,NULL);CHKERRQ(ierr);
  }
  ierr = TSComputeRHSJacobian(adj_ctx->fwdts,ft,U,A,P);CHKERRQ(ierr);
  if (type > TS_LINEAR && !adj_ctx->discrete) {
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,&U,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* The adjoint formulation used assumes the problem written as H(U,Udot,t) = 0

   -> the forward DAE is Udot - G(U) = 0 ( -> H(U,Udot,t) := Udot - G(U) )
      the first-order adjoint DAE is F - L^T * G_U - Ldot^T in backward time (F the derivative of the objective wrt U)
      the first-order adjoint DAE is Ldot^T = L^T * G_U - F in forward time
   -> the second-order adjoint differs only by the forcing term :
      F = O_UM * direction + O_UU * eta + (L \otimes I_N)(tH_UM * direction + tH_UU * eta + tH_UUdot * etadot)
      with eta the solution of the tangent linear model
*/
static PetscErrorCode AdjointTSRHSFunctionLinear(TS adjts, PetscReal time, Vec U, Vec F, void *ctx)
{
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  ierr = AdjointTSComputeForcing(adjts,time,NULL,NULL,NULL,NULL,NULL,NULL,&has,F);CHKERRQ(ierr);
  /* force recomputation of RHS Jacobian XXX CHECK WITH RK FOR CACHING */
  adjts->rhsjacobian.time = PETSC_MIN_REAL;
  ierr = TSComputeRHSJacobian(adjts,time,U,adjts->Arhs,adjts->Brhs);CHKERRQ(ierr);
  if (has) {
    ierr = VecScale(F,-1.0);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(adjts->Arhs,U,F,F);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(adjts->Arhs,U,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Given the forward DAE : H(U,Udot,t) = 0

   -> the first-order adjoint DAE is : F - L^T * (H_U - d/dt H_Udot) - Ldot^T H_Udot = 0 (in backward time)
      the first-order adjoint DAE is : Ldot^T H_Udot + L^T * (H_U + d/dt H_Udot) + F = 0 (in forward time)
      with F = dObjectiveIntegrand/dU (O_U in short)
   -> the second-order adjoint DAE differs only by the forcing term :
      F = O_UM * direction + O_UU * eta + (L \otimes I_N)(tH_UM * direction + tH_UU * eta + tH_UUdot * etadot) +
                                        - (Ldot \otimes I_N)(tH_UdotM * direction + tH_UdotU * eta + tH_UdotUdot * etadot) +
      with eta the solution of the tangent linear model and tH_U = H_U + d/dt H_Udot

   TODO : add support for augmented system when d/dt H_Udot != 0 ?
*/
static PetscErrorCode AdjointTSIFunctionLinear(TS adjts, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  Mat            J_U = NULL, J_Udot = NULL;
  PetscReal      fwdt;
  PetscBool      has = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  ierr = AdjointTSComputeForcing(adjts,time,NULL,NULL,NULL,NULL,NULL,NULL,&has,F);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSUpdateSplitJacobiansFromHistory_Private(adj_ctx->fwdts,fwdt);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL,&J_Udot,NULL);CHKERRQ(ierr);
  if (has) {
    ierr = MatMultTransposeAdd(J_U,U,F,F);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(J_U,U,F);CHKERRQ(ierr);
  }
  ierr = MatMultTransposeAdd(J_Udot,Udot,F,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* use U and Udot to sample the Jacobian for discrete adjoints */
static PetscErrorCode AdjointTSIJacobian(TS adjts, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  if (!adj_ctx->discrete) {
    ierr = TSComputeIJacobianWithSplits_Private(adj_ctx->fwdts,time,U,Udot,shift,A,B,ctx);CHKERRQ(ierr);
  } else {
    ierr = TSComputeIJacobian(adj_ctx->fwdts,time,U,Udot,shift,A,B,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Handles the detection of Dirac's delta forcing terms in the adjoint equations
    - first-order adjoint f_x(state,design,t = fixed)
    - second-order adjoint f_xx(state,design,t = fixed) or f_xm(state,design,t = fixed)
*/
static PetscErrorCode AdjointTSEventFunction(TS adjts, PetscReal t, Vec U, PetscScalar fvalue[], void *ctx)
{
  AdjointCtx     *adj_ctx;
  TSObj          link;
  PetscInt       cnt = 0;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - t + adj_ctx->t0;
  link = adj_ctx->tsobj;
  if (adj_ctx->direction) { /* second-order adjoint */
    while (link) { fvalue[cnt++] = ((link->f_xx || link->f_xm) && link->fixedtime > PETSC_MIN_REAL) ?  link->fixedtime - fwdt : 1.0; link = link->next; }
  } else if (adj_ctx->design) { /* gradient computations */
    while (link) { fvalue[cnt++] = (link->f_x && link->fixedtime > PETSC_MIN_REAL) ?  link->fixedtime - fwdt : 1.0; link = link->next; }
  }
  PetscFunctionReturn(0);
}

/* Dirac's delta integration H_Udot^T ( L(+) - L(-) )  = - f_U -> L(+) = - H_Udot^-T f_U + L(-)
   We store the increment - H_Udot^-T f_U in adj_ctx->workinit and apply it during the AdjointTSPostStep
   AdjointTSComputeInitialConditions supports index-1 DAEs too (singular H_Udot).
*/
static PetscErrorCode AdjointTSPostEvent(TS adjts, PetscInt nevents, PetscInt event_list[], PetscReal t, Vec U, PetscBool forwardsolve, void* ctx)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  ierr = VecLockReadPush(U);CHKERRQ(ierr);
  ierr = AdjointTSComputeInitialConditions(adjts,NULL,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecLockReadPop(U);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  adj_ctx->dirac_delta = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSPostStep(TS adjts)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  if (adjts->reason < 0) PetscFunctionReturn(0);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  /* We detected Dirac's delta terms -> add the increment here */
  if (adj_ctx->dirac_delta) {
    Vec lambda;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = VecAXPY(lambda,1.0,adj_ctx->workinit);CHKERRQ(ierr);
  }
  adj_ctx->dirac_delta = PETSC_FALSE;
  /* the sanity checks comparisons need the exact floating point value */
  if (adjts->reason == TS_CONVERGED_TIME) {
    PetscReal time;

    ierr = TSGetTime(adjts,&time);CHKERRQ(ierr);
    adj_ctx->tf = time;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AdjointTSSetUpStep(TS adjts)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscCheckAdjointTS(adjts);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  if (!adj_ctx->cstep) adj_ctx->cstep = adjts->ops->step;
  if (adj_ctx->discrete) {
    PetscBool rk,theta,cn;

    ierr = PetscObjectTypeCompare((PetscObject)adjts,TSRK,&rk);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)adjts,TSTHETA,&theta);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)adjts,TSCN,&cn);CHKERRQ(ierr);
    if (rk)               adjts->ops->step = TSStep_Adjoint_RK;
    else if (theta || cn) adjts->ops->step = TSStep_Adjoint_Theta;
    else {
      TSType tstype;

      ierr = TSGetType(adjts,&tstype);CHKERRQ(ierr);
      SETERRQ1(PetscObjectComm((PetscObject)adjts),PETSC_ERR_SUP,"Discrete adjoint not available for type %s\n",tstype);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSSetUp(TS adjts)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AdjointTSSetUpStep(adjts);CHKERRQ(ierr);
  PetscCheckAdjointTS(adjts);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  if (adj_ctx->setup) { ierr = (*adj_ctx->setup)(adjts);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSOptionsHandler(PetscOptionItems *PetscOptionsObject,PetscObject obj,void *ctx)
{
  TS             adjts = (TS)obj;
  AdjointCtx     *adj_ctx;
  PetscContainer container;
  PetscBool      jcon,rksp,flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"AdjointTS options");CHKERRQ(ierr);
  jcon = PETSC_FALSE;
  ierr = PetscOptionsBool("-constjacobians","Whether or not the DAE Jacobians are constant",NULL,jcon,&jcon,NULL);CHKERRQ(ierr);
  rksp = PETSC_FALSE;
  ierr = PetscOptionsBool("-reuseksp","Reuse the KSP solver from the nonlinear model",NULL,rksp,&rksp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-discrete","Use discrete adjoints (not available for all methods)",NULL,adj_ctx->discrete,&adj_ctx->discrete,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = AdjointTSSetUpStep(adjts);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_splitJac",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    TSSplitJacobians *splitJ;

    ierr = PetscContainerGetPointer(container,(void**)&splitJ);CHKERRQ(ierr);
    splitJ->jacconsts = jcon;
  }
  if (jcon) {
    TSRHSFunction rhsfunc;

    ierr = TSGetRHSFunction(adjts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
    if (rhsfunc) { /* just to make sure we have a correct Jacobian */
      DM  dm;
      Mat A,B;
      Vec U;

      ierr = TSGetRHSMats_Private(adj_ctx->fwdts,&A,&B);CHKERRQ(ierr);
      ierr = TSGetDM(adj_ctx->fwdts,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobian(adj_ctx->fwdts,PETSC_MIN_REAL,U,A,B);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(adjts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    }
  }
  if (rksp) { /* reuse the same KSP */
    SNES snes;
    KSP  ksp;

    ierr = TSGetSNES(adj_ctx->fwdts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = TSGetSNES(adjts,&snes);CHKERRQ(ierr);
    ierr = SNESSetKSP(snes,ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AdjointTSIsDiscrete(TS adjts, PetscBool *flg)
{
  PetscContainer c;
  AdjointCtx     *adj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscValidPointer(flg,2);
  ierr = AdjointTSSetUpStep(adjts);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj);CHKERRQ(ierr);
  *flg = adj->discrete;
  PetscFunctionReturn(0);
}

/*@C
   TSCreateAdjointTS - Creates a TS object that can be used to solve the adjoint DAE.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
-  ts - the model TS context obtained from TSCreate()

   Output Parameters:
-  adjts - the new TS context for the adjoint DAE

   Options Database Keys:
+  -adjoint_constjacobians <0> - if the Jacobians are constant
-  -adjoint_reuseksp <0> - if the AdjointTS should reuse the same KSP object used to solve the model DAE

   Notes: Given the DAE in implicit form F(t,x,xdot) = 0, the AdjointTS solves the linear DAE F_xdot^T L_dot + (F_x - d/dt F_xdot)^T L + forcing = 0.

          Note that the adjoint DAE is solved forward in time.

          Both the IFunction/IJacobian and the RHSFunction/RHSJacobian interfaces are supported.

          The Jacobians needed to perform the adjoint solve are automatically constructed for the case d/dt F_xdot = 0. Alternatively, the user can compose the "TSComputeSplitJacobians_C" function in the model TS to compute them.

          The forcing term depends on the objective functions set via TSAddObjective() and, for second-order adjoints, on some of the Hessian terms set with TSSetHessianDAE(). The forcing term is zero if those functions are not called on the model TS before a TSSolve() with the AdjointTS.

          AdjointTSSetTimeLimits() must be called before TSSolve() on the AdjointTS for gradient and Hessian computations.
          Initialization of the adjoint variables is automatically performed within AdjointTSComputeInitialConditions().

          For gradient computations, the linearized dependency of the DAE on the parameters must be set with TSSetGradientDAE() on the model TS.
          Use AdjointTSSetDirectionVec() for second-order adjoints; in this case, second order information must be attached to the model TS with TSSetHessianDAE() and TSSetHessianIC().

          Initial condition dependency for the model TS must be provided via TSSetGradientIC() for gradient computations and with both TSSetGradientIC() and TSSetHessianIC() for second-order adjoints.

          For nonzero forcing terms, the design vector for the current paramaters of the DAE must be passed via AdjointTSSetDesign().
          Gradients and Hessian matrix-vector results are obtained through a quadrature (currently trapezoidal rule); relevant API is AdjointTSSetQuadratureVec(), AdjointTSFinalizeQuadrature().

          The AdjointTS inherits the prefix of the model TS. E.g. if the model TS has prefix "burgers", the options prefix for the AdjointTS is -adjoint_burgers_. The user needs to call TSSetFromOptions() on the AdjointTS to trigger its customization.

   References:
.vb
   [1] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: algorithms and software, JCAM 149, 2002.
   [2] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: the adjoint DAE system and its numerical solution, SISC 24, 2003.
   [3] Ozyurt, Barton. Cheap second order directional derivatives of stiff ODE embedded functionals, SISC 26, 2005.
.ve

   Level: developer

.seealso: AdjointTSSetQuadratureVec(), AdjointTSFinalizeQuadrature(), AdjointTSSetDesignVec(), AdjointTSSetDirectionVec(), AdjointTSComputeInitialConditions(), TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC(), TSComputeSplitJacobians()
@*/
PetscErrorCode TSCreateAdjointTS(TS ts, TS* adjts)
{
  SNES             snes;
  KSP              ksp;
  Mat              A,B;
  Vec              vatol,vrtol;
  PetscContainer   container;
  AdjointCtx       *adj;
  TSIFunction      ifunc;
  TSRHSFunction    rhsfunc;
  TSI2Function     i2func;
  TSEquationType   eqtype;
  const char       *prefix;
  KSPType          ksptype;
  PetscReal        atol,rtol,dtol;
  PetscInt         maxits;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = TSGetEquationType(ts,&eqtype);CHKERRQ(ierr);
  if (eqtype != TS_EQ_UNSPECIFIED && eqtype != TS_EQ_EXPLICIT && eqtype != TS_EQ_ODE_EXPLICIT &&
      eqtype != TS_EQ_IMPLICIT && eqtype != TS_EQ_ODE_IMPLICIT && eqtype != TS_EQ_DAE_SEMI_EXPLICIT_INDEX1)
      SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSEquationType %D\n",eqtype);
  ierr = TSGetI2Function(ts,NULL,&i2func,NULL);CHKERRQ(ierr);
  if (i2func) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Second order DAEs are not supported");
  ierr = TSCreateWithTS(ts,adjts);CHKERRQ(ierr);
  ierr = TSGetTolerances(ts,&atol,&vatol,&rtol,&vrtol);CHKERRQ(ierr);
  ierr = TSSetTolerances(*adjts,atol,vatol,rtol,vrtol);CHKERRQ(ierr);
  if (ts->adapt) {
    ierr = TSAdaptCreate(PetscObjectComm((PetscObject)*adjts),&(*adjts)->adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType((*adjts)->adapt,((PetscObject)ts->adapt)->type_name);CHKERRQ(ierr);
  }

  /* application context */
  ierr = PetscNew(&adj);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*adjts,(void *)adj);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  adj->fwdts = ts;
  adj->setup = (*adjts)->ops->setup;
  (*adjts)->ops->setup = AdjointTSSetUp;

  /* TODO: this needs a better sharing mechanism */
  ierr = TSGetTSObj(ts,&adj->tsobj);CHKERRQ(ierr);

  /* invalidate time limits, that need to be set by AdjointTSSetTimeLimits */
  adj->t0 = adj->tf = PETSC_MAX_REAL;

  /* wrap application context in a container, so that it will be destroyed when calling TSDestroy on adjts */
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(*adjts)),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,adj);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,AdjointTSDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*adjts),"_ts_adjctx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  /* AdjointTS prefix: i.e. options called as -adjoint_ts_monitor or -adjoint_fwdtsprefix_ts_monitor */
  ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(*adjts,"adjoint_");CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*adjts,prefix);CHKERRQ(ierr);

  /* setup callbacks for adjoint DAE: we reuse the same jacobian matrices of the forward solve */
  ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
  ierr = TSGetRHSFunction(ts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (ifunc) {
    TSSplitJacobians *splitJ;

    ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
    ierr = TSSetIFunction(*adjts,NULL,AdjointTSIFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(*adjts,A,B,AdjointTSIJacobian,NULL);CHKERRQ(ierr);
    /* setup _ts_splitJac container */
    ierr = TSGetSplitJacobians(ts,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    /* caching to prevent from recomputation of Jacobians */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&container);CHKERRQ(ierr);
    if (!container) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing _ts_splitJac container");
    ierr = PetscContainerGetPointer(container,(void**)&splitJ);CHKERRQ(ierr);

    splitJ->splitdone = PETSC_FALSE;
  } else {
    TSRHSJacobian rhsjacfunc;

    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");
    ierr = TSSetRHSFunction(*adjts,NULL,AdjointTSRHSFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobian(ts,NULL,NULL,&rhsjacfunc,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSMats_Private(ts,&A,&B);CHKERRQ(ierr);
    if (rhsjacfunc == TSComputeRHSJacobianConstant) {
      ierr = TSSetRHSJacobian(*adjts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(*adjts,A,B,AdjointTSRHSJacobian,NULL);CHKERRQ(ierr);
    }
  }

  /* the adjoint DAE is linear */
  ierr = TSSetProblemType(*adjts,TS_LINEAR);CHKERRQ(ierr);

  /* use KSPSolveTranspose to solve the adjoint */
  if (ts->snes) {
    ierr = TSGetSNES(*adjts,&snes);CHKERRQ(ierr);
    ierr = SNESSetType(snes,SNESKSPTRANSPOSEONLY);CHKERRQ(ierr);

    /* adjointTS linear solver */
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
    ierr = KSPGetTolerances(ksp,&rtol,&atol,&dtol,&maxits);CHKERRQ(ierr);
    ierr = TSGetSNES(*adjts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    if (ksptype) { ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr); }
    ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxits);CHKERRQ(ierr);
  }

  /* set special purpose post step method for handling of discontinuities */
  ierr = TSSetPostStep(*adjts,AdjointTSPostStep);CHKERRQ(ierr);

  /* handle specific AdjointTS options */
  ierr = PetscObjectAddOptionsHandler((PetscObject)(*adjts),AdjointTSOptionsHandler,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSGetModelTS - Gets the forward model TS.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
-  ats - the adjoint TS context obtained from TSCreateAdjointTS()

   Output Parameters:
-  fts - the TS context for the forward DAE

   Level: developer

.seealso: TSCreateAdjointTS()
@*/
PetscErrorCode AdjointTSGetModelTS(TS ats, TS* fts)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode (*f)(TS,TS*);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ats,TS_CLASSID,1);
  PetscValidPointer(fts,2);
  ierr = PetscObjectQueryFunction((PetscObject)ats,"AdjointTSGetModelTS_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ats,fts);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscCheckAdjointTS(ats);
  ierr = TSGetApplicationContext(ats,(void*)&adj_ctx);CHKERRQ(ierr);
  *fts = adj_ctx->fwdts;
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSComputeForcing - Computes the forcing term for the AdjointTS

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  adjts - the adjoint TS context obtained from TSCreateAdjointTS()
.  time - the current backward time
.  U - state vector used to sample the forcing term (can be NULL)
.  Udot - state vector used to sample the forcing term (can be NULL)
.  L - adjoint vector used to sample the forcing term (can be NULL, only used for second order adjoints)
.  Ldot - adjoint vector used to sample the forcing term (can be NULL, only used for second order adjoints)
.  lU - TLM vector used to sample the forcing term (can be NULL, only used for second order adjoints)
-  lUdot - TLM vector used to sample the forcing term (can be NULL, only used for second order adjoints)

   Output Parameters:
+  hasf - PETSC_TRUE if F contains valid data
-  F - the output vector

   Level: developer

   Notes: If any of the vectors are NULL, AdjointTS computes the sampling data from the relative trajectory.

.seealso: TSCreateAdjointTS()
@*/
PetscErrorCode AdjointTSComputeForcing(TS adjts, PetscReal time, Vec U, Vec Udot, Vec L, Vec Ldot, Vec lU, Vec lUdot, PetscBool* hasf, Vec F)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec,PetscBool*,Vec);
  TSOpt          tsopt;
  PetscReal      fwdt;
  PetscBool      has = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(adjts,time,2);
  if (U) PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidPointer(hasf,4);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  ierr = PetscObjectQueryFunction((PetscObject)adjts,"AdjointTSComputeForcing_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(adjts,time,U,Udot,L,Ldot,lU,lUdot,hasf,F);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscCheckAdjointTS(adjts);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSGetTSOpt(adj_ctx->fwdts,&tsopt);CHKERRQ(ierr);
  if (adj_ctx->direction) { /* second-order adjoint */
    TS        fwdts = adj_ctx->fwdts;
    TS        tlmts = adj_ctx->tlmts;
    TS        foats = adj_ctx->foats;
    DM        dm;
    Vec       soawork0,soawork1;
    Vec       FWDH,TLMH;
    PetscBool hast,HFhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                                  {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                                  {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};

    ierr = VecSet(F,0.0);CHKERRQ(ierr);
    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork1);CHKERRQ(ierr);
    if (!U) { ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH,NULL);CHKERRQ(ierr); }
    else FWDH = U;
    if (!lU) { ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,&TLMH,NULL);CHKERRQ(ierr); }
    else TLMH = lU;
    ierr = TSObjEval_UU(adj_ctx->tsobj,FWDH,adj_ctx->design,fwdt,TLMH,soawork0,&has,soawork1);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
    }
    ierr = TSObjEval_UM(adj_ctx->tsobj,FWDH,adj_ctx->design,fwdt,adj_ctx->direction,soawork0,&hast,soawork1);CHKERRQ(ierr);
    if (hast) {
      ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
      has  = PETSC_TRUE;
    }

    if (foats) { /* if present, not a Gauss-Newton Hessian */
      TSIFunction ifunc;
      PetscInt    i;

      ierr = TSGetIFunction(fwdts,NULL,&ifunc,NULL);CHKERRQ(ierr);
      ierr = TSOptHasHessianDAE(tsopt,HFhas);CHKERRQ(ierr);
      if (!ifunc) for (i=0;i<3;i++) HFhas[i][1] = HFhas[1][i] = PETSC_FALSE;
    }
    if (HFhas[0][0] || HFhas[0][1] || HFhas[0][2]) {
      Vec FWDHdot,FOAH;

      if (!U && !Udot) { ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,NULL,&FWDHdot);CHKERRQ(ierr); }
      else FWDHdot = Udot;
      if (!L) { ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,time,&FOAH,NULL);CHKERRQ(ierr); }
      else FOAH = L;
      if (HFhas[0][0]) { /* (L^T \otimes I_N) H_XX \eta, \eta the TLM solution */
        ierr = TSOptEvalHessianDAE(tsopt,0,0,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,TLMH,soawork1);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (HFhas[0][1]) { /* (L^T \otimes I_N) H_XXdot \etadot, \eta the TLM solution */
        Vec TLMHdot;

        if (!lUdot) { ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,NULL,&TLMHdot);CHKERRQ(ierr); }
        else TLMHdot = lUdot;
        ierr = TSOptEvalHessianDAE(tsopt,0,1,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,TLMHdot,soawork1);CHKERRQ(ierr);
        if (!lUdot) { ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,NULL,&TLMHdot);CHKERRQ(ierr); }
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (HFhas[0][2]) { /* (L^T \otimes I_N) H_XM direction */
        ierr = TSOptEvalHessianDAE(tsopt,0,2,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,adj_ctx->direction,soawork1);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (!U && !Udot) { ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,NULL,&FWDHdot);CHKERRQ(ierr); }
      if (!L) { ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,&FOAH,NULL);CHKERRQ(ierr); }
    }
    /* these terms are computed against Ldot ->
       The formulas have a minus sign in front of them, but this cancels with time inversion of Ldot */
    if (HFhas[1][0] || HFhas[1][1] || HFhas[1][2]) {
      Vec FOAHdot,FWDHdot;

      if (!U && !Udot) { ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,NULL,&FWDHdot);CHKERRQ(ierr); }
      else FWDHdot = Udot;
      if (!Ldot) { ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,time,NULL,&FOAHdot);CHKERRQ(ierr); }
      else FOAHdot = Ldot;
      if (HFhas[1][0]) { /* (Ldot^T \otimes I_N) H_XdotX \eta, \eta the TLM solution */
        ierr = TSOptEvalHessianDAE(tsopt,1,0,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAHdot,TLMH,soawork1);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (HFhas[1][1]) { /* (Ldot^T \otimes I_N) H_XdotXdot \etadot, \eta the TLM solution */
        Vec TLMHdot;

        if (!lUdot) { ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,NULL,&TLMHdot);CHKERRQ(ierr); }
        else TLMHdot = lUdot;
        ierr = TSOptEvalHessianDAE(tsopt,1,1,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAHdot,TLMHdot,soawork1);CHKERRQ(ierr);
        if (!lUdot) { ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,NULL,&TLMHdot);CHKERRQ(ierr); }
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (HFhas[1][2]) { /* (Ldot^T \otimes I_N) H_XdotM direction */
        ierr = TSOptEvalHessianDAE(tsopt,1,2,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAHdot,adj_ctx->direction,soawork1);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (!U && !Udot) { ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,NULL,&FOAHdot);CHKERRQ(ierr); }
      if (!Ldot) { ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,NULL,&FWDHdot);CHKERRQ(ierr); }
    }
    if (!U)  { ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH,NULL);CHKERRQ(ierr); }
    if (!lU) { ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,&TLMH,NULL);CHKERRQ(ierr); }
    ierr = DMRestoreGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork1);CHKERRQ(ierr);
  } else if (adj_ctx->design) { /* gradient computations */
    TS  fwdts = adj_ctx->fwdts;
    DM  dm;
    Vec FWDH,W;

    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
    if (!U) {
      ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH,NULL);CHKERRQ(ierr);
    } else FWDH = U;
    ierr = TSObjEval_U(adj_ctx->tsobj,FWDH,adj_ctx->design,fwdt,W,&has,F);CHKERRQ(ierr);
    if (!U) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH,NULL);CHKERRQ(ierr);
    }
    ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
  }
  *hasf = has;
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSSetQuadratureVec - Sets the vector to store the quadrature to be computed.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  adjts - the TS context obtained from TSCreateAdjointTS()
-  q     - the vector where to accumulate the quadrature computation

   Notes: The vector is not zeroed.
          Currently, two kind of quadratures are supported: gradient computations and Hessian matrix-vector products.
          Pass NULL if you want to destroy the quadrature vector stored inside the AdjointTS.

   Level: developer

.seealso: TSCreateAdjointTS(), AdjointTSFinalizeQuadrature(), AdjointTSSetDesignVec(), AdjointTSSetDirectionVec(), AdjointTSComputeInitialConditions(), TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode AdjointTSSetQuadratureVec(TS adjts, Vec q)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscCheckAdjointTS(adjts);
  if (q) {
    PetscValidHeaderSpecific(q,VEC_CLASSID,2);
    PetscCheckSameComm(adjts,1,q,2);
  }
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)q);CHKERRQ(ierr);
  ierr = VecDestroy(&adj_ctx->quadvec);CHKERRQ(ierr);
  adj_ctx->quadvec = q;
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSComputeInitialConditions - Initializes the adjoint variables.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  adjts - the TS context obtained from TSCreateAdjointTS()
.  svec  - optional state vector to be used to sample the relevant objective functions
-  apply - if the initial conditions must be applied

   Notes: svec is needed only when we perform MatMultTranspose with the MatPropagator().
          ODEs and index-1 DAEs are supported for gradient computations.
          For Hessian computations, index-1 DAEs are not supported for point-form functionals.
          AdjointTSSetTimeLimits() should be called first.

   Level: developer

.seealso: TSCreateAdjointTS(), AdjointTSSetQuadratureVec(). AdjointTSFinalizeQuadrature(), AdjointTSSetDesignVec(), AdjointTSSetDirectionVec(), AdjointTSSetTimeLimits(), TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode AdjointTSComputeInitialConditions(TS adjts, Vec svec, PetscBool apply)
{
  PetscReal      fwdt,time;
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  TSOpt          tsopt;
  PetscErrorCode ierr;
  TSIJacobian    ijac;
  PetscBool      has_g = PETSC_FALSE;
  TSEquationType eqtype;
  PetscBool      rsve = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscCheckAdjointTS(adjts);
  if (svec) {
    PetscValidHeaderSpecific(svec,VEC_CLASSID,2);
    PetscCheckSameComm(adjts,1,svec,2);
  }
  PetscValidLogicalCollectiveBool(adjts,apply,3);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (adj_ctx->direction && !adj_ctx->tlmts) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Missing TLMTS! You need to call AdjointTSSetTLMTSAndFOATS");
  ierr = TSGetTSOpt(adj_ctx->fwdts,&tsopt);CHKERRQ(ierr);
  ierr = TSGetTime(adjts,&time);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  if (!svec) {
    ierr = TSTrajectoryGetUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,&svec,NULL);CHKERRQ(ierr);
    rsve = PETSC_TRUE;
  }
  {
    Vec L;
    ierr = TSGetSolution(adjts,&L);CHKERRQ(ierr);
    if (!L) {
      Vec U;
      ierr = TSGetSolution(adj_ctx->fwdts,&U);CHKERRQ(ierr);
      ierr = VecDuplicate(U,&L);CHKERRQ(ierr);
      ierr = TSSetSolution(adjts,L);CHKERRQ(ierr);
      ierr = VecDestroy(&L);CHKERRQ(ierr);
    }
  }
  /* only AdjointTSPostEvent and AdjointTSComputeInitialConditions can modify workinit */
  if (!adj_ctx->workinit) {
    Vec lambda;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = VecDuplicate(lambda,&adj_ctx->workinit);CHKERRQ(ierr);
    ierr = VecLockReadPush(adj_ctx->workinit);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(adj_ctx->workinit);CHKERRQ(ierr);
  ierr = VecSet(adj_ctx->workinit,0.0);CHKERRQ(ierr);

  if (adj_ctx->direction) {
    TS        fwdts = adj_ctx->fwdts;
    TS        tlmts = adj_ctx->tlmts;
    TS        foats = adj_ctx->foats;
    DM        dm;
    Vec       soawork0,soawork1,TLMH[2];
    PetscBool hast,HFhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                                  {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                                  {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};

    ierr = TSGetDM(adj_ctx->fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork1);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,&TLMH[0],&TLMH[1]);CHKERRQ(ierr);
    ierr = TSObjEvalFixed_UU(adj_ctx->tsobj,svec,adj_ctx->design,fwdt,TLMH[0],soawork0,&has_g,soawork1);CHKERRQ(ierr);
    if (has_g) {
      ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
    }
    ierr = TSObjEvalFixed_UM(adj_ctx->tsobj,svec,adj_ctx->design,fwdt,adj_ctx->direction,soawork0,&hast,soawork1);CHKERRQ(ierr);
    if (hast) {
      ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
      has_g = PETSC_TRUE;
    }
    if (rsve) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&svec,NULL);CHKERRQ(ierr);
      rsve = PETSC_FALSE;
    }
    /* these terms need to be evaluated only if we have point-form functionals,
       and only in the case of full Hessians for continuous adjoints */
    if (foats && !adj_ctx->discrete) {
      ierr = TSOptHasHessianDAE(tsopt,HFhas);CHKERRQ(ierr);
    }
    if (has_g && (HFhas[1][0] || HFhas[1][1] || HFhas[1][2])) {
      Vec FOAH,FWDH[2];

      ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,time,&FOAH,NULL);CHKERRQ(ierr);
      if (HFhas[1][0]) { /* (L^T \otimes I_N) H_XdotX \eta, \eta the TLM solution */
        ierr  = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = TSOptEvalHessianDAE(tsopt,1,0,fwdt,FWDH[0],FWDH[1],adj_ctx->design,FOAH,TLMH[0],soawork1);CHKERRQ(ierr);
        ierr  = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
        has_g = PETSC_TRUE;
      }
      if (HFhas[1][1]) { /* (L^T \otimes I_N) H_XdotXdot \etadot, \eta the TLM solution */
        ierr  = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = TSOptEvalHessianDAE(tsopt,1,1,fwdt,FWDH[0],FWDH[1],adj_ctx->design,FOAH,TLMH[1],soawork1);CHKERRQ(ierr);
        ierr  = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
        has_g = PETSC_TRUE;
      }
      if (HFhas[1][2]) { /* (L^T \otimes I_N) H_XdotM direction */
        ierr  = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = TSOptEvalHessianDAE(tsopt,1,2,fwdt,FWDH[0],FWDH[1],adj_ctx->design,FOAH,adj_ctx->direction,soawork1);CHKERRQ(ierr);
        ierr  = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
        has_g = PETSC_TRUE;
      }
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,&FOAH,NULL);CHKERRQ(ierr);
    }
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,&TLMH[0],&TLMH[1]);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork1);CHKERRQ(ierr);
  } else if (adj_ctx->design) { /* gradient computations */
    TS  fwdts = adj_ctx->fwdts;
    DM  dm;
    Vec W;

    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
    ierr = TSObjEvalFixed_U(adj_ctx->tsobj,svec,adj_ctx->design,fwdt,W,&has_g,adj_ctx->workinit);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
    if (rsve) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&svec,NULL);CHKERRQ(ierr);
      rsve = PETSC_FALSE;
    }
  }
  ierr = TSGetEquationType(adj_ctx->fwdts,&eqtype);CHKERRQ(ierr);
  ierr = TSGetIJacobian(adjts,NULL,NULL,&ijac,NULL);CHKERRQ(ierr);
  if (eqtype == TS_EQ_DAE_SEMI_EXPLICIT_INDEX1 && adj_ctx->design && !adj_ctx->discrete) { /* details in [1,Section 4.2] */
    KSP       kspM,kspD;
    Mat       M = NULL,B = NULL,C = NULL,D = NULL,pM = NULL,pD = NULL;
    Mat       J_U,J_Udot,pJ_U,pJ_Udot;
    PetscInt  m,n,N;
    DM        dm;
    IS        diff = NULL,alg = NULL;
    Vec       f_x,W;
    PetscBool has_f;

    ierr = VecDuplicate(adj_ctx->workinit,&f_x);CHKERRQ(ierr);
    if (!svec) {
      ierr = TSTrajectoryGetUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,&svec,NULL);CHKERRQ(ierr);
      rsve = PETSC_TRUE;
    }
    ierr = TSGetDM(adj_ctx->fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
    ierr = TSObjEval_U(adj_ctx->tsobj,svec,adj_ctx->design,fwdt,W,&has_f,f_x);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
    if (rsve) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,&svec,NULL);CHKERRQ(ierr);
    }
    if (!has_f && !has_g) {
      ierr = VecDestroy(&f_x);CHKERRQ(ierr);
      goto initialize;
    }
    if (adj_ctx->direction) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_SUP,"Second order adjoint for INDEX-1 DAE not yet coded");
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_algebraic_is",(PetscObject*)&alg);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_differential_is",(PetscObject*)&diff);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_dae_BMat",(PetscObject*)&B);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_dae_CMat",(PetscObject*)&C);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjoint_index1_kspM",(PetscObject*)&kspM);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjoint_index1_kspD",(PetscObject*)&kspD);CHKERRQ(ierr);
    if (!kspD) {
      const char *prefix;

      ierr = TSGetOptionsPrefix(adjts,&prefix);CHKERRQ(ierr);
      ierr = KSPCreate(PetscObjectComm((PetscObject)adjts),&kspD);CHKERRQ(ierr);
      ierr = KSPSetTolerances(kspD,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(kspD,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(kspD,"index1_D_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(kspD);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)adjts,"_ts_adjoint_index1_kspD",(PetscObject)kspD);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)kspD);CHKERRQ(ierr);
    } else {
      ierr = KSPGetOperators(kspD,&D,&pD);CHKERRQ(ierr);
    }
    if (!kspM) {
      const char *prefix;

      ierr = TSGetOptionsPrefix(adjts,&prefix);CHKERRQ(ierr);
      ierr = KSPCreate(PetscObjectComm((PetscObject)adjts),&kspM);CHKERRQ(ierr);
      ierr = KSPSetTolerances(kspM,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(kspM,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(kspM,"index1_M_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(kspM);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)adjts,"_ts_adjoint_index1_kspM",(PetscObject)kspM);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)kspM);CHKERRQ(ierr);
    } else {
      ierr = KSPGetOperators(kspM,&M,&pM);CHKERRQ(ierr);
    }
    ierr = TSUpdateSplitJacobiansFromHistory_Private(adj_ctx->fwdts,fwdt);CHKERRQ(ierr);
    ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,&pJ_U,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(J_Udot,&m,&n);CHKERRQ(ierr);
    if (!diff) {
      if (alg) {
        ierr = ISComplement(alg,m,n,&diff);CHKERRQ(ierr);
      } else {
        ierr = MatChop(J_Udot,PETSC_SMALL);CHKERRQ(ierr);
        ierr = MatFindNonzeroRows(J_Udot,&diff);CHKERRQ(ierr);
        if (!diff) SETERRQ(PetscObjectComm((PetscObject)adj_ctx->fwdts),PETSC_ERR_USER,"The DAE does not appear to have algebraic variables");
      }
      ierr = PetscObjectCompose((PetscObject)adj_ctx->fwdts,"_ts_differential_is",(PetscObject)diff);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)diff);CHKERRQ(ierr);
    }
    if (!alg) {
      ierr = ISComplement(diff,m,n,&alg);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)adj_ctx->fwdts,"_ts_algebraic_is",(PetscObject)alg);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)alg);CHKERRQ(ierr);
    }
    ierr = ISGetSize(alg,&N);CHKERRQ(ierr);
    if (!N) SETERRQ(PetscObjectComm((PetscObject)adj_ctx->fwdts),PETSC_ERR_USER,"The DAE does not have algebraic variables");
    if (M) { ierr = PetscObjectReference((PetscObject)M);CHKERRQ(ierr); }
    if (B) { ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr); }
    if (C) { ierr = PetscObjectReference((PetscObject)C);CHKERRQ(ierr); }
    if (D) { ierr = PetscObjectReference((PetscObject)D);CHKERRQ(ierr); }
    ierr = MatCreateSubMatrix(J_Udot,diff,diff,M ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&M);CHKERRQ(ierr);
    if (pJ_Udot != J_Udot) {
      if (pM) { ierr = PetscObjectReference((PetscObject)pM);CHKERRQ(ierr); }
      ierr = MatCreateSubMatrix(pJ_Udot,diff,diff,pM ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&pM);CHKERRQ(ierr);
    } else {
      if (pM && pM != M) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Amat and Pmat don't match");
      ierr = PetscObjectReference((PetscObject)M);CHKERRQ(ierr);
      pM   = M;
    }
    ierr = MatCreateSubMatrix(J_U,diff,alg ,B ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(J_U,alg ,diff,C ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(J_U,alg ,alg ,D ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    if (pJ_U != J_U) {
      if (pD) { ierr = PetscObjectReference((PetscObject)pD);CHKERRQ(ierr); }
      ierr = MatCreateSubMatrix(pJ_U,alg,alg,pD ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&pD);CHKERRQ(ierr);
    } else {
      if (pD && pD != D) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Amat and Pmat don't match");
      ierr = PetscObjectReference((PetscObject)D);CHKERRQ(ierr);
      pD   = D;
    }
    ierr = PetscObjectCompose((PetscObject)adj_ctx->fwdts,"_ts_dae_BMat",(PetscObject)B);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)adj_ctx->fwdts,"_ts_dae_CMat",(PetscObject)C);CHKERRQ(ierr);

    /* we first compute the contribution of the g(x,T,p) terms,
       the initial conditions are consistent by construction with the adjointed algebraic constraints, i.e.
       B^T lambda_d + D^T lambda_a = 0 */
    if (has_g) {
      Vec       g_d,g_a;
      PetscReal norm;

      ierr = VecGetSubVector(adj_ctx->workinit,diff,&g_d);CHKERRQ(ierr);
      ierr = VecGetSubVector(adj_ctx->workinit,alg,&g_a);CHKERRQ(ierr);
      ierr = VecNorm(g_a,NORM_2,&norm);CHKERRQ(ierr);
      if (norm) {

        ierr = KSPSetOperators(kspD,D,pD);CHKERRQ(ierr);
        ierr = KSPSolveTranspose(kspD,g_a,g_a);CHKERRQ(ierr);
        ierr = VecScale(g_a,-1.0);CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(C,g_a,g_d,g_d);CHKERRQ(ierr);
        if (adj_ctx->quadvec) {
          Mat adjF_m;

          ierr = TSOptEvalGradientDAE(tsopt,fwdt,NULL,NULL,adj_ctx->design,NULL,&adjF_m);CHKERRQ(ierr);
          if (adjF_m) { /* add fixed term to the gradient */
            Mat       subadjF_m;
            IS        all;
            PetscInt  n,st;
            PetscBool hasop;

            ierr = MatGetOwnershipRange(adjF_m,&st,NULL);CHKERRQ(ierr);
            ierr = MatGetLocalSize(adjF_m,&n,NULL);CHKERRQ(ierr);
            ierr = ISCreateStride(PetscObjectComm((PetscObject)adjF_m),n,st,1,&all);CHKERRQ(ierr);
            ierr = MatCreateSubMatrix(adjF_m,all,alg,MAT_INITIAL_MATRIX,&subadjF_m);CHKERRQ(ierr);
            ierr = ISDestroy(&all);CHKERRQ(ierr);
            ierr = MatHasOperation(subadjF_m,MATOP_MULT_ADD,&hasop);CHKERRQ(ierr);
            if (hasop) {
              ierr = MatMultAdd(subadjF_m,g_a,adj_ctx->quadvec,adj_ctx->quadvec);CHKERRQ(ierr);
            } else {
              Vec w;

              ierr = VecDuplicate(adj_ctx->quadvec,&w);CHKERRQ(ierr);
              ierr = MatMult(subadjF_m,g_a,w);CHKERRQ(ierr);
              ierr = VecAXPY(adj_ctx->quadvec,1.0,w);CHKERRQ(ierr);
              ierr = VecDestroy(&w);CHKERRQ(ierr);
            }
            ierr = MatDestroy(&subadjF_m);CHKERRQ(ierr);
          }
        }
      }
      ierr = KSPSetOperators(kspM,M,pM);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspM,g_d,g_d);CHKERRQ(ierr);
      ierr = MatMultTranspose(B,g_d,g_a);CHKERRQ(ierr);
      ierr = KSPSetOperators(kspD,D,pD);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspD,g_a,g_a);CHKERRQ(ierr);
      ierr = VecScale(g_d,-1.0);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->workinit,diff,&g_d);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->workinit,alg,&g_a);CHKERRQ(ierr);
#if 0
      {
        Mat J_U;
        Vec test,test_a;
        PetscReal norm;

        ierr = VecDuplicate(adj_ctx->workinit,&test);CHKERRQ(ierr);
        ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL,NULL,NULL);CHKERRQ(ierr);
        ierr = MatMultTranspose(J_U,adj_ctx->workinit,test);CHKERRQ(ierr);
        ierr = VecGetSubVector(test,alg,&test_a);CHKERRQ(ierr);
        ierr = VecNorm(test_a,NORM_2,&norm);CHKERRQ(ierr);
        ierr = PetscPrintf(PetscObjectComm((PetscObject)test),"This should be zero %1.16e\n",norm);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(test,alg,&test_a);CHKERRQ(ierr);
        ierr = VecDestroy(&test);CHKERRQ(ierr);
      }
#endif
    }
    /* we then compute, and add, admissible initial conditions for the algebraic variables, since the rhs of the adjoint system will depend
       on the derivative of the intergrand terms in the objective function w.r.t to the state */
    if (has_f) {
      Vec f_a,lambda_a;

      ierr = VecGetSubVector(f_x,alg,&f_a);CHKERRQ(ierr);
      ierr = VecGetSubVector(adj_ctx->workinit,alg,&lambda_a);CHKERRQ(ierr);
      ierr = KSPSetOperators(kspD,D,pD);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspD,f_a,f_a);CHKERRQ(ierr);
      ierr = VecAXPY(lambda_a,-1.0,f_a);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->workinit,alg,&lambda_a);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(f_x,alg,&f_a);CHKERRQ(ierr);
    }
#if 0
    {
      Mat J_U;
      Vec test,test_a;
      PetscReal norm;

      ierr = VecDuplicate(adj_ctx->workinit,&test);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_U,adj_ctx->workinit,test);CHKERRQ(ierr);
      ierr = VecGetSubVector(test,alg,&test_a);CHKERRQ(ierr);
      ierr = VecNorm(test_a,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)test),"FINAL: This should be zero %1.16e\n",norm);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(test,alg,&test_a);CHKERRQ(ierr);
      ierr = VecDestroy(&test);CHKERRQ(ierr);
    }
#endif
    ierr = VecDestroy(&f_x);CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
    ierr = MatDestroy(&pM);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&pD);CHKERRQ(ierr);
  } else {
    if (has_g) {
      if (adj_ctx->discrete) { /* adjoint variable is scaled by -1.0 only to make it comparable with the continuous adjoint case */
        ierr = VecScale(adj_ctx->workinit,-1.0);CHKERRQ(ierr);
      } else {
        if (ijac) { /* lambda_T(T) = (J_Udot)^T D_x, D_x the gradients of the functionals that sample the solution at the final time */
          KSP       ksp;
          Mat       J_Udot, pJ_Udot;
          DM        dm;
          Vec       W;

          ierr = TSUpdateSplitJacobiansFromHistory_Private(adj_ctx->fwdts,fwdt);CHKERRQ(ierr);
          ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjointinit_ksp",(PetscObject*)&ksp);CHKERRQ(ierr);
          if (!ksp) {
            SNES       snes;
            PC         pc;
            KSPType    ksptype;
            PCType     pctype;
            const char *prefix;

            ierr = TSGetSNES(adjts,&snes);CHKERRQ(ierr);
            ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
            ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
            ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
            ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
            ierr = KSPCreate(PetscObjectComm((PetscObject)adjts),&ksp);CHKERRQ(ierr);
            ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
            if (ksptype) { ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr); }
            ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
            if (pctype) { ierr = PCSetType(pc,pctype);CHKERRQ(ierr); }
            ierr = TSGetOptionsPrefix(adjts,&prefix);CHKERRQ(ierr);
            ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
            ierr = KSPAppendOptionsPrefix(ksp,"initlambda_");CHKERRQ(ierr);
            ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
            ierr = PetscObjectCompose((PetscObject)adjts,"_ts_adjointinit_ksp",(PetscObject)ksp);CHKERRQ(ierr);
            ierr = PetscObjectDereference((PetscObject)ksp);CHKERRQ(ierr);
          }
          ierr = TSGetSplitJacobians(adj_ctx->fwdts,NULL,NULL,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
          ierr = KSPSetOperators(ksp,J_Udot,pJ_Udot);CHKERRQ(ierr);
          ierr = TSGetDM(adjts,&dm);CHKERRQ(ierr);
          ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
          ierr = VecCopy(adj_ctx->workinit,W);CHKERRQ(ierr);
          ierr = KSPSolveTranspose(ksp,W,adj_ctx->workinit);CHKERRQ(ierr);
          ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
          /* destroy inner vectors to avoid ABA issues when destroying the DM */
          ierr = VecDestroy(&ksp->vec_rhs);CHKERRQ(ierr);
          ierr = VecDestroy(&ksp->vec_sol);CHKERRQ(ierr);
        }
        /* the lambdas we use are equivalent to -lambda_T in [1] */
        ierr = VecScale(adj_ctx->workinit,-1.0);CHKERRQ(ierr);
      }
    }
  }
initialize:
  ierr = VecLockReadPush(adj_ctx->workinit);CHKERRQ(ierr);
  if (apply) {
    Vec lambda;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = VecCopy(adj_ctx->workinit,lambda);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSSetDesignVec - Sets the vector that stores the current design.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  adjts  - the TS context obtained from TSCreateAdjointTS()
-  design - the vector that stores the current values of the parameters

   Notes: The presence of the design vector activates the code for gradient or Hessian computations. Pass NULL if you want to destroy the design vector stored inside the AdjointTS.

   Level: developer

.seealso: TSCreateAdjointTS(), AdjointTSSetQuadratureVec(), AdjointTSSetDirectionVec(), TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode AdjointTSSetDesignVec(TS adjts, Vec design)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscCheckAdjointTS(adjts);
  if (design) {
    PetscValidHeaderSpecific(design,VEC_CLASSID,2);
    PetscCheckSameComm(adjts,1,design,2);
  }
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (design) {
    ierr = PetscObjectReference((PetscObject)design);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&adj_ctx->design);CHKERRQ(ierr);
  adj_ctx->design = design;
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSSetDirectionVec - Sets the vector that stores the input vector for the Hessian matrix vector product.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  adjts     - the TS context obtained from TSCreateAdjointTS()
-  direction - the vector

   Notes: The presence of the direction vector activates the code for the second-order adjoint. Pass NULL if you want to destroy the direction vector stored inside the AdjointTS.

   Level: developer

.seealso: TSCreateAdjointTS(), AdjointTSSetQuadratureVec(), AdjointTSSetDesignVec(), TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode AdjointTSSetDirectionVec(TS adjts, Vec direction)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscCheckAdjointTS(adjts);
  if (direction) {
    PetscValidHeaderSpecific(direction,VEC_CLASSID,2);
    PetscCheckSameComm(adjts,1,direction,2);
  }
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (direction) {
    ierr = PetscObjectReference((PetscObject)direction);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&adj_ctx->direction);CHKERRQ(ierr);
  adj_ctx->direction = direction;
  PetscFunctionReturn(0);
}

PetscErrorCode AdjointTSGetDirectionVec(TS adjts, Vec *direction)
{
  PetscContainer c;
  PetscErrorCode (*f)(TS,Vec*);
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscValidPointer(direction,2);
  ierr = PetscObjectQueryFunction((PetscObject)adjts,"AdjointTSGetDirectionVec_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(adjts,direction);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  *direction = adj_ctx->direction;
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSSetTLMTSAndFOATS - Sets the Tangent Linear Model TS and the first-order adjoint TS, needed for Hessian matrix-vector products.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  soats - the TS context obtained from TSCreateAdjointTS() (second order adjoint)
.  tlmts - the TS context obtained from TSCreateTLMTS()
-  foats - the TS context obtained from TSCreateAdjointTS() (can be NULL)

   Notes: To activate second order adjoints, you should call AdjointTSSetDirectionVec() first.

   Level: developer

.seealso: TSCreateAdjointTS(), AdjointTSSetQuadratureVec(), AdjointTSSetDesignVec(), TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode AdjointTSSetTLMTSAndFOATS(TS soats, TS tlmts, TS foats)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(soats,TS_CLASSID,1);
  PetscCheckAdjointTS(soats);
  PetscValidHeaderSpecific(tlmts,TS_CLASSID,2);
  PetscCheckSameComm(soats,1,tlmts,2);
  PetscCheckTLMTS(tlmts);
  if (foats) {
    PetscValidHeaderSpecific(foats,TS_CLASSID,3);
    PetscCheckSameComm(soats,1,foats,3);
    PetscCheckAdjointTS(foats);
  }
  if (soats == foats) SETERRQ(PetscObjectComm((PetscObject)soats),PETSC_ERR_SUP,"The two AdjointTS should be different");
  ierr = PetscObjectQuery((PetscObject)soats,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)tlmts);CHKERRQ(ierr);
  if (foats) {
    ierr = PetscObjectReference((PetscObject)foats);CHKERRQ(ierr);
  }
  ierr = TSDestroy(&adj_ctx->tlmts);CHKERRQ(ierr);
  ierr = TSDestroy(&adj_ctx->foats);CHKERRQ(ierr);
  adj_ctx->tlmts = tlmts;
  adj_ctx->foats = foats;
  PetscFunctionReturn(0);
}

PetscErrorCode AdjointTSGetTLMTSAndFOATS(TS adjts, TS *tlmts, TS *foats)
{
  PetscContainer c;
  PetscErrorCode (*f)(TS,TS*,TS*);
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  ierr = PetscObjectQueryFunction((PetscObject)adjts,"AdjointTSGetTLMTSAndFOATS_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(adjts,tlmts,foats);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (tlmts) *tlmts = adj_ctx->tlmts;
  if (foats) *foats = adj_ctx->foats;
  PetscFunctionReturn(0);
}
/*@C
   AdjointTSSetTimeLimits - Sets the forward time interval where to perform the adjoint simulation.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
+  adjts - the TS context obtained from TSCreateAdjointTS()
.  t0    - the initial time
-  tf    - the final time

   Notes: You should call AdjointTSSetTimeLimits() before any TSSolve() with the AdjointTS.
          The values are needed to recover the forward time from the backward.

   Level: developer

.seealso: TSCreateAdjointTS(), AdjointTSComputeInitialConditions(), TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode AdjointTSSetTimeLimits(TS adjts, PetscReal t0, PetscReal tf)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscCheckAdjointTS(adjts);
  PetscValidLogicalCollectiveReal(adjts,t0,2);
  PetscValidLogicalCollectiveReal(adjts,tf,3);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = TSSetTime(adjts,t0);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(adjts,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetMaxTime(adjts,tf);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(adjts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  adj_ctx->tf = tf;
  adj_ctx->t0 = t0;
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSEventHandler - Initializes the TSEvent() if needed.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
.  adjts - the TS context obtained from TSCreateAdjointTS()

   Notes: This needs to be called if you are evaluating gradients with point-form functionals sampled in between the simulation.

   Level: developer

.seealso: TSCreateAdjointTS(), TSAddObjective()
@*/
PetscErrorCode AdjointTSEventHandler(TS adjts)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;
  PetscInt       cnt;
  PetscBool      has;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscCheckAdjointTS(adjts);
  ierr = TSEventDestroy(&adjts->event);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = TSObjGetNumObjectives(adj_ctx->tsobj,&cnt);CHKERRQ(ierr);
  if (!cnt) PetscFunctionReturn(0);
  ierr = TSObjHasObjectiveFixed(adj_ctx->tsobj,adj_ctx->t0,adj_ctx->tf,NULL,&has,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (has) {
    PetscInt  *dir;
    PetscBool *term;

    ierr = PetscCalloc2(cnt,&dir,cnt,&term);CHKERRQ(ierr);
    ierr = TSSetEventHandler(adjts,cnt,dir,term,AdjointTSEventFunction,AdjointTSPostEvent,NULL);CHKERRQ(ierr);
    ierr = PetscFree2(dir,term);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   AdjointTSFinalizeQuadrature - Finalizes the quadrature to be computed, by adding the dependency on the initial conditions.

   Synopsis:
   #include <petsc/private/tsadjointtsimpl.h>

   Logically Collective on TS

   Input Parameters:
.  adjts - the TS context obtained from TSCreateAdjointTS()

   Notes: This needs to be called after a successful TSSolve() with the AdjointTS.

   Level: developer

.seealso: TSCreateAdjointTS(), AdjointTSSetQuadratureVec(), AdjointTSComputeInitialConditions(), TSAddObjective(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode AdjointTSFinalizeQuadrature(TS adjts)
{
  PetscReal      tf;
  AdjointCtx     *adj_ctx;
  PetscContainer c;
  TSOpt          tsopt;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscCheckAdjointTS(adjts);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (!adj_ctx->quadvec) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Missing quadrature vector. You should call AdjointTSSetQuadratureVec() first");
  if (!adj_ctx->design) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing design vector. You should call AdjointTSSetDesignVec() first");
  ierr = TSGetTime(adjts,&tf);CHKERRQ(ierr);
  if (PetscAbsReal(tf - adj_ctx->tf) > PETSC_SMALL) SETERRQ3(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Backward solve did not complete %1.14e (%g %g)\nMaybe you forgot to call AdjointTSSetTimeLimits() and TSSolve() on the AdjointTS",tf-adj_ctx->tf,tf,adj_ctx->tf);

  ierr = TSGetTSOpt(adj_ctx->fwdts,&tsopt);CHKERRQ(ierr);

  /* initial condition contribution to the gradient */
  ierr = TSOptHasGradientIC(tsopt,&has);CHKERRQ(ierr);
  if (has) {
    TS          fwdts = adj_ctx->fwdts;
    Vec         lambda, FWDH[2], work;
    TSIJacobian ijacfunc = NULL;
    Mat         J_Udot = NULL;
    DM          adm;

    ierr = TSGetDM(adjts,&adm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(adm,&work);CHKERRQ(ierr);
    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    if (!adj_ctx->discrete) { /* no need to adjust for IJacobian for discrete adjoints */
      ierr = TSGetIJacobian(adjts,NULL,NULL,&ijacfunc,NULL);CHKERRQ(ierr);
    }
    if (!ijacfunc) {
      ierr = VecCopy(lambda,work);CHKERRQ(ierr);
    } else {
      ierr = TSUpdateSplitJacobiansFromHistory_Private(fwdts,adj_ctx->t0);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(fwdts,NULL,NULL,&J_Udot,NULL);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_Udot,lambda,work);CHKERRQ(ierr);
    }
    if (!adj_ctx->wquad) {
      ierr = VecDuplicate(adj_ctx->quadvec,&adj_ctx->wquad);CHKERRQ(ierr);
    }
    if (!adj_ctx->direction) { /* first-order adjoint in gradient computations */
      ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,adj_ctx->t0,&FWDH[0],NULL);CHKERRQ(ierr);
      ierr = TSLinearizedICApply_Private(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,work,adj_ctx->wquad,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],NULL);CHKERRQ(ierr);
      ierr = VecAXPY(adj_ctx->quadvec,1.0,adj_ctx->wquad);CHKERRQ(ierr);
    } else { /* second-order adjoint in Hessian computations */
      TS        foats = adj_ctx->foats;
      TS        tlmts = adj_ctx->tlmts;
      Vec       soawork0 = NULL,soawork1 = NULL,FOAH = NULL,FWDH[2] = {NULL, NULL},TLMH,TLMHdot = NULL;
      PetscBool HGhas[2][2] = {{PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE}};
      PetscBool HFhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};

      if (foats) { /* Full (not Gauss-Newton) Hessian IC contributions */
        ierr = DMGetGlobalVector(adm,&soawork0);CHKERRQ(ierr);
        ierr = DMGetGlobalVector(adm,&soawork1);CHKERRQ(ierr);
        ierr = TSOptHasHessianIC(tsopt,HGhas);CHKERRQ(ierr);
        if (!adj_ctx->discrete) { /* for discrete adjoints, we only need to sample the IC terms */
          ierr = TSOptHasHessianDAE(tsopt,HFhas);CHKERRQ(ierr);
        }
        ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,adj_ctx->tf,&FOAH,NULL);CHKERRQ(ierr);
        ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,adj_ctx->t0,&TLMH,HFhas[1][1] ? &TLMHdot : NULL);CHKERRQ(ierr);
        ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,adj_ctx->t0,&FWDH[0],(HFhas[1][0] || HFhas[1][1] || HFhas[1][2]) ? &FWDH[1] : NULL);CHKERRQ(ierr);
        if (!J_Udot) {
          ierr = VecCopy(FOAH,soawork1);CHKERRQ(ierr);
        } else {
          ierr = MatMultTranspose(J_Udot,FOAH,soawork1);CHKERRQ(ierr);
        }
        /* XXX Hack to just solve for G_x (if any) -> compute mu */
        ierr = TSLinearizedICApply_Private(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork1,soawork0,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
        if (HGhas[0][0]) { /* (\mu^T \otimes I_N) G_XX \eta, \eta the TLM solution */
          ierr = TSOptEvalHessianIC(tsopt,0,0,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork0,TLMH,soawork1);CHKERRQ(ierr);
          ierr = VecAXPY(work,-1.0,soawork1);CHKERRQ(ierr);
        }
        if (HGhas[0][1]) { /* (\mu^T \otimes I_N) G_XM direction */
          ierr = TSOptEvalHessianIC(tsopt,0,1,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork0,adj_ctx->direction,soawork1);CHKERRQ(ierr);
          ierr = VecAXPY(work,-1.0,soawork1);CHKERRQ(ierr);
        }
        if (HFhas[1][0]) { /* (L^T \otimes I_N) H_XdotX \eta, \eta the TLM solution */
          ierr = TSOptEvalHessianDAE(tsopt,1,0,adj_ctx->t0,FWDH[0],FWDH[1],adj_ctx->design,FOAH,TLMH,soawork1);CHKERRQ(ierr);
          ierr = VecAXPY(work,1.0,soawork1);CHKERRQ(ierr);
        }
        if (HFhas[1][1]) { /* (L^T \otimes I_N) H_XdotXdot \etadot, \eta the TLM solution */
          ierr = TSOptEvalHessianDAE(tsopt,1,1,adj_ctx->t0,FWDH[0],FWDH[1],adj_ctx->design,FOAH,TLMHdot,soawork1);CHKERRQ(ierr);
          ierr = VecAXPY(work,1.0,soawork1);CHKERRQ(ierr);
        }
        if (HFhas[1][2]) { /* (L^T \otimes I_N) H_XdotM direction */
          ierr = TSOptEvalHessianDAE(tsopt,1,2,adj_ctx->t0,FWDH[0],FWDH[1],adj_ctx->design,FOAH,adj_ctx->direction,soawork1);CHKERRQ(ierr);
          ierr = VecAXPY(work,1.0,soawork1);CHKERRQ(ierr);
        }
      } else {
        ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,adj_ctx->t0,&FWDH[0],NULL);CHKERRQ(ierr);
      }
      /* With Gauss-Newton Hessians, all terms are zero except for this */
      ierr = TSLinearizedICApply_Private(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,work,adj_ctx->wquad,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
      ierr = VecAXPY(adj_ctx->quadvec,1.0,adj_ctx->wquad);CHKERRQ(ierr);
      if (foats) { /* full Hessian */
        if (HGhas[1][1]) { /* (\mu^T \otimes I_M) G_MM direction */
          ierr = TSOptEvalHessianIC(tsopt,1,1,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork0,adj_ctx->direction,adj_ctx->wquad);CHKERRQ(ierr);
          ierr = VecAXPY(adj_ctx->quadvec,1.0,adj_ctx->wquad);CHKERRQ(ierr);
        }
        if (HGhas[1][0]) { /* (\mu^T \otimes I_M) G_MX  \eta, \eta the TLM solution */
          ierr = TSOptEvalHessianIC(tsopt,1,0,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork0,TLMH,adj_ctx->wquad);CHKERRQ(ierr);
          ierr = VecAXPY(adj_ctx->quadvec,1.0,adj_ctx->wquad);CHKERRQ(ierr);
        }
        if (foats) {
          ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,&FOAH,NULL);CHKERRQ(ierr);
        }
        ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],FWDH[1] ? &FWDH[1] : NULL);CHKERRQ(ierr);
        ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,&TLMH,TLMHdot ? &TLMHdot : NULL);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(adm,&soawork0);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(adm,&soawork1);CHKERRQ(ierr);
      } else {
        ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],NULL);CHKERRQ(ierr);
      }
    }
    ierr = DMRestoreGlobalVector(adm,&work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <petscopt/augmentedts.h>

static PetscErrorCode JacCoupling(TS qts, PetscReal t, Vec L, Vec Ldot, PetscReal s, Mat A, Mat B, void* ctx)
{
  AdjEvalQuadCtx *adjq;
  TSQuadCtx      *qctx;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(qts,(void*)&qctx);CHKERRQ(ierr);
  adjq = qctx->evalquadctx;
  ierr = AdjointTSIsDiscrete(adjq->adjts,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)qts),PETSC_ERR_PLIB,"This should not happen");
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Ldot) {
    ierr = MatScale(A,-1.0);CHKERRQ(ierr);
    if (A && A != B) { ierr = MatScale(B,-1.0);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode __mat_shell_mult(Mat B, Vec x, Vec y)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(B,(void **)&A);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode __mat_shell_multtranspose(Mat B, Vec x, Vec y)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(B,(void **)&A);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode __mat_shell_destroy(Mat B)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(B,(void **)&A);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateShellWithMat(Mat A, PetscBool trans, Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       m,n,M,N;
  PetscInt       i;
  void           (*f[])(void) = { (void (*)())__mat_shell_mult,
                                  (void (*)())__mat_shell_multtranspose,
                                  (void (*)())__mat_shell_destroy };
  void           (*t[])(void) = { (void (*)())__mat_shell_multtranspose,
                                  (void (*)())__mat_shell_mult,
                                  (void (*)())__mat_shell_destroy };
  MatOperation   ops[] = { MATOP_MULT,
                           MATOP_MULT_TRANSPOSE,
                           MATOP_DESTROY };

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (trans) {
    ierr = MatCreateShell(PetscObjectComm((PetscObject)A),n,m,N,M,A,B);CHKERRQ(ierr);
  } else {
    ierr = MatCreateShell(PetscObjectComm((PetscObject)A),m,n,M,N,A,B);CHKERRQ(ierr);
  }
  for (i=0;i<sizeof(ops)/sizeof(MatOperation);i++) {
    ierr = MatShellSetOperation(*B,ops[i],trans ? t[i] : f[i]);CHKERRQ(ierr);
  }
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AdjointTSComputeQuadrature(TS ts, PetscReal t, Vec U, Vec Udot, Vec L, Vec FOAL, Vec FOALdot, Vec TLMU, Vec TLMUdot, PetscBool* has, Vec F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  if (Udot) PetscValidHeaderSpecific(Udot,VEC_CLASSID,4);
  PetscValidHeaderSpecific(L,VEC_CLASSID,5);
  if (FOAL) PetscValidHeaderSpecific(FOAL,VEC_CLASSID,6);
  if (FOALdot) PetscValidHeaderSpecific(FOALdot,VEC_CLASSID,7);
  if (TLMU) PetscValidHeaderSpecific(TLMU,VEC_CLASSID,8);
  if (TLMUdot) PetscValidHeaderSpecific(TLMUdot,VEC_CLASSID,9);
  PetscValidPointer(has,10);
  PetscValidHeaderSpecific(F,VEC_CLASSID,11);
  *has = PETSC_FALSE;
  ierr = PetscTryMethod(ts,"AdjointTSComputeQuadrature_C",(TS,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec,Vec,PetscBool*,Vec),
                                                          (ts,t,U,Udot,L,FOAL,FOALdot,TLMU,TLMUdot,has,F));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AdjointTSSolveWithQuadrature_Private(TS adjts)
{
  AdjointCtx     *adj_ctx;
  TSOpt          tsopt;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckAdjointTS(adjts);
  ierr = TSSetUp(adjts);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  ierr = TSGetTSOpt(adj_ctx->fwdts,&tsopt);CHKERRQ(ierr);
  ierr = TSOptHasGradientDAE(tsopt,&flg,NULL);CHKERRQ(ierr);
  if (!flg) {
    if (!adj_ctx->direction) {
      ierr = TSObjHasObjectiveIntegrand(adj_ctx->tsobj,NULL,NULL,&flg,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    if (adj_ctx->direction && adj_ctx->discrete) { /* for continuous adjoints, these terms are evaluated during the continuous TLM */
      TSIFunction ifunc;
      PetscBool   has1, has2, Hhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                                            {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                                            {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};

      ierr = TSObjHasObjectiveIntegrand(adj_ctx->tsobj,NULL,NULL,NULL,NULL,&has1,&has2);CHKERRQ(ierr);
      if (adj_ctx->foats) {
        ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
      }
      flg  = (PetscBool)(flg || has1 || has2);
      ierr = TSGetIFunction(adj_ctx->fwdts,NULL,&ifunc,NULL);CHKERRQ(ierr);
      if (!ifunc) Hhas[2][1] = PETSC_FALSE;
      flg = (PetscBool)(flg || Hhas[2][0] || Hhas[2][1] || Hhas[2][2]);
    }
  }
  if (flg) {
    TS             ats,qts;
    AdjEvalQuadCtx adjq;
    TSQuadCtx      qctx;
    Mat            Ac = NULL,Bc = NULL;
    PetscErrorCode (*qup)(TS,Vec,Vec) = QuadTSUpdateStates;
    TSIJacobian    coup = adj_ctx->discrete ? NULL : JacCoupling;
    PetscBool      diffrhs = (adjts->Arhs != adjts->Brhs) ? PETSC_TRUE : PETSC_FALSE;

    if (!adj_ctx->quadvec) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Missing quadrature vector. You should call AdjointTSSetQuadratureVec() first");

    adjq.adjts = adjts;
    ierr = VecDuplicate(adj_ctx->quadvec,&adjq.work1);CHKERRQ(ierr);
    ierr = VecDuplicate(adj_ctx->quadvec,&adjq.work2);CHKERRQ(ierr);

    qctx.evalquad       = EvalQuadIntegrand_ADJ;
    qctx.evalquad_fixed = NULL;
    qctx.evalquadctx    = &adjq;
    qctx.U              = NULL;
    qctx.Udot           = NULL;
    qctx.design         = NULL;

    ierr = TSCreateQuadTS(PetscObjectComm((PetscObject)adjts),adj_ctx->quadvec,diffrhs,&qctx,&qts);CHKERRQ(ierr);
    ierr = TSSetProblemType(qts,TS_LINEAR);CHKERRQ(ierr);

    if (!adj_ctx->discrete) {
      ierr = MatCreateShellWithMat(tsopt->adjF_m,PETSC_TRUE,&Ac);CHKERRQ(ierr);
      if (diffrhs) {
        ierr = MatCreateShellWithMat(tsopt->adjF_m,PETSC_TRUE,&Bc);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)Ac);CHKERRQ(ierr);
        Bc   = Ac;
      }
    }
    ierr = TSCreateAugmentedTS(adjts,1,&qts,NULL,&qup,&coup,&Ac,&Bc,PETSC_TRUE,&ats);CHKERRQ(ierr);
    ierr = MatDestroy(&Ac);CHKERRQ(ierr);
    ierr = MatDestroy(&Bc);CHKERRQ(ierr);
    ierr = TSDestroy(&qts);CHKERRQ(ierr);
    if (ats->snes) {
      ierr = SNESSetType(ats->snes,SNESKSPTRANSPOSEONLY);CHKERRQ(ierr);
    }
    ierr = TSSetFromOptions(ats);CHKERRQ(ierr);
    ierr = AugmentedTSInitialize(ats);CHKERRQ(ierr);
    ierr = TSSolve(ats,NULL);CHKERRQ(ierr);
    ierr = AugmentedTSFinalize(ats);CHKERRQ(ierr);
    ierr = TSDestroy(&ats);CHKERRQ(ierr);
    ierr = VecDestroy(&adjq.work1);CHKERRQ(ierr);
    ierr = VecDestroy(&adjq.work2);CHKERRQ(ierr);
  } else {
    ierr = TSSolve(adjts,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

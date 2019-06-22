#include <petscopt/private/tspdeconstrainedutilsimpl.h>
#include <petscopt/private/tsobjimpl.h>
#include <petscopt/private/tsoptimpl.h>
#include <petsc/private/tsimpl.h> /* adapt and poststep dependency */
#include <petsc/private/kspimpl.h>
#include <petsc/private/petscimpl.h>
#include <petscopt/tlmts.h>
#include <petscdm.h>

/* ------------------ Wrappers for quadrature evaluation ----------------------- */

PetscErrorCode TSQuadratureCtxDestroy_Private(void *ptr)
{
  TSQuadratureCtx* q = (TSQuadratureCtx*)ptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(5,&q->wquad);CHKERRQ(ierr);
  ierr = PetscFree(q->veval_ctx);CHKERRQ(ierr);
  ierr = PetscFree(q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  XXX_FWD are evaluated during the forward run
  XXX_TLM are evaluated during the tangent linear model run within Hessian computations
  XXX_ADJ are evaluated during the adjoint run
*/

typedef struct {
  TSObj obj;
  Vec   design;
  Vec   work;
} FWDEvalQuadCtx;

static PetscErrorCode EvalQuadObj_FWD(Vec U, PetscReal t, PetscReal *f, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEval(evalctx->obj,U,evalctx->design,t,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadObjFixed_FWD(Vec U, PetscReal t, PetscReal *f, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEvalFixed(evalctx->obj,U,evalctx->design,t,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrand_FWD(Vec U, PetscReal t, Vec F, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscBool      has_m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEval_M(evalctx->obj,U,evalctx->design,t,evalctx->work,&has_m,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrandFixed_FWD(Vec U, PetscReal t, Vec F, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscBool      has_m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEvalFixed_M(evalctx->obj,U,evalctx->design,t,evalctx->work,&has_m,F);CHKERRQ(ierr);
  if (!has_m) { /* can be called with a non-matching time */
    ierr = VecSet(F,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct {
  TSObj     obj; /* TODO : access via TSGetTSObj? */
  TS        fwdts;
  TS        adjts;
  TS        tlmts;
  PetscReal t0,tf;
  Vec       design;
  Vec       direction;
  Vec       work1;
  Vec       work2;
  PetscBool init;
} TLMEvalQuadCtx;

/* computes d^2 f / dp^2 direction + d^2 f / dp dx U + (L^T \otimes I_M)(H_MM direction + H_MU U + H_MUdot Udot) during TLM runs */
static PetscErrorCode EvalQuadIntegrand_TLM(Vec U, PetscReal t, Vec F, void* ctx)
{
  TLMEvalQuadCtx *q = (TLMEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  TS             adjts = q->adjts;
  TS             tlmts = q->tlmts;
  TSTrajectory   tj,atj = NULL,ltj;
  TSOpt          tsopt;
  Vec            FWDH[2],FOAH = NULL;
  PetscReal      adjt  = q->tf - t + q->t0;
  PetscBool      AXPY, rest = PETSC_FALSE;
  PetscBool      Hhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTrajectory(fwdts,&tj);CHKERRQ(ierr);
  if (adjts) { /* If not present, Gauss-Newton Hessian */
    ierr = TSGetTrajectory(adjts,&atj);CHKERRQ(ierr);
  }
  ierr = TSGetTrajectory(tlmts,&ltj);CHKERRQ(ierr);
  ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,fwdts,t,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  ierr = TSObjEval_MU(q->obj,FWDH[0],q->design,t,U,q->work1,&AXPY,F);CHKERRQ(ierr);
  if (!AXPY) {
    ierr = TSObjEval_MM(q->obj,FWDH[0],q->design,t,q->direction,q->work1,&AXPY,F);CHKERRQ(ierr);
  } else {
    PetscBool has;

    ierr = TSObjEval_MM(q->obj,FWDH[0],q->design,t,q->direction,q->work1,&has,q->work2);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr);
    }
  }
  ierr = TSGetTSOpt(fwdts,&tsopt);CHKERRQ(ierr);
  if (adjts) {
    ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
  }
  if (Hhas[2][0] || (Hhas[2][1] && q->init) || Hhas[2][2]) { /* Not for GN */
    rest = PETSC_TRUE;
    ierr = TSTrajectoryGetUpdatedHistoryVecs(atj,adjts,adjt,&FOAH,NULL);CHKERRQ(ierr);
  }
  if (Hhas[2][2]) { /* (L^T \otimes I_M) H_MM direction */
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,2,t,FWDH[0],FWDH[1],q->design,FOAH,q->direction,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,2,t,FWDH[0],FWDH[1],q->design,FOAH,q->direction,F);CHKERRQ(ierr);
      AXPY = PETSC_TRUE;
    }
  }
  if (Hhas[2][0]) { /* (L^T \otimes I_M) H_MX \eta, \eta (=U) the TLM solution */
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,0,t,FWDH[0],FWDH[1],q->design,FOAH,U,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,0,t,FWDH[0],FWDH[1],q->design,FOAH,U,F);CHKERRQ(ierr);
      AXPY = PETSC_TRUE;
    }
  }
  if (Hhas[2][1] && q->init) { /* (L^T \otimes I_M) H_MXdot \etadot, \eta the TLM solution */
    Vec TLMHdot;
    DM  dm;

    ierr = TSGetDM(tlmts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&TLMHdot);CHKERRQ(ierr);
    ierr = TSTrajectoryGetVecs(ltj,NULL,PETSC_DECIDE,&t,NULL,TLMHdot);CHKERRQ(ierr);
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,1,t,FWDH[0],FWDH[1],q->design,FOAH,TLMHdot,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,1,t,FWDH[0],FWDH[1],q->design,FOAH,TLMHdot,F);CHKERRQ(ierr);
    }
    ierr = DMRestoreGlobalVector(dm,&TLMHdot);CHKERRQ(ierr);
  }
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  if (rest) {
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(atj,&FOAH,NULL);CHKERRQ(ierr);
  }
  q->init = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrandFixed_TLM(Vec U, PetscReal t, Vec F, void* ctx)
{
  TLMEvalQuadCtx *q = (TLMEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  TSTrajectory   tj;
  PetscBool      has;
  Vec            FWDH;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTrajectory(fwdts,&tj);CHKERRQ(ierr);
  ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,fwdts,t,&FWDH,NULL);CHKERRQ(ierr);
  ierr = TSObjEvalFixed_MU(q->obj,FWDH,q->design,t,U,q->work1,&has,F);CHKERRQ(ierr);
  if (!has) {
    ierr = TSObjEvalFixed_MM(q->obj,FWDH,q->design,t,q->direction,q->work1,&has,F);CHKERRQ(ierr);
    if (!has) { /* can be called with a non-matching time */
      ierr = VecSet(F,0);CHKERRQ(ierr);
    }
  } else {
    ierr = TSObjEvalFixed_MM(q->obj,FWDH,q->design,t,q->direction,q->work1,&has,q->work2);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr);
    }
  }
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&FWDH,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSQuadraturePostStep_Private(TS ts)
{
  PetscContainer    container;
  Vec               solution;
  TSQuadratureCtx   *qeval_ctx;
  PetscReal         squad = 0.0;
  PetscReal         dt,time,ptime;
  TSConvergedReason reason;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing evaluate_quadrature container");
  ierr = PetscContainerGetPointer(container,(void**)&qeval_ctx);CHKERRQ(ierr);
  if (qeval_ctx->user && !qeval_ctx->userafter) {
    PetscStackPush("User post-step function");
    ierr = (*qeval_ctx->user)(ts);CHKERRQ(ierr);
    PetscStackPop;
  }

  ierr = TSGetSolution(ts,&solution);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  if (reason == TS_CONVERGED_TIME) {
    ierr = TSGetMaxTime(ts,&time);CHKERRQ(ierr);
  }

  /* time step used */
  ierr = TSGetPrevTime(ts,&ptime);CHKERRQ(ierr);
  dt   = time - ptime;

  /* scalar quadrature (psquad have been initialized with the first function evaluation) */
  if (qeval_ctx->seval) {
    PetscStackPush("TS scalar quadrature function");
    ierr = (*qeval_ctx->seval)(solution,time,&squad,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
    qeval_ctx->squad += dt*(squad+qeval_ctx->psquad)/2.0;
    qeval_ctx->psquad = squad;
  }

  /* scalar quadrature (qeval_ctx->wquad[qeval_ctx->old] have been initialized with the first function evaluation) */
  if (qeval_ctx->veval) {
    PetscScalar t[2];
    PetscInt    tmp;

    PetscStackPush("TS vector quadrature function");
    ierr = (*qeval_ctx->veval)(solution,time,qeval_ctx->wquad[qeval_ctx->cur],qeval_ctx->veval_ctx);CHKERRQ(ierr);
    PetscStackPop;

    /* trapezoidal rule */
    t[0] = dt/2.0;
    t[1] = dt/2.0;
    ierr = VecMAXPY(qeval_ctx->vquad,2,t,qeval_ctx->wquad);CHKERRQ(ierr);

    /* swap pointers */
    tmp            = qeval_ctx->cur;
    qeval_ctx->cur = qeval_ctx->old;
    qeval_ctx->old = tmp;
  }
  if (qeval_ctx->user && qeval_ctx->userafter) {
    PetscStackPush("User post-step function");
    ierr = (*qeval_ctx->user)(ts);CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

/*
   Apply "Jacobians" of initial conditions
   if transpose is false : y =   G_x^-1 G_m x or G_x^-1 x if G_m == 0
   if transpose is true  : y = G_m^T G_x^-T x or G_x^-T x if G_m == 0
   (x0,design) are the variables one needs to linearize against to get the partial Jacobians G_x and G_m
   The case for useGm == PETSC_FALSE is a hack to reuse the same code for the second-order adjoint
*/
PetscErrorCode TSLinearizedICApply_Private(TS ts, PetscReal t0, Vec x0, Vec design, Vec x, Vec y, PetscBool transpose, PetscBool useGm)
{
  KSP            ksp = NULL;
  Mat            G_x, G_m = NULL;
  Vec            workvec = NULL;
  TSOpt          tsopt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidHeaderSpecific(x0,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(x,VEC_CLASSID,5);
  PetscValidHeaderSpecific(y,VEC_CLASSID,6);
  PetscValidLogicalCollectiveBool(ts,transpose,7);
  ierr = TSGetTSOpt(ts,&tsopt);CHKERRQ(ierr);
  ierr = TSOptEvalGradientIC(tsopt,t0,x0,design,&G_x,useGm ? &G_m : NULL);CHKERRQ(ierr);
  if (useGm && !G_m) {
    ierr = VecSet(y,0.);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (G_x) { /* this is optional. If not provided, identity is assumed */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradientIC_G",(PetscObject*)&ksp);CHKERRQ(ierr);
    if (!ksp) {
      const char *prefix;
      ierr = KSPCreate(PetscObjectComm((PetscObject)ts),&ksp);CHKERRQ(ierr);
      ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(ksp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"jactsic_");CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,G_x,G_x);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_G",(PetscObject)ksp);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)ksp);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(ksp,G_x,G_x);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradientIC_GW",(PetscObject*)&workvec);CHKERRQ(ierr);
    if (!workvec) {
      ierr = MatCreateVecs(G_x,&workvec,NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_GW",(PetscObject)workvec);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)workvec);CHKERRQ(ierr);
    }
  }
  if (transpose) {
    if (ksp) {
      if (useGm) {
        ierr = KSPSolveTranspose(ksp,x,workvec);CHKERRQ(ierr);
        ierr = MatMultTranspose(G_m,workvec,y);CHKERRQ(ierr);
      } else {
        ierr = KSPSolveTranspose(ksp,x,y);CHKERRQ(ierr);
      }
    } else {
      if (useGm) {
        ierr = MatMultTranspose(G_m,x,y);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(x,y);CHKERRQ(ierr);
      }
    }
  } else {
    if (ksp) {
      if (useGm) {
        ierr = MatMult(G_m,x,workvec);CHKERRQ(ierr);
        ierr = KSPSolve(ksp,workvec,y);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(ksp,x,y);CHKERRQ(ierr);
      }
    } else {
      if (useGm) {
        ierr = MatMult(G_m,x,y);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(x,y);CHKERRQ(ierr);
      }
    }
  }
  if (ksp) { /* destroy inner vectors to avoid ABA issues when destroying the DM */
    ierr = VecDestroy(&ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecDestroy(&ksp->vec_sol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* auxiliary function to solve a forward model with a quadrature */
PetscErrorCode TSSolveWithQuadrature_Private(TS ts, Vec X, Vec design, Vec direction, Vec quadvec, PetscReal *quadscalar)
{
  TS              model = NULL;
  Vec             U;
  PetscContainer  container;
  TSQuadratureCtx *qeval_ctx;
  PetscReal       t0,tf,tfup,dt;
  PetscBool       fidt,stop;
  SQuadEval       seval_fixed, seval;
  VQuadEval       veval_fixed, veval;
  FWDEvalQuadCtx  qfwd;
  TLMEvalQuadCtx  qtlm;
  TSObj           funchead;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (direction) {
    PetscValidHeaderSpecific(direction,VEC_CLASSID,4);
    PetscCheckSameComm(ts,1,direction,4);
  }
  if (quadvec) {
    PetscValidHeaderSpecific(quadvec,VEC_CLASSID,5);
    PetscCheckSameComm(ts,1,quadvec,5);
  }
  if (direction && !quadvec) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Cannot compute Hessian without quadrature vector");
  if (direction) { /* when the direction is present, the ts need to be a TLMTS */
    ierr = TLMTSGetModelTS(ts,&model);CHKERRQ(ierr);
    ierr = TSGetTSObj(model,&funchead);CHKERRQ(ierr);
  } else {
    ierr = TSGetTSObj(ts,&funchead);CHKERRQ(ierr);
  }

  /* solution vector */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  if (!U) {
    ierr = VecDuplicate(X,&U);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)U);CHKERRQ(ierr);
  }
  if (X) {
    ierr = VecCopy(X,U);CHKERRQ(ierr);
  }

  /* quadrature evaluations */
  seval       = direction ? NULL : (quadscalar ? EvalQuadObj_FWD      : NULL);
  seval_fixed = direction ? NULL : (quadscalar ? EvalQuadObjFixed_FWD : NULL);
  veval       = quadvec ? (direction ? EvalQuadIntegrand_TLM      : EvalQuadIntegrand_FWD)      : NULL;
  veval_fixed = quadvec ? (direction ? EvalQuadIntegrandFixed_TLM : EvalQuadIntegrandFixed_FWD) : NULL;

  /* init contexts for quadrature evaluations */
  qfwd.obj    = funchead;
  qfwd.design = design;
  qtlm.obj    = funchead;
  qtlm.design = design;

  /* set special purpose post step method for quadrature evaluation */
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) {
    ierr = PetscNew(&qeval_ctx);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,(void *)qeval_ctx);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,TSQuadratureCtxDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)container);CHKERRQ(ierr);
  } else {
    ierr = PetscContainerGetPointer(container,(void**)&qeval_ctx);CHKERRQ(ierr);
  }

  qeval_ctx->user      = ts->poststep;
  qeval_ctx->userafter = PETSC_FALSE;
  qeval_ctx->seval     = seval;
  qeval_ctx->seval_ctx = &qfwd;
  qeval_ctx->squad     = 0.0;
  qeval_ctx->psquad    = 0.0;
  qeval_ctx->veval     = veval;
  qeval_ctx->vquad     = quadvec;
  qeval_ctx->cur       = 0;
  qeval_ctx->old       = 1;

  ierr = TSSetPostStep(ts,TSQuadraturePostStep_Private);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  ierr = TSGetMaxTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  /* fixed term at t0 for scalar quadrature */
  if (seval_fixed) {
    Vec sol;

    ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
    PetscStackPush("TS scalar quadrature function (fixed time)");
    ierr = (*seval_fixed)(sol,t0,&qeval_ctx->squad,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  /* initialize trapz rule for scalar quadrature */
  if (qeval_ctx->seval) {
    Vec sol;

    ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
    PetscStackPush("TS scalar quadrature function");
    ierr = (*qeval_ctx->seval)(sol,t0,&qeval_ctx->psquad,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  if (qeval_ctx->veval) {
    PetscBool has;

    if (direction) { /* Hessian computations */
      PetscBool has1,has2,Hhas[3][3];
      TSOpt     tsopt;

      ierr = TSGetTSOpt(model,&tsopt);CHKERRQ(ierr);
      ierr = TSObjHasObjectiveIntegrand(funchead,NULL,NULL,NULL,NULL,&has1,&has2);CHKERRQ(ierr);
      has  = (PetscBool)(has1 || has2);
      ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
      if (Hhas[2][0] || Hhas[2][1] || Hhas[2][2]) has = PETSC_TRUE;
    } else {
      ierr = TSObjHasObjectiveIntegrand(funchead,NULL,NULL,&has,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    if (!has) { /* cost integrands not present */
      qeval_ctx->veval = NULL;
    }
    if (!qeval_ctx->wquad) {
      ierr = VecDuplicateVecs(qeval_ctx->vquad,5,&qeval_ctx->wquad);CHKERRQ(ierr);
    }
    if (direction) { /* we use PETSC_SMALL since some of the fixed terms can be at the initial time */
      PetscBool has1,has2;

      ierr = TSObjHasObjectiveFixed(funchead,t0-PETSC_SMALL,tf,NULL,NULL,NULL,NULL,&has1,&has2,NULL);CHKERRQ(ierr);
      has  = (PetscBool)(has1 || has2);
    } else {
      ierr = TSObjHasObjectiveFixed(funchead,t0-PETSC_SMALL,tf,NULL,NULL,&has,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    if (!has) veval_fixed = NULL;
  }
  /* for gradient computations, we just need the design vector and one work vector for the function evaluation
     for Hessian computations, we need extra data */
  if (qeval_ctx->veval || veval_fixed) {
    if (!direction) {
      qfwd.work = qeval_ctx->wquad[2];
      qeval_ctx->veval_ctx = &qfwd;
    } else {
      /* qtlm.adjts may be NULL for Gauss-Newton approximations */
      ierr = PetscObjectQuery((PetscObject)model,"_ts_hessian_foats",(PetscObject*)&qtlm.adjts);CHKERRQ(ierr);
      qtlm.fwdts     = model;
      qtlm.tlmts     = ts;
      qtlm.t0        = t0;
      qtlm.tf        = tf;
      qtlm.design    = design;
      qtlm.direction = direction;
      qtlm.work1     = qeval_ctx->wquad[2];
      qtlm.work2     = qeval_ctx->wquad[3];
      qtlm.init      = PETSC_FALSE; /* skip etadot terms when initializing the quadrature */
      qeval_ctx->veval_ctx = &qtlm;
    }

    if (veval_fixed) { /* Fixed term at t0: we use wquad[4] since wquad[3] can be used by the TLM quadrature */
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS vector quadrature function (fixed time)");
      ierr = (*veval_fixed)(sol,t0,qeval_ctx->wquad[4],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
      ierr = VecAXPY(qeval_ctx->vquad,1.0,qeval_ctx->wquad[4]);CHKERRQ(ierr);
    }

    /* initialize trapz rule for vector quadrature */
    if (qeval_ctx->veval) {
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS vector quadrature function");
      ierr = (*qeval_ctx->veval)(sol,t0,qeval_ctx->wquad[qeval_ctx->old],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
    }
  }

  /* forward solve */
  fidt = PETSC_TRUE;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (ts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&fidt);CHKERRQ(ierr);
  }

  /* determine if there are functionals, gradients or Hessians wrt parameters of the type f(U,M,t=fixed) to be evaluated */
  /* we don't use events since there's no API to add new events to a pre-existing set */
  tfup = tf;
  do {
    PetscBool has_f = PETSC_FALSE, has_m = PETSC_FALSE;
    PetscReal tt;

    ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
    if (direction) { /* we always stop at the selected times */
      PetscBool has1,has2;

      ierr  = TSObjHasObjectiveFixed(funchead,t0,tf,NULL,NULL,NULL,NULL,&has1,&has2,&tfup);CHKERRQ(ierr);
      has_m = (PetscBool)(has1 || has2);
    } else {
      ierr = TSObjHasObjectiveFixed(funchead,t0,tf,&has_f,NULL,&has_m,NULL,NULL,NULL,&tfup);CHKERRQ(ierr);
    }
    ierr = TSSetMaxTime(ts,tfup);CHKERRQ(ierr);
    tt   = tfup;
    ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

    /* determine if TS finised before the max time requested */
    ierr = TSGetTime(ts,&tfup);CHKERRQ(ierr);
    stop = (PetscAbsReal(tt-tfup) < PETSC_SMALL) ? PETSC_FALSE : PETSC_TRUE;
    if (has_f && seval_fixed) {
      Vec       sol;
      PetscReal v;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS scalar quadrature function (fixed time)");
      ierr = (*seval_fixed)(sol,tfup,&v,qeval_ctx->seval_ctx);CHKERRQ(ierr);
      PetscStackPop;
      qeval_ctx->squad += v;
    }
    if (has_m && veval_fixed) { /* we use wquad[4] since wquad[3] can be used by the TLM quadrature */
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS vector quadrature function (fixed time)");
      ierr = (*veval_fixed)(sol,tfup,qeval_ctx->wquad[4],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
      ierr = VecAXPY(qeval_ctx->vquad,1.0,qeval_ctx->wquad[4]);CHKERRQ(ierr);
    }
    if (fidt) { /* restore fixed time step */
      ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    }
  } while (PetscAbsReal(tfup - tf) > PETSC_SMALL && !stop);

  /* zero contexts for quadrature */
  qeval_ctx->seval_ctx = NULL;
  qeval_ctx->veval_ctx = NULL;

  /* restore user PostStep */
  ierr = TSSetPostStep(ts,qeval_ctx->user);CHKERRQ(ierr);

  /* get back scalar value */
  if (quadscalar) *quadscalar = qeval_ctx->squad;
  PetscFunctionReturn(0);
}

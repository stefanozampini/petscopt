#include <petscopt/adjointts.h>
#include <petscopt/tlmts.h>
#include <petscopt/private/tsobjimpl.h>
#include <petscopt/private/tsoptimpl.h>
#include <petscopt/private/adjointtsimpl.h>
#include <petscopt/private/tspdeconstrainedutilsimpl.h>
#include <petsc/private/tshistoryimpl.h>

/*
   TODO: add custom fortran wrappers ?
*/

/* ------------------ Routines for the Hessian matrix ----------------------- */

typedef struct {
  TS           model;    /* nonlinear DAE */
  TS           tlmts;    /* tangent linear model solver */
  TS           foats;    /* first-order adjoint solver */
  TS           soats;    /* second-order adjoint solver */
  Vec          x0;       /* initial conditions */
  PetscReal    t0,dt,tf;
  Vec          design;
  TSTrajectory modeltj;  /* nonlinear model trajectory */
  PetscBool    GN;
} TSHessian;

static PetscErrorCode TSHessianReset_Private(void *ptr)
{
  TSHessian*     tshess = (TSHessian*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSDestroy(&tshess->model);CHKERRQ(ierr);
  ierr = TSDestroy(&tshess->tlmts);CHKERRQ(ierr);
  ierr = TSDestroy(&tshess->foats);CHKERRQ(ierr);
  ierr = TSDestroy(&tshess->soats);CHKERRQ(ierr);
  ierr = VecDestroy(&tshess->x0);CHKERRQ(ierr);
  ierr = VecDestroy(&tshess->design);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tshess->modeltj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSHessianDestroy_Private(void *ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSHessianReset_Private(ptr);CHKERRQ(ierr);
  ierr = PetscFree(ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_TSHessian(Mat H, Vec x, Vec y)
{
  PetscContainer c;
  TSHessian      *tshess;
  TSTrajectory   otrj;
  TSAdapt        adapt;
  PetscReal      dt;
  PetscBool      istr,soadisc,tlmdisc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)H,"_ts_hessian_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)H),PETSC_ERR_PLIB,"Not a valid Hessian matrix");
  ierr = PetscContainerGetPointer(c,(void**)&tshess);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)tshess->model,"_ts_obj_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) PetscFunctionReturn(0);

  otrj = tshess->model->trajectory;
  tshess->model->trajectory = tshess->modeltj;

  /* Need to setup the model TS, as the tlm and soa solvers in the following depend on it (relevant callbacks) */
  ierr = TSSetUpFromDesign(tshess->model,tshess->x0,tshess->design);CHKERRQ(ierr);

  /* solve tangent linear model */
  ierr = AdjointTSIsDiscrete(tshess->soats,&soadisc);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tshess->tlmts->trajectory);CHKERRQ(ierr); /* XXX add Reset method to TSTrajectory */
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)tshess->tlmts),&tshess->tlmts->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(tshess->tlmts->trajectory,tshess->tlmts,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(tshess->tlmts->trajectory,soadisc ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(tshess->tlmts->trajectory,tshess->tlmts);CHKERRQ(ierr);
  tshess->tlmts->trajectory->adjoint_solve_mode = PETSC_FALSE;

  ierr = TLMTSIsDiscrete(tshess->tlmts,&tlmdisc);CHKERRQ(ierr);
  ierr = TLMTSSetPerturbationVec(tshess->tlmts,x);CHKERRQ(ierr);
  ierr = TLMTSComputeInitialConditions(tshess->tlmts,tshess->t0,tshess->x0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(tshess->tlmts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(tshess->tlmts);CHKERRQ(ierr);
  ierr = TSSetTime(tshess->tlmts,tshess->t0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(tshess->tlmts,tshess->tf);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(tshess->modeltj->tsh,PETSC_FALSE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tshess->tlmts,dt);CHKERRQ(ierr);
  ierr = TSGetAdapt(tshess->tlmts,&adapt);CHKERRQ(ierr);
  if (tlmdisc) {
    ierr = TSAdaptSetType(adapt,TSADAPTHISTORY);CHKERRQ(ierr);
  }
  ierr = TSAdaptHistorySetTSHistory(adapt,tshess->modeltj->tsh,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTHISTORY,&istr);CHKERRQ(ierr);
  if (istr) {
    PetscInt n;

    ierr = TSTrajectoryGetNumSteps(tshess->modeltj,&n);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(tshess->tlmts,n-1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->tlmts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  } else {
    ierr = TSSetMaxSteps(tshess->tlmts,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->tlmts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  }

  /* XXX should we add the AdjointTS to the TS private data? */
  ierr = PetscObjectCompose((PetscObject)tshess->model,"_ts_hessian_foats",tshess->GN ? NULL : (PetscObject)tshess->foats);CHKERRQ(ierr);
  ierr = TSSolveWithQuadrature_Private(tshess->tlmts,NULL,tshess->design,x,y,NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)tshess->model,"_ts_hessian_foats",NULL);CHKERRQ(ierr);
  ierr = TLMTSSetPerturbationVec(tshess->tlmts,NULL);CHKERRQ(ierr);

  /* second-order adjoint solve */
  ierr = AdjointTSSetTimeLimits(tshess->soats,tshess->t0,tshess->tf);CHKERRQ(ierr);
  ierr = AdjointTSSetDirectionVec(tshess->soats,x);CHKERRQ(ierr);
  ierr = AdjointTSSetTLMTSAndFOATS(tshess->soats,tshess->tlmts,tshess->GN ? NULL : tshess->foats);CHKERRQ(ierr);
  ierr = AdjointTSSetQuadratureVec(tshess->soats,y);CHKERRQ(ierr);
  ierr = AdjointTSComputeInitialConditions(tshess->soats,NULL,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetStepNumber(tshess->soats,0);CHKERRQ(ierr);
  ierr = TSRestartStep(tshess->soats);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(tshess->modeltj->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tshess->soats,dt);CHKERRQ(ierr);
  ierr = TSGetAdapt(tshess->soats,&adapt);CHKERRQ(ierr);
  if (soadisc) {
    ierr = TSAdaptSetType(adapt,TSADAPTHISTORY);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTHISTORY,&istr);CHKERRQ(ierr);
  if (!istr) {
    PetscBool isnone;

    ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTNONE,&isnone);CHKERRQ(ierr);
    if (isnone && tshess->dt > 0.0) {
      ierr = TSSetTimeStep(tshess->soats,tshess->dt);CHKERRQ(ierr);
    }
    ierr = TSSetMaxSteps(tshess->soats,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->soats,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  } else { /* follow trajectory -> fix number of time steps */
    PetscInt nsteps;

    ierr = TSAdaptHistorySetTSHistory(adapt,tshess->modeltj->tsh,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSTrajectoryGetNumSteps(tshess->modeltj,&nsteps);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(tshess->soats,nsteps-1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->soats,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  }
  ierr = AdjointTSSolveWithQuadrature_Private(tshess->soats);CHKERRQ(ierr);
  ierr = AdjointTSFinalizeQuadrature(tshess->soats);CHKERRQ(ierr);

  ierr = AdjointTSSetQuadratureVec(tshess->soats,NULL);CHKERRQ(ierr);
  ierr = AdjointTSSetDirectionVec(tshess->soats,NULL);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tshess->tlmts->trajectory);CHKERRQ(ierr); /* XXX add Reset method to TSTrajectory */
  tshess->model->trajectory = otrj;
  PetscFunctionReturn(0);
}

/* private functions for objective, gradient and Hessian evaluation */
static PetscErrorCode TSComputeObjectiveAndGradient_Private(TS ts, Vec X, Vec design, Vec gradient, PetscReal *val)
{
  TS             adjts = NULL;
  TSTrajectory   otrj = NULL;
  PetscReal      t0,tf,dt;
  PetscContainer c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (gradient) {
    ierr = VecSet(gradient,0.0);CHKERRQ(ierr);
  }
  if (val) *val = 0.0;
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_obj_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) PetscFunctionReturn(0);
  if (!X) {
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
    if (!X) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing solution vector");
  }
  ierr = TSSetUpFromDesign(ts,X,design);CHKERRQ(ierr);
  otrj = ts->trajectory;
  ts->trajectory = NULL;
  if (gradient) {
    const char* prefix;
    char        *prefix_cp;
    PetscBool   flg;

    ierr = TSCreateAdjointTS(ts,&adjts);CHKERRQ(ierr);
    ierr = TSGetOptionsPrefix(adjts,&prefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(prefix,&prefix_cp);CHKERRQ(ierr);
    ierr = TSSetOptionsPrefix(adjts,"tsgradient_");CHKERRQ(ierr);
    ierr = TSAppendOptionsPrefix(adjts,prefix_cp);CHKERRQ(ierr);
    ierr = PetscFree(prefix_cp);CHKERRQ(ierr);
    ierr = TSSetFromOptions(adjts);CHKERRQ(ierr);
    ierr = AdjointTSSetUpStep(adjts);CHKERRQ(ierr);
    ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
    ierr = TSTrajectorySetType(ts->trajectory,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
    ierr = TSTrajectorySetFromOptions(ts->trajectory,ts);CHKERRQ(ierr);
    ierr = AdjointTSIsDiscrete(adjts,&flg);CHKERRQ(ierr);
    ierr = TSTrajectorySetSolutionOnly(ts->trajectory,flg ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
    /* we don't have an API for this right now */
    ts->trajectory->adjoint_solve_mode = PETSC_FALSE;
  }

  /* forward solve */
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  ierr = TSSolveWithQuadrature_Private(ts,X,design,NULL,gradient,val);CHKERRQ(ierr);

  /* adjoint */
  if (gradient) {
    ierr = TSHistoryGetTimeStep(ts->trajectory->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
    ierr = TSSetTimeStep(adjts,dt);CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
    ierr = AdjointTSSetDesignVec(adjts,design);CHKERRQ(ierr);
    ierr = AdjointTSSetQuadratureVec(adjts,gradient);CHKERRQ(ierr);
    ierr = AdjointTSComputeInitialConditions(adjts,NULL,PETSC_TRUE);CHKERRQ(ierr);
    ierr = AdjointTSEventHandler(adjts);CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
    if (adjts->adapt) {
      PetscBool istr;

      ierr = PetscObjectTypeCompare((PetscObject)adjts->adapt,TSADAPTHISTORY,&istr);CHKERRQ(ierr);
      ierr = TSAdaptHistorySetTSHistory(adjts->adapt,ts->trajectory->tsh,PETSC_TRUE);CHKERRQ(ierr);
      if (!istr) { /* indepently adapting the time step */
        ierr = TSSetMaxSteps(adjts,PETSC_MAX_INT);CHKERRQ(ierr);
        ierr = TSSetExactFinalTime(adjts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
      } else { /* follow trajectory -> fix number of time steps */
        PetscInt nsteps;

        ierr = TSTrajectoryGetNumSteps(ts->trajectory,&nsteps);CHKERRQ(ierr);
        ierr = TSSetMaxSteps(adjts,nsteps-1);CHKERRQ(ierr);
        ierr = TSSetExactFinalTime(adjts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
      }
    }
    ierr = AdjointTSSolveWithQuadrature_Private(adjts);CHKERRQ(ierr);
    ierr = AdjointTSFinalizeQuadrature(adjts);CHKERRQ(ierr);
  }
  ierr = TSDestroy(&adjts);CHKERRQ(ierr);
  /* restore TS to its original state */
  ierr = TSTrajectoryDestroy(&ts->trajectory);CHKERRQ(ierr);
  ts->trajectory = otrj;
  PetscFunctionReturn(0);
}

typedef struct {
  TS        ts;
  PetscReal t0,dt,tf;
  Vec       X;
} TSHessian_MFFD;

static PetscErrorCode TSHessianMFFDDestroy_Private(void *ptr)
{
  TSHessian_MFFD *mffd = (TSHessian_MFFD*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSDestroy(&mffd->ts);CHKERRQ(ierr);
  ierr = VecDestroy(&mffd->X);CHKERRQ(ierr);
  ierr = PetscFree(ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeHessianMFFD_Private(void* ctx, Vec P, Vec G)
{
  TSHessian_MFFD *mffd = (TSHessian_MFFD*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSComputeObjectiveAndGradient(mffd->ts,mffd->t0,mffd->dt,mffd->tf,mffd->X,P,G,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeHessian_MFFD(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Mat H)
{
  PetscContainer c;
  TSHessian_MFFD *mffd;
  Vec            G;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&mffd);CHKERRQ(ierr);
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&c);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,mffd);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(c,TSHessianMFFDDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)H,"_ts_hessianmffd_ctx",(PetscObject)c);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  mffd->ts = ts;
  mffd->t0 = t0;
  if (dt < 0) { ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr); }
  mffd->dt = dt;
  mffd->tf = tf;
  if (!X) {
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(X,&mffd->X);CHKERRQ(ierr);

  ierr = MatSetType(H,MATMFFD);CHKERRQ(ierr);
  ierr = MatSetUp(H);CHKERRQ(ierr);

  /* sample at linearization point */
  ierr = VecDuplicate(design,&G);CHKERRQ(ierr);
  ierr = TSComputeHessianMFFD_Private(mffd,design,G);CHKERRQ(ierr);
  ierr = MatMFFDSetBase(H,design,G);CHKERRQ(ierr);
  /* MATMFFD does not take ownership of the base vector */
  ierr = PetscObjectCompose((PetscObject)H,"__tsopt_mffd_base",(PetscObject)G);CHKERRQ(ierr);
  ierr = VecDestroy(&G);CHKERRQ(ierr);

  ierr = MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))TSComputeHessianMFFD_Private,mffd);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeHessian_Private(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Mat H)
{
  PetscContainer c;
  TSHessian      *tshess;
  Vec            U;
  TSTrajectory   otrj;
  TSAdapt        adapt;
  PetscBool      has,istr;
  PetscBool      tlmdisc,foadisc,soadisc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_obj_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (c) {
    TSObj tsobj;

    ierr = PetscContainerGetPointer(c,(void**)&tsobj);CHKERRQ(ierr);
    ierr = TSObjHasObjectiveFixed(tsobj,t0,tf-PETSC_SMALL,NULL,&has,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    if (has) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Point-form functionals in between the simulations are not supported! Use -tshessian_mffd");
  }
  ierr = PetscObjectQuery((PetscObject)H,"_ts_hessian_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) {
    ierr = PetscNew(&tshess);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&c);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(c,tshess);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(c,TSHessianDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)H,"_ts_hessian_ctx",(PetscObject)c);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)c);CHKERRQ(ierr);
  }

  ierr = MatSetType(H,MATSHELL);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H,MATOP_MULT,(void (*)())MatMult_TSHessian);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H,MATOP_MULT_TRANSPOSE,(void (*)())MatMult_TSHessian);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tshess);CHKERRQ(ierr);

  /* Gauss-Newton approximation */
  ierr = PetscOptionsGetBool(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-tshessian_gn",&tshess->GN,NULL);CHKERRQ(ierr);

  /* nonlinear model */
  if (ts != tshess->model) {
    ierr = TSHessianReset_Private(tshess);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
    tshess->model = ts;
  }

  if (!X) {
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
    if (!X) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing solution vector");
  }
  if (!tshess->x0) {
    ierr = VecDuplicate(X,&tshess->x0);CHKERRQ(ierr);
  }
  if (!tshess->design) {
    ierr = VecDuplicate(design,&tshess->design);CHKERRQ(ierr);
  }
  ierr = VecCopy(design,tshess->design);CHKERRQ(ierr);
  ierr = TSSetUpFromDesign(ts,X,tshess->design);CHKERRQ(ierr);
  ierr = VecCopy(X,tshess->x0);CHKERRQ(ierr);
  tshess->t0 = t0;
  tshess->dt = dt;
  tshess->tf = tf;

  /* tangent linear model solver */
  if (!tshess->tlmts) {
    const char* prefix;
    char        *prefix_cp;

    ierr = TSCreateTLMTS(tshess->model,&tshess->tlmts);CHKERRQ(ierr);
    ierr = TSGetOptionsPrefix(tshess->tlmts,&prefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(prefix,&prefix_cp);CHKERRQ(ierr);
    ierr = TSSetOptionsPrefix(tshess->tlmts,"tshessian_");CHKERRQ(ierr);
    ierr = TSAppendOptionsPrefix(tshess->tlmts,prefix_cp);CHKERRQ(ierr);
    ierr = PetscFree(prefix_cp);CHKERRQ(ierr);
    ierr = TSSetFromOptions(tshess->tlmts);CHKERRQ(ierr);
    ierr = TSSetTime(tshess->tlmts,tshess->t0);CHKERRQ(ierr);
    ierr = TSSetMaxTime(tshess->tlmts,tshess->tf);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(tshess->tlmts,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->tlmts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  }
  ierr = TLMTSSetUpStep(tshess->tlmts);CHKERRQ(ierr);
  ierr = TLMTSIsDiscrete(tshess->tlmts,&tlmdisc);CHKERRQ(ierr);
  ierr = TLMTSSetDesignVec(tshess->tlmts,design);CHKERRQ(ierr);

  /* first-order adjoint solver */
  if (!tshess->foats) {
    const char* prefix;
    char        *prefix_cp;

    ierr = TSCreateAdjointTS(tshess->model,&tshess->foats);CHKERRQ(ierr);
    ierr = TSGetOptionsPrefix(tshess->foats,&prefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(prefix,&prefix_cp);CHKERRQ(ierr);
    ierr = TSSetOptionsPrefix(tshess->foats,"tshessian_fo");CHKERRQ(ierr);
    ierr = TSAppendOptionsPrefix(tshess->foats,prefix_cp);CHKERRQ(ierr);
    ierr = PetscFree(prefix_cp);CHKERRQ(ierr);
    ierr = TSSetFromOptions(tshess->foats);CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(tshess->foats,tshess->t0,tshess->tf);CHKERRQ(ierr);
    ierr = AdjointTSEventHandler(tshess->foats);CHKERRQ(ierr);
  }
  ierr = AdjointTSSetUpStep(tshess->foats);CHKERRQ(ierr);
  ierr = AdjointTSIsDiscrete(tshess->foats,&foadisc);CHKERRQ(ierr);
  ierr = AdjointTSSetDesignVec(tshess->foats,design);CHKERRQ(ierr);
  ierr = AdjointTSSetQuadratureVec(tshess->foats,NULL);CHKERRQ(ierr);

  /* second-order adjoint solver */
  if (!tshess->soats) {
    const char* prefix;
    char        *prefix_cp;

    ierr = TSCreateAdjointTS(tshess->model,&tshess->soats);CHKERRQ(ierr);
    ierr = TSGetOptionsPrefix(tshess->soats,&prefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(prefix,&prefix_cp);CHKERRQ(ierr);
    ierr = TSSetOptionsPrefix(tshess->soats,"tshessian_so");CHKERRQ(ierr);
    ierr = TSAppendOptionsPrefix(tshess->soats,prefix_cp);CHKERRQ(ierr);
    ierr = PetscFree(prefix_cp);CHKERRQ(ierr);
    ierr = TSSetFromOptions(tshess->soats);CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(tshess->soats,tshess->t0,tshess->tf);CHKERRQ(ierr);
    ierr = AdjointTSEventHandler(tshess->soats);CHKERRQ(ierr);
  }
  ierr = AdjointTSSetUpStep(tshess->soats);CHKERRQ(ierr);
  ierr = AdjointTSIsDiscrete(tshess->soats,&soadisc);CHKERRQ(ierr);
  ierr = AdjointTSSetDesignVec(tshess->soats,design);CHKERRQ(ierr);
  ierr = AdjointTSSetQuadratureVec(tshess->soats,NULL);CHKERRQ(ierr);

  /* sanity check XXX */
  if (tlmdisc != foadisc || (!tshess->GN && soadisc != foadisc)) SETERRQ3(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Cannot mix discrete and continuous adjoints! TLM %d, FOA %d, SOA %d",tlmdisc,foadisc,soadisc);

  /* sample nonlinear model */
  otrj = ts->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(ts->trajectory,ts,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(ts->trajectory,foadisc ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(ts->trajectory,ts);CHKERRQ(ierr);
  ts->trajectory->adjoint_solve_mode = PETSC_FALSE;
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(ts);CHKERRQ(ierr);
  ierr = TSSetTime(ts,tshess->t0);CHKERRQ(ierr);
  if (tshess->dt > 0) {
    ierr = TSSetTimeStep(ts,tshess->dt);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts,tshess->tf);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  if (!U) {
    ierr = VecDuplicate(tshess->x0,&U);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)U);CHKERRQ(ierr);
  }
  ierr = VecCopy(tshess->x0,U);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tshess->modeltj);CHKERRQ(ierr);
  tshess->modeltj = ts->trajectory;

  /* model sampling can terminate before tf due to events */
  ierr = TSGetTime(ts,&tshess->tf);CHKERRQ(ierr);

  /* sample first-order adjoint (not needed for Gauss-Newton approximation) */
  ierr = TSTrajectoryDestroy(&tshess->foats->trajectory);CHKERRQ(ierr); /* XXX add Reset method to TSTrajectory */
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)tshess->foats),&tshess->foats->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(tshess->foats->trajectory,tshess->foats,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(tshess->foats->trajectory,foadisc ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(tshess->foats->trajectory,tshess->foats);CHKERRQ(ierr);
  tshess->foats->trajectory->adjoint_solve_mode = PETSC_FALSE;
  ierr = TSSetStepNumber(tshess->foats,0);CHKERRQ(ierr);
  ierr = TSRestartStep(tshess->foats);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(tshess->modeltj->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tshess->foats,dt);CHKERRQ(ierr);
  ierr = AdjointTSSetTimeLimits(tshess->foats,tshess->t0,tshess->tf);CHKERRQ(ierr);
  if (!tshess->GN) {
    ierr = AdjointTSComputeInitialConditions(tshess->foats,NULL,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSGetAdapt(tshess->foats,&adapt);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTHISTORY,&istr);CHKERRQ(ierr);
    if (istr) {
      PetscInt nsteps;

      ierr = TSAdaptHistorySetTSHistory(adapt,tshess->modeltj->tsh,PETSC_TRUE);CHKERRQ(ierr);
      ierr = TSTrajectoryGetNumSteps(tshess->modeltj,&nsteps);CHKERRQ(ierr);
      ierr = TSSetMaxSteps(tshess->foats,nsteps-1);CHKERRQ(ierr);
      ierr = TSSetExactFinalTime(tshess->foats,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
    }
    ierr = TSSolve(tshess->foats,NULL);CHKERRQ(ierr);
  }
  /* restore old TSTrajectory (if any) */
  ts->trajectory = otrj;
  ierr = MatSetUp(H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSComputeObjectiveAndGradient - Evaluates the sum of the objective functions set with TSAddObjective, together with the gradient with respect to the parameters.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context for the model DAE
.  t0     - initial time
.  dt     - initial time step
.  tf     - final time
.  X      - the initial vector for the state (can be NULL)
-  design - current design vector

   Output Parameters:
+  gradient - the computed gradient
-  obj      - the value of the objective function

   Notes: If gradient is NULL, just a forward solve will be performed to compute the objective function. Otherwise, forward and backward solves are performed.
          The dt argument is ignored when smaller or equal to zero. If X is NULL, the initial state is given by the current TS solution vector.

          The dependency of the TS and X from the design parameters can be set with TSSetSetUpFromDesign().

          Options for the adjoint DAE solver are prefixed with -tsgradient_adjoint_XXX, where XXX is the prefix for the model DAE.

   Level: advanced

.seealso: TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetSolution(), TSComputeHessian(), TSSetUpFromDesign(), TSSetSetUpFromDesign()
@*/
PetscErrorCode TSComputeObjectiveAndGradient(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Vec gradient, PetscReal *obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidLogicalCollectiveReal(ts,dt,3);
  PetscValidLogicalCollectiveReal(ts,tf,4);
  if (X) {
    PetscValidHeaderSpecific(X,VEC_CLASSID,5);
    PetscCheckSameComm(ts,1,X,5);
  }
  PetscValidHeaderSpecific(design,VEC_CLASSID,6);
  PetscCheckSameComm(ts,1,design,6);
  if (gradient) {
    PetscValidHeaderSpecific(gradient,VEC_CLASSID,7);
    PetscCheckSameComm(ts,1,gradient,7);
  }
  if (obj) PetscValidPointer(obj,8);
  if (!gradient && !obj) PetscFunctionReturn(0);

  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(ts);CHKERRQ(ierr);
  ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
  if (dt > 0.0) {
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = VecLockReadPush(design);CHKERRQ(ierr);
  ierr = TSComputeObjectiveAndGradient_Private(ts,X,design,gradient,obj);CHKERRQ(ierr);
  ierr = VecLockReadPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSComputeHessian - Setup the Hessian matrix with respect to the parameters for the objective functions set with TSAddObjective.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context for the model DAE
.  t0     - initial time
.  dt     - initial time step
.  tf     - final time
.  X      - the initial vector for the state (can be NULL)
-  design - current design vector

   Output Parameters:
.  H - the Hessian matrix

   Options Database Keys:
.  -tshessian_mffd <false>  - activates Matrix-Free Finite Differencing of the gradient code

   Notes: The Hessian matrix is not computed explictly; the only operation implemented for H is MatMult().
          The dt argument is ignored when smaller or equal to zero. If X is NULL, the initial state is given by the current TS solution vector.

          The dependency of the TS and X from the design parameters can be set with TSSetSetUpFromDesign().

          Internally, one forward solve and one backward solve (first-order adjoint) are performed within this call. Every MatMult() call solves one tangent linear and one second order adjoint problem.

          Options for the DAE solvers are prefixed with

$ -tshessian_foadjoint_XXX
$ -tshessian_tlm_XXX
$ -tshessian_soadjoint_XXX

          where XXX is the prefix for the model DAE.

   Level: advanced

.seealso: TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC(), TSComputeObjectiveAndGradient(), TSSetSetUpFromDesign()
@*/
PetscErrorCode TSComputeHessian(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Mat H)
{
  PetscBool      mffd;
  PetscErrorCode ierr;
  PetscInt       n,N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidLogicalCollectiveReal(ts,dt,3);
  PetscValidLogicalCollectiveReal(ts,tf,4);
  if (X) {
    PetscValidHeaderSpecific(X,VEC_CLASSID,5);
    PetscCheckSameComm(ts,1,X,5);
  }
  PetscValidHeaderSpecific(design,VEC_CLASSID,6);
  PetscCheckSameComm(ts,1,design,6);
  PetscValidHeaderSpecific(H,MAT_CLASSID,7);
  PetscCheckSameComm(ts,1,H,7);

  ierr = VecGetLocalSize(design,&n);CHKERRQ(ierr);
  ierr = VecGetSize(design,&N);CHKERRQ(ierr);
  ierr = MatSetSizes(H,n,n,N,N);CHKERRQ(ierr);

  mffd = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-tshessian_mffd",&mffd,NULL);CHKERRQ(ierr);
  if (mffd) {
    ierr = TSComputeHessian_MFFD(ts,t0,dt,tf,X,design,H);CHKERRQ(ierr);
  } else {
    ierr = TSComputeHessian_Private(ts,t0,dt,tf,X,design,H);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSTaylorTest - Performs a Taylor's reminders test at the prescribed design point.

   Collective on TS

   Input Parameters:
+  ts      - the TS context for the model DAE
.  t0      - initial time
.  dt      - initial time step
.  tf      - final time
.  X       - the initial vector for the state (can be NULL)
.  design  - current design vector
-  ddesign - design direction to be tested (can be NULL)

   Options Database Keys (prepended by ts prefix, if any):
.  -taylor_ts_hessian <false> - activates tests for the Hessian
.  -taylor_ts_h <0.125>       - initial increment
.  -taylor_ts_steps <4>       - number of refinements

   Notes: If the direction design is not passed, a random perturbation is generated. Options for the internal PetscRandom() object are prefixed by -XXX_taylor_ts, with XXX the prefix for the TS solver.

   Level: advanced

.seealso: TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetHessianIC(), TSComputeObjectiveAndGradient(), TSComputeHessian(), TSSetSetUpFromDesign()
@*/
PetscErrorCode TSTaylorTest(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Vec ddesign)
{
  Vec            G,M,dM;
  PetscScalar    v;
  PetscReal      h,*tG,*tH,obj,res1,res2 = 0.0;
  PetscInt       i,n;
  PetscBool      hess;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidLogicalCollectiveReal(ts,dt,3);
  PetscValidLogicalCollectiveReal(ts,tf,4);
  if (X) {
    PetscValidHeaderSpecific(X,VEC_CLASSID,5);
    PetscCheckSameComm(ts,1,X,5);
  }
  PetscValidHeaderSpecific(design,VEC_CLASSID,6);
  PetscCheckSameComm(ts,1,design,6);
  if (ddesign) {
    PetscValidHeaderSpecific(ddesign,VEC_CLASSID,7);
    PetscCheckSameComm(ts,1,ddesign,7);
  }

  ierr = PetscOptionsGetBool(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-taylor_ts_hessian",(hess=PETSC_FALSE,&hess),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-taylor_ts_h",(h=0.125,&h),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-taylor_ts_steps",(n=4,&n),NULL);CHKERRQ(ierr);

  ierr = PetscCalloc2(n,&tG,n,&tH);CHKERRQ(ierr);

  if (!ddesign) {
    PetscRandom r;

    ierr = VecDuplicate(design,&dM);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PetscObjectComm((PetscObject)ts),&r);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)r,"taylor_ts_");CHKERRQ(ierr);
    ierr = PetscObjectAppendOptionsPrefix((PetscObject)r,((PetscObject)ts)->prefix);CHKERRQ(ierr);
    ierr = VecSetRandom(dM,r);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = VecRealPart(dM);CHKERRQ(ierr);
    ierr = VecViewFromOptions(dM,NULL,"-taylor_ts_rand_vec_view");CHKERRQ(ierr);
  } else dM = ddesign;
  ierr = VecDuplicate(design,&M);CHKERRQ(ierr);
  ierr = VecDuplicate(design,&G);CHKERRQ(ierr);

  /* Sample gradient and Hessian at design point */
  ierr = TSComputeObjectiveAndGradient(ts,t0,dt,tf,X,design,G,&obj);CHKERRQ(ierr);
  ierr = VecDot(G,dM,&v);CHKERRQ(ierr);
  res1 = PetscRealPart(v);
  if (hess) {
    Mat       H;
    Vec       M2;
    PetscBool expl;

    ierr = PetscOptionsGetBool(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-taylor_ts_hessian_explicit",(expl=PETSC_FALSE,&expl),NULL);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)ts),&H);CHKERRQ(ierr);
    ierr = TSComputeHessian(ts,t0,dt,tf,X,design,H);CHKERRQ(ierr);
    if (expl) {
      Mat He;

      ierr = MatComputeOperator(H,NULL,&He);CHKERRQ(ierr);
      ierr = MatDestroy(&H);CHKERRQ(ierr);
      H    = He;
    }
    ierr = VecDuplicate(M,&M2);CHKERRQ(ierr);
    ierr = MatMult(H,dM,M2);CHKERRQ(ierr);
    ierr = VecDot(M2,dM,&v);CHKERRQ(ierr);
    ierr = VecDestroy(&M2);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
    res2 = PetscRealPart(v);
  }

  /*
    Taylor test:
     - obj(M + h*dM) - obj(M) - h * (G^T * dM) should be O(h^2)
     - obj(M + h*dM) - obj(M) - h * (G^T * dM) - 1/2 * h^2 * (dM^T * H * dM) should be O(h^3)
  */
  for (i = 0; i < n; i++, h /= 2.) {
    PetscReal   objtest;

    ierr  = VecWAXPY(M,h,dM,design);CHKERRQ(ierr);
    ierr  = TSComputeObjectiveAndGradient(ts,t0,dt,tf,X,M,NULL,&objtest);CHKERRQ(ierr);
    tG[i] = PetscAbsReal(objtest-obj-h*res1);
    if (hess) {
      tH[i] = PetscAbsReal(objtest-obj-h*res1-h*h*res2/2.);
    }
  }

  ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ts),&viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"-------------------------- Taylor test results ---------------------------\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\t\tGradient");CHKERRQ(ierr);
  if (hess) {
    ierr = PetscViewerASCIIPrintf(viewer,"\t\t\t\tHessian");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"\t\t\tHessian not tested");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n--------------------------------------------------------------------------\n");CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    PetscReal rate;

    rate = (i > 0 && tG[i] != tG[i-1]) ? -PetscLogReal(tG[i]/tG[i-1])/PetscLogReal(2.0) : 0.0;
    ierr = PetscViewerASCIIPrintf(viewer,"%-#8g\t%-#8g\t%D",(double)tG[i],(double)rate,(PetscInt)PetscMin(PetscRoundReal(rate),2));CHKERRQ(ierr);
    rate = (i > 0 && tH[i] != tH[i-1]) ? -PetscLogReal(tH[i]/tH[i-1])/PetscLogReal(2.0) : 0.0;
    ierr = PetscViewerASCIIPrintf(viewer,"\t%-#8g\t%-#8g\t%D",(double)tH[i],(double)rate,(PetscInt)PetscMin(PetscRoundReal(rate),3));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }

  ierr = PetscFree2(tG,tH);CHKERRQ(ierr);
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = VecDestroy(&G);CHKERRQ(ierr);
  if (!ddesign) {
    ierr = VecDestroy(&dM);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <petscopt/private/tssplitjacimpl.h>
#include <petscdm.h>
/* MFFD data struct to check Hessian terms */
typedef struct {
  Vec L;
  Vec U;
  Vec Udot;
  Vec M;

  PetscBool ic;
  PetscInt  deriv;
  PetscInt  sample;

  TS ts;
  PetscReal t;

} MFFDCtx;

static PetscErrorCode ResidualHessian_Private(void *ctx, Vec X, Vec Y)
{
  MFFDCtx*       mffd = (MFFDCtx*)(ctx);
  TSOpt          tsopt;
  Mat            G;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTSOpt(mffd->ts,&tsopt);CHKERRQ(ierr);
  if (mffd->ic) {
    ierr = TSOptEvalGradientIC(tsopt,mffd->t,(mffd->deriv == 0) ? X : mffd->U,
                                             (mffd->deriv == 1) ? X : mffd->M,
                                             mffd->sample ? NULL : &G,
                                             mffd->sample ? &G : NULL);CHKERRQ(ierr);
  } else {
    if (mffd->sample < 2) {
      Mat J_U,pJ_U,J_Udot,pJ_Udot;

      if (mffd->deriv == 2) {
        DM  dm;
        Vec U0;

        ierr = TSGetDM(mffd->ts,&dm);CHKERRQ(ierr);
        ierr = DMGetGlobalVector(dm,&U0);CHKERRQ(ierr);
        ierr = TSSetUpFromDesign(mffd->ts,U0,X);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(dm,&U0);CHKERRQ(ierr);
      }
      ierr = TSGetSplitJacobians(mffd->ts,&J_U,&pJ_U,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
      ierr = TSComputeSplitJacobians(mffd->ts,mffd->t,(mffd->deriv == 0) ? X : mffd->U,
                                                      (mffd->deriv == 1) ? X : mffd->Udot,
                                                      J_U,pJ_U,J_Udot,pJ_Udot);CHKERRQ(ierr);
      G = mffd->sample ? J_Udot : J_U;
    } else {
      ierr = TSOptEvalGradientDAE(tsopt,mffd->t,(mffd->deriv == 0) ? X : mffd->U,
                                                (mffd->deriv == 1) ? X : mffd->Udot,
                                                (mffd->deriv == 2) ? X : mffd->M,&G,NULL);CHKERRQ(ierr);
    }
  }
  if (G) {
    ierr = MatMultTranspose(G,mffd->L,Y);CHKERRQ(ierr);
  } else {
    ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode HessianMult_Private(Mat H, Vec X, Vec Y)
{
  MFFDCtx*       mffd;
  TSOpt          tsopt;
  PetscInt       w0,w1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(H,(void**)&mffd);CHKERRQ(ierr);
  w0   = mffd->sample;
  w1   = mffd->deriv;
  ierr = TSGetTSOpt(mffd->ts,&tsopt);CHKERRQ(ierr);
  if (mffd->ic) {
    ierr = TSOptEvalHessianIC(tsopt,w0,w1,mffd->t,mffd->U,mffd->M,mffd->L,X,Y);CHKERRQ(ierr);
  } else {
    ierr = TSOptEvalHessianDAE(tsopt,w0,w1,mffd->t,mffd->U,mffd->Udot,mffd->M,mffd->L,X,Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSCheckHessian_Private(TS ts, PetscReal t, Vec U, Vec Udot, Vec design, Vec L, PetscBool ic)
{
  MFFDCtx        mffd;
  Mat            H,He;
  Vec            base;
  PetscInt       samples,i,j,m,n,M,N;
  MPI_Comm       comm;
  char           dstr[8],sstr[8];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mffd.ts   = ts;
  mffd.ic   = ic;
  mffd.t    = t;
  mffd.U    = U;
  mffd.Udot = Udot;
  mffd.M    = design;
  mffd.L    = L;
  samples   = ic ? 2 : 3;
  for (i=0;i<samples;i++) {
    mffd.sample = i;
    for (j=0;j<samples;j++) {
      mffd.deriv = j;

      comm = PetscObjectComm((PetscObject)ts);
      if (mffd.deriv == samples - 1) {
        ierr = PetscStrcpy(dstr,"M");CHKERRQ(ierr);
        ierr = VecGetSize(design,&N);CHKERRQ(ierr);
        ierr = VecGetLocalSize(design,&n);CHKERRQ(ierr);
        base = mffd.M;
      } else {
        if (mffd.deriv == 0) {
          ierr = PetscStrcpy(dstr,"U");CHKERRQ(ierr);
          base = mffd.U;
        } else {
          ierr = PetscStrcpy(dstr,"Udot");CHKERRQ(ierr);
          base = mffd.Udot;
        }
        ierr = VecGetSize(U,&N);CHKERRQ(ierr);
        ierr = VecGetLocalSize(U,&n);CHKERRQ(ierr);
      }
      if (mffd.sample == samples - 1) {
        ierr = PetscStrcpy(sstr,"M");CHKERRQ(ierr);
        ierr = VecGetSize(design,&M);CHKERRQ(ierr);
        ierr = VecGetLocalSize(design,&m);CHKERRQ(ierr);
      } else {
        if (mffd.sample == 0) {
          ierr = PetscStrcpy(sstr,"U");CHKERRQ(ierr);
        } else {
          ierr = PetscStrcpy(sstr,"Udot");CHKERRQ(ierr);
        }
        ierr = VecGetSize(U,&M);CHKERRQ(ierr);
        ierr = VecGetLocalSize(U,&m);CHKERRQ(ierr);
      }
      ierr = MatCreate(comm,&H);CHKERRQ(ierr);
      ierr = MatSetSizes(H,m,n,M,N);CHKERRQ(ierr);
      ierr = MatSetType(H,MATMFFD);CHKERRQ(ierr);
      ierr = MatSetUp(H);CHKERRQ(ierr);

      ierr = MatMFFDSetBase(H,base,NULL);CHKERRQ(ierr);
      ierr = MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))ResidualHessian_Private,&mffd);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatComputeOperator(H,NULL,&He);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Hessian %s MFFD_%s%s\n",ic ? "IC" : "DAE",sstr,dstr);CHKERRQ(ierr);
      ierr = MatView(He,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&He);CHKERRQ(ierr);
      ierr = MatDestroy(&H);CHKERRQ(ierr);

      ierr = MatCreateShell(comm,m,n,M,N,&mffd,&H);CHKERRQ(ierr);
      ierr = MatShellSetOperation(H,MATOP_MULT,(void(*)(void))HessianMult_Private);CHKERRQ(ierr);
      ierr = MatComputeOperator(H,NULL,&He);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Hessian %s SHELL_%s%s\n",ic ? "IC" : "DAE",sstr,dstr);CHKERRQ(ierr);
      ierr = MatView(He,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&He);CHKERRQ(ierr);
      ierr = MatDestroy(&H);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSCheckHessianIC(TS ts, PetscReal t0, Vec U0, Vec design, Vec L)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCheckHessian_Private(ts,t0,U0,NULL,design,L,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSCheckHessianDAE(TS ts, PetscReal t, Vec U, Vec Udot, Vec design, Vec L)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCheckHessian_Private(ts,t,U,Udot,design,L,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

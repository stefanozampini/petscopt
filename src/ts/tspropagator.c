#include <petscopt/tsopt.h>
#include <petscopt/tsobj.h>
#include <petscopt/adjointts.h>
#include <petscopt/tlmts.h>
#include <petscopt/private/tsoptimpl.h>
#include <petscopt/private/adjointtsimpl.h>
#include <petsc/private/tshistoryimpl.h>

/* ------------------ Routines for the Mat that represents the linearized propagator ----------------------- */
typedef struct {
  TS           model;
  TS           lts;
  TS           adjlts;
  Vec          x0;
  Mat          P;
  TSTrajectory tj;
  PetscReal    t0;
  PetscReal    tf;
} MatPropagator_Ctx;

static PetscErrorCode MatDestroy_Propagator(Mat A)
{
  MatPropagator_Ctx *prop;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&prop->tj);CHKERRQ(ierr);
  ierr = VecDestroy(&prop->x0);CHKERRQ(ierr);
  ierr = TSDestroy(&prop->adjlts);CHKERRQ(ierr);
  ierr = TSDestroy(&prop->lts);CHKERRQ(ierr);
  ierr = TSDestroy(&prop->model);CHKERRQ(ierr);
  ierr = MatDestroy(&prop->P);CHKERRQ(ierr);
  ierr = PetscFree(prop);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Dummy objective function to not have TSAddObjective complaining about a null objective */
/* LCOV_EXCL_START */
static PetscErrorCode TLMTS_dummyOBJ(Vec U, Vec M, PetscReal time, PetscReal *f, void *ctx)
{
  PetscFunctionBegin;
  *f = 0.0;
  PetscFunctionReturn(0);
}
/* LCOV_EXCL_STOP */

/* Just a silly function to pass information to initialize the adjoint variables */
static PetscErrorCode TLMTS_dummyRHS(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(U,grad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Propagator(Mat A, Vec x, Vec y)
{
  MatPropagator_Ctx *prop;
  PetscErrorCode    ierr;
  PetscBool         istr,dadj,dtlm,done;
  PetscReal         dt;
  Vec               tlmdesign,tlmworkrhs;
  TSTrajectory      otrj;

  PetscFunctionBegin;
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);
  ierr = TSSetUp(prop->lts);CHKERRQ(ierr);
  if (!prop->adjlts) {
    Vec L;

    ierr = TSCreateAdjointTS(prop->lts,&prop->adjlts);CHKERRQ(ierr);
    ierr = TSSetFromOptions(prop->adjlts);CHKERRQ(ierr);
    ierr = TSGetSolution(prop->adjlts,&L);CHKERRQ(ierr);
    if (!L) {
      ierr = VecDuplicate(prop->x0,&L);CHKERRQ(ierr);
      ierr = TSSetSolution(prop->adjlts,L);CHKERRQ(ierr);
      ierr = VecDestroy(&L);CHKERRQ(ierr);
    }
  }
  otrj = prop->model->trajectory;
  prop->model->trajectory = prop->tj;
  prop->lts->trajectory = prop->tj;

  /* Need to setup the model TS, as the tlm solver depends on it (relevant callbacks)
     Note that x0 should not be changed, as we are using the same design vector
     used in MatPropagatorUpdate_Propagator */
  ierr = TLMTSGetDesignVec(prop->lts,&tlmdesign);CHKERRQ(ierr);
  ierr = VecLockReadPop(prop->x0);CHKERRQ(ierr);
  ierr = TSSetUpFromDesign(prop->model,prop->x0,tlmdesign);CHKERRQ(ierr);
  ierr = VecLockReadPush(prop->x0);CHKERRQ(ierr);
  ierr = AdjointTSSetDesignVec(prop->adjlts,tlmdesign);CHKERRQ(ierr);
  ierr = AdjointTSSetTimeLimits(prop->adjlts,prop->t0,prop->tf);CHKERRQ(ierr);
  ierr = AdjointTSSetQuadratureVec(prop->adjlts,y);CHKERRQ(ierr);
  /* Initialize adjoint variables using P^T x or x */
  ierr = TLMTSGetRHSVec(prop->lts,&tlmworkrhs);CHKERRQ(ierr);
  if (prop->P) {
    ierr = MatMultTranspose(prop->P,x,tlmworkrhs);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,tlmworkrhs);CHKERRQ(ierr);
  }
  ierr = AdjointTSComputeInitialConditions(prop->adjlts,tlmworkrhs,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetStepNumber(prop->adjlts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(prop->adjlts);CHKERRQ(ierr);
  ierr = TSSetTime(prop->adjlts,prop->t0);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(prop->tj->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->adjlts,dt);CHKERRQ(ierr);
  istr = PETSC_FALSE;
  if (prop->adjlts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)prop->adjlts->adapt,TSADAPTHISTORY,&istr);CHKERRQ(ierr);
    ierr = TSAdaptHistorySetTSHistory(prop->adjlts->adapt,prop->tj->tsh,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (!istr) { /* if we don't follow the trajectory, we need to match the final time */
    ierr = TSSetMaxSteps(prop->adjlts,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(prop->adjlts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  } else {
    PetscInt nsteps;

    ierr = TSHistoryGetNumSteps(prop->tj->tsh,&nsteps);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(prop->adjlts,nsteps-1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(prop->adjlts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(prop->adjlts,prop->tf);CHKERRQ(ierr);
  ierr = TSSetUp(prop->adjlts);CHKERRQ(ierr);
  ierr = TLMTSIsDiscrete(prop->lts,&dtlm);CHKERRQ(ierr);
  ierr = AdjointTSIsDiscrete(prop->adjlts,&dadj);CHKERRQ(ierr);
  if (dtlm != dadj) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot mix %s TLM and %s adjoint",dtlm ? "discrete" : "continuous",dadj ? "discrete" : "continuous");
  ierr = AdjointTSSolveWithQuadrature_Private(prop->adjlts,&done);CHKERRQ(ierr);
  /* An error here, if any, is unrecoverable */
  if (done) {
    ierr = AdjointTSFinalizeQuadrature(prop->adjlts);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_NOT_CONVERGED,"Adjoint of tangent linear model TS did not converge");
  prop->lts->trajectory = NULL;
  prop->tj = prop->model->trajectory;
  prop->model->trajectory = otrj;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Propagator(Mat A, Vec x, Vec y)
{
  MatPropagator_Ctx *prop;
  PetscErrorCode    ierr;
  PetscReal         dt;
  PetscBool         istr;
  Vec               sol,tlmdesign;
  TSConvergedReason reason;
  TSTrajectory      otrj;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);
  otrj = prop->model->trajectory;
  prop->model->trajectory = prop->tj;

  /* Need to setup the model TS, as the tlm solver depends on it (relevant callbacks)
     Note that x0 should not be changed, as we are using the same design vector
     used in MatPropagatorUpdate_Propagator */
  ierr = TLMTSGetDesignVec(prop->lts,&tlmdesign);CHKERRQ(ierr);
  ierr = VecLockReadPop(prop->x0);CHKERRQ(ierr);
  ierr = TSSetUpFromDesign(prop->model,prop->x0,tlmdesign);CHKERRQ(ierr);
  ierr = VecLockReadPush(prop->x0);CHKERRQ(ierr);

  istr = PETSC_FALSE;
  if (prop->lts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)prop->lts->adapt,TSADAPTHISTORY,&istr);CHKERRQ(ierr);
    ierr = TSAdaptHistorySetTSHistory(prop->lts->adapt,prop->tj->tsh,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = TSHistoryGetTimeStep(prop->tj->tsh,PETSC_FALSE,0,&dt);CHKERRQ(ierr);
  ierr = TLMTSSetPerturbationVec(prop->lts,x);CHKERRQ(ierr);
  ierr = TLMTSComputeInitialConditions(prop->lts,prop->t0,prop->x0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(prop->lts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(prop->lts);CHKERRQ(ierr);
  ierr = TSSetTime(prop->lts,prop->t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->lts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(prop->lts,prop->tf);CHKERRQ(ierr);
  if (istr) {
    PetscInt nsteps;

    ierr = TSHistoryGetNumSteps(prop->tj->tsh,&nsteps);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(prop->lts,nsteps-1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(prop->lts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  } else {
    ierr = TSSetMaxSteps(prop->lts,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(prop->lts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  }
  ierr = TSSolve(prop->lts,NULL);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(prop->lts,&reason);CHKERRQ(ierr);
  ierr = TLMTSSetPerturbationVec(prop->lts,NULL);CHKERRQ(ierr);
  /* This error is unrecoverable */
  if (reason <= TS_CONVERGED_ITERATING) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_NOT_CONVERGED,"Tangent linear model TS did not converge");
  ierr = TSGetSolution(prop->lts,&sol);CHKERRQ(ierr);
  if (prop->P) {
    ierr = MatMult(prop->P,sol,y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(sol,y);CHKERRQ(ierr);
  }
  prop->tj = prop->model->trajectory;
  prop->model->trajectory = otrj;
  PetscFunctionReturn(0);
}

/* solves the forward model and stores its trajectory */
static PetscErrorCode MatPropagatorUpdate_Propagator(Mat A, PetscReal t0, PetscReal dt, PetscReal tf, Vec x0, Vec design)
{
  Vec               osol;
  TSTrajectory      otrj;
  MatPropagator_Ctx *prop;
  TSConvergedReason reason;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);

  /* Need to setup the model TS, as the tlm solver depends on it (relevant callbacks) */
  ierr = TSSetUpFromDesign(prop->model,x0,design);CHKERRQ(ierr);

  ierr = VecLockReadPop(prop->x0);CHKERRQ(ierr);
  ierr = VecCopy(x0,prop->x0);CHKERRQ(ierr);
  ierr = VecLockReadPush(prop->x0);CHKERRQ(ierr);
  prop->t0 = t0;
  prop->tf = tf;
  ierr = TSTrajectoryDestroy(&prop->tj);CHKERRQ(ierr);

  /* Customize nonlinear model */
  ierr = TSSetStepNumber(prop->model,0);CHKERRQ(ierr);
  ierr = TSRestartStep(prop->model);CHKERRQ(ierr);
  ierr = TSSetTime(prop->model,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->model,dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(prop->model,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetMaxTime(prop->model,tf);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(prop->model,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /* Create trajectory object */
  otrj = prop->model->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)prop->model),&prop->model->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(prop->model->trajectory,prop->model,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  ierr = TLMTSIsDiscrete(prop->lts,&flg);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(prop->model->trajectory,flg ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(prop->model->trajectory,prop->model);CHKERRQ(ierr);
  /* we don't have an API for this right now */
  prop->model->trajectory->adjoint_solve_mode = PETSC_FALSE;

  /* Solve the forward nonlinear model in the given time window */
  ierr = TSGetSolution(prop->model,&osol);CHKERRQ(ierr);
  if (!osol) {
    ierr = VecDuplicate(prop->x0,&osol);CHKERRQ(ierr);
    ierr = TSSetSolution(prop->model,osol);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)osol);CHKERRQ(ierr);
  }
  ierr = VecCopy(prop->x0,osol);CHKERRQ(ierr);
  ierr = TSSolve(prop->model,NULL);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(prop->model,&reason);CHKERRQ(ierr);
  /* This error is unrecoverable */
  if (reason <= TS_CONVERGED_ITERATING) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_NOT_CONVERGED,"Model TS did not converge");
  prop->tj = prop->model->trajectory;
  prop->model->trajectory = otrj;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSCreatePropagatorMat_Private(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec x0, Vec design, Mat P, Mat *A)
{
  MatPropagator_Ctx *prop;
  TSOpt             tsopt;
  TSAdapt           adapt;
  Vec               X;
  PetscInt          M,N,m,n,rbs,cbs;
  PetscBool         has,hasm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&prop);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  prop->model = ts;
  if (P) {
    PetscBool   match;
    PetscLayout pmap,map;

    ierr = PetscObjectReference((PetscObject)P);CHKERRQ(ierr);
    ierr = MatDestroy(&prop->P);CHKERRQ(ierr);
    prop->P = P;
    ierr = MatGetLayouts(P,NULL,&pmap);CHKERRQ(ierr);
    ierr = VecGetLayout(x0,&map);CHKERRQ(ierr);
    ierr = PetscLayoutCompare(map,pmap,&match);CHKERRQ(ierr);
    if (!match) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The layouts of P and x0 do not match");
    ierr = MatGetSize(P,&M,NULL);CHKERRQ(ierr);
    ierr = MatGetLocalSize(P,&m,NULL);CHKERRQ(ierr);
    ierr = MatGetBlockSizes(P,&rbs,NULL);CHKERRQ(ierr);
  } else {
    ierr = VecGetSize(x0,&M);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x0,&m);CHKERRQ(ierr);
    ierr = VecGetBlockSize(x0,&rbs);CHKERRQ(ierr);
  }
  ierr = TSGetTSOpt(prop->model,&tsopt);CHKERRQ(ierr);
  if (!design) {
    Mat G_m;

    ierr = TSOptEvalGradientIC(tsopt,0.0,NULL,NULL,NULL,&G_m);CHKERRQ(ierr);
    if (G_m) {
      ierr = MatCreateVecs(G_m,&design,NULL);CHKERRQ(ierr);
    } else {
      ierr = VecDuplicate(x0,&design);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscObjectReference((PetscObject)design);CHKERRQ(ierr);
  }
  ierr = VecGetSize(design,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(design,&n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(design,&cbs);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(*A,rbs,cbs);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSHELL);CHKERRQ(ierr);
  ierr = MatShellSetContext(*A,(void *)prop);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_MULT,(void (*)())MatMult_Propagator);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,(void (*)())MatMultTranspose_Propagator);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_DESTROY,(void (*)())MatDestroy_Propagator);CHKERRQ(ierr);

  /* create the tangent linear model solver */
  ierr = TSCreateTLMTS(prop->model,&prop->lts);CHKERRQ(ierr);
  ierr = VecDuplicate(x0,&X);CHKERRQ(ierr);
  ierr = TSSetSolution(prop->lts,X);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSGetAdapt(prop->lts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTHISTORY);CHKERRQ(ierr);
  ierr = TLMTSSetDesignVec(prop->lts,design);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)design);CHKERRQ(ierr);
  ierr = TSSetFromOptions(prop->lts);CHKERRQ(ierr);
  ierr = TSResetObjective(prop->lts);CHKERRQ(ierr);
  ierr = TLMTSSetUpStep(prop->lts);CHKERRQ(ierr);

  ierr = VecDuplicate(x0,&prop->x0);CHKERRQ(ierr);
  ierr = VecCopy(x0,prop->x0);CHKERRQ(ierr);
  ierr = VecLockReadPush(prop->x0);CHKERRQ(ierr); /* this vector is locked since it stores the initial conditions */
  ierr = MatPropagatorUpdate_Propagator(*A,t0,dt,tf,x0,design);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);

  /* model sampling can terminate before tf due to events */
  ierr = TSGetTime(prop->model,&prop->tf);CHKERRQ(ierr);

  ierr = TSAddObjective(prop->lts,prop->tf,TLMTS_dummyOBJ,TLMTS_dummyRHS,NULL,
                        NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /* we need to call this since we will then compute the adjoint of the TLM */
  ierr = TSSetTSOpt(prop->lts,tsopt);CHKERRQ(ierr);

  ierr = TSOptHasGradientIC(tsopt,&has);CHKERRQ(ierr);
  ierr = TSOptHasGradientDAE(tsopt,&hasm,NULL);CHKERRQ(ierr);
  if (!has && !hasm) { /* we compute a linear dependence on u_0 by default */
    ierr = TSSetGradientIC(prop->lts,NULL,NULL,TSEvalGradientICDefault,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSCreatePropagatorMat - Creates a Mat object that behaves like a linearized propagator of a time stepper on the time window [t0,tf].

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context
.  t0     - the initial time
.  dt     - the initial time step
.  tf     - the final time
.  x0     - the vector of initial conditions
.  design - the vector of design
-  P      - an optional projection

   Output Parameters:
.  A  - the Mat object

   Notes: Internally, the Mat object solves the Tangent Linear Model (TLM) during MatMult() and the adjoint of the TLM during MatMultTranspose().
          The design vector can be NULL if the Jacobians (wrt to the parameters) of the DAE and of the initial conditions does not explicitly depend on it.
          The projection P is intended to analyze problems in Generalized Stability Theory of the type

$           argmax ||P du_T||^2
$          ||du_0||=1

          when one can be interested in the norm of the final state in a subspace.
          The projector is applied (via MatMult) on the final state computed by the forward Tangent Linear Model.
          The transposed action of P is instead used to initialize the adjoint of the Tangent Linear Model.
          Note that the role of P is somewhat different from that of the matrix representing the norm in the state variables.
          If P is provided, the row layout of A is the same of that of P. Otherwise, it is the same of that of x0.
          The column layout of A is the same of that of the design vector. If the latter is not provided, it is inherited from x0.
          Note that the column layout of P should be compatible with that of x0.

   Level: developer

.seealso: TSSetGradientDAE(), TSSetGradientIC()
@*/
PetscErrorCode TSCreatePropagatorMat(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec x0, Vec design, Mat P, Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidLogicalCollectiveReal(ts,dt,3);
  PetscValidLogicalCollectiveReal(ts,tf,4);
  PetscValidHeaderSpecific(x0,VEC_CLASSID,5);
  if (design) PetscValidHeaderSpecific(design,VEC_CLASSID,6);
  if (P) PetscValidHeaderSpecific(P,MAT_CLASSID,7);
  PetscValidPointer(A,8);
  ierr = MatCreate(PetscObjectComm((PetscObject)ts),A);CHKERRQ(ierr);
  ierr = TSCreatePropagatorMat_Private(ts,t0,dt,tf,x0,design,P,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

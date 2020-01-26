#include <petscopt/private/discretetsimpl.h>
#include <petscopt/adjointts.h>
#include <petscopt/tlmts.h>
#include <petsc/private/tsimpl.h>
#include <petsc/private/tshistoryimpl.h>
#include <petscdm.h>

#define TS_RK_DISCRETE_MAXSTAGES 32

PetscErrorCode TSStep_Adjoint_RK(TS ts)
{
  TS              fwdts;
  DM              dm;
  Vec             *Y,L,*LY,F,Thetas[TS_RK_DISCRETE_MAXSTAGES],Lt[TS_RK_DISCRETE_MAXSTAGES+1];
  Vec             direction,*FOAY = NULL,*TLMY = NULL,Q;
  const PetscReal *A,*b,*c;
  PetscScalar     w[TS_RK_DISCRETE_MAXSTAGES+1];
  PetscReal       h = ts->time_step, at = ts->ptime, dummy;
  PetscInt        step,astep,tstep;
  PetscInt        s,sf,i,j,k,skps,acts;
  PetscBool       flg,sonly,quad;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = AdjointTSGetModelTS(ts,&fwdts);CHKERRQ(ierr);
#if PETSC_VERSION_GE(3,13,0)
  ierr = TSRKGetTableau(fwdts,&s,&A,&b,&c,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
#else
  SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"RK discrete adjoints need PETSc version greater or equal 3.13");
#endif
  if (s > TS_RK_DISCRETE_MAXSTAGES) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Not coded for %D stages",s);
  ierr = TSTrajectoryGetSolutionOnly(fwdts->trajectory,&sonly);CHKERRQ(ierr);
  if (sonly) SETERRQ(PetscObjectComm((PetscObject)fwdts->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
  ierr = TSGetStepNumber(ts,&astep);CHKERRQ(ierr);
  ierr = TSTrajectoryGetNumSteps(fwdts->trajectory,&tstep);CHKERRQ(ierr);
  step = tstep - astep - 1;
  ierr = TSTrajectoryGet(fwdts->trajectory,fwdts,step,&dummy);CHKERRQ(ierr);
  ierr = TSGetStages(fwdts,&sf,&Y);CHKERRQ(ierr);
  if (s != sf) SETERRQ2(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != %D",s,sf);
  for (i=0; i<s; i++) {
    ierr = VecLockReadPush(Y[i]);CHKERRQ(ierr);
  }
  ierr = TSGetStages(ts,&sf,&LY);CHKERRQ(ierr);
  if (s != sf) SETERRQ2(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != %D",s,sf);
  ierr = AdjointTSGetDirectionVec(ts,&direction);CHKERRQ(ierr);
  if (direction) { /* second order adjoints need tangent linear model and first-order adjoint (if not GN) history */
    TS        tlmts,foats;
    PetscReal dummy;

    ierr = AdjointTSGetTLMTSAndFOATS(ts,&tlmts,&foats);CHKERRQ(ierr);
    ierr = TSTrajectoryGetSolutionOnly(tlmts->trajectory,&sonly);CHKERRQ(ierr);
    if (sonly) SETERRQ(PetscObjectComm((PetscObject)tlmts->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
    ierr = TSTrajectoryGet(tlmts->trajectory,tlmts,step,&dummy);CHKERRQ(ierr);
    ierr = TSGetStages(tlmts,&sf,&TLMY);CHKERRQ(ierr);
    if (s != sf) SETERRQ2(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != %D",s,sf);
    for (i=0; i<s; i++) {
      ierr = VecLockReadPush(TLMY[i]);CHKERRQ(ierr);
    }
    if (foats) {
      ierr = TSTrajectoryGetSolutionOnly(foats->trajectory,&sonly);CHKERRQ(ierr);
      if (sonly) SETERRQ(PetscObjectComm((PetscObject)foats->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
      ierr = TSTrajectoryGet(foats->trajectory,foats,astep+1,&dummy);CHKERRQ(ierr);
      ierr = TSGetStages(foats,&sf,&FOAY);CHKERRQ(ierr);
      if (s != sf) SETERRQ2(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != %D",s,sf);
      for (i=0; i<s; i++) {
        ierr = VecLockReadPush(FOAY[i]);CHKERRQ(ierr);
      }
    }
  }
  ierr = TSGetSolution(ts,&L);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    ierr = DMGetGlobalVector(dm,&Thetas[i]);CHKERRQ(ierr);
  }
  ierr = DMGetGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&Q);CHKERRQ(ierr);
  /* If b[s-1] == 0, skip last forward stage */
  skps = !b[s-1] ? 1 : 0;
  acts = s-skps;
  quad = PETSC_FALSE;
  for (k=skps; k<s; k++) {
    Vec       FOAL,TLMU;
    PetscInt  i = s - k - 1;
    PetscReal astage_time = at + (1. - c[i]) * h;
    PetscInt  cnt = 0;
    PetscReal bs = b[i] ? b[i] : 1.0;

    if (b[i]) {
      w[0]  = 1.0;
      Lt[0] = L;
      cnt++;
    }
    for (j=i+1; j<s; j++) {
      PetscReal aw = A[j*s+i];
      if (!aw) continue;
      w[cnt]  = aw / bs;
      Lt[cnt] = Thetas[j];
      cnt++;
    }
    ierr = VecSet(LY[k],0.0);CHKERRQ(ierr);
    if (cnt) {
      Mat J,Jp;

      ts->rhsjacobian.time = PETSC_MIN_REAL;
      ierr = TSGetRHSJacobian(ts,&J,&Jp,NULL,NULL);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobian(ts,astage_time,Y[i],J,Jp);CHKERRQ(ierr);
      ierr = VecMAXPY(LY[k],cnt,w,Lt);CHKERRQ(ierr);
      ierr = MatMultTranspose(J,LY[k],Thetas[i]);CHKERRQ(ierr);
      ierr = VecScale(Thetas[i],h*bs);CHKERRQ(ierr);
    } else {
      ierr = VecSet(Thetas[i],0.0);CHKERRQ(ierr);
    }

    FOAL = FOAY ? FOAY[k] : NULL;
    TLMU = TLMY ? TLMY[i] : NULL;
    if (b[i]) {
      ierr = AdjointTSComputeForcing(ts,astage_time,Y[i],NULL,FOAL,NULL,TLMU,NULL,&flg,F);CHKERRQ(ierr);
      if (flg) {
        ierr = VecAXPY(Thetas[i],-h*b[i],F);CHKERRQ(ierr);
      }
    }
    if (cnt) {
      ierr = VecSet(F,0.0);CHKERRQ(ierr);
      ierr = AdjointTSComputeQuadrature(ts,astage_time,Y[i],NULL,LY[k],FOAL,NULL,TLMU,NULL,&flg,F);CHKERRQ(ierr);
      if (flg) {
        if (!quad) { ierr = VecSet(Q,0.0);CHKERRQ(ierr); }
        quad = PETSC_TRUE;
        ierr = VecAXPY(Q,h*bs,F);CHKERRQ(ierr);
      }
    }
  }

  for (j=0; j<acts; j++) w[j] = 1.0;
  ierr = VecMAXPY(L,acts,w,Thetas);CHKERRQ(ierr);
  if (quad) {
    ierr = VecAXPY(L,1.0,Q);CHKERRQ(ierr);
  }
  for (i=0; i<s; i++) {
    ierr = DMRestoreGlobalVector(dm,&Thetas[i]);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(dm,&Q);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&F);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    ierr = VecLockReadPop(Y[i]);CHKERRQ(ierr);
  }
  if (TLMY) {
    for (i=0; i<s; i++) {
      ierr = VecLockReadPop(TLMY[i]);CHKERRQ(ierr);
    }
  }
  if (FOAY) {
    for (i=0; i<s; i++) {
      ierr = VecLockReadPop(FOAY[i]);CHKERRQ(ierr);
    }
  }
  ts->ptime += h;

  ierr = TSHistoryGetTimeStep(fwdts->trajectory->tsh,PETSC_TRUE,astep+1,&ts->time_step);CHKERRQ(ierr);
  if (ts->time_step == 0.0) ts->reason = TS_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

PetscErrorCode TSStep_TLM_RK(TS ts)
{
  TS              fwdts;
  DM              dm;
  Vec             *Y,U,YdotRHS[TS_RK_DISCRETE_MAXSTAGES],*fwdY,F;
  PetscScalar     w[TS_RK_DISCRETE_MAXSTAGES];
  PetscReal       t = ts->ptime,dummy;
  PetscReal       h = ts->time_step;
  const PetscReal *A,*c,*b;
  PetscInt        s,i,j;
  PetscBool       flg,sonly;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TLMTSGetModelTS(ts,&fwdts);CHKERRQ(ierr);
#if PETSC_VERSION_GE(3,13,0)
  ierr = TSRKGetTableau(fwdts,&s,&A,&b,&c,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
#else
  SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"RK discrete tangent linear models need PETSc version greater or equal 3.13");
#endif
  if (s > TS_RK_DISCRETE_MAXSTAGES) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Not coded for %D stages",s);
  ierr = TSTrajectoryGetSolutionOnly(fwdts->trajectory,&sonly);CHKERRQ(ierr);
  if (sonly) SETERRQ(PetscObjectComm((PetscObject)fwdts->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
  ierr = TSGetStepNumber(ts,&i);CHKERRQ(ierr);
  ierr = TSTrajectoryGet(fwdts->trajectory,fwdts,i+1,&dummy);CHKERRQ(ierr);
  ierr = TSGetStages(fwdts,&i,&fwdY);CHKERRQ(ierr);
  if (s != i) SETERRQ2(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != %D",s,i);
  ierr = TSGetStages(ts,NULL,&Y);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    ierr = VecLockReadPush(fwdY[i]);CHKERRQ(ierr);
  }
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    ierr = DMGetGlobalVector(dm,&YdotRHS[i]);CHKERRQ(ierr);
  }
  ierr = DMGetGlobalVector(dm,&F);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    Mat       J,Jp;
    PetscReal stage_time = t + h*c[i];

    ierr = VecCopy(U,Y[i]);CHKERRQ(ierr);
    for (j=0; j<i; j++) w[j] = h*A[i*s+j];
    ierr = VecMAXPY(Y[i],i,w,YdotRHS);CHKERRQ(ierr);

    ts->rhsjacobian.time = PETSC_MIN_REAL;
    ierr = TSGetRHSJacobian(ts,&J,&Jp,NULL,NULL);CHKERRQ(ierr);
    ierr = TSComputeRHSJacobian(ts,stage_time,fwdY[i],J,Jp);CHKERRQ(ierr);
    ierr = MatMult(J,Y[i],YdotRHS[i]);CHKERRQ(ierr);
    ierr = TLMTSComputeForcing(ts,stage_time,fwdY[i],NULL,&flg,F);CHKERRQ(ierr);
    if (flg) {
      ierr = VecAXPY(YdotRHS[i],-1.0,F);CHKERRQ(ierr);
    }
  }
  for (j=0; j<s; j++) w[j] = h*b[j];
  ierr = VecMAXPY(U,s,w,YdotRHS);CHKERRQ(ierr);

  for (i=0; i<s; i++) {
    ierr = DMRestoreGlobalVector(dm,&YdotRHS[i]);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(dm,&F);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    ierr = VecLockReadPop(fwdY[i]);CHKERRQ(ierr);
  }
  ts->ptime += h;
  ierr = TSGetStepNumber(ts,&i);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(fwdts->trajectory->tsh,PETSC_FALSE,i+1,&ts->time_step);CHKERRQ(ierr);
  if (ts->time_step == 0.0) ts->reason = TS_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

/* The forward step of the endpoint variant assumes linear time-independent mass matrix */
/* Otherwise, we should sample J_xdot at stage_time, and J_x at the beginning of the step and compute s * J_xdot + (theta-1)/theta J_x */
PetscErrorCode TSStep_Adjoint_Theta(TS ts)
{
  TS             fwdts, tlmts = NULL, foats = NULL;
  SNES           snes;
  KSP            ksp;
  DM             dm;
  Mat            J, Jp;
  Vec            *LY,L,*fwdY,fwdYdot,fwdYSol,F,direction,Q;
  Vec            *FOAY = NULL,*TLMY = NULL,TLMSol = NULL,FOAL = NULL,FOASol = NULL,FOALdot = NULL,TLMU = NULL,TLMUdot = NULL;
  Vec            beY0 = NULL, beTLM0 = NULL;
  PetscReal      at = ts->ptime, dummy;
  PetscReal      h = ts->time_step;
  PetscReal      theta, astage_time, s;
  PetscInt       i, j, astep, tstep, step;
  PetscBool      endpoint, flg, quad, beuler = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AdjointTSGetModelTS(ts,&fwdts);CHKERRQ(ierr);
  ierr = TSThetaGetTheta(fwdts,&theta);CHKERRQ(ierr);
  if (theta == 1.0) beuler = PETSC_TRUE;
  ierr = TSThetaGetEndpoint(fwdts,&endpoint);CHKERRQ(ierr);
  ierr = TSTrajectoryGetSolutionOnly(fwdts->trajectory,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)fwdts->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
  ierr = TSGetStepNumber(ts,&astep);CHKERRQ(ierr);
  ierr = TSTrajectoryGetNumSteps(fwdts->trajectory,&tstep);CHKERRQ(ierr);
  step = tstep - astep - 1;
  if (beuler && !endpoint) {
    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&beY0);CHKERRQ(ierr);
    ierr = TSTrajectoryGet(fwdts->trajectory,fwdts,step-1,&dummy);CHKERRQ(ierr);
    ierr = TSGetSolution(fwdts,&fwdYSol);CHKERRQ(ierr);
    ierr = VecCopy(fwdYSol,beY0);CHKERRQ(ierr);
  }
  ierr = TSTrajectoryGet(fwdts->trajectory,fwdts,step,&dummy);CHKERRQ(ierr);

  ierr = TSGetSolution(fwdts,&fwdYSol);CHKERRQ(ierr);
  ierr = TSGetStages(fwdts,&i,&fwdY);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&j,&LY);CHKERRQ(ierr);
  if (i != j || i != 1) SETERRQ3(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != %D || %D != 1",i,j,i);
  ierr = VecLockReadPush(fwdYSol);CHKERRQ(ierr);
  ierr = VecLockReadPush(fwdY[0]);CHKERRQ(ierr);
  ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&fwdYdot);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&L);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&Q);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&J,&Jp,NULL,NULL);CHKERRQ(ierr);

  astage_time = at + (endpoint ? 0.0 : 1.0 - theta)*h;
  quad = PETSC_FALSE;
  ierr = AdjointTSGetDirectionVec(ts,&direction);CHKERRQ(ierr);
  if (direction) { /* second order adjoints need tangent linear model and first-order adjoint (if not GN) history */
    PetscReal dummy;

    ierr = AdjointTSGetTLMTSAndFOATS(ts,&tlmts,&foats);CHKERRQ(ierr);
    ierr = TSTrajectoryGetSolutionOnly(tlmts->trajectory,&flg);CHKERRQ(ierr);
    if (flg) SETERRQ(PetscObjectComm((PetscObject)tlmts->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
    if (beuler && !endpoint) {
      ierr = TSGetDM(tlmts,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&beTLM0);CHKERRQ(ierr);
      ierr = TSTrajectoryGet(tlmts->trajectory,tlmts,step-1,&dummy);CHKERRQ(ierr);
      ierr = TSGetSolution(tlmts,&TLMSol);CHKERRQ(ierr);
      ierr = VecCopy(TLMSol,beTLM0);CHKERRQ(ierr);
    }
    ierr = TSTrajectoryGet(tlmts->trajectory,tlmts,step,&dummy);CHKERRQ(ierr);
    ierr = TSGetStages(tlmts,&i,&TLMY);CHKERRQ(ierr);
    if (i != 1) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != 1",i);
    ierr = TSGetSolution(tlmts,&TLMSol);CHKERRQ(ierr);
    ierr = VecLockReadPush(TLMSol);CHKERRQ(ierr);
    ierr = VecLockReadPush(TLMY[0]);CHKERRQ(ierr);
    ierr = TSGetDM(tlmts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&TLMUdot);CHKERRQ(ierr);
    if (foats) {
      ierr = TSTrajectoryGetSolutionOnly(foats->trajectory,&flg);CHKERRQ(ierr);
      if (flg) SETERRQ(PetscObjectComm((PetscObject)foats->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
      ierr = TSTrajectoryGet(foats->trajectory,foats,astep+1,&dummy);CHKERRQ(ierr);
      ierr = TSGetStages(foats,&i,&FOAY);CHKERRQ(ierr);
      if (i != 1) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != 1",i);
      ierr = TSGetSolution(foats,&FOASol);CHKERRQ(ierr);
      ierr = VecLockReadPush(FOASol);CHKERRQ(ierr);
      ierr = VecLockReadPush(FOAY[0]);CHKERRQ(ierr);
      ierr = TSGetDM(foats,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&FOALdot);CHKERRQ(ierr);
    }
  }
  if (endpoint && !beuler) {
    s    = 1.0/(theta*h);
    ierr = VecAXPBYPCZ(fwdYdot,s,-s,0.0,fwdYSol,fwdY[0]);CHKERRQ(ierr);
    if (direction) {
      FOAL = FOAY ? FOAY[0] : NULL;
      ierr = VecAXPBYPCZ(TLMUdot,s,-s,0.0,TLMSol,TLMY[0]);CHKERRQ(ierr);
      TLMU = TLMSol;
      if (FOAL) {
        ierr = VecAXPBY(FOALdot,s,0.0,FOAL);CHKERRQ(ierr);
      }
    }

    ierr = AdjointTSComputeForcing(ts,astage_time,fwdYSol,fwdYdot,FOAL,FOALdot,TLMU,TLMUdot,&flg,F);CHKERRQ(ierr);
    ierr = VecScale(L,s);CHKERRQ(ierr);
    if (flg) {
      ierr = VecAXPY(L,-1.0,F);CHKERRQ(ierr);
    }
    ierr = TSComputeIJacobian(ts,astage_time,fwdYSol,fwdYdot,s,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,J,Jp);CHKERRQ(ierr);
    /* Adjoint stage */
    ierr = KSPSolveTranspose(ksp,L,LY[0]);CHKERRQ(ierr);
    ierr = VecLockReadPush(LY[0]);CHKERRQ(ierr);
    ierr = VecSet(Q,0.0);CHKERRQ(ierr);
    ierr = AdjointTSComputeQuadrature(ts,astage_time,fwdYSol,fwdYdot,LY[0],FOAL,NULL,TLMU,TLMUdot,&quad,Q);CHKERRQ(ierr);
    if (quad) { ierr = VecScale(Q,-theta/(theta-1.0));CHKERRQ(ierr); }

    ierr = VecZeroEntries(fwdYdot);CHKERRQ(ierr);
    s    = 1.0/((theta-1.0)*h);
    ierr = TSComputeIJacobian(ts,at+h,fwdY[0],fwdYdot,s,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    if (direction) {
      FOAL = FOAY ? FOAY[0] : NULL;
      ierr = VecZeroEntries(TLMUdot);CHKERRQ(ierr);
      TLMU = TLMY[0];
      if (FOAL) {
        ierr = VecAXPBY(FOALdot,s,0.0,FOAL);CHKERRQ(ierr);
      }
    }
    ierr = AdjointTSComputeForcing(ts,at+h,fwdY[0],fwdYdot,FOAL,FOALdot,TLMU,TLMUdot,&flg,F);CHKERRQ(ierr);
    if (flg) {
      ierr = MatMultTransposeAdd(J,LY[0],F,F);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(J,LY[0],F);CHKERRQ(ierr);
    }
    if (quad) {
      Vec Q2;

      ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&Q2);CHKERRQ(ierr);
      ierr = VecSet(Q2,0.0);CHKERRQ(ierr);
      ierr = AdjointTSComputeQuadrature(ts,at+h,fwdY[0],fwdYdot,LY[0],FOAL,NULL,TLMU,TLMUdot,&flg,Q2);CHKERRQ(ierr);
      ierr = VecAXPY(Q,1.0,Q2);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&Q2);CHKERRQ(ierr);
    }

    s = (theta-1.0)*h;
    if (quad) {
      ierr = VecAXPBYPCZ(L,-s,s,0.0,Q,F);CHKERRQ(ierr);
    } else {
      ierr = VecAXPBY(L,s,0.0,F);CHKERRQ(ierr);
    }
  } else { /* theta case or beuler (endpoint or not) */
    Vec fwdU = fwdY[0];
    /* need to reconstruct x0 from loaded solution (next time step) and stage solution
       to properly compute fwdYdot.
       This is not doable with BEULER, unless we have the endpoint variant which stores
       it as a stage vector.
    */
    if (endpoint) { beY0 = fwdY[0]; beTLM0 = TLMY ? TLMY[0] : NULL; fwdU = fwdYSol; }
    if (!beY0) {
      s    = 1.0/(h*(theta-1.0));
      ierr = VecAXPBYPCZ(fwdYdot,-s,s,0.0,fwdYSol,fwdY[0]);CHKERRQ(ierr);
    } else {
      s    = 1.0/(theta*h);
      ierr = VecAXPBYPCZ(fwdYdot,s,-s,0.0,fwdYSol,beY0);CHKERRQ(ierr);
    }
    if (direction) {
      if (!beTLM0) {
        s    = 1.0/(h*(theta-1.0));
        ierr = VecAXPBYPCZ(TLMUdot,-s,s,0.0,TLMSol,TLMY[0]);CHKERRQ(ierr);
      } else {
        s    = 1.0/(theta*h);
        ierr = VecAXPBYPCZ(TLMUdot,s,-s,0.0,TLMSol,beTLM0);CHKERRQ(ierr);
      }
      TLMU = endpoint ? TLMSol : TLMY[0];
      FOAL = FOAY ? FOAY[0] : NULL;
      if (FOAL) {
        s    = 1.0/(h*theta);
        ierr = VecAXPBY(FOALdot,s,0.0,FOAL);CHKERRQ(ierr);
      }
    }

    s    = 1.0/(h*theta);
    ierr = AdjointTSComputeForcing(ts,astage_time,fwdU,fwdYdot,FOAL,FOALdot,TLMU,TLMUdot,&flg,F);CHKERRQ(ierr);
    if (flg) {
      ierr = VecAXPBY(F,s,-1.0,L);CHKERRQ(ierr);
    } else {
      ierr = VecAXPBY(F,s,0.0,L);CHKERRQ(ierr);
    }
    ierr = TSComputeIJacobian(ts,astage_time,fwdU,fwdYdot,s,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,J,Jp);CHKERRQ(ierr);
    /* Adjoint stage */
    ierr = KSPSolveTranspose(ksp,F,LY[0]);CHKERRQ(ierr);
    ierr = VecLockReadPush(LY[0]);CHKERRQ(ierr);
    ierr = VecSet(Q,0.0);CHKERRQ(ierr);
    ierr = AdjointTSComputeQuadrature(ts,astage_time,fwdU,fwdYdot,LY[0],FOAL,NULL,TLMU,TLMUdot,&quad,Q);CHKERRQ(ierr);
    if (quad) {
      ierr = VecAXPY(L,h,Q);CHKERRQ(ierr);
    }
    ierr = TSComputeIJacobian(ts,astage_time,fwdU,fwdYdot,0.0,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecZeroEntries(fwdYdot);CHKERRQ(ierr);
    if (TLMUdot) { ierr = VecZeroEntries(TLMUdot);CHKERRQ(ierr); }
    if (FOALdot) { ierr = VecZeroEntries(FOALdot);CHKERRQ(ierr); }
    ierr = AdjointTSComputeForcing(ts,astage_time,fwdU,fwdYdot,FOAL,FOALdot,TLMU,TLMUdot,&flg,F);CHKERRQ(ierr);
    if (flg) {
      ierr = VecAXPY(L,-h,F);CHKERRQ(ierr);
    }
    ierr = MatMultTranspose(J,LY[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(L,-h,F);CHKERRQ(ierr);
    if (endpoint) { beY0 = NULL; beTLM0 = NULL; }
  }
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&Q);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&fwdYdot);CHKERRQ(ierr);
  ierr = VecLockReadPop(fwdY[0]);CHKERRQ(ierr);
  ierr = VecLockReadPop(fwdYSol);CHKERRQ(ierr);
  ierr = VecLockReadPop(LY[0]);CHKERRQ(ierr);
  if (TLMY) {
    ierr = VecLockReadPop(TLMY[0]);CHKERRQ(ierr);
  }
  if (TLMSol) {
    ierr = VecLockReadPop(TLMSol);CHKERRQ(ierr);
  }
  if (FOAY) {
    ierr = VecLockReadPop(FOAY[0]);CHKERRQ(ierr);
  }
  if (FOASol) {
    ierr = VecLockReadPop(FOASol);CHKERRQ(ierr);
  }
  if (tlmts) {
    ierr = TSGetDM(tlmts,&dm);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&TLMUdot);CHKERRQ(ierr);
  }
  if (foats) {
    ierr = TSGetDM(foats,&dm);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&FOALdot);CHKERRQ(ierr);
  }
  if (beY0) {
    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&beY0);CHKERRQ(ierr);
  }
  if (beTLM0) {
    ierr = TSGetDM(tlmts,&dm);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&beTLM0);CHKERRQ(ierr);
  }
  ts->ptime += h;
  ierr = TSHistoryGetTimeStep(fwdts->trajectory->tsh,PETSC_TRUE,astep+1,&ts->time_step);CHKERRQ(ierr);
  if (ts->time_step == 0.0) ts->reason = TS_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

PetscErrorCode TSStep_TLM_Theta(TS ts)
{
  TS             fwdts;
  SNES           snes;
  KSP            ksp;
  DM             dm;
  Mat            J, Jp;
  Vec            *Y,U,*fwdY,fwdYdot,fwdYSol,F,F2,beY0 = NULL;
  PetscReal      t = ts->ptime,dummy;
  PetscReal      h = ts->time_step;
  PetscReal      theta, stage_time, s;
  PetscInt       i, j;
  PetscBool      endpoint,flg,beuler = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TLMTSGetModelTS(ts,&fwdts);CHKERRQ(ierr);
  ierr = TSThetaGetTheta(fwdts,&theta);CHKERRQ(ierr);
  if (theta == 1.0) beuler = PETSC_TRUE;
  ierr = TSThetaGetEndpoint(fwdts,&endpoint);CHKERRQ(ierr);
  ierr = TSTrajectoryGetSolutionOnly(fwdts->trajectory,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)fwdts->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
  ierr = TSGetStepNumber(ts,&i);CHKERRQ(ierr);
  if (beuler && !endpoint) {
    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&beY0);CHKERRQ(ierr);
    ierr = TSTrajectoryGet(fwdts->trajectory,fwdts,i,&dummy);CHKERRQ(ierr);
    ierr = TSGetSolution(fwdts,&fwdYSol);CHKERRQ(ierr);
    ierr = VecCopy(fwdYSol,beY0);CHKERRQ(ierr);
  }
  ierr = TSTrajectoryGet(fwdts->trajectory,fwdts,i+1,&dummy);CHKERRQ(ierr);
  ierr = TSGetSolution(fwdts,&fwdYSol);CHKERRQ(ierr);
  ierr = TSGetStages(fwdts,&i,&fwdY);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&j,&Y);CHKERRQ(ierr);
  if (i != j || i != 1) SETERRQ3(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Mismatch number of stages %D != %D || %D != 1",i,j,i);
  ierr = VecLockReadPush(fwdYSol);CHKERRQ(ierr);
  ierr = VecLockReadPush(fwdY[0]);CHKERRQ(ierr);
  ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&fwdYdot);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&F2);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&J,&Jp,NULL,NULL);CHKERRQ(ierr);

  stage_time = t + (endpoint ? 1.0 : theta)*h;

  if (endpoint && !beuler) {
    /* TLM stage */
    ierr = VecCopy(U,Y[0]);CHKERRQ(ierr);
    ierr = VecLockReadPush(Y[0]);CHKERRQ(ierr);
    /* forward step assumes linear time-independent mass matrix */
    /* otherwise, we should sample J_xdot at stage_time, and J_x at the beginning of the step and compute s * J_xdot + (theta-1)/theta J_x */
    ierr = VecZeroEntries(fwdYdot);CHKERRQ(ierr);
    s    = 1.0/((theta-1.0)*h);
    ierr = TSComputeIJacobian(ts,t,fwdY[0],fwdYdot,s,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TLMTSComputeForcing(ts,t,fwdY[0],fwdYdot,&flg,F);CHKERRQ(ierr);
    if (flg) {
      ierr = MatMultAdd(J,U,F,F);CHKERRQ(ierr);
    } else {
      ierr = MatMult(J,U,F);CHKERRQ(ierr);
    }
    ierr = VecScale(F,(theta-1.0)/theta);CHKERRQ(ierr);
    s    = 1.0/(theta*h);
    ierr = VecAXPBYPCZ(fwdYdot,s,-s,0.0,fwdYSol,fwdY[0]);CHKERRQ(ierr);
    ierr = TSComputeIJacobian(ts,stage_time,fwdYSol,fwdYdot,s,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TLMTSComputeForcing(ts,stage_time,fwdYSol,fwdYdot,&flg,F2);CHKERRQ(ierr);
    if (flg) {
      ierr = VecAXPY(F,-1.0,F2);CHKERRQ(ierr);
    }
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,J,Jp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,F,U);CHKERRQ(ierr);
    ierr = VecLockReadPop(Y[0]);CHKERRQ(ierr);
  } else {
    Vec fwdU = fwdY[0];
    if (endpoint) { beY0 = fwdY[0]; fwdU = fwdYSol; }
    /* need to reconstruct x0 from loaded solution (next time step) and stage solution
       to properly compute fwdYdot */
    if (!beY0) {
      s    = 1.0/(h*(theta-1.0));
      ierr = VecAXPBYPCZ(fwdYdot,-s,s,0.0,fwdYSol,fwdY[0]);CHKERRQ(ierr);
    } else {
      s    = 1.0/(theta*h);
      ierr = VecAXPBYPCZ(fwdYdot,s,-s,0.0,fwdYSol,beY0);CHKERRQ(ierr);
    }
    ierr = TSComputeIJacobian(ts,stage_time,fwdU,fwdYdot,0.0,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TLMTSComputeForcing(ts,stage_time,fwdU,fwdYdot,&flg,F);CHKERRQ(ierr);
    if (flg) {
      ierr = MatMultAdd(J,U,F,F);CHKERRQ(ierr);
    } else {
      ierr = MatMult(J,U,F);CHKERRQ(ierr);
    }
    ierr = VecScale(F,-1.0);CHKERRQ(ierr);
    s    = 1.0/(h*theta);
    ierr = TSComputeIJacobian(ts,stage_time,fwdU,fwdYdot,s,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,J,Jp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,F,F);CHKERRQ(ierr);
    /* TLM stage */
    if (endpoint) { ierr = VecCopy(U,Y[0]);CHKERRQ(ierr); }
    else { ierr = VecWAXPY(Y[0],1.0,F,U);CHKERRQ(ierr); }
    /* Advance TLM solution */
    ierr = VecAXPY(U,1.0/theta,F);CHKERRQ(ierr);
    if (endpoint) { beY0 = NULL; }
  }

  ierr = DMRestoreGlobalVector(dm,&F2);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&fwdYdot);CHKERRQ(ierr);
  ierr = VecLockReadPop(fwdY[0]);CHKERRQ(ierr);
  ierr = VecLockReadPop(fwdYSol);CHKERRQ(ierr);
  if (beY0) {
    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&beY0);CHKERRQ(ierr);
  }

  ts->ptime += h;
  ierr = TSGetStepNumber(ts,&i);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(fwdts->trajectory->tsh,PETSC_FALSE,i+1,&ts->time_step);CHKERRQ(ierr);
  if (ts->time_step == 0.0) ts->reason = TS_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

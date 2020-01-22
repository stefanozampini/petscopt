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
  PetscReal       h = ts->time_step, at = ts->ptime, t, dummy;
  PetscInt        step,astep,tstep;
  PetscInt        s,sf,i,j,k,skps,acts;
  PetscBool       flg,sonly,quad;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
#if PETSC_VERSION_GE(3,13,0)
  ierr = TSRKGetTableau(ts,&s,&A,&b,&c,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
#else
  SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"RK discrete adjoints need PETSc version greater or equal 3.13");
#endif
  if (s > TS_RK_DISCRETE_MAXSTAGES) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Not coded for %D stages",s);
  ierr = AdjointTSGetModelTS(ts,&fwdts);CHKERRQ(ierr);
  ierr = TSTrajectoryGetSolutionOnly(fwdts->trajectory,&sonly);CHKERRQ(ierr);
  if (sonly) SETERRQ(PetscObjectComm((PetscObject)fwdts->trajectory),PETSC_ERR_SUP,"TSTrajectory did not save the stages! Rerun with TSTrajectorySetSolutionOnly(tj,PETSC_TRUE)");
  ierr = TSGetStepNumber(ts,&astep);CHKERRQ(ierr);
  ierr = TSTrajectoryGetNumSteps(fwdts->trajectory,&tstep);CHKERRQ(ierr);
  step = tstep - astep - 1;
  /* XXX t may be garbage... */
  ierr = TSTrajectoryGet(fwdts->trajectory,fwdts,step,&dummy);CHKERRQ(ierr);
  ierr = TSHistoryGetTime(fwdts->trajectory->tsh,PETSC_FALSE,step,&t);CHKERRQ(ierr);
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
      ierr = AdjointTSComputeQuadrature(ts,astage_time,Y[i],NULL,LY[k],NULL,FOAL,NULL,TLMU,NULL,&flg,F);CHKERRQ(ierr);
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

  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(fwdts->trajectory->tsh,PETSC_TRUE,step+1,&ts->time_step);CHKERRQ(ierr);
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
#if PETSC_VERSION_GE(3,13,0)
  ierr = TSRKGetTableau(ts,&s,&A,&b,&c,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
#else
  SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"RK discrete tangent linear models need PETSc version greater or equal 3.13");
#endif
  if (s > TS_RK_DISCRETE_MAXSTAGES) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Not coded for %D stages",s);
  ierr = TLMTSGetModelTS(ts,&fwdts);CHKERRQ(ierr);
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

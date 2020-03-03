#include <petscopt/private/tssplitjacimpl.h>
#include <petsc/private/petscimpl.h>

/* ------------------ Helper routines to compute split Jacobians ----------------------- */
PetscErrorCode TSSplitJacobiansDestroy_Private(void *ptr)
{
  TSSplitJacobians* s = (TSSplitJacobians*)ptr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&s->pJ_U);CHKERRQ(ierr);
  ierr = MatDestroy(&s->pJ_Udot);CHKERRQ(ierr);
  ierr = MatDestroy(&s->J_U);CHKERRQ(ierr);
  ierr = MatDestroy(&s->J_Udot);CHKERRQ(ierr);
  ierr = PetscFree(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSComputeSplitJacobians - Given a DAE in implicit form F(t,U,Udot), it computes F_U and F_Udot.

   Synopsis:
   #include <petsc/private/tssplitjacimpl.h>

   Logically Collective on TS

   Input Parameters:
+  ts   - the TS context obtained from TSCreate()
.  time - current solution time
.  U    - current state
-  Udot - current time derivative

   Output Parameters:
+  A  - the F_U Jacobian matrix
.  pA - the preconditioning matrix for A
.  B  - the F_Udot Jacobian matrix
-  pB - the preconditioning matrix for B

Notes:
   The user can supersed the default implementation by calling
$             PetscObjectComposeFunction((PetscObject)(ts),"TSComputeSplitJacobians_C",MyImplementationOfSplitJacobians);CHKERRQ(ierr);

   Level: developer

.seealso: TSCreateTLMSTS(), TSCreateAdjointTS(), TSComputeIJacobian()
@*/
PetscErrorCode TSComputeSplitJacobians(TS ts, PetscReal time, Vec U, Vec Udot, Mat A, Mat pA, Mat B, Mat pB)
{
  PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,Mat,Mat,Mat,Mat);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Udot,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,MAT_CLASSID,5);
  PetscValidHeaderSpecific(pA,MAT_CLASSID,6);
  PetscValidHeaderSpecific(B,MAT_CLASSID,7);
  PetscValidHeaderSpecific(pB,MAT_CLASSID,8);
  if (A == B) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"A and B must be different matrices");
  if (pA == pB) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"pA and pB must be different matrices");
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSComputeSplitJacobians_C",&f);CHKERRQ(ierr);
  if (!f) {
    ierr = TSComputeIJacobian(ts,time,U,Udot,0.0,A,pA,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TSComputeIJacobian(ts,time,U,Udot,1.0,B,pB,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    if (pB && pB != B) {
      ierr = MatAXPY(pB,-1.0,pA,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  } else {
    ierr = (*f)(ts,time,U,Udot,A,pA,B,pB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSGetSplitJacobians - Gets the matrices to be used by TSComputeSplitJacobians()

   Synopsis:
   #include <petsc/private/tssplitjacimpl.h>

   Logically Collective on TS

   Input Parameters:
.  ts   - the TS context obtained from TSCreate()

   Output Parameters:
+  A  - the F_Udot Jacobian matrix
.  pA - the preconditioning matrix for A
.  B  - the F_U Jacobian matrix
-  pB - the preconditioning matrix for B

   Level: developer

.seealso: TSComputeSplitJacobians()
@*/
PetscErrorCode TSGetSplitJacobians(TS ts, Mat* JU, Mat* pJU, Mat *JUdot, Mat* pJUdot)
{
  PetscErrorCode    ierr;
  PetscContainer    c;
  Mat               A,B;
  TSSplitJacobians *splitJ;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (JU) PetscValidPointer(JU,2);
  if (pJU) PetscValidPointer(pJU,3);
  if (JUdot) PetscValidPointer(JUdot,4);
  if (pJUdot) PetscValidPointer(pJUdot,5);
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) {
    ierr = PetscNew(&splitJ);CHKERRQ(ierr);
    splitJ->Astate    = -1;
    splitJ->Aid       = PETSC_MIN_INT;
    splitJ->shift     = PETSC_MIN_REAL;
    splitJ->splitdone = PETSC_FALSE;
    splitJ->jacconsts = PETSC_FALSE;
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&c);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(c,splitJ);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(c,TSSplitJacobiansDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"_ts_splitJac",(PetscObject)c);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)c);CHKERRQ(ierr);
  }
  if (!JU && !pJU && !JUdot && !pJUdot) PetscFunctionReturn(0);

  ierr = PetscContainerGetPointer(c,(void**)&splitJ);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
  if (JU) {
    if (!splitJ->J_U) {
      ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&splitJ->J_U);CHKERRQ(ierr);
    }
    *JU = splitJ->J_U;
  }
  if (pJU) {
    if (!splitJ->pJ_U) {
      if (B && B != A) {
        ierr = MatDuplicate(B,MAT_SHARE_NONZERO_PATTERN,&splitJ->pJ_U);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)splitJ->J_U);CHKERRQ(ierr);
        splitJ->pJ_U = splitJ->J_U;
      }
    }
    *pJU = splitJ->pJ_U;
  }
  if (JUdot) {
    if (!splitJ->J_Udot) {
      ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&splitJ->J_Udot);CHKERRQ(ierr);
    }
    *JUdot = splitJ->J_Udot;
  }
  if (pJUdot) {
    if (!splitJ->pJ_Udot) {
      if (B && B != A) {
        ierr = MatDuplicate(B,MAT_SHARE_NONZERO_PATTERN,&splitJ->pJ_Udot);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)splitJ->J_Udot);CHKERRQ(ierr);
        splitJ->pJ_Udot = splitJ->J_Udot;
      }
    }
    *pJUdot = splitJ->pJ_Udot;
  }
  PetscFunctionReturn(0);
}

/* Updates splitJ->J_Udot and splitJ->J_U at a given time */
PetscErrorCode TSUpdateSplitJacobiansFromHistory_Private(TS ts, PetscReal time)
{
  PetscContainer   c;
  TSSplitJacobians *splitJ;
  Mat              J_U = NULL,J_Udot = NULL,pJ_U = NULL,pJ_Udot = NULL;
  Vec              U,Udot;
  TSProblemType    type;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing splitJac container");
  ierr = PetscContainerGetPointer(c,(void**)&splitJ);CHKERRQ(ierr);
  if (splitJ->jacconsts && splitJ->splitdone) PetscFunctionReturn(0);
  ierr = TSGetProblemType(ts,&type);CHKERRQ(ierr);
  if (type > TS_LINEAR) {
    TSTrajectory     tj;

    ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,ts,time,&U,&Udot);CHKERRQ(ierr);
  } else { /* use fake U and Udot */
    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    Udot = U;
  }
  ierr = TSGetSplitJacobians(ts,&J_U,&pJ_U,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
  ierr = TSComputeSplitJacobians(ts,time,U,Udot,J_U,pJ_U,J_Udot,pJ_Udot);CHKERRQ(ierr);
  if (type > TS_LINEAR) {
    TSTrajectory     tj;

    ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&U,&Udot);CHKERRQ(ierr);
  }
  splitJ->splitdone = splitJ->jacconsts ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* This function is used in AdjointTSIJacobian and (optionally) in TLMTSIJacobian.
   The assumption here is that the IJacobian routine is called after the IFunction (called with same time, U and Udot)
   This is why the time, U and Udot arguments are currently ignored (TODO: add checks?) */
PetscErrorCode TSComputeIJacobianWithSplits_Private(TS ts, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  PetscObjectState Astate;
  PetscObjectId    Aid;
  PetscContainer   c;
  TSSplitJacobians *splitJ;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing splitJac container");
  ierr = PetscContainerGetPointer(c,(void**)&splitJ);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)A,&Astate);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)A,&Aid);CHKERRQ(ierr);
  if (splitJ->jacconsts && PetscAbsScalar(splitJ->shift - shift) < PETSC_SMALL &&
      splitJ->Astate == Astate && splitJ->Aid == Aid) {
    PetscFunctionReturn(0);
  }
  if (A == splitJ->J_U) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"A should be different from J_U");
  if (A == splitJ->J_Udot) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"A should be different from J_Udot");
  ierr = MatCopy(splitJ->J_U,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(A,shift,splitJ->J_Udot,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)A,&splitJ->Astate);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)A,&splitJ->Aid);CHKERRQ(ierr);
  splitJ->shift = shift;
  if (B && A != B) {
    if (B == splitJ->pJ_U) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"B should be different from pJ_U");
    if (B == splitJ->pJ_Udot) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"B should be different from pJ_Udot");
    ierr = MatCopy(splitJ->pJ_U,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(B,shift,splitJ->pJ_Udot,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

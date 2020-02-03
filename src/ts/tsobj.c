#include <petscopt/private/petscoptimpl.h>
#include <petscopt/private/tsobjimpl.h>
#include <petsc/private/petscimpl.h>

/* ------------------ Helper routines for PDE-constrained support to evaluate objective functions, gradients and Hessian terms ----------------------- */

/* Evaluates objective functions of the type f(state,design,t) */
PetscErrorCode TSObjEval(TSObj funchead, Vec state, Vec design, PetscReal time, PetscReal *val)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidPointer(val,5);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  ierr = VecLockReadPush(state);CHKERRQ(ierr);
  ierr = VecLockReadPush(design);CHKERRQ(ierr);
  *val = 0.0;
  while (link) {
    if (link->f && link->fixedtime <= PETSC_MIN_REAL) {
      PetscReal v;
      ierr = (*link->f)(state,design,time,&v,link->f_ctx);CHKERRQ(ierr);
      *val += v;
    }
    link = link->next;
  }
  ierr = VecLockReadPop(state);CHKERRQ(ierr);
  ierr = VecLockReadPop(design);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates objective functions of the type f(state,design,t = fixed) */
PetscErrorCode TSObjEvalFixed(TSObj funchead, Vec state, Vec design, PetscReal time, PetscReal *val)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidPointer(val,5);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  ierr = VecLockReadPush(state);CHKERRQ(ierr);
  ierr = VecLockReadPush(design);CHKERRQ(ierr);
  *val = 0.0;
  while (link) {
    if (link->f && time == link->fixedtime) {
      PetscReal v;
      ierr = (*link->f)(state,design,link->fixedtime,&v,link->f_ctx);CHKERRQ(ierr);
      *val += v;
    }
    link = link->next;
  }
  ierr = VecLockReadPop(state);CHKERRQ(ierr);
  ierr = VecLockReadPop(design);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the state) of objective functions of the type f(state,design,t) */
PetscErrorCode TSObjEval_U(TSObj funchead, Vec state, Vec design, PetscReal time, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidPointer(has,6);
  PetscValidHeaderSpecific(out,VEC_CLASSID,7);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_x && link->fixedtime <= PETSC_MIN_REAL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    while (link) {
      if (link->f_x && link->fixedtime <= PETSC_MIN_REAL) {
        if (!firstdone) {
          ierr = (*link->f_x)(state,design,time,out,link->f_ctx);CHKERRQ(ierr);
        } else {
          ierr = (*link->f_x)(state,design,time,work,link->f_ctx);CHKERRQ(ierr);
          ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the state) of objective functions of the type f(state,design,t = fixed)
   These may lead to Dirac's delta terms in the adjoint DAE if the fixed time is in between (t0,tf) */
PetscErrorCode TSObjEvalFixed_U(TSObj funchead, Vec state, Vec design, PetscReal time, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidPointer(has,6);
  PetscValidHeaderSpecific(out,VEC_CLASSID,7);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_x && link->fixedtime > PETSC_MIN_REAL && PetscAbsReal(link->fixedtime-time) < PETSC_SMALL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    while (link) {
      if (link->f_x && link->fixedtime > PETSC_MIN_REAL && PetscAbsReal(link->fixedtime-time) < PETSC_SMALL) {
        if (!firstdone) {
          ierr = (*link->f_x)(state,design,link->fixedtime,out,link->f_ctx);CHKERRQ(ierr);
        } else {
          ierr = (*link->f_x)(state,design,link->fixedtime,work,link->f_ctx);CHKERRQ(ierr);
          ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the parameters) of objective functions of the type f(state,design,t) */
PetscErrorCode TSObjEval_M(TSObj funchead, Vec state, Vec design, PetscReal time, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidPointer(has,6);
  PetscValidHeaderSpecific(out,VEC_CLASSID,7);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_m && link->fixedtime <= PETSC_MIN_REAL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    while (link) {
      if (link->f_m && link->fixedtime <= PETSC_MIN_REAL) {
        if (!firstdone) {
          ierr = (*link->f_m)(state,design,time,out,link->f_ctx);CHKERRQ(ierr);
        } else {
          ierr = (*link->f_m)(state,design,time,work,link->f_ctx);CHKERRQ(ierr);
          ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the parameters) of objective functions of the type f(state,design,t = tfixed) */
PetscErrorCode TSObjEvalFixed_M(TSObj funchead, Vec state, Vec design, PetscReal time, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidPointer(has,6);
  PetscValidHeaderSpecific(out,VEC_CLASSID,7);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_m && time == link->fixedtime) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    while (link) {
      if (link->f_m && time == link->fixedtime) {
        if (!firstdone) {
          ierr = (*link->f_m)(state,design,link->fixedtime,out,link->f_ctx);CHKERRQ(ierr);
        } else {
          ierr = (*link->f_m)(state,design,link->fixedtime,work,link->f_ctx);CHKERRQ(ierr);
          ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates Hessian (wrt the state) matvec of objective functions of the type f(state,design,t) */
PetscErrorCode TSObjEval_UU(TSObj funchead, Vec state, Vec design, PetscReal time, Vec direction, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(direction,VEC_CLASSID,5);
  PetscValidHeaderSpecific(work,VEC_CLASSID,6);
  PetscValidPointer(has,7);
  PetscValidHeaderSpecific(out,VEC_CLASSID,8);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_XX && link->fixedtime <= PETSC_MIN_REAL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = VecLockReadPush(direction);CHKERRQ(ierr);
    while (link) {
      if (link->f_XX && link->fixedtime <= PETSC_MIN_REAL) {
        if (link->f_xx) { /* non-constant dependence */
          ierr = (*link->f_xx)(state,design,time,link->f_XX,link->f_ctx);CHKERRQ(ierr);
        }
        if (!firstdone) {
          ierr = MatMult(link->f_XX,direction,out);CHKERRQ(ierr);
        } else {
          PetscBool hasop;

          ierr = MatHasOperation(link->f_XX,MATOP_MULT_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultAdd(link->f_XX,direction,out,out);CHKERRQ(ierr);
          } else {
            ierr = MatMult(link->f_XX,direction,work);CHKERRQ(ierr);
            ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
          }
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
    ierr = VecLockReadPop(direction);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates Hessian (wrt the state) matvec of objective functions of the type f(state,design,t = tfixed) */
PetscErrorCode TSObjEvalFixed_UU(TSObj funchead, Vec state, Vec design, PetscReal time, Vec direction, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(direction,VEC_CLASSID,5);
  PetscValidHeaderSpecific(work,VEC_CLASSID,6);
  PetscValidPointer(has,7);
  PetscValidHeaderSpecific(out,VEC_CLASSID,8);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_XX && time == link->fixedtime) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = VecLockReadPush(direction);CHKERRQ(ierr);
    while (link) {
      if (link->f_XX && time == link->fixedtime) {
        if (link->f_xx) { /* non-constant dependence */
          ierr = (*link->f_xx)(state,design,time,link->f_XX,link->f_ctx);CHKERRQ(ierr);
        }
        if (!firstdone) {
          ierr = MatMult(link->f_XX,direction,out);CHKERRQ(ierr);
        } else {
          PetscBool hasop;

          ierr = MatHasOperation(link->f_XX,MATOP_MULT_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultAdd(link->f_XX,direction,out,out);CHKERRQ(ierr);
          } else {
            ierr = MatMult(link->f_XX,direction,work);CHKERRQ(ierr);
            ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
          }
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
    ierr = VecLockReadPop(direction);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates Hessian (mixed state-parameters) matvec of objective functions of the type f(state,design,t) */
PetscErrorCode TSObjEval_UM(TSObj funchead, Vec state, Vec design, PetscReal time, Vec direction, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(direction,VEC_CLASSID,5);
  PetscValidHeaderSpecific(work,VEC_CLASSID,6);
  PetscValidPointer(has,7);
  PetscValidHeaderSpecific(out,VEC_CLASSID,8);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_XM && link->fixedtime <= PETSC_MIN_REAL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = VecLockReadPush(direction);CHKERRQ(ierr);
    while (link) {
      if (link->f_XM && link->fixedtime <= PETSC_MIN_REAL) {
        if (link->f_xm) { /* non-constant dependence */
          ierr = (*link->f_xm)(state,design,time,link->f_XM,link->f_ctx);CHKERRQ(ierr);
        }
        if (!firstdone) {
          ierr = MatMult(link->f_XM,direction,out);CHKERRQ(ierr);
        } else {
          PetscBool hasop;

          ierr = MatHasOperation(link->f_XM,MATOP_MULT_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultAdd(link->f_XM,direction,out,out);CHKERRQ(ierr);
          } else {
            ierr = MatMult(link->f_XM,direction,work);CHKERRQ(ierr);
            ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
          }
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
    ierr = VecLockReadPop(direction);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates Hessian (mixed state-parameters) matvec of objective functions of the type f(state,design,t=tfixed) */
PetscErrorCode TSObjEvalFixed_UM(TSObj funchead, Vec state, Vec design, PetscReal time, Vec direction, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(direction,VEC_CLASSID,5);
  PetscValidHeaderSpecific(work,VEC_CLASSID,6);
  PetscValidPointer(has,7);
  PetscValidHeaderSpecific(out,VEC_CLASSID,8);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_XM && time == link->fixedtime) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = VecLockReadPush(direction);CHKERRQ(ierr);
    while (link) {
      if (link->f_XM && time == link->fixedtime) {
        if (link->f_xm) { /* non-constant dependence */
          ierr = (*link->f_xm)(state,design,time,link->f_XM,link->f_ctx);CHKERRQ(ierr);
        }
        if (!firstdone) {
          ierr = MatMult(link->f_XM,direction,out);CHKERRQ(ierr);
        } else {
          PetscBool hasop;

          ierr = MatHasOperation(link->f_XM,MATOP_MULT_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultAdd(link->f_XM,direction,out,out);CHKERRQ(ierr);
          } else {
            ierr = MatMult(link->f_XM,direction,work);CHKERRQ(ierr);
            ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
          }
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
    ierr = VecLockReadPop(direction);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates Hessian (mixed parameters-state) matvec of objective functions of the type f(state,design,t) */
PetscErrorCode TSObjEval_MU(TSObj funchead, Vec state, Vec design, PetscReal time, Vec direction, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(direction,VEC_CLASSID,5);
  PetscValidHeaderSpecific(work,VEC_CLASSID,6);
  PetscValidPointer(has,7);
  PetscValidHeaderSpecific(out,VEC_CLASSID,8);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_XM && link->fixedtime <= PETSC_MIN_REAL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = VecLockReadPush(direction);CHKERRQ(ierr);
    while (link) {
      if (link->f_XM && link->fixedtime <= PETSC_MIN_REAL) {
        if (link->f_xm) { /* non-constant dependence */
          ierr = (*link->f_xm)(state,design,time,link->f_XM,link->f_ctx);CHKERRQ(ierr);
        }
        if (!firstdone) {
          ierr = MatMultTranspose(link->f_XM,direction,out);CHKERRQ(ierr);
        } else {
          PetscBool hasop;

          ierr = MatHasOperation(link->f_XM,MATOP_MULT_TRANSPOSE_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultTransposeAdd(link->f_XM,direction,out,out);CHKERRQ(ierr);
          } else {
            ierr = MatMultTranspose(link->f_XM,direction,work);CHKERRQ(ierr);
            ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
          }
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
    ierr = VecLockReadPop(direction);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates Hessian (mixed parameters-state) matvec of objective functions of the type f(state,design,t=tfixed) */
PetscErrorCode TSObjEvalFixed_MU(TSObj funchead, Vec state, Vec design, PetscReal time, Vec direction, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(direction,VEC_CLASSID,5);
  PetscValidHeaderSpecific(work,VEC_CLASSID,6);
  PetscValidPointer(has,7);
  PetscValidHeaderSpecific(out,VEC_CLASSID,8);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_XM && time == link->fixedtime) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = VecLockReadPush(direction);CHKERRQ(ierr);
    while (link) {
      if (link->f_XM && time == link->fixedtime) {
        if (link->f_xm) { /* non-constant dependence */
          ierr = (*link->f_xm)(state,design,time,link->f_XM,link->f_ctx);CHKERRQ(ierr);
        }
        if (!firstdone) {
          ierr = MatMultTranspose(link->f_XM,direction,out);CHKERRQ(ierr);
        } else {
          PetscBool hasop;

          ierr = MatHasOperation(link->f_XM,MATOP_MULT_TRANSPOSE_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultTransposeAdd(link->f_XM,direction,out,out);CHKERRQ(ierr);
          } else {
            ierr = MatMultTranspose(link->f_XM,direction,work);CHKERRQ(ierr);
            ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
          }
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
    ierr = VecLockReadPop(direction);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates Hessian (wrt the parameters) matvec of objective functions of the type f(state,design,t) */
PetscErrorCode TSObjEval_MM(TSObj funchead, Vec state, Vec design, PetscReal time, Vec direction, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(direction,VEC_CLASSID,5);
  PetscValidHeaderSpecific(work,VEC_CLASSID,6);
  PetscValidPointer(has,7);
  PetscValidHeaderSpecific(out,VEC_CLASSID,8);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_MM && link->fixedtime <= PETSC_MIN_REAL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = VecLockReadPush(direction);CHKERRQ(ierr);
    while (link) {
      if (link->f_MM && link->fixedtime <= PETSC_MIN_REAL) {
        if (link->f_mm) { /* non-constant dependence */
          ierr = (*link->f_mm)(state,design,time,link->f_MM,link->f_ctx);CHKERRQ(ierr);
        }
        if (!firstdone) {
          ierr = MatMult(link->f_MM,direction,out);CHKERRQ(ierr);
        } else {
          PetscBool hasop;

          ierr = MatHasOperation(link->f_MM,MATOP_MULT_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultAdd(link->f_MM,direction,out,out);CHKERRQ(ierr);
          } else {
            ierr = MatMult(link->f_MM,direction,work);CHKERRQ(ierr);
            ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
          }
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
    ierr = VecLockReadPop(direction);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates Hessian (wrt the parameters) matvec of objective functions of the type f(state,design,t = tfixed) */
PetscErrorCode TSObjEvalFixed_MM(TSObj funchead, Vec state, Vec design, PetscReal time, Vec direction, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  TSObj          link = funchead;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidLogicalCollectiveReal(state,time,4);
  PetscValidHeaderSpecific(direction,VEC_CLASSID,5);
  PetscValidHeaderSpecific(work,VEC_CLASSID,6);
  PetscValidPointer(has,7);
  PetscValidHeaderSpecific(out,VEC_CLASSID,8);
  ierr = PetscLogEventBegin(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  *has = PETSC_FALSE;
  if (work == out) SETERRQ(PetscObjectComm((PetscObject)out),PETSC_ERR_USER,"work and out vectors need to be different");
  while (link) {
    if (link->f_MM && time == link->fixedtime) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = funchead;
    ierr = VecLockReadPush(state);CHKERRQ(ierr);
    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = VecLockReadPush(direction);CHKERRQ(ierr);
    while (link) {
      if (link->f_MM && time == link->fixedtime) {
        if (link->f_mm) { /* non-constant dependence */
          ierr = (*link->f_mm)(state,design,time,link->f_MM,link->f_ctx);CHKERRQ(ierr);
        }
        if (!firstdone) {
          ierr = MatMult(link->f_MM,direction,out);CHKERRQ(ierr);
        } else {
          PetscBool hasop;

          ierr = MatHasOperation(link->f_MM,MATOP_MULT_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultAdd(link->f_MM,direction,out,out);CHKERRQ(ierr);
          } else {
            ierr = MatMult(link->f_MM,direction,work);CHKERRQ(ierr);
            ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
          }
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockReadPop(state);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
    ierr = VecLockReadPop(direction);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Obj_Eval,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSObjGetNumObjectives(TSObj link, PetscInt *n)
{
  PetscFunctionBeginHot;
  *n = 0;
  while (link) {
    (*n)++;
    link = link->next;
  }
  PetscFunctionReturn(0);
}

/* Inquires the presence of integrand terms */
PetscErrorCode TSObjHasObjectiveIntegrand(TSObj link, PetscBool *has, PetscBool *has_x, PetscBool *has_m, PetscBool *has_xx, PetscBool *has_xm, PetscBool *has_mm)
{
  PetscFunctionBeginHot;
  if (has)    PetscValidPointer(has,2);
  if (has_x)  PetscValidPointer(has_x,3);
  if (has_m)  PetscValidPointer(has_m,4);
  if (has_xx) PetscValidPointer(has_xx,5);
  if (has_xm) PetscValidPointer(has_xm,6);
  if (has_mm) PetscValidPointer(has_mm,7);
  if (has)    *has    = PETSC_FALSE;
  if (has_x)  *has_x  = PETSC_FALSE;
  if (has_m)  *has_m  = PETSC_FALSE;
  if (has_xx) *has_xx = PETSC_FALSE;
  if (has_xm) *has_xm = PETSC_FALSE;
  if (has_mm) *has_mm = PETSC_FALSE;
  while (link) {
    if (link->fixedtime <= PETSC_MIN_REAL) {
      if (has    && link->f)    *has    = PETSC_TRUE;
      if (has_x  && link->f_x)  *has_x  = PETSC_TRUE;
      if (has_m  && link->f_m)  *has_m  = PETSC_TRUE;
      if (has_xx && link->f_XX) *has_xx = PETSC_TRUE;
      if (has_xm && link->f_XM) *has_xm = PETSC_TRUE;
      if (has_mm && link->f_MM) *has_mm = PETSC_TRUE;
    }
    link = link->next;
  }
  PetscFunctionReturn(0);
}

/* Inquires the presence of point-form functionals in a given time interval (t0,tf] and returns the minimum among the requested ones and tf */
PetscErrorCode TSObjHasObjectiveFixed(TSObj linkin, PetscReal t0, PetscReal tf, PetscBool *has, PetscBool *has_x, PetscBool *has_m, PetscBool *has_xx, PetscBool *has_xm, PetscBool *has_mm, PetscReal *time)
{
  TSObj link = linkin;

  PetscFunctionBeginHot;
  if (has)    PetscValidPointer(has,3);
  if (has_x)  PetscValidPointer(has_x,4);
  if (has_m)  PetscValidPointer(has_m,5);
  if (has_xx) PetscValidPointer(has_xx,6);
  if (has_xm) PetscValidPointer(has_xm,7);
  if (has_mm) PetscValidPointer(has_mm,8);
  if (time)   PetscValidPointer(time,9);
  if (has)    *has    = PETSC_FALSE;
  if (has_x)  *has_x  = PETSC_FALSE;
  if (has_m)  *has_m  = PETSC_FALSE;
  if (has_xx) *has_xx = PETSC_FALSE;
  if (has_xm) *has_xm = PETSC_FALSE;
  if (has_mm) *has_mm = PETSC_FALSE;
  if (time)   *time   = tf;
  while (link) {
    if (t0 < link->fixedtime && link->fixedtime <= tf) {
      if ((has    && link->f   ) || (has_x  && link->f_x ) || (has_m  && link->f_m ) ||
          (has_xx && link->f_XX) || (has_xm && link->f_XM) || (has_mm && link->f_MM))
        tf = PetscMax(t0,PetscMin(link->fixedtime,tf));
    }
    link = link->next;
  }
  link = linkin;
  while (link) {
    if (link->fixedtime == tf) {
      if (has    && link->f)    *has    = PETSC_TRUE;
      if (has_x  && link->f_x)  *has_x  = PETSC_TRUE;
      if (has_m  && link->f_m)  *has_m  = PETSC_TRUE;
      if (has_xx && link->f_XX) *has_xx = PETSC_TRUE;
      if (has_xm && link->f_XM) *has_xm = PETSC_TRUE;
      if (has_mm && link->f_MM) *has_mm = PETSC_TRUE;
    }
    link = link->next;
  }
  if (time) *time = tf;
  PetscFunctionReturn(0);
}

/*@
   TSResetObjective - Resets the list of objective functions set with TSAddObjective().

   Logically Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Level: advanced

.seealso: TSAddObjective()
@*/
PetscErrorCode TSResetObjective(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_obj_ctx",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSObjDestroy_Private(void *ptr)
{
  TSObj          link = (TSObj)ptr;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  while (link) {
    TSObj olink = link;

    link = link->next;
    ierr = MatDestroy(&olink->f_XX);CHKERRQ(ierr);
    ierr = MatDestroy(&olink->f_XM);CHKERRQ(ierr);
    ierr = MatDestroy(&olink->f_MM);CHKERRQ(ierr);
    ierr = PetscFree(olink);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSGetTSObj(TS ts, TSObj *obj)
{
  PetscErrorCode ierr;
  PetscContainer c;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(obj,2);
  *obj = NULL;
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_obj_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (c) {
    TSObj link;

    ierr = PetscContainerGetPointer(c,(void**)&link);CHKERRQ(ierr);
    *obj = link;
  }
  PetscFunctionReturn(0);
}

/*@C
   TSAddObjective - Sets a cost functional callback together with its gradient and Hessian terms.

   Logically Collective on TS

   Input Parameters:
+  ts      - the TS context obtained from TSCreate()
.  fixtime - the time at which the functional has to be evaluated (use PETSC_MIN_REAL for integrand terms)
.  f       - the function evaluation routine
.  f_x     - the function evaluation routine for the derivative wrt the state variables (can be NULL)
.  f_m     - the function evaluation routine for the derivative wrt the design variables (can be NULL)
.  f_XX    - the Mat object to hold f_xx(x,m,t) (can be NULL)
.  f_xx    - the function evaluation routine for the second derivative wrt the state variables (can be NULL)
.  f_XM    - the Mat object to hold f_xm(x,m,t) (can be NULL)
.  f_xm    - the function evaluation routine for the mixed derivative (can be NULL)
.  f_MM    - the Mat object to hold f_mm(x,m,t) (can be NULL)
.  f_mm    - the function evaluation routine for the second derivative wrt the design variables (can be NULL)
-  f_ctx   - user-defined context (can be NULL)

   Calling sequence of f:
$  f(Vec u,Vec m,PetscReal t,PetscReal *out,void *ctx);

+  u   - state vector
.  m   - design vector
.  t   - time at step/stage being solved
.  out - output value
-  ctx - [optional] user-defined context

   Calling sequence of f_x and f_m:
$  f(Vec u,Vec m,PetscReal t,Vec out,void *ctx);

+  u   - state vector
.  m   - design vector
.  t   - time at step/stage being solved
.  out - output vector
-  ctx - [optional] user-defined context

   Calling sequence of f_xx, f_xm and f_mm:
$  f(Vec u,Vec m,PetscReal t,Mat A,void *ctx);

+  u   - state vector
.  m   - design vector
.  t   - time at step/stage being solved
.  A   - the output matrix
-  ctx - [optional] user-defined context

   Notes: the functions passed in are appended to a list. More functions can be passed by simply calling TSAddObjective multiple times.
          The functionals are intendended to be used as integrand terms of a time integration (if fixtime == PETSC_MIN_REAL) or as evaluation at a given specific time.
          Regularizers fall into the latter category: use f_x = NULL, and pass f and f_m with any time in between the interval [t0, tf] (i.e. start and end of the forward solve).
          For f_x, the size of the output vector equals the size of the state vector; for f_m it equals the size of the design vector.
          The hessian matrices do not need to be in assembled form, just the MatMult() action is needed. If f_XM is present, the action of f_MX is obtained by calling MatMultTranspose().
          If any of the second derivative matrices is constant, the associated function pointers can be NULL, with the matrix passed properly setup.
          It misses the Fortran wrapper.

   Level: advanced

.seealso: TSSetGradientDAE(), TSSetHessianDAE(), TSComputeObjectiveAndGradient(), TSSetGradientIC(), TSSetHessianIC(), MATSHELL
@*/
PetscErrorCode TSAddObjective(TS ts, PetscReal fixtime, TSEvalObjective f,
                              TSEvalObjectiveGradient f_x, TSEvalObjectiveGradient f_m,
                              Mat f_XX, TSEvalObjectiveHessian f_xx,
                              Mat f_XM, TSEvalObjectiveHessian f_xm,
                              Mat f_MM, TSEvalObjectiveHessian f_mm, void* f_ctx)
{
  TSObj          link;
  PetscErrorCode ierr;
  PetscContainer c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,fixtime,2);
  ierr = PetscOptInitializePackage();CHKERRQ(ierr);
  if (f_XX) {
    PetscValidHeaderSpecific(f_XX,MAT_CLASSID,9);
    PetscCheckSameComm(ts,1,f_XX,9);
  }
  if (f_XM) {
    PetscValidHeaderSpecific(f_XM,MAT_CLASSID,12);
    PetscCheckSameComm(ts,1,f_XM,12);
  }
  if (f_MM) {
    PetscValidHeaderSpecific(f_MM,MAT_CLASSID,15);
    PetscCheckSameComm(ts,1,f_MM,15);
  }
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_obj_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) {
    ierr = PetscNew(&link);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&c);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(c,link);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(c,TSObjDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"_ts_obj_ctx",(PetscObject)c);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)c);CHKERRQ(ierr);
  } else {
    ierr = PetscContainerGetPointer(c,(void**)&link);CHKERRQ(ierr);
    while (link->next) link = link->next;
    ierr = PetscNew(&link->next);CHKERRQ(ierr);
    link = link->next;
  }
  link->f = f;
  if (f_x && !f) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing f with non-zero f_x");
  link->f_x = f_x;
  if (f_m && !f) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing f with non-zero f_m");
  link->f_m = f_m;
  if (f_XX) {
    if (!f_x) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing f_x with non-zero f_XX");
    ierr       = PetscObjectReference((PetscObject)f_XX);CHKERRQ(ierr);
    link->f_XX = f_XX;
    link->f_xx = f_xx;
  }
  if (f_XM) {
    if (!f_x) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing f_x with non-zero f_XM");
    if (!f_m) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing f_m with non-zero f_XM");
    ierr       = PetscObjectReference((PetscObject)f_XM);CHKERRQ(ierr);
    link->f_XM = f_XM;
    link->f_xm = f_xm;
  }
  if (f_MM) {
    if (!f_m) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing f_m with non-zero f_MM");
    ierr       = PetscObjectReference((PetscObject)f_MM);CHKERRQ(ierr);
    link->f_MM = f_MM;
    link->f_mm = f_mm;
  }
  link->f_ctx     = f_ctx;
  link->fixedtime = fixtime;
  PetscFunctionReturn(0);
}

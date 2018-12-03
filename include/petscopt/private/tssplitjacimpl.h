#ifndef __PETSCOPT_TSSPLITJACIMPL_H
#define __PETSCOPT_TSSPLITJACIMPL_H

#include <petscts.h>

typedef struct {
  PetscScalar      shift;
  PetscObjectState Astate;
  PetscObjectId    Aid;
  PetscBool        splitdone;
  PetscBool        jacconsts; /* true if the DAE is A Udot + B U -f = 0 */
  Mat              J_U;       /* Jacobian : F_U (U,Udot,t) */
  Mat              J_Udot;    /* Jacobian : F_Udot(U,Udot,t) */
  Mat              pJ_U;      /* Jacobian : F_U (U,Udot,t) (to be preconditioned) */
  Mat              pJ_Udot;   /* Jacobian : F_Udot(U,Udot,t) (to be preconditioned) */
} TSSplitJacobians;

PETSC_EXTERN PetscErrorCode TSComputeSplitJacobians(TS,PetscReal,Vec,Vec,Mat,Mat,Mat,Mat);
PETSC_EXTERN PetscErrorCode TSGetSplitJacobians(TS,Mat*,Mat*,Mat*,Mat*);
PETSC_INTERN PetscErrorCode TSSplitJacobiansDestroy_Private(void*);
PETSC_INTERN PetscErrorCode TSUpdateSplitJacobiansFromHistory_Private(TS,PetscReal);
PETSC_INTERN PetscErrorCode TSComputeIJacobianWithSplits_Private(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
#endif

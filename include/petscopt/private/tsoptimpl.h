#ifndef __PETSCOPT_TSOPTIMPL_H
#define __PETSCOPT_TSOPTIMPL_H

#include <petscopt/tsopt.h>

typedef struct _TSOpt *TSOpt;

PETSC_EXTERN PetscLogEvent TSOPT_Opt_Eval_Grad_DAE;
PETSC_EXTERN PetscLogEvent TSOPT_Opt_Eval_Grad_IC;
PETSC_EXTERN PetscLogEvent TSOPT_Opt_Eval_Hess_DAE;
PETSC_EXTERN PetscLogEvent TSOPT_Opt_Eval_Hess_IC;
PETSC_EXTERN PetscLogEvent TSOPT_Opt_SetUp;
PETSC_EXTERN PetscBool TSOPT_OptPackageInitialized;

PETSC_INTERN PetscErrorCode TSGetTSOpt(TS,TSOpt*);
PETSC_INTERN PetscErrorCode TSSetTSOpt(TS,TSOpt);

PETSC_INTERN PetscErrorCode TSOptHasGradientDAE(TSOpt,PetscBool*,PetscBool*);
PETSC_INTERN PetscErrorCode TSOptHasHessianDAE(TSOpt,PetscBool[3][3]);
PETSC_INTERN PetscErrorCode TSOptHasGradientIC(TSOpt,PetscBool*);
PETSC_INTERN PetscErrorCode TSOptHasHessianIC(TSOpt,PetscBool[2][2]);
PETSC_INTERN PetscErrorCode TSOptEvalGradientDAE(TSOpt,PetscReal,Vec,Vec,Vec,Mat*,Mat*);
PETSC_INTERN PetscErrorCode TSOptEvalGradientIC(TSOpt,PetscReal,Vec,Vec,Mat*,Mat*);
PETSC_INTERN PetscErrorCode TSOptEvalHessianDAE(TSOpt,PetscInt,PetscInt,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode TSOptEvalHessianIC(TSOpt,PetscInt,PetscInt,PetscReal,Vec,Vec,Vec,Vec,Vec);

#endif

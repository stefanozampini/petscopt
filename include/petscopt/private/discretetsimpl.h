#ifndef __PETSCOPT_DISCRETETSIMPL_H
#define __PETSCOPT_DISCRETETSIMPL_H

#include <petscts.h>

PETSC_INTERN PetscErrorCode TSStep_Adjoint_RK(TS);
PETSC_INTERN PetscErrorCode TSStep_TLM_RK(TS);

#endif

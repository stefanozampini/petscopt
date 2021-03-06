#if !defined(__PETSCOPT_ADJOINTTS_H)
#define __PETSCOPT_ADJOINTTS_H

#include <petscts.h>

PETSC_EXTERN PetscErrorCode TSCreateAdjointTS(TS,TS*);
PETSC_EXTERN PetscErrorCode AdjointTSGetModelTS(TS,TS*);
PETSC_EXTERN PetscErrorCode AdjointTSComputeForcing(TS,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec,PetscBool*,Vec);
PETSC_EXTERN PetscErrorCode AdjointTSComputeInitialConditions(TS,Vec,PetscBool);
PETSC_EXTERN PetscErrorCode AdjointTSSetQuadratureVec(TS,Vec);
PETSC_EXTERN PetscErrorCode AdjointTSSetDesignVec(TS,Vec);
PETSC_EXTERN PetscErrorCode AdjointTSSetDirectionVec(TS,Vec);
PETSC_EXTERN PetscErrorCode AdjointTSGetDirectionVec(TS,Vec*);
PETSC_EXTERN PetscErrorCode AdjointTSSetTLMTSAndFOATS(TS,TS,TS);
PETSC_EXTERN PetscErrorCode AdjointTSGetTLMTSAndFOATS(TS,TS*,TS*);
PETSC_EXTERN PetscErrorCode AdjointTSSetTimeLimits(TS,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode AdjointTSEventHandler(TS);
PETSC_EXTERN PetscErrorCode AdjointTSFinalizeQuadrature(TS);
PETSC_EXTERN PetscErrorCode AdjointTSIsDiscrete(TS,PetscBool*);
PETSC_EXTERN PetscErrorCode AdjointTSSetUpStep(TS);
PETSC_EXTERN PetscErrorCode AdjointTSComputeQuadrature(TS,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec,Vec,PetscBool*,Vec);

#endif

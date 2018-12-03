#ifndef __PETSCOPT_TLMTS_H
#define __PETSCOPT_TLMTS_H

#include <petscts.h>

PETSC_EXTERN PetscErrorCode TSCreateTLMTS(TS,TS*);
PETSC_EXTERN PetscErrorCode TLMTSGetRHSVec(TS,Vec*);
PETSC_EXTERN PetscErrorCode TLMTSComputeInitialConditions(TS,PetscReal,Vec);
PETSC_EXTERN PetscErrorCode TLMTSSetPerturbationVec(TS,Vec);
PETSC_EXTERN PetscErrorCode TLMTSSetDesignVec(TS,Vec);
PETSC_EXTERN PetscErrorCode TLMTSGetDesignVec(TS,Vec*);
PETSC_EXTERN PetscErrorCode TLMTSGetModelTS(TS,TS*);

#endif

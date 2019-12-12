#ifndef __PETSCOPT_AUGMENTEDTS_H
#define __PETSCOPT_AUGMENTEDTS_H

#include <petscts.h>

PETSC_EXTERN PetscErrorCode AugmentedTSUpdateModelSolution(TS);
PETSC_EXTERN PetscErrorCode AugmentedTSInitialize(TS);
PETSC_EXTERN PetscErrorCode AugmentedTSFinalize(TS);
PETSC_EXTERN PetscErrorCode TSCreateAugmentedTS(TS,PetscInt,TS[],PetscBool[],PetscErrorCode(*[])(TS,Vec,Vec),TSIJacobian[],Mat[],Mat[],PetscBool,TS*);

#endif

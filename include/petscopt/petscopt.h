#if !defined(__PETSCOPT_PETSCOPT_H)
#define __PETSCOPT_PETSCOPT_H

#include <petscsys.h>

#define KSPHILBERTCG     "hilbertcg"
#define KSPAUGTRIANGULAR "augtriangular"
#define SNESAUGMENTED    "augmented"

PETSC_EXTERN PetscErrorCode PetscOptInitialize(int*,char***,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscOptFinalize();
PETSC_EXTERN PetscBool PetscOptInitialized();

#endif

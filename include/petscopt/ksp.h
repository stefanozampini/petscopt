#if !defined(__PETSCOPT_KSP_H)
#define __PETSCOPT_KSP_H

#include <petscksp.h>

PETSC_EXTERN PetscErrorCode KSPHilbertCGSetMaps(KSP,PetscErrorCode (*)(Vec,Vec,PetscScalar*,void*),PetscErrorCode (*)(Vec,Vec,void*),void*);

#endif

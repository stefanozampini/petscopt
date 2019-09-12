#if !defined(__PETSCOPT_SYS_CL_H)
#define __PETSCOPT_SYS_CL_H

#include <petscsys.h>

#if PETSC_VERSION_GE(3,12,0)
#define PetscRoundReal(A) ( PetscAbsReal(PetscCeilReal((A))-(A)) < PetscAbsReal(PetscFloorReal((A))-(A)) ? PetscCeilReal((A)) : PetscFloorReal((A)))
#endif

#endif

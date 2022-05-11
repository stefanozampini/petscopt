#if !defined(__PETSCOPT_SYS_CL_H)
#define __PETSCOPT_SYS_CL_H

#include <petscsys.h>

#if PETSC_VERSION_GE(3,12,0)
#define PetscRoundReal(A) ( PetscAbsReal(PetscCeilReal((A))-(A)) < PetscAbsReal(PetscFloorReal((A))-(A)) ? PetscCeilReal((A)) : PetscFloorReal((A)))
#endif

#if PETSC_VERSION_LT(3,18,0)
#define PetscOptionsHeadBegin(a,b) CHKERRQ(PetscOptionsHead(a,b))
#define PetscOptionsHeadEnd() CHKERRQ(PetscOptionsTail())
#endif

#endif

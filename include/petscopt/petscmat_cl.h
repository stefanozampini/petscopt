#if !defined(__PETSCOPT_CL_MAT_H)
#define __PETSCOPT_CL_MAT_H

#include <petscmat.h>

#if PETSC_VERSION_LT(3,12,0)
#define MatComputeOperator(A,B,C) MatComputeExplicitOperator(A,C)
#define MatComputeOperatorTranspose(A,B,C) MatComputeExplicitOperatorTranspose(A,C)
#endif

#include <petscopt/petscsys_cl.h>

#endif

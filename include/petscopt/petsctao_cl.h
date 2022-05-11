#if !defined(__PETSCOPT_CL_TAO_H)
#define __PETSCOPT_CL_TAO_H

#include <petsctao.h>

#if PETSC_VERSION_LT(3,17,0)
#define TaoSetObjective(A,B,C) TaoSetObjectiveRoutine(A,B,C)
#define TaoSetGradient(A,B,C,D) TaoSetGradientRoutine(A,C,D)
#define TaoSetObjectiveAndGradient(A,B,C,D) TaoSetObjectiveAndGradientRoutine(A,C,D)
#define TaoSetHessian(A,B,C,D,E) TaoSetHessianRoutine(A,B,C,D,E)
#define TaoSetSolution(A,B) TaoSetInitialVector(A,B)
#endif

#include <petscopt/petscsys_cl.h>

#endif

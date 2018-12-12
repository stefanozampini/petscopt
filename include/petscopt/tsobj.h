#if !defined(__PETSCOPT_TSOBJ_H)
#define __PETSCOPT_TSOBJ_H

/* ---------------------- DAE-constrained optimization support -----------------------------*/
/* Targets problems of the type

     min/max obj(x(m),m) , obj(x,m) = \int^{TF}_{T0} (\sum_j f_j(x,m,t)) dt + \sum_j g_j(x,m,t=tfixed_j), with f_i, g_j : X \times M \times R -> R
        m

     subject to
       F(x,x_t,t;m) = 0  Parameter dependent DAE in implicit form
       G(x(t0),m)   = 0  Initial conditions

   Here we define the API to specify the objective functions
*/

#include <petscts.h>

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSEvalObjective)(Vec,Vec,PetscReal,PetscReal*,void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSEvalObjectiveGradient)(Vec,Vec,PetscReal,Vec,void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSEvalObjectiveHessian)(Vec,Vec,PetscReal,Mat,void*);
PETSC_EXTERN PetscErrorCode TSResetObjective(TS);
PETSC_EXTERN PetscErrorCode TSAddObjective(TS,PetscReal,TSEvalObjective,TSEvalObjectiveGradient,TSEvalObjectiveGradient,
                                           Mat,TSEvalObjectiveHessian,Mat,TSEvalObjectiveHessian,Mat,TSEvalObjectiveHessian,void*);

#endif

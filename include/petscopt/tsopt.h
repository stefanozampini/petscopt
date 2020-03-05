#if !defined(__PETSCOPT_TSOPT_H)
#define __PETSCOPT_TSOPT_H

/* ---------------------- DAE-constrained optimization support -----------------------------*/
/* Targets problems of the type

     min/max obj(x(m),m) , obj(x,m) = \int^{TF}_{T0} (\sum_j f_j(x,m,t)) dt + \sum_j g_j(x,m,t=tfixed_j), with f_i, g_j : X \times M \times R -> R
        m

     subject to
       F(x,x_t,t;m) = 0  Parameter dependent DAE in implicit form
       G(x(t0),m)   = 0  Initial conditions

   Here we define the API to customize the parameter dependency of the DAE
*/

#include <petscts.h>
#include <petscopt/petscts_cl.h>

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSEvalGradientIC)(TS,PetscReal,Vec,Vec,Mat,Mat,void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSEvalGradientDAE)(TS,PetscReal,Vec,Vec,Vec,Mat,void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSEvalHessianDAE)(TS,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec,void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSEvalHessianIC)(TS,PetscReal,Vec,Vec,Vec,Vec,Vec,void*);

PETSC_EXTERN PetscErrorCode TSEvalGradientICDefault(TS,PetscReal,Vec,Vec,Mat,Mat,void*);
PETSC_EXTERN PetscErrorCode TSSetGradientDAE(TS,Mat,TSEvalGradientDAE,void*);
PETSC_EXTERN PetscErrorCode TSSetHessianDAE(TS,TSEvalHessianDAE,TSEvalHessianDAE,TSEvalHessianDAE,
                                               TSEvalHessianDAE,TSEvalHessianDAE,TSEvalHessianDAE,
                                               TSEvalHessianDAE,TSEvalHessianDAE,TSEvalHessianDAE,void*);
PETSC_EXTERN PetscErrorCode TSSetGradientIC(TS,Mat,Mat,TSEvalGradientIC,void*);
PETSC_EXTERN PetscErrorCode TSSetHessianIC(TS,TSEvalHessianIC,TSEvalHessianIC,TSEvalHessianIC,TSEvalHessianIC,void*);
PETSC_EXTERN PetscErrorCode TSSetSetUpFromDesign(TS,PetscErrorCode (*)(TS,Vec,Vec,void*),void*);
PETSC_EXTERN PetscErrorCode TSSetUpFromDesign(TS,Vec,Vec);

PETSC_EXTERN PetscErrorCode TSComputeObjectiveAndGradient(TS,PetscReal,PetscReal,PetscReal,Vec,Vec,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode TSComputeHessian(TS,PetscReal,PetscReal,PetscReal,Vec,Vec,Mat);
PETSC_EXTERN PetscErrorCode TSCreatePropagatorMat(TS,PetscReal,PetscReal,PetscReal,Vec,Vec,Mat,Mat*);

PETSC_EXTERN PetscErrorCode TSTaylorTest(TS,PetscReal,PetscReal,PetscReal,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSCheckGradientDAE(TS,PetscReal,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSCheckHessianIC(TS,PetscReal,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSCheckHessianDAE(TS,PetscReal,Vec,Vec,Vec,Vec);

#endif

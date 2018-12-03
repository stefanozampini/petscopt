#ifndef __PETSCOPT_TSPDECONSTRAINEDUTILSIMPL_H
#define __PETSCOPT_TSPDECONSTRAINEDUTILSIMPL_H

//#include <petscopt/private/tsobjimpl.h>
#include <petscts.h>

/* prototypes for cost integral evaluation */
typedef PetscErrorCode (*SQuadEval)(Vec,PetscReal,PetscReal*,void*);
typedef PetscErrorCode (*VQuadEval)(Vec,PetscReal,Vec,void*);

typedef struct {
  PetscErrorCode (*user)(TS); /* user post step method */
  PetscBool      userafter;   /* call user-defined poststep after quadrature evaluation */
  SQuadEval      seval;       /* scalar function to be evaluated */
  void           *seval_ctx;  /* context for scalar function */
  PetscReal      squad;       /* scalar function value */
  PetscReal      psquad;      /* previous scalar function value (for trapezoidal rule) */
  VQuadEval      veval;       /* vector function to be evaluated */
  void           *veval_ctx;  /* context for vector function */
  Vec            vquad;       /* used for vector quadrature */
  Vec            *wquad;      /* quadrature work vectors used by the trapezoidal rule + 3 extra work vectors */
  PetscInt       cur,old;     /* pointers to current and old wquad vectors for trapezoidal rule */
} TSQuadratureCtx;

PETSC_INTERN PetscErrorCode TSQuadraturePostStep_Private(TS);
PETSC_INTERN PetscErrorCode TSQuadratureCtxDestroy_Private(void*);
PETSC_INTERN PetscErrorCode TSLinearizedICApply_Private(TS, PetscReal,Vec,Vec,Vec,Vec,PetscBool,PetscBool);
PETSC_INTERN PetscErrorCode TSSolveWithQuadrature_Private(TS,Vec,Vec,Vec,Vec,PetscReal*);
#endif

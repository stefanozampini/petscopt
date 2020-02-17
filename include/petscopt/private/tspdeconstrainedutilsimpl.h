#ifndef __PETSCOPT_TSPDECONSTRAINEDUTILSIMPL_H
#define __PETSCOPT_TSPDECONSTRAINEDUTILSIMPL_H

#include <petscts.h>

/* prototypes for cost integral evaluation */
typedef PetscErrorCode (*QuadEval)(Vec,Vec,PetscReal,Vec,void*);
typedef PetscErrorCode (*QuadJacEval)(Vec,Vec,PetscReal,PetscReal,Mat,Mat,void*);
typedef struct {
  Vec         U;
  Vec         Udot;
  Vec         design;
  QuadEval    evalquad;
  QuadEval    evalquad_fixed;
  QuadJacEval evaljaccoup;
  void        *evalquadctx;
} TSQuadCtx;

PETSC_INTERN PetscErrorCode TSCreateQuadTS(TS,Vec,PetscBool,TSQuadCtx*,TS*);
PETSC_INTERN PetscErrorCode QuadTSUpdateStates(TS,Vec,Vec);
PETSC_INTERN PetscErrorCode TSQuadraturePostStep_Private(TS);
PETSC_INTERN PetscErrorCode TSQuadratureCtxDestroy_Private(void*);
PETSC_INTERN PetscErrorCode TSLinearizedICApply_Private(TS, PetscReal,Vec,Vec,Vec,Vec,PetscBool,PetscBool);
PETSC_INTERN PetscErrorCode TSSolveWithQuadrature_Private(TS,Vec,Vec,Vec,Vec,PetscReal*,PetscBool*);
#endif

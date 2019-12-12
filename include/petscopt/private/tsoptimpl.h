#ifndef __PETSCOPT_TSOPTIMPL_H
#define __PETSCOPT_TSOPTIMPL_H

#include <petscopt/tsopt.h>

struct _TSOpt {
  TS                ts;         /* pointer to the parent TS (not reference counted) */
  TSEvalGradientIC  Ggrad;      /* compute the IC Jacobian terms G_m(x(t0),m) and G_x(x(t0),m) */
  Mat               G_x;
  Mat               G_m;
  void              *Ggrad_ctx;
  TSEvalHessianIC   HG[2][2];   /* compute the IC Hessian terms G_xx, G_xm, G_mx and G_mm */
  void              *HGctx;
  TSEvalGradientDAE F_m_f;      /* compute the DAE Jacobian term F_m(x,x_t,t;m) */
  Mat               F_m;
  Mat               adjF_m;
  void              *F_m_ctx;
  TSEvalHessianDAE  HF[3][3];   /* compute the DAE Hessian terms F_{x|x_t|m}{x|x_t|m} */
  void              *HFctx;
  PetscErrorCode    (*setupfromdesign)(TS,Vec,Vec,void*);
  void              *setupfromdesignctx;
};
typedef struct _TSOpt *TSOpt;

PETSC_EXTERN PetscLogEvent TSOPT_Opt_Eval_Grad_DAE;
PETSC_EXTERN PetscLogEvent TSOPT_Opt_Eval_Grad_IC;
PETSC_EXTERN PetscLogEvent TSOPT_Opt_Eval_Hess_DAE;
PETSC_EXTERN PetscLogEvent TSOPT_Opt_Eval_Hess_IC;
PETSC_EXTERN PetscLogEvent TSOPT_Opt_SetUp;
PETSC_EXTERN PetscBool TSOPT_OptPackageInitialized;

PETSC_INTERN PetscErrorCode TSGetTSOpt(TS,TSOpt*);
PETSC_INTERN PetscErrorCode TSSetTSOpt(TS,TSOpt);

PETSC_INTERN PetscErrorCode TSOptHasGradientDAE(TSOpt,PetscBool*,PetscBool*);
PETSC_INTERN PetscErrorCode TSOptHasHessianDAE(TSOpt,PetscBool[3][3]);
PETSC_INTERN PetscErrorCode TSOptHasGradientIC(TSOpt,PetscBool*);
PETSC_INTERN PetscErrorCode TSOptHasHessianIC(TSOpt,PetscBool[2][2]);
PETSC_INTERN PetscErrorCode TSOptEvalGradientDAE(TSOpt,PetscReal,Vec,Vec,Vec,Mat*,Mat*);
PETSC_INTERN PetscErrorCode TSOptEvalGradientIC(TSOpt,PetscReal,Vec,Vec,Mat*,Mat*);
PETSC_INTERN PetscErrorCode TSOptEvalHessianDAE(TSOpt,PetscInt,PetscInt,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode TSOptEvalHessianIC(TSOpt,PetscInt,PetscInt,PetscReal,Vec,Vec,Vec,Vec,Vec);

#endif

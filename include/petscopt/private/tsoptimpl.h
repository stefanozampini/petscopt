#ifndef __PETSCOPT_TSOPTIMPL_H
#define __PETSCOPT_TSOPTIMPL_H

#include <petscopt/tsopt.h>

typedef struct _TSOpt *TSOpt;

PETSC_INTERN PetscErrorCode TSGetTSOpt(TS,TSOpt*);

struct _TSOpt {
  TSEvalGradientIC  Ggrad;      /* compute the IC Jacobian terms G_m(x(t0),m) and G_x(x(t0),m) */
  Mat               G_x;
  Mat               G_m;
  void              *Ggrad_ctx;
  TSEvalHessianIC   HG[2][2];   /* compute the IC Hessian terms G_xx, G_xm, G_mx and G_mm */
  void              *HGctx;
  TSEvalGradientDAE F_m_f;      /* compute the DAE Jacobian term F_m(x,x_t,t;m) */
  Mat               F_m;
  void              *F_m_ctx;
  TSEvalHessianDAE  HF[3][3];   /* compute the DAE Hessian terms F_{x|x_t|m}{x|x_t|m} */
  void              *HFctx;
  PetscErrorCode    (*setupfromdesign)(TS,Vec,Vec,void*);
  void              *setupfromdesignctx;
};

#endif

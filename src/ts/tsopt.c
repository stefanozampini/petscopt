#include <petscopt/private/tsoptimpl.h>
#include <petsc/private/petscimpl.h>

PetscLogEvent TSOPT_Opt_Eval_Grad_DAE = 0;
PetscLogEvent TSOPT_Opt_Eval_Grad_IC  = 0;
PetscLogEvent TSOPT_Opt_Eval_Hess_DAE = 0;
PetscLogEvent TSOPT_Opt_Eval_Hess_IC  = 0;
PetscLogEvent TSOPT_Opt_SetUp         = 0;
PetscBool TSOPT_OptPackageInitialized = PETSC_FALSE;

static PetscErrorCode TSOptDestroy_Private(void *ptr)
{
  TSOpt          tsopt = (TSOpt)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&tsopt->G_x);CHKERRQ(ierr);
  ierr = MatDestroy(&tsopt->G_m);CHKERRQ(ierr);
  ierr = MatDestroy(&tsopt->F_m);CHKERRQ(ierr);
  ierr = MatDestroy(&tsopt->adjF_m);CHKERRQ(ierr);
  ierr = PetscFree(tsopt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSGetTSOpt(TS ts, TSOpt *tsopt)
{
  TSOpt          t;
  PetscContainer c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_opt_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) {
    ierr = PetscNew(&t);CHKERRQ(ierr);
    t->ts = ts;
    /* zeros pointers for Hessian callbacks */
    t->HF[0][0] = t->HF[0][1] = t->HF[0][2] = NULL;
    t->HF[1][0] = t->HF[1][1] = t->HF[1][2] = NULL;
    t->HF[2][0] = t->HF[2][1] = t->HF[2][2] = NULL;
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&c);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(c,t);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(c,TSOptDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"_ts_opt_ctx",(PetscObject)c);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)c);CHKERRQ(ierr);
  } else {
    ierr = PetscContainerGetPointer(c,(void**)&t);CHKERRQ(ierr);
  }
  *tsopt = t;
  PetscFunctionReturn(0);
}

PetscErrorCode TSSetTSOpt(TS ts, TSOpt tsopt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSSetGradientDAE(ts,tsopt->F_m,tsopt->F_m_f,tsopt->F_m_ctx);CHKERRQ(ierr);
  ierr = TSSetGradientIC(ts,tsopt->G_x,tsopt->G_m,tsopt->Ggrad,tsopt->Ggrad_ctx);CHKERRQ(ierr);
  ierr = TSSetHessianDAE(ts,tsopt->HF[0][0],tsopt->HF[0][1],tsopt->HF[0][2],
                            tsopt->HF[1][0],tsopt->HF[1][1],tsopt->HF[1][2],
                            tsopt->HF[2][0],tsopt->HF[2][1],tsopt->HF[2][2],tsopt->HFctx);CHKERRQ(ierr);
  ierr = TSSetHessianIC(ts,tsopt->HG[0][0],tsopt->HG[0][1],
                           tsopt->HG[1][0],tsopt->HG[1][1],tsopt->HGctx);CHKERRQ(ierr);
  ierr = TSSetSetUpFromDesign(ts,tsopt->setupfromdesign,tsopt->setupfromdesignctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSOptHasGradientDAE(TSOpt tsopt, PetscBool *has, PetscBool *hasnc)
{
  PetscFunctionBegin;
  if (has)   *has   = tsopt->F_m   ? PETSC_TRUE : PETSC_FALSE;
  if (hasnc) *hasnc = tsopt->F_m_f ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSOptEvalGradientDAE(TSOpt tsopt, PetscReal t, Vec U, Vec Udot, Vec M, Mat* F_m, Mat *adjF_m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(TSOPT_Opt_Eval_Grad_DAE,0,0,0,0);CHKERRQ(ierr);
  if (tsopt->F_m_f && M) { /* non constant dependence */
    TSTrajectory tj;
    Vec          W[2];

    W[0] = U;
    W[1] = Udot;
    ierr = TSGetTrajectory(tsopt->ts,&tj);CHKERRQ(ierr);
    if (!U || !Udot) {
      ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,tsopt->ts,t,U ? NULL : &W[0],Udot ? NULL : &W[1]);CHKERRQ(ierr);
    }
    ierr = (*tsopt->F_m_f)(tsopt->ts,t,W[0],W[1],M,tsopt->F_m,tsopt->F_m_ctx);CHKERRQ(ierr);
    if (!U || !Udot) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,U ? NULL : &W[0],Udot ? NULL : &W[1]);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(TSOPT_Opt_Eval_Grad_DAE,0,0,0,0);CHKERRQ(ierr);
  /* In the future we can probably support different discretization spaces for
     forward and adjoint states */
  if (F_m)       *F_m = tsopt->F_m;
  if (adjF_m) *adjF_m = tsopt->adjF_m;
  PetscFunctionReturn(0);
}

PetscErrorCode TSOptHasGradientIC(TSOpt tsopt, PetscBool *has)
{
  PetscFunctionBegin;
  if (has) *has = tsopt->G_m ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSOptEvalGradientIC(TSOpt tsopt, PetscReal t, Vec x, Vec M, Mat* G_x, Mat *G_m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(TSOPT_Opt_Eval_Grad_IC,0,0,0,0);CHKERRQ(ierr);
  if (tsopt->Ggrad && M) {
    ierr = (*tsopt->Ggrad)(tsopt->ts,t,x,M,tsopt->G_x,tsopt->G_m,tsopt->Ggrad_ctx);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Opt_Eval_Grad_IC,0,0,0,0);CHKERRQ(ierr);
  if (G_x) *G_x = tsopt->G_x;
  if (G_m) *G_m = tsopt->G_m;
  PetscFunctionReturn(0);
}

PetscErrorCode TSOptHasHessianDAE(TSOpt tsopt, PetscBool has[3][3])
{
  PetscFunctionBegin;
  has[0][0] = tsopt->HF[0][0] ? PETSC_TRUE : PETSC_FALSE;
  has[0][1] = tsopt->HF[0][1] ? PETSC_TRUE : PETSC_FALSE;
  has[0][2] = tsopt->HF[0][2] ? PETSC_TRUE : PETSC_FALSE;
  has[1][0] = tsopt->HF[1][0] ? PETSC_TRUE : PETSC_FALSE;
  has[1][1] = tsopt->HF[1][1] ? PETSC_TRUE : PETSC_FALSE;
  has[1][2] = tsopt->HF[1][2] ? PETSC_TRUE : PETSC_FALSE;
  has[2][0] = tsopt->HF[2][0] ? PETSC_TRUE : PETSC_FALSE;
  has[2][1] = tsopt->HF[2][1] ? PETSC_TRUE : PETSC_FALSE;
  has[2][2] = tsopt->HF[2][2] ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSOptEvalHessianDAE(TSOpt tsopt, PetscInt w0, PetscInt w1, PetscReal t, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y)
{
  TSTrajectory   tj;
  Vec            W[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(TSOPT_Opt_Eval_Hess_DAE,0,0,0,0);CHKERRQ(ierr);
  if (tsopt->HF[w0][w1]) {
    W[0] = U;
    W[1] = Udot;
    ierr = TSGetTrajectory(tsopt->ts,&tj);CHKERRQ(ierr);
    if (!U || !Udot) {
      ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,tsopt->ts,t,U ? NULL : &W[0],Udot ? NULL : &W[1]);CHKERRQ(ierr);
    }
    ierr = (*tsopt->HF[w0][w1])(tsopt->ts,t,W[0],W[1],M,L,X,Y,tsopt->HFctx);CHKERRQ(ierr);
    if (!U || !Udot) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,U ? NULL : &W[0],Udot ? NULL : &W[1]);CHKERRQ(ierr);
    }
  } else {
    ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Opt_Eval_Hess_DAE,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSOptHasHessianIC(TSOpt tsopt, PetscBool has[2][2])
{
  PetscFunctionBegin;
  has[0][0] = tsopt->HG[0][0] ? PETSC_TRUE : PETSC_FALSE;
  has[0][1] = tsopt->HG[0][1] ? PETSC_TRUE : PETSC_FALSE;
  has[1][0] = tsopt->HG[1][0] ? PETSC_TRUE : PETSC_FALSE;
  has[1][1] = tsopt->HG[1][1] ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSOptEvalHessianIC(TSOpt tsopt, PetscInt w0, PetscInt w1, PetscReal t, Vec U, Vec M, Vec L, Vec X, Vec Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(TSOPT_Opt_Eval_Hess_IC,0,0,0,0);CHKERRQ(ierr);
  if (tsopt->HG[w0][w1]) {
    ierr = (*tsopt->HG[w0][w1])(tsopt->ts,t,U,M,L,X,Y,tsopt->HGctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Opt_Eval_Hess_IC,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSOptFinalizePackage(void)
{
  PetscFunctionBegin;
  TSOPT_OptPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSOptInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSOPT_OptPackageInitialized) PetscFunctionReturn(0);
  TSOPT_OptPackageInitialized = PETSC_TRUE;
  /* Register Events */
  ierr = PetscLogEventRegister("TSOptEvalGrad",  0,&TSOPT_Opt_Eval_Grad_DAE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalGradIC",0,&TSOPT_Opt_Eval_Grad_IC);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHess",  0,&TSOPT_Opt_Eval_Hess_DAE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHessIC",0,&TSOPT_Opt_Eval_Hess_IC);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptSetUp",     0,&TSOPT_Opt_SetUp);CHKERRQ(ierr);
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  /* LCOV_EXCL_START */
  if (opt) {
    ierr = PetscStrInList("tsopt",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {
      ierr = PetscLogEventDeactivate(TSOPT_Opt_Eval_Grad_DAE);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_Opt_Eval_Grad_IC);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_Opt_Eval_Hess_DAE);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_Opt_Eval_Hess_IC);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_Opt_SetUp);CHKERRQ(ierr);
    }
  }
  /* LCOV_EXCL_STOP */
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(TSOptFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSSetGradientDAE - Sets the callback for the evaluation of the Jacobian matrix F_m(t,x(t),x_t(t);m) of a parameter dependent DAE, written in the implicit form F(t,x(t),x_t(t);m) = 0.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  J   - the Mat object to hold F_m(t,x(t),x_t(t);m)
.  f   - the function evaluation routine
-  ctx - user-defined context for the function evaluation routine (can be NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec m,Mat J,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  m   - design vector
.  J   - the jacobian
-  ctx - [optional] user-defined context

   Notes: The ij entry of F_m is given by \frac{\partial F_i}{\partial m_j}, where F_i is the i-th component of the DAE and m_j the j-th design variable.
          The row and column layouts of the J matrix have to be compatible with those of the state and design vector, respectively.
          The matrix doesn't need to be in assembled form. For propagator computations, J needs to implement MatMult() and MatMultTranspose().
          For gradient and Hessian computations, both MatMult() and MatMultTranspose() need to be implemented.
          Pass NULL for J if you want to cancel the DAE dependence on the parameters.

   Level: advanced

.seealso: TSAddObjective(), TSSetHessianDAE(), TSComputeObjectiveAndGradient(), TSSetGradientIC(), TSSetHessianIC(), TSCreatePropagatorMat()
@*/
PetscErrorCode TSSetGradientDAE(TS ts, Mat J, TSEvalGradientDAE f, void *ctx)
{
  TSOpt          tsopt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSOptInitializePackage();CHKERRQ(ierr);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(ts,1,J,2);
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
  }
  ierr = TSGetTSOpt(ts,&tsopt);CHKERRQ(ierr);

  ierr           = MatDestroy(&tsopt->adjF_m);CHKERRQ(ierr);
  ierr           = MatDestroy(&tsopt->F_m);CHKERRQ(ierr);
  tsopt->F_m     = J;
  tsopt->F_m_f   = J ? f : NULL;
  tsopt->F_m_ctx = J ? ctx : NULL;
  if (tsopt->F_m) { /* TODO ADJ: need a placeholder and a setup for the adjoint matrix */
    ierr = MatCreateTranspose(tsopt->F_m,&tsopt->adjF_m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSSetHessianDAE - Sets the callbacks for the evaluation of Hessian terms of a parameter dependent DAE, written in the implicit form F(t,x(t),x_t(t);m) = 0.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context obtained from TSCreate()
.  f_xx   - the function evaluation routine for second order state derivative
.  f_xxt  - the function evaluation routine for second order mixed x,x_t derivative
.  f_xm   - the function evaluation routine for second order mixed state and parameter derivative
.  f_xtx  - the function evaluation routine for second order mixed x_t,x derivative
.  f_xtxt - the function evaluation routine for second order x_t,x_t derivative
.  f_xtm  - the function evaluation routine for second order mixed x_t and parameter derivative
.  f_mx   - the function evaluation routine for second order mixed m,x derivative
.  f_mxt  - the function evaluation routine for second order mixed m,x_t derivative
.  f_mm   - the function evaluation routine for second order parameter derivative
-  ctx    - user-defined context for the function evaluation routines (can be NULL)

   Calling sequence of each function evaluation routine:
$  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec m,Vec L,Vec X,Vec Y,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  m   - design vector
.  L   - input vector (adjoint variable)
.  X   - input vector (state or parameter variable)
.  Y   - output vector (state or parameter variable)
-  ctx - [optional] user-defined context

   Notes: the callbacks need to return

$  f_xx   : Y = (L^T \otimes I_N)*F_UU*X
$  f_xxt  : Y = (L^T \otimes I_N)*F_UUdot*X
$  f_xm   : Y = (L^T \otimes I_N)*F_UM*X
$  f_xtx  : Y = (L^T \otimes I_N)*F_UdotU*X
$  f_xtxt : Y = (L^T \otimes I_N)*F_UdotUdot*X
$  f_xtm  : Y = (L^T \otimes I_N)*F_UdotM*X
$  f_mx   : Y = (L^T \otimes I_P)*F_MU*X
$  f_mxt  : Y = (L^T \otimes I_P)*F_MUdot*X
$  f_mm   : Y = (L^T \otimes I_P)*F_MM*X

   where L is a vector of size N (the number of DAE equations), I_x the identity matrix of size x, \otimes is the Kronecker product, X an input vector of appropriate size, and F_AB an N*size(A) x size(B) matrix given as

$            | F^1_AB |
$     F_AB = |   ...  |, A = {U|Udot|M}, B = {U|Udot|M}.
$            | F^N_AB |

   Each F^k_AB block term has dimension size(A) x size(B), with {F^k_AB}_ij = \frac{\partial^2 F_k}{\partial b_j \partial a_i}, where F_k is the k-th component of the DAE, a_i the i-th variable of A and b_j the j-th variable of B.
   For example, {F^k_UM}_ij = \frac{\partial^2 F_k}{\partial m_j \partial u_i}.
   Developing the Kronecker product, we get Y = (\sum_k L_k*F^k_AB)*X, with L_k the k-th entry of the adjoint variable L.
   Pass NULL if F_AB is zero for some A and B.

   Level: advanced

.seealso: TSAddObjective(), TSSetGradientDAE(), TSComputeObjectiveAndGradient(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode TSSetHessianDAE(TS ts, TSEvalHessianDAE f_xx,  TSEvalHessianDAE f_xxt,  TSEvalHessianDAE f_xm,
                                      TSEvalHessianDAE f_xtx, TSEvalHessianDAE f_xtxt, TSEvalHessianDAE f_xtm,
                                      TSEvalHessianDAE f_mx,  TSEvalHessianDAE f_mxt,  TSEvalHessianDAE f_mm, void *ctx)
{
  TSOpt          tsopt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSOptInitializePackage();CHKERRQ(ierr);
  ierr = TSGetTSOpt(ts,&tsopt);CHKERRQ(ierr);

  tsopt->HF[0][0] = f_xx;
  tsopt->HF[0][1] = f_xxt;
  tsopt->HF[0][2] = f_xm;
  tsopt->HF[1][0] = f_xtx;
  tsopt->HF[1][1] = f_xtxt;
  tsopt->HF[1][2] = f_xtm;
  tsopt->HF[2][0] = f_mx;
  tsopt->HF[2][1] = f_mxt;
  tsopt->HF[2][2] = f_mm;
  tsopt->HFctx    = ctx;
  PetscFunctionReturn(0);
}

/*@C
   TSSetGradientIC - Sets the callback to compute the Jacobian matrices G_x(x0,m) and G_m(x0,m), with parameter dependent initial conditions implicitly defined by the function G(x(0),m) = 0.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  J_x - the Mat object to hold G_x(x0,m) (optional, if NULL identity is assumed)
.  J_m - the Mat object to hold G_m(x0,m)
.  f   - the function evaluation routine
-  ctx - user-defined context for the function evaluation routine (can be NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec m,Mat Gx,Mat Gm,void *ctx);

+  t   - initial time
.  u   - state vector (at initial time)
.  m   - design vector
.  Gx  - the Mat object to hold the Jacobian wrt the state variables
.  Gm  - the Mat object to hold the Jacobian wrt the design variables
-  ctx - [optional] user-defined context

   Notes: J_x is a square matrix of the same size of the state vector. J_m is a rectangular matrix with "state size" rows and "design size" columns.
          If f is not provided, J_x is assumed constant. The J_m matrix doesn't need to assembled; only MatMult() and MatMultTranspose() are needed.
          Currently, the initial condition vector should be computed by the user.
          Pass NULL for J_m if you want to cancel the initial condition dependency from the parameters.

   Level: advanced

.seealso: TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetHessianIC(), TSComputeObjectiveAndGradient(), MATSHELL
@*/
PetscErrorCode TSSetGradientIC(TS ts, Mat J_x, Mat J_m, TSEvalGradientIC f, void *ctx)
{
  TSOpt          tsopt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSOptInitializePackage();CHKERRQ(ierr);
  if (J_x) {
    PetscValidHeaderSpecific(J_x,MAT_CLASSID,2);
    PetscCheckSameComm(ts,1,J_x,2);
  }
  if (J_m) {
    PetscValidHeaderSpecific(J_m,MAT_CLASSID,3);
    PetscCheckSameComm(ts,1,J_m,3);
  } else J_x = NULL;

  ierr = TSGetTSOpt(ts,&tsopt);CHKERRQ(ierr);

  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_G",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_GW",NULL);CHKERRQ(ierr);

  if (J_x) {
    ierr = PetscObjectReference((PetscObject)J_x);CHKERRQ(ierr);
  }
  ierr       = MatDestroy(&tsopt->G_x);CHKERRQ(ierr);
  tsopt->G_x = J_x;

  if (J_m) {
    ierr = PetscObjectReference((PetscObject)J_m);CHKERRQ(ierr);
  }
  ierr             = MatDestroy(&tsopt->G_m);CHKERRQ(ierr);
  tsopt->G_m       = J_m;
  tsopt->Ggrad     = f;
  tsopt->Ggrad_ctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   TSSetHessianIC - Sets the callbacks to compute the action of the Hessian matrices G_xx(x0,m), G_xm(x0,m), G_mx(x0,m) and G_mm(x0,m), with parameter dependent initial conditions implicitly defined by the function G(x(0),m) = 0.

   Logically Collective on TS

   Input Parameters:
+  ts   - the TS context obtained from TSCreate()
.  g_xx - the function evaluation routine for second order state derivative
.  g_xm - the function evaluation routine for second order mixed x,m derivative
.  g_mx - the function evaluation routine for second order mixed m,x derivative
.  g_mm - the function evaluation routine for second order parameter derivative
-  ctx  - user-defined context for the function evaluation routines (can be NULL)

   Calling sequence of each function evaluation routine:
$  f(TS ts,PetscReal t,Vec u,Vec m,Vec L,Vec X,Vec Y,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  m   - design vector
.  L   - input vector (adjoint variable)
.  X   - input vector (state or parameter variable)
.  Y   - output vector (state or parameter variable)
-  ctx - [optional] user-defined context

   Notes: the callbacks need to return

$  g_xx   : Y = (L^T \otimes I_N)*G_UU*X
$  g_xm   : Y = (L^T \otimes I_N)*G_UM*X
$  g_mx   : Y = (L^T \otimes I_P)*G_MU*X
$  g_mm   : Y = (L^T \otimes I_P)*G_MM*X

   where L is a vector of size N (the number of DAE equations), I_x the identity matrix of size x, \otimes is the Kronecker product, X an input vector of appropriate size, and G_AB an N*size(A) x size(B) matrix given as

$            | G^1_AB |
$     G_AB = |   ...  | , A = {U|M}, B = {U|M}.
$            | G^N_AB |

   Each G^k_AB block term has dimension size(A) x size(B), with {G^k_AB}_ij = \frac{\partial^2 G_k}{\partial b_j \partial a_i}, where G_k is the k-th component of the implicit function G that determines the initial conditions, a_i the i-th variable of A and b_j the j-th variable of B.
   For example, {G^k_UM}_ij = \frac{\partial^2 G_k}{\partial m_j \partial u_i}.
   Developing the Kronecker product, we get Y = (\sum_k L_k*G^k_AB)*X, with L_k the k-th entry of the adjoint variable L.
   Pass NULL if G_AB is zero for some A and B.

   Level: advanced

.seealso: TSAddObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSComputeObjectiveAndGradient(), TSSetGradientIC()
@*/
PetscErrorCode TSSetHessianIC(TS ts, TSEvalHessianIC g_xx,  TSEvalHessianIC g_xm,  TSEvalHessianIC g_mx, TSEvalHessianIC g_mm, void *ctx)
{
  TSOpt          tsopt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSOptInitializePackage();CHKERRQ(ierr);
  ierr = TSGetTSOpt(ts,&tsopt);CHKERRQ(ierr);

  tsopt->HG[0][0] = g_xx;
  tsopt->HG[0][1] = g_xm;
  tsopt->HG[1][0] = g_mx;
  tsopt->HG[1][1] = g_mm;
  tsopt->HGctx    = ctx;
  PetscFunctionReturn(0);
}

/*@
  TSSetSetUpFromDesign - Set the function to be run when the parameters change

  Collective on TS

  Input Parameter:
+ ts       - The TS context
. setup    - The setup function
- setupctx - The setup function context (can be NULL)

  Calling sequence of setup:
$  setup(TS ts, Vec x0, Vec design, void *ctx);

+  ts     - the TS context
.  x0     - the vector of initial conditions
.  design - the vector of parameters
-  ctx    - [optional] context for setup function

  Level: developer

.keywords: TS
.seealso: TSCreate(), TSSetType(), TSSetGradientDAE(), TSSetGradientIC(), TSSetHessianDAE(), TSSetHessianIC(), TSAddObjective()
@*/
PetscErrorCode TSSetSetUpFromDesign(TS ts,PetscErrorCode (*setup)(TS,Vec,Vec,void*),void* setupctx)
{
  PetscErrorCode ierr;
  TSOpt          tsopt;

  PetscFunctionBegin;
  ierr = TSOptInitializePackage();CHKERRQ(ierr);
  ierr = TSGetTSOpt(ts,&tsopt);CHKERRQ(ierr);
  tsopt->setupfromdesign    = setup;
  tsopt->setupfromdesignctx = setupctx;
  PetscFunctionReturn(0);
}

/*@
  TSSetUpFromDesign - Runs user-defined setup function from parameters

  Collective on TS

  Input Parameters:
+ ts     - The TS context
- design - The vector of parameters

  Output Parameters:
. x0     - The vector of initial conditions

  Level: developer

.keywords: TS
.seealso: TSCreate(), TSSetSetUpFromDesign()
@*/
PetscErrorCode TSSetUpFromDesign(TS ts,Vec x0,Vec design)
{
  PetscErrorCode ierr;
  TSOpt          tsopt;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(TSOPT_Opt_SetUp,0,0,0,0);CHKERRQ(ierr);
  ierr = TSGetTSOpt(ts,&tsopt);CHKERRQ(ierr);
  if (tsopt->setupfromdesign) {

    ierr = VecLockReadPush(design);CHKERRQ(ierr);
    ierr = (*tsopt->setupfromdesign)(ts,x0,design,tsopt->setupfromdesignctx);CHKERRQ(ierr);
    ierr = VecLockReadPop(design);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(TSOPT_Opt_SetUp,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

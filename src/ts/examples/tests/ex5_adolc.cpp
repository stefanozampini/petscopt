static const char help[] = "Tests state dependent mass matrix. AD version using ADOL-C";
/*
  This is example 6.1.2 in http://people.cs.vt.edu/~ycao/publication/adj_part2.pdf
  Computes the gradient of

    Obj(u,m) = u0 + u1

  where u = [u0,u1] obeys

     |  u0   u1 | | u0_dot |   |       0      |
     |          | |        | = |              |
     | -u1   u0 | | u1_dot |   | -u0^2 - u1^2 |

  In this version, we use ADOL-C to differentiate the terms.
  TODO: make the ADOL-C enabled callbacks general and part of the library?
*/
#include <petscopt.h>
#include <adolc/adolc.h>

/* trace the objective function */
static PetscErrorCode trace_obj(PetscBool paper)
{
  double  uin[2] = {0.0,0.0},objout;
  adouble obj;
  adouble u[2];

  trace_on(1);
  /* independent variables */
  u[0] <<= uin[0];
  u[1] <<= uin[1];
  /* dependent variable */
  if (paper) obj = u[0] + u[1];
  else       obj = u[0]*u[0] + u[1]*u[1];
  obj >>= objout;
  trace_off();
  return 0;
}

static PetscErrorCode EvalObjective(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  const PetscScalar *u;
  PetscErrorCode    ierr;
  int               n = 2,keep = 0,tagobj = 1;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);

  /* call ADOL-C functionality to sample the objective */
  zos_forward(tagobj,1,n,keep,u,val);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,val,1,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)U));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveGradient_U(Vec U, Vec M, PetscReal time, Vec G, void *ctx)
{
  const PetscScalar *u;
  PetscScalar       *g,dummy,one = 1.0;
  PetscErrorCode    ierr;
  int               n = 2,keep = 1,tagobj = 1;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);

  /* we could as well use the "gradient" API of ADOL-C */
  zos_forward(tagobj,1,n,keep,u,&dummy);
  fos_reverse(tagobj,1,n,&one,g);

  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscScalar U0[2];
} ObjHctx;

static PetscErrorCode ObjectiveHessianDestroy_UU(Mat H)
{
  ObjHctx        *mctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(H,(void**)&mctx);CHKERRQ(ierr);
  ierr = PetscFree(mctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ObjectiveHessianMult_UU(Mat H, Vec X, Vec Y)
{
  PetscScalar       Xw[2][2];
  PetscScalar       *hXw[2] = {&Xw[0][0],&Xw[1][0]};
  PetscScalar       Yw,YTw,*y,one = 1.0;
  const PetscScalar *x;
  ObjHctx           *ctx;
  PetscErrorCode    ierr;
  int               n = 2,deg = 1,keep = 2,tagobj = 1;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(H,&ctx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);

  /* we could as well use the "hess_vec" API of ADOL-C */
  fos_forward(tagobj,1,n,keep,ctx->U0,(PetscScalar*)x,&Yw,&YTw);
  hos_reverse(tagobj,1,n,deg,&one,hXw);
  y[0] = Xw[0][1];
  y[1] = Xw[1][1];

  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveHessian_UU(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  const PetscScalar *u;
  PetscInt          n;
  ObjHctx           *mctx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(U,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = MatShellGetContext(H,&mctx);CHKERRQ(ierr);
  if (!mctx) {
    ierr = PetscNew(&mctx);CHKERRQ(ierr);
    ierr = MatShellSetContext(H,mctx);CHKERRQ(ierr);
    ierr = MatShellSetOperation(H,MATOP_DESTROY,(void(*)(void))ObjectiveHessianDestroy_UU);CHKERRQ(ierr);
    ierr = MatShellSetOperation(H,MATOP_MULT,(void(*)(void))ObjectiveHessianMult_UU);CHKERRQ(ierr);
    ierr = MatSetUp(H);CHKERRQ(ierr);
  }
  /* save linearization point */
  ierr = PetscArraycpy(mctx->U0,u,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* trace the nonlinear ode functions */
static PetscErrorCode trace_ifunc(void)
{
  double  uin[4] = {0.0,0.0,0.0,0.0},fout[2];
  double  p = 0;
  adouble f[2];
  adouble u[2];
  adouble udot[2];

  /* full case : 2 dependent variables, 4 independent variables */
  trace_on(2);
  /* independent variables */
  u[0] <<= uin[0];
  u[1] <<= uin[1];
  udot[0] <<= uin[2];
  udot[1] <<= uin[3];
  /* dependent variable */
  f[0] = ( u[0] * udot[0] + u[1] * udot[1]);
  f[1] = (-u[1] * udot[0] + u[0] * udot[1]) + u[0]*u[0] + u[1]*u[1];
  f[0] >>= fout[0];
  f[1] >>= fout[1];
  trace_off();

  /* partial u : 2 dependent variables, 2 independent variables , udot values enter as parameters */
  trace_on(3);
  /* independent variables */
  u[0] <<= uin[0];
  u[1] <<= uin[1];
  /* parameters */
  locint pudot0 = mkparam_idx(p);
  locint pudot1 = mkparam_idx(p);
  /* dependent variables */
  f[0] = ( u[0] * getparam(pudot0) + u[1] * getparam(pudot1));
  f[1] = (-u[1] * getparam(pudot0) + u[0] * getparam(pudot1)) + u[0]*u[0] + u[1]*u[1];
  f[0] >>= fout[0];
  f[1] >>= fout[1];
  trace_off();

  /* partial udot : 2 dependent variables, 2 independent variables , u values enter as parameters */
  trace_on(4);
  /* independent variables */
  udot[0] <<= uin[0];
  udot[1] <<= uin[1];
  /* parameters */
  locint pu0 = mkparam_idx(p);
  locint pu1 = mkparam_idx(p);
  locint pu2 = mkparam_idx(p);
  /* dependent variables */
  f[0] = ( getparam(pu0) * udot[0] + getparam(pu1) * udot[1]);
  f[1] = (-getparam(pu1) * udot[0] + getparam(pu0) * udot[1]) + getparam(pu2);
  f[0] >>= fout[0];
  f[1] >>= fout[1];
  trace_off();

  return 0;
}

static PetscErrorCode FormIFunction(TS ts,PetscReal time,Vec U,Vec Udot,Vec F,void* ctx)
{
  const PetscScalar *u;
  const PetscScalar *udot;
  PetscScalar       *f,uf[4];
  PetscErrorCode    ierr;
  int               n = 2,keep = 0,tagifunc = 2;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);

  /* call ADOL-C functionality to sample the residual equations */
  uf[0] = u[0];
  uf[1] = u[1];
  uf[2] = udot[0];
  uf[3] = udot[1];
  zos_forward(tagifunc,n,n*2,keep,uf,f);

  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscBool   full;
  PetscScalar U0[4];
  PetscReal   s;
} IJacobianctx;

/* we lazily attach contexts to the shell matrices
   each will own its context */
static PetscErrorCode IJacobianDestroy(Mat A)
{
  IJacobianctx   *mctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&mctx);CHKERRQ(ierr);
  ierr = PetscFree(mctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobianCopy(Mat A, Mat B, MatStructure str)
{
  IJacobianctx   *actx,*bctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,&actx);CHKERRQ(ierr);
  ierr = MatShellGetContext(B,&bctx);CHKERRQ(ierr);
  if (bctx && bctx != actx) {
    PetscErrorCode (*f)(Mat);

    ierr = MatShellGetOperation(A,MATOP_DESTROY,(void(**)(void))&f);CHKERRQ(ierr);
    if (f != IJacobianDestroy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot copy");
  } else { /* matrix has been obtained from MatDuplicate_Shell -> set new pointer and possibly copy over */
    ierr = PetscNew(&bctx);CHKERRQ(ierr);
    ierr = MatShellSetContext(B,bctx);CHKERRQ(ierr);
  }
  if (actx) {
    bctx->full  = actx->full;
    bctx->U0[0] = actx->U0[0];
    bctx->U0[1] = actx->U0[1];
    bctx->U0[2] = actx->U0[2];
    bctx->U0[3] = actx->U0[3];
    bctx->s     = actx->s;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobianMult_Private(Mat A, Vec X, Vec Y, PetscBool trans)
{
  PetscScalar       *y,*u,*udot;
  const PetscScalar *x;
  PetscScalar       params[3],yu[2] = {0.0,0.0},yudot[2] = {0.0,0.0};
  IJacobianctx      *ctx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);
  u    = ctx->U0;
  udot = ctx->U0 + 2;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  if (trans) { /* for the transposed action, we need to split the contributions */
    int m = 2, n = 2, keep = 1, tagifunc_u = 3, tagifunc_udot = 4;

    /* F_U^T*x */
    if (ctx->full || ctx->s == 0.0) {
      params[0] = udot[0];
      params[1] = udot[1];
      set_param_vec(tagifunc_u,2,params);
      zos_forward(tagifunc_u,m,n,keep,u,yu/*dummy*/);
      fos_reverse(tagifunc_u,m,n,(PetscScalar*)x,yu);
    }

    /* s*F_Udot^T*x */
    if (ctx->s != 0.0) {
      params[0] = u[0];
      params[1] = u[1];
      params[2] = u[0]*u[0] + u[1]*u[1];
      set_param_vec(tagifunc_udot,3,params);
      zos_forward(tagifunc_udot,m,n,keep,u,yudot/*dummy*/);
      fos_reverse(tagifunc_udot,m,n,(PetscScalar*)x,yudot);
    }

    /* assemble output */
    y[0] = yu[0] + ctx->s * yudot[0];
    y[1] = yu[1] + ctx->s * yudot[1];
  } else { /* forward action */
    if (ctx->full) { /* s*J_Udot + J_U */
      PetscScalar full[4];
      int         m = 2,n = 4,keep = 0,tagifunc = 2;

      full[0] = x[0];
      full[1] = x[1];
      full[2] = ctx->s*x[0];
      full[3] = ctx->s*x[1];
      fos_forward(tagifunc,m,n,keep,ctx->U0,full,yu/*dummy*/,y);
    } else if (ctx->s == 0.0) { /* J_U */
      int m = 2,n = 2,keep = 0,tagifunc_u = 3;

      params[0] = udot[0];
      params[1] = udot[1];
      set_param_vec(tagifunc_u,2,params);
      fos_forward(tagifunc_u,m,n,keep,u,(PetscScalar*)x,yu/*dummy*/,y);
    } else if (ctx->s == 1.0) { /* J_Udot */
      int m = 2,n = 2,keep = 0,tagifunc_udot = 4;

      params[0] = u[0];
      params[1] = u[1];
      params[2] = u[0]*u[0] + u[1]*u[1];
      set_param_vec(tagifunc_udot,3,params);
      fos_forward(tagifunc_udot,m,n,keep,udot,(PetscScalar*)x,yu/*dummy*/,y);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This should not happen");
  }

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobianMult(Mat A, Vec X, Vec Y)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = IJacobianMult_Private(A,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobianMultTranspose(Mat A, Vec X, Vec Y)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = IJacobianMult_Private(A,X,Y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIJacobian(TS ts,PetscReal time,Vec U,Vec Udot,PetscReal s,Mat A,Mat P,void* ctx)
{
  IJacobianctx      *mctx;
  const PetscScalar *u;
  const PetscScalar *udot;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,&mctx);CHKERRQ(ierr);
  if (!mctx) { /* lazy setup */
    ierr = PetscNew(&mctx);CHKERRQ(ierr);
    ierr = MatShellSetContext(A,mctx);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_DESTROY,(void(*)(void))IJacobianDestroy);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)(void))IJacobianMult);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void(*)(void))IJacobianMultTranspose);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_COPY,(void(*)(void))IJacobianCopy);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
  }

  /* flag full of split jacobians */
  mctx->full = (s != 0.0 && s != PETSC_MIN_REAL) ? PETSC_TRUE : PETSC_FALSE;
  mctx->s    = s == PETSC_MIN_REAL ? 1.0 : s;

  /* save linearization point */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  mctx->U0[0] = u[0];
  mctx->U0[1] = u[1];
  mctx->U0[2] = udot[0];
  mctx->U0[3] = udot[1];

  /* flag assembly of Jacobians */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Explicitly compute jacobians when a preconditioner is requested */
  if (P && A != P) {
    PetscInt    st,i[2];
    PetscScalar v[2][2],params[3];
    PetscScalar *J[2] = {&v[0][0],&v[1][0]};
    int         n = 2,tagifunc_u = 3,tagifunc_udot = 4;

    ierr = MatZeroEntries(P);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(P,&st,NULL);CHKERRQ(ierr);
    i[0] = st;
    i[1] = st+1;

    /* F_U */
    if (mctx->full || mctx->s == 0.0) {
      params[0] = udot[0];
      params[1] = udot[1];
      set_param_vec(tagifunc_u,2,params);
      jacobian(tagifunc_u,n,n,u,J);
      ierr = MatSetValues(P,2,i,2,i,(PetscScalar*)v,ADD_VALUES);CHKERRQ(ierr);
    }

    /* s*F_Udot */
    if (mctx->s != 0.0) {
      params[0] = u[0];
      params[1] = u[1];
      params[2] = u[0]*u[0] + u[1]*u[1];
      set_param_vec(tagifunc_udot,3,params);
      jacobian(tagifunc_udot,n,n,udot,J);
      v[0][0] = mctx->s*v[0][0];
      v[0][1] = mctx->s*v[0][1];
      v[1][0] = mctx->s*v[1][0];
      v[1][1] = mctx->s*v[1][1];
      ierr = MatSetValues(P,2,i,2,i,(PetscScalar*)v,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatShift(P,PETSC_SMALL);CHKERRQ(ierr); /* prevent from null-pivots */
  }

  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* special callback used to sample J_U and J_Udot separately */
static PetscErrorCode TSComputeSplitJacobians_AD(TS ts,PetscReal time,Vec U,Vec Udot,Mat J_U,Mat pJ_U,Mat J_Udot,Mat pJ_Udot)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* FormIJacobian has special cases for these */
  ierr = FormIJacobian(ts,time,U,Udot,0.0,J_U,pJ_U,NULL);CHKERRQ(ierr);
  ierr = FormIJacobian(ts,time,U,Udot,PETSC_MIN_REAL,J_Udot,pJ_Udot,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_UU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *x,*l,*u,*udot;
  PetscScalar       *y,yd1[2],yd2[2];
  PetscScalar       params[2];
  PetscScalar       v[2][2];
  PetscScalar       *Z[2] = {&v[0][0],&v[1][0]};
  int               m = 2,n = 2,keep = 2,tagifunc_u = 3;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);

  /* we use the split trace on u only */
  params[0] = udot[0];
  params[1] = udot[1];
  set_param_vec(tagifunc_u,2,params);
  fos_forward(tagifunc_u,m,n,keep,u,(PetscScalar*)x,yd1/*dummy*/,yd2/*dummy*/);
  hos_reverse(tagifunc_u,m,n,1,(PetscScalar*)l,Z);
  y[0] = v[0][1];
  y[1] = v[1][1];

  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_UUdot(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *x,*l,*u,*udot;
  PetscScalar       *y,yd1[2],yd2[2];
  PetscScalar       xfull[4],ufull[4];
  PetscScalar       v[4][2];
  PetscScalar       *Z[4] = {&v[0][0],&v[1][0],&v[2][0],&v[3][0]};
  int               m = 2,n = 2,keep = 2,tagifunc = 2;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);

  /* Compute from full trace
     we could probably use the externally differentiated functionality */
  ufull[0] = u[0];
  ufull[1] = u[1];
  ufull[2] = udot[0];
  ufull[3] = udot[1];
  xfull[0] = 0.0;
  xfull[1] = 0.0;
  xfull[2] = x[0];
  xfull[3] = x[1];
  fos_forward(tagifunc,m,2*n,keep,ufull,xfull,yd1/*dummy*/,yd2/*dummy*/);
  hos_reverse(tagifunc,m,2*n,1,(PetscScalar*)l,Z);
  y[0] = v[0][1];
  y[1] = v[1][1];

  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_UdotU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *x,*l,*u,*udot;
  PetscScalar       *y,yd1[2],yd2[2];
  PetscScalar       xfull[4],ufull[4];
  PetscScalar       v[4][2];
  PetscScalar       *Z[4] = {&v[0][0],&v[1][0],&v[2][0],&v[3][0]};
  int               m = 2,n = 2,keep = 2,tagifunc = 2;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);

  /* Compute from full trace
     we could probably use the externally differentiated functionality */
  ufull[0] = u[0];
  ufull[1] = u[1];
  ufull[2] = udot[0];
  ufull[3] = udot[1];
  xfull[0] = x[0];
  xfull[1] = x[1];
  xfull[2] = 0.0;
  xfull[3] = 0.0;
  fos_forward(tagifunc,m,2*n,keep,ufull,xfull,yd1/*dummy*/,yd2/*dummy*/);
  hos_reverse(tagifunc,m,2*n,1,(PetscScalar*)l,Z);
  y[0] = v[2][1];
  y[1] = v[3][1];

  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This term is zero, we can also avoid setting the callback in TSSetHessianDAE */
static PetscErrorCode EvalHessianDAE_UdotUdot(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *x,*l,*u,*udot;
  PetscScalar       *y,yd1[2],yd2[2];
  PetscScalar       params[3];
  PetscScalar       v[2][2];
  PetscScalar       *Z[2] = {&v[0][0],&v[1][0]};
  int               m = 2,n = 2,keep = 2,tagifunc_udot = 4;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);

  /* we use the split trace on udot only */
  params[0] = u[0];
  params[1] = u[1];
  params[2] = u[0]*u[0] + u[1]*u[1];
  set_param_vec(tagifunc_udot,3,params);
  fos_forward(tagifunc_udot,m,n,keep,u,(PetscScalar*)x,yd1/*dummy*/,yd2/*dummy*/);
  hos_reverse(tagifunc_udot,m,n,1,(PetscScalar*)l,Z);
  y[0] = v[0][1];
  y[1] = v[1][1];

  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Setup ODE solver from parameters */
typedef struct {
  TSTrajectory tj;
} AppCtx;

static PetscErrorCode MyTSSetUpFromDesign(TS ts, Vec u0, Vec M, void *ctx)
{
  AppCtx         *tctx = (AppCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(M,u0);CHKERRQ(ierr);
  if (tctx->tj) {
    TSAdapt adapt;

    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType(adapt,TSADAPTHISTORY);CHKERRQ(ierr);
    ierr = TSAdaptHistorySetTrajectory(adapt,tctx->tj,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Callbacks to be used within TAO (TODO: make the part of the library?) */
typedef struct {
  TS        ts;
  PetscReal t0,dt,tf;
} OptCtx;

static PetscErrorCode FormFunctionHessian(Tao tao, Vec M, Mat H, Mat Hpre, void *ctx)
{
  OptCtx         *octx = (OptCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSComputeHessian(octx->ts,octx->t0,octx->dt,octx->tf,NULL,M,H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormFunctionGradient(Tao tao,Vec M,PetscReal *obj,Vec G,void *ctx)
{
  OptCtx         *octx = (OptCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSComputeObjectiveAndGradient(octx->ts,octx->t0,octx->dt,octx->tf,NULL,M,G,obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormGradient(Tao tao,Vec M,Vec G,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = FormFunctionGradient(tao,M,NULL,G,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormFunction(Tao tao,Vec M,PetscReal *obj,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = FormFunctionGradient(tao,M,obj,NULL,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char* argv[])
{
  AppCtx         app;
  OptCtx         opt;
  TS             ts;
  TSAdapt        adapt;
  Mat            J,Jp,H;
  Vec            U,Uobj,M,G;
  PetscScalar    *g;
  PetscReal      t0,tf,dt,obj,objnull;
  PetscInt       st;
  PetscBool      paper = PETSC_TRUE, testtao = PETSC_FALSE, testtlm = PETSC_FALSE, testtaylor = PETSC_FALSE;
  PetscBool      testhistory = PETSC_FALSE, flg, check_dae = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscOptInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Command line options */
  t0   = 0.0;
  tf   = 1.57;
  dt   = 1.e-3;
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"","");
  ierr = PetscOptionsReal("-t0","Initial time","",t0,&t0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","Final time","",tf,&tf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","Initial time step","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check","Check Hessian DAE terms","",check_dae,&check_dae,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tao","Solve the optimization problem","",testtao,&testtao,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tlm","Test Tangent Linear Model to compute the gradient","",testtlm,&testtlm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_taylor","Run Taylor test","",testtaylor,&testtaylor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_history","Run objective using the initially generated history","",testhistory,&testhistory,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-paper","Use objective from the paper","",paper,&paper,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();

  /* trace functions used by ADOL-C once */
  trace_obj(paper);
  trace_ifunc();

  /* state vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,2,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);

  /* ODE solver */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSComputeSplitJacobians_C",TSComputeSplitJacobians_AD);CHKERRQ(ierr);
  ierr = TSSetTolerances(ts,1.e-8,NULL,1.e-8,NULL);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(J,MATSHELL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&Jp);CHKERRQ(ierr);
  ierr = MatSetSizes(Jp,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Jp,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Jp,2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Jp,2,NULL,0,NULL);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FormIFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,Jp,FormIJacobian,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&Jp);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTBASIC);CHKERRQ(ierr);
  if (testhistory) {
    TSTrajectory tj;

    ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
    ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
    ierr = TSTrajectorySetType(tj,ts,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&opt.t0);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&opt.dt);CHKERRQ(ierr);
  /* override command line */
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /* initial condition */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U,&st,NULL);CHKERRQ(ierr);
  ierr = VecSetValue(U,0+st,0.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(U,1+st,1.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(U);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(U);CHKERRQ(ierr);

  /* design vector (initial conditions) */
  ierr = VecDuplicate(U,&M);CHKERRQ(ierr);
  ierr = VecCopy(U,M);CHKERRQ(ierr);

  /* sample nonlinear model */
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  /* update optimization contexts */
  ierr = TSGetTime(ts,&opt.tf);CHKERRQ(ierr);
  opt.ts = ts;
  ierr = TSGetTrajectory(ts,&app.tj);CHKERRQ(ierr);

  /* store final time and state */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Uobj);CHKERRQ(ierr);
  ierr = VecCopy(U,Uobj);CHKERRQ(ierr);
  ierr = EvalObjective(Uobj,M,opt.tf,&objnull,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nFinal state\n");CHKERRQ(ierr);
  ierr = VecView(Uobj,NULL);CHKERRQ(ierr);

  /* sensitivity callbacks */
  ierr = MatCreate(PETSC_COMM_WORLD,&H);CHKERRQ(ierr);
  ierr = MatSetSizes(H,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(H,MATSHELL);CHKERRQ(ierr);
  /* objective function as final state sampling */
  ierr = TSAddObjective(ts,opt.tf,EvalObjective,EvalObjectiveGradient_U,NULL,
                                H,EvalObjectiveHessian_UU,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);
  /* set callback to setup model solver after a design vector changes */
  ierr = TSSetSetUpFromDesign(ts,MyTSSetUpFromDesign,&app);CHKERRQ(ierr);
  /* we compute sensitivity wrt initial condition */
  ierr = TSSetGradientIC(ts,NULL,NULL,TSEvalGradientICDefault,NULL);CHKERRQ(ierr);
  /* callbacks for Hessian terms of residual function */
  ierr = TSSetHessianDAE(ts,EvalHessianDAE_UU,   EvalHessianDAE_UUdot,   NULL,
                            EvalHessianDAE_UdotU,EvalHessianDAE_UdotUdot,NULL,
                            NULL,                NULL,                   NULL,
                            NULL);CHKERRQ(ierr);

  /* check Hessian terms */
  if (check_dae) {
    PetscRandom r;
    Vec         L,Udot;

    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&L);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&Udot);CHKERRQ(ierr);
    ierr = VecSetRandom(U,r);CHKERRQ(ierr);
    ierr = VecSetRandom(Udot,r);CHKERRQ(ierr);
    ierr = VecSetRandom(L,r);CHKERRQ(ierr);
    ierr = TSCheckGradientDAE(ts,0.0,U,Udot,M);CHKERRQ(ierr);
    ierr = TSCheckHessianDAE(ts,0.0,U,Udot,M,L);CHKERRQ(ierr);
    ierr = VecDestroy(&Udot);CHKERRQ(ierr);
    ierr = VecDestroy(&L);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  }

  /* null test */
  ierr = FormFunction(NULL,M,&obj,&opt);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Null test: %g (%g)\n",(double)(PetscAbsReal(obj-objnull)),(double)obj);CHKERRQ(ierr);

  /* check gradient */
  ierr = VecDuplicate(M,&G);CHKERRQ(ierr);
  ierr = FormGradient(NULL,M,G,&opt);CHKERRQ(ierr);
  ierr = VecGetArrayRead(G,(const PetscScalar**)&g);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD, "[%d] Gradient: %1.14e %1.14e\n",PetscGlobalRank,g[0],g[1]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(G,(const PetscScalar**)&g);CHKERRQ(ierr);

  /* check tangent linear model */
  if (testtlm) {
    Mat Phi,Phie,PhiT,PhiTe,TLMe;
    Vec T,G2;

    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = VecCopy(M,U);CHKERRQ(ierr);
    ierr = TSCreatePropagatorMat(ts,opt.t0,opt.dt,opt.tf,U,M,NULL,&Phi);CHKERRQ(ierr);
    ierr = MatCreateTranspose(Phi,&PhiT);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTLM matrices (Phi, Phi^T and Phi-(Phi^T)^T)\n");
    ierr = MatComputeOperator(Phi,NULL,&Phie);CHKERRQ(ierr);
    ierr = MatView(Phie,NULL);CHKERRQ(ierr);

    ierr = MatComputeOperator(PhiT,NULL,&PhiTe);CHKERRQ(ierr);
    ierr = MatView(PhiTe,NULL);CHKERRQ(ierr);

    ierr = MatTranspose(PhiTe,MAT_INITIAL_MATRIX,&TLMe);CHKERRQ(ierr);
    ierr = MatAXPY(TLMe,-1.0,Phie,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(TLMe,NULL);CHKERRQ(ierr);

    ierr = VecDuplicate(G,&T);CHKERRQ(ierr);
    ierr = VecDuplicate(G,&G2);CHKERRQ(ierr);
    ierr = EvalObjectiveGradient_U(Uobj,M,opt.tf,T,NULL);CHKERRQ(ierr);
    ierr = MatMultTranspose(Phie,T,G2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGradient via TLM (explicit fwd via multtrans)\n");
    ierr = VecView(G2,NULL);CHKERRQ(ierr);
    ierr = VecAXPY(G2,-1.0,G);CHKERRQ(ierr);
    ierr = VecView(G2,NULL);CHKERRQ(ierr);

    ierr = MatMult(PhiTe,T,G2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGradient via TLM (explicit adj via mult)\n");
    ierr = VecView(G2,NULL);CHKERRQ(ierr);
    ierr = VecAXPY(G2,-1.0,G);CHKERRQ(ierr);
    ierr = VecView(G2,NULL);CHKERRQ(ierr);

    ierr = VecDestroy(&T);CHKERRQ(ierr);
    ierr = VecDestroy(&G2);CHKERRQ(ierr);
    ierr = MatDestroy(&TLMe);CHKERRQ(ierr);
    ierr = MatDestroy(&Phie);CHKERRQ(ierr);
    ierr = MatDestroy(&Phi);CHKERRQ(ierr);
    ierr = MatDestroy(&PhiTe);CHKERRQ(ierr);
    ierr = MatDestroy(&PhiT);CHKERRQ(ierr);
  }

  /* check gradient and hessian with Tao */
  if (testtao) {
    Tao tao;
    Vec X;

    ierr = VecDuplicate(M,&X);CHKERRQ(ierr);
    ierr = VecSetRandom(X,NULL);CHKERRQ(ierr);
    ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
    ierr = TaoSetSolution(tao,X);CHKERRQ(ierr);
    ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);
    ierr = TaoSetObjective(tao,FormFunction,&opt);CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,&opt);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&H);CHKERRQ(ierr);
    ierr = TaoSetHessian(tao,H,H,FormFunctionHessian,&opt);CHKERRQ(ierr);
    ierr = TaoComputeGradient(tao,X,G);CHKERRQ(ierr);
    ierr = TaoComputeHessian(tao,X,H,H);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
    ierr = TaoDestroy(&tao);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
  }

  /* view hessian */
  ierr = PetscOptionsHasName(NULL,NULL,"-tshessian_view",&flg);CHKERRQ(ierr);
  if (flg) {
    Mat He,HeT;

    ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&H);CHKERRQ(ierr);
    ierr = TSComputeHessian(ts,opt.t0,opt.dt,opt.tf,NULL,M,H);CHKERRQ(ierr);
    ierr = MatComputeOperator(H,MATAIJ,&He);CHKERRQ(ierr);
    ierr = MatConvert(He,MATDENSE,MAT_INPLACE_MATRIX,&He);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)He,"H");CHKERRQ(ierr);
    ierr = MatViewFromOptions(He,NULL,"-tshessian_view");CHKERRQ(ierr);
    ierr = MatTranspose(He,MAT_INITIAL_MATRIX,&HeT);CHKERRQ(ierr);
    ierr = MatAXPY(HeT,-1.0,He,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatViewFromOptions(HeT,NULL,"-tshessian_view");CHKERRQ(ierr);
    ierr = MatDestroy(&HeT);CHKERRQ(ierr);
    ierr = MatDestroy(&He);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
  }

  /* run taylor test */
  if (testtaylor) {
    ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);
    ierr = TSTaylorTest(ts,opt.t0,opt.dt,opt.tf,NULL,M,NULL);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&G);CHKERRQ(ierr);
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = VecDestroy(&Uobj);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscOptFinalize();
  return ierr;
}

/*TEST
  build:
    requires: double !complex adolc

  test:
    suffix: paper_discrete
    filter: sed -e "s/1 MPI process/1 MPI process/g"
    args: -paper -ts_trajectory_type memory -check -test_tao -test_tlm -tsgradient_adjoint_discrete -tao_test_gradient -tlm_discrete -adjoint_tlm_discrete -tshessian_foadjoint_discrete -tshessian_tlm_discrete -tshessian_soadjoint_discrete -tshessian_view

  test:
    suffix: hessian_discrete
    filter: sed -e "s/1 MPI process/1 MPI process/g"
    args: -paper 0 -ts_trajectory_type memory -check -test_tao -test_tlm -tsgradient_adjoint_discrete -tao_test_hessian -tshessian_foadjoint_discrete -tshessian_tlm_discrete -tshessian_soadjoint_discrete -tlm_discrete -adjoint_tlm_discrete -ts_rtol 1.e-4 -ts_atol 1.e-4 -test_history {{0 1}separate output} -tshessian_view

TEST*/


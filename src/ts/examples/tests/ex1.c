static const char help[] = "Demonstrates the use of PETScOpt.";
/*
  Computes the gradient of

    Obj(u,m) = \int^{TF}_{T0} f(u,m) dt + other_terms_tested

  where u obeys the ODEs:

      (mm^2+1)*udot = b*u^p
               u(0) = a^2

  The integrand of the objective function is either f(u) = ||u||^2 or f(u) = Sum(u).

  The design variables are:
    - with the implicit (IFunction/IJacobian) interface: m = [a,b,p,mm].
    - with the explicit (RHSFunction/RHSJacobian) interface: m = [a,b,p].

  It also shows how to compute gradients with point-functionals of the type f(u,T,m), with T in (T0,TF].

  For parallel runs, we replicate the same solution on each process.
*/
#include <petscopt.h>

typedef struct {
  PetscBool isnorm;
} UserObjective;

/* returns f(u) -> ||u||^2  or Sum(u), depending on the objective function selected */
static PetscErrorCode EvalObjective(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  UserObjective  *user = (UserObjective*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (user->isnorm) {
    ierr  = VecNorm(U,NORM_2,val);CHKERRQ(ierr);
    *val *= *val;
  } else {
    PetscScalar sval;
    ierr = VecSum(U,&sval);CHKERRQ(ierr);
    *val = PetscRealPart(sval);
  }
  PetscFunctionReturn(0);
}

/* returns \partial_u f(u) ->  2*u or 1, depending on the objective function selected */
static PetscErrorCode EvalObjectiveGradient_U(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  UserObjective  *user = (UserObjective*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (user->isnorm) {
    ierr = VecCopy(U,grad);CHKERRQ(ierr);
    ierr = VecScale(grad,2.0);CHKERRQ(ierr);
  } else {
    ierr = VecSet(grad,1.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* returns \partial_uu f(u) -> 2 or 0, depending on the objective function selected */
static PetscErrorCode EvalObjectiveHessian_UU(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  UserObjective  *user = (UserObjective*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  if (user->isnorm) {
    ierr = MatShift(H,2.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* returns \partial_m f(u) = 0, the functional does not depend on the parameters  */
static PetscErrorCode EvalObjectiveGradient_M(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(grad,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* testing functionals */
static PetscReal store_Event = 0.0;
static PetscBool general_fixed = PETSC_FALSE;

static PetscErrorCode EvalObjective_Const(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = 1.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjective_Event(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr  = VecNorm(U,NORM_2,val);CHKERRQ(ierr);
  *val *= *val;
  store_Event += *val;
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveGradient_U_Event(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(U,grad);CHKERRQ(ierr);
  ierr = VecScale(grad,2.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveHessian_UU_Event(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  ierr = MatShift(H,2.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* f(u,m,t) = -||u||^2 + 3*||m||^2 + 0.004*||u||^2*||m||^2 */
static PetscErrorCode EvalObjective_Gen(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      v1,v2;

  PetscFunctionBeginUser;
  ierr = VecNorm(M,NORM_2,&v1);CHKERRQ(ierr);
  ierr = VecNorm(U,NORM_2,&v2);CHKERRQ(ierr);
  *val = 3.*v1*v1 - v2*v2 + 0.004*v1*v1*v2*v2;
  if (general_fixed) store_Event += *val;
  PetscFunctionReturn(0);
}

/* f_u(u,m,t) = 2*(-1 + 0.004*||m||^2)*u */
static PetscErrorCode EvalObjective_U_Gen(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscReal      v1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecNorm(M,NORM_2,&v1);CHKERRQ(ierr);
  ierr = VecCopy(U,grad);CHKERRQ(ierr);
  ierr = VecScale(grad,2.0*(-1.0 + 0.004*v1*v1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* f_m(u,m,t) = 2(3 + 0.004*||u||^2*)*m */
static PetscErrorCode EvalObjective_M_Gen(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscReal      v1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecNorm(U,NORM_2,&v1);CHKERRQ(ierr);
  ierr = VecCopy(M,grad);CHKERRQ(ierr);
  ierr = VecScale(grad,2.0*(3.0+0.004*v1*v1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*               | 2*(-1 + 0.004*||m||^2)       0            |
   f_uu(u,m,t) = |                                           |
                 |          0         2*(-1 + 0.004*||m||^2) | */
static PetscErrorCode EvalObjective_UU_Gen(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  PetscReal      v1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecNorm(M,NORM_2,&v1);CHKERRQ(ierr);
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  ierr = MatShift(H,2.0*(-1.0 + 0.004*v1*v1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* H_ij = 0.016*u_i*m_j -> we test a MatShell implementation */
typedef struct {
  Vec U;
  Vec M;
} Gen_UM_ctx;

static PetscErrorCode MatDestroy_Gen_UM(Mat H)
{
  Gen_UM_ctx     *mctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(H,(void**)&mctx);CHKERRQ(ierr);
  ierr = VecDestroy(&mctx->M);CHKERRQ(ierr);
  ierr = VecDestroy(&mctx->U);CHKERRQ(ierr);
  ierr = PetscFree(mctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Gen_UM(Mat H, Vec X, Vec Y)
{
  Gen_UM_ctx     *mctx;
  PetscScalar    v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(H,(void**)&mctx);CHKERRQ(ierr);
  ierr = VecDot(X,mctx->M,&v);CHKERRQ(ierr);
  ierr = VecCopy(mctx->U,Y);CHKERRQ(ierr);
  ierr = VecScale(Y,v*0.016);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Gen_UM(Mat H, Vec X, Vec Y)
{
  Gen_UM_ctx     *mctx;
  PetscScalar    v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(H,(void**)&mctx);CHKERRQ(ierr);
  ierr = VecDot(X,mctx->U,&v);CHKERRQ(ierr);
  ierr = VecCopy(mctx->M,Y);CHKERRQ(ierr);
  ierr = VecScale(Y,v*0.016);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjective_UM_Gen(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  Gen_UM_ctx     *mctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(H,(void**)&mctx);CHKERRQ(ierr);
  ierr = VecCopy(U,mctx->U);CHKERRQ(ierr);
  ierr = VecCopy(M,mctx->M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*               | 2*(3 + 0.004*||u||^2)       0            |
   f_mm(u,m,t) = |                                          |
                 |         0          2*(3 + 0.004*||u||^2) | */
static PetscErrorCode EvalObjective_MM_Gen(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  PetscBool      flg1,flg2; /* I don't know why the operations pop back in the matrix */
  PetscReal      v1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatHasOperation(H,MATOP_MULT_ADD,&flg1);CHKERRQ(ierr);
  ierr = MatHasOperation(H,MATOP_MULT_TRANSPOSE_ADD,&flg2);CHKERRQ(ierr);
  ierr = VecNorm(U,NORM_2,&v1);CHKERRQ(ierr);
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  /* MatShift for some reason puts back the original operations in place */
  ierr = MatShift(H,2.0*(3.0+0.004*v1*v1));CHKERRQ(ierr);
  if (!flg1) { ierr = MatSetOperation(H,MATOP_MULT_ADD,(void(*)(void))NULL);CHKERRQ(ierr); }
  if (!flg2) { ierr = MatSetOperation(H,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))NULL);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* DAE callbacks */

typedef struct {
  PetscScalar a;
  PetscScalar b;
  PetscReal   p;
  PetscScalar mm;
  VecScatter  Msct;
  Vec         M;
  Mat         F_MM; /* Hessian workspace */
  Mat         F_UM; /* Hessian workspace */
  Mat         F_UU; /* Hessian workspace */
} UserDAE;

/* returns \partial_m F(U,Udot,t;M) for a fixed design M, where F(U,Udot,t;M) is the parameter dependent ODE in implicit form */
static PetscErrorCode EvalGradientDAE(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Mat J, void *ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscInt       rst,ren,r,ls;
  PetscScalar    *arr, *arrd;
  PetscReal      p;
  PetscScalar    b, mm = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* this scatter here is just for debugging purposes
     we could have used the User ctx without the need for the scatters */
  ierr = VecScatterBegin(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->M,&ls);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  b    = arr[1];
  p    = PetscRealPart(arr[2]);
  if (ls > 3) mm = arr[3];
  ierr = VecRestoreArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,(const PetscScalar**)&arrd);CHKERRQ(ierr);
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(J,&rst,&ren);CHKERRQ(ierr);
  for (r = rst; r < ren; r++) {
    /* F_a  : 0 */
    /* F_b  : -x^p */
    /* F_p  : -b*x^p*log(x) */
    /* F_mm : 2*mm*xdot */
    PetscInt    c = 1;
    PetscScalar lx, v = -PetscPowScalarReal(arr[r-rst],p);

    ierr = MatSetValues(J,1,&r,1,&c,&v,INSERT_VALUES);CHKERRQ(ierr);
    c    = 2;
    lx   = PetscLogScalar(arr[r-rst]);
    if (PetscIsInfOrNanScalar(lx)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Error in log! Invalid combination of parameters");
    v   *= b*lx;
    ierr = MatSetValues(J,1,&r,1,&c,&v,INSERT_VALUES);CHKERRQ(ierr);
    if (ls > 3) { /* IFunction interface */
      c    = 3;
      v    = 2.0*mm*arrd[r-rst];
      ierr = MatSetValues(J,1,&r,1,&c,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,(const PetscScalar**)&arrd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_N)*F_UU*X, where L and X are vectors of size N, I_N is the identity matrix of size N, \otimes is the Kronecker product,
   and F_UU is the N^2 x N matrix with entries

          | F^1_UU |
   F_UU = |   ...  |, F^k has dimension NxN, with {F^k_UU}_ij = \frac{\partial^2 F_k}{\partial u_j \partial u_i}, where F_k is the k-th component of the DAE
          | F^N_UU |

   with u_i the i-th state variable, and N the number of state variables.

   The output should be computed as: Y = (\sum_k L_k*F^k_UU)*X, with L_k the k-th entry of the adjoint variable L.

   In this example, {F^k_UU}_ij != 0 only when i == j == k, with nonzero value -p*b*(p-1)*u_k^-2.
*/
static PetscErrorCode EvalHessianDAE_UU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *arr;
  PetscReal      p;
  PetscScalar    b,v,l;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecScatterBegin(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  b    = arr[1];
  p    = PetscRealPart(arr[2]);
  ierr = VecRestoreArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  if (!user->F_UU) { /* create the workspace matrix once */
    ierr = MatCreate(PETSC_COMM_WORLD,&user->F_UU);CHKERRQ(ierr);
    ierr = MatSetSizes(user->F_UU,1,1,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetUp(user->F_UU);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(user->F_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->F_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  l    = arr[0];
  ierr = VecRestoreArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  if (p != 2.0) v = -l*p*b*(p-1.0)*PetscPowScalarReal(arr[0],p - 2.0);
  else          v = -l*p*b*(p-1.0);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = MatZeroEntries(user->F_UU);CHKERRQ(ierr);
  ierr = MatShift(user->F_UU,v);CHKERRQ(ierr);
  ierr = MatMult(user->F_UU,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_N)*F_UM*X, where L and X are vectors of size N and P respectively, I_N is the identity matrix of size N, \otimes is the Kronecker product,
   and F_UM is the N^2 x P matrix with entries

          | F^1_UM |
   F_UM = |   ...  |, F^k has dimension NxP, with {F^k_UM}_ij = \frac{\partial^2 F_k}{\partial m_j \partial u_i}, where F_k is the k-th component of the DAE
          | F^N_UM |

   with m_j the j-th design variable and u_i the i-th state variable, with N the number of state variables and P the number of design variables.

   The output should be computed as:  Y = (\sum_k L_k*F^k_UM)*X, with L_k the k-th entry of the adjoint variable L.

   In this example, {F^k_UM}_ij != 0 only when i == k, with the k-th row of F^k_UM given by [0, p*u_k^(p-1), -p*u_k^(p-1)*(1+p*log(u_k)), 0].
*/
static PetscErrorCode EvalHessianDAE_UM(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *arr;
  PetscReal      p;
  PetscScalar    b,v[3],tmp,x,l;
  PetscInt       rst,id[3] = {0, 1, 2};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecScatterBegin(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  b    = arr[1];
  p    = PetscRealPart(arr[2]);
  ierr = VecRestoreArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  if (!user->F_UM) { /* create the workspace matrix once */
    PetscInt dsize;

    ierr = VecGetSize(M,&dsize);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&user->F_UM);CHKERRQ(ierr);
    ierr = MatSetSizes(user->F_UM,1,PETSC_DECIDE,PETSC_DECIDE,dsize);CHKERRQ(ierr);
    ierr = MatSetUp(user->F_UM);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(user->F_UM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->F_UM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  l    = arr[0];
  ierr = VecRestoreArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  x    = arr[0];
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  if (p != 1.0) tmp = PetscPowScalarReal(x,p - 1.0);
  else          tmp = 1.0;
  v[0] = 0.0;
  v[1] = -l*p*tmp;
  v[2] = -l*b*tmp*(1.0+p*PetscLogReal(x));
  ierr = MatZeroEntries(user->F_UM);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->F_UM,&rst,NULL);CHKERRQ(ierr);
  ierr = MatSetValues(user->F_UM,1,&rst,3,id,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(user->F_UM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->F_UM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMult(user->F_UM,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_P)*F_MU*X, where L and X are vectors of size N, I_P is the identity matrix of size P, \otimes is the Kronecker product,
   and F_MU is the N*P x N matrix with entries

          | F^1_MU |
   F_MU = |   ...  |, F^k has dimension PxN, with {F^k_MU}_ij = \frac{\partial^2 F_k}{\partial u_j \partial m_i}, where F_k is the k-th component of the DAE
          | F^N_MU |

   with u_j the j-th state variable and m_i the i-th design variable, with N the number of state variables and P the number of design variables.

   The output should be computed as:  Y = (\sum_k L_k*F^k_MU)*X = (\sum_k L_k*(F^k_MU)^T)*X, with L_k the k-th entry of the adjoint variable L.
*/
static PetscErrorCode EvalHessianDAE_MU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *arr;
  PetscReal      p;
  PetscScalar    b,v[3],tmp,x,l;
  PetscInt       rst,id[3] = {0, 1, 2};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecScatterBegin(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  b    = arr[1];
  p    = PetscRealPart(arr[2]);
  ierr = VecRestoreArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  if (!user->F_UM) { /* create the workspace matrix once */
    PetscInt dsize;

    ierr = VecGetSize(M,&dsize);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&user->F_UM);CHKERRQ(ierr);
    ierr = MatSetSizes(user->F_UM,1,PETSC_DECIDE,PETSC_DECIDE,dsize);CHKERRQ(ierr);
    ierr = MatSetUp(user->F_UM);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(user->F_UM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->F_UM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  l    = arr[0];
  ierr = VecRestoreArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  x    = arr[0];
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  if (p != 1.0) tmp = PetscPowScalarReal(x,p - 1.0);
  else          tmp = 1.0;
  v[0] = 0.0;
  v[1] = -l*p*tmp;
  v[2] = -l*b*tmp*(1.0+p*PetscLogReal(x));
  ierr = MatZeroEntries(user->F_UM);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->F_UM,&rst,NULL);CHKERRQ(ierr);
  ierr = MatSetValues(user->F_UM,1,&rst,3,id,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(user->F_UM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->F_UM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->F_UM,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_N)*F_UdotM*X, where L and X are vectors of size N and P respectively, I_N is the identity matrix of size N, \otimes is the Kronecker product,
   and F_UdotM is the N^2 x P matrix with entries

             | F^1_UdotM |
   F_UdotM = |    ...    |, F^k has dimension NxP, with {F^k_UdotM}_ij = \frac{\partial^2 F_k}{\partial m_j \partial udot_i}, where F_k is the k-th component of the DAE
             | F^N_UdotM |

   with m_j the j-th design variable and udot_i the time derivative of the i-th state variable, with N the number of state variables and P the number of design variables.

   The output should be computed as:  Y = (\sum_k L_k*F^k_UdotM)*X, with L_k the k-th entry of the adjoint variable L.

   In this example, {F^k_UdotM}_ij != 0 only when i == k, with the i-th row of F^k_UdotM given by [0, 0, 0, 2*m].
*/
static PetscErrorCode EvalHessianDAE_UdotM(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *arr;
  PetscScalar    x,l;
  PetscInt       dsize,rst;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  ierr = VecGetSize(M,&dsize);CHKERRQ(ierr);
  if (dsize < 4) { /* with the RHSFunction interface, this term is zero */
    PetscFunctionReturn(0);
  }
  ierr = VecGetArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  l    = arr[0];
  ierr = VecRestoreArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecScatterBegin(user->Msct,X,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(user->Msct,X,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  x    = arr[3];
  ierr = VecRestoreArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Y,&rst,NULL);CHKERRQ(ierr);
  ierr = VecSetValue(Y,rst,l*2.0*user->mm*x,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_P)*F_MUdot*X, where L and X are vectors of size N, I_P is the identity matrix of size P, \otimes is the Kronecker product,
   and F_MUdot is the N*P x N matrix with entries

          | F^1_MUdot |
   F_MU = |     ...   |, F^k has dimension PxN, with {F^k_MUdot}_ij = \frac{\partial^2 F_k}{\partial udot_j \partial m_i}, where F_k is the k-th component of the DAE
          | F^N_MUdot |

   with udot_j the time derivative of the j-th state variable and m_i the i-th design variable, with N the number of state variables and P the number of design variables.

   The output should be computed as:  Y = (\sum_k L_k*F^k_MUdot)*X = (\sum_k L_k*(F^k_MUdot)^T)*X, with L_k the k-th entry of the adjoint variable L.
*/
static PetscErrorCode EvalHessianDAE_MUdot(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *arr;
  PetscScalar    x,l;
  PetscInt       dsize;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  ierr = VecGetSize(M,&dsize);CHKERRQ(ierr);
  if (dsize < 4) { /* with the RHSFunction interface, this term is zero */
    PetscFunctionReturn(0);
  }
  ierr = VecGetArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  l    = arr[0];
  ierr = VecRestoreArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,(const PetscScalar**)&arr);CHKERRQ(ierr);
  x    = arr[0];
  ierr = VecRestoreArrayRead(X,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecSetValue(Y,3,l*2.0*user->mm*x,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_P)*F_MM*X, where L and X are vectors of size N and P respectively, I_P is the identity matrix of size P, \otimes is the Kronecker product,
   and F_MM is the N*P x P matrix with entries

          | F^1_MM |
   F_MM = |   ...  |, F^k has dimension PxP, with {F^k_MM}_ij = \frac{\partial^2 F_k}{\partial m_j \partial m_i}, where F_k is the k-th component of the DAE
          | F^N_MM |

   with m_j the j-th design variable, N the number of state variables, and P the number of design variables.

   The output should be computed as:  Y = (\sum_k L_k*F^k_MM)*X, with L_k the k-th entry of the adjoint variable L.

                             | 0       0                   0           0        |
   In this example, F^k_MM = | 0       0           -u_k^p*log(u_k)     0        |.
                             | 0 -u_k^p*log(u_k) -b*log^2(u_k)*u_k^p   0        |
                             | 0       0                   0           2*udot_k |
*/
static PetscErrorCode EvalHessianDAE_MM(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *arr;
  PetscReal      p;
  PetscScalar    b,v[9],tmp,x,l,ll;
  PetscInt       dsize, id[3] = {0, 1, 2};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecScatterBegin(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(user->Msct,M,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  b    = arr[1];
  p    = PetscRealPart(arr[2]);
  ierr = VecRestoreArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  if (!user->F_MM) { /* create the workspace matrix once */
    PetscInt dsize;

    ierr = VecGetSize(M,&dsize);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&user->F_MM);CHKERRQ(ierr);
    ierr = MatSetSizes(user->F_MM,PETSC_DECIDE,PETSC_DECIDE,dsize,dsize);CHKERRQ(ierr);
    ierr = MatSetUp(user->F_MM);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(user->F_MM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->F_MM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  l    = arr[0];
  ierr = VecRestoreArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  x    = arr[0];
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  if (p != 0.0) tmp = PetscPowScalarReal(x,p);
  else          tmp = 1.0;
  ll = PetscLogScalar(x);
  if (PetscIsInfOrNanScalar(ll)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Error in log! Invalid combination of parameters");
  v[0*3+0] = 0.0;
  v[0*3+1] = 0.0;
  v[0*3+2] = 0.0;
  v[1*3+0] = 0.0;
  v[1*3+1] = 0.0;
  v[1*3+2] = -l*tmp*ll;
  v[2*3+0] = 0.0;
  v[2*3+1] = -l*tmp*ll;
  v[2*3+2] = -l*b*ll*ll*tmp;
  ierr = MatZeroEntries(user->F_MM);CHKERRQ(ierr);
  ierr = MatSetValues(user->F_MM,3,id,3,id,v,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->M,&dsize);CHKERRQ(ierr);
  if (dsize > 3) {
    PetscInt r = 3;

    ierr = VecGetArrayRead(Udot,(const PetscScalar**)&arr);CHKERRQ(ierr);
    x    = l*2.0*arr[0];
    ierr = VecRestoreArrayRead(Udot,(const PetscScalar**)&arr);CHKERRQ(ierr);
    ierr = MatSetValues(user->F_MM,1,&r,1,&r,&x,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(user->F_MM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->F_MM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMult(user->F_MM,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns \partial_u0 G and \partial_m G, with G the initial conditions in implicit form, i.e. G(u0,m) = 0 */
static PetscErrorCode EvalGradientIC(TS ts, PetscReal t0, Vec u0, Vec M, Mat G_u0, Mat G_m, void *ctx)
{
  PetscInt       rst,ren,r;
  PetscScalar    *a = (PetscScalar*)(ctx);
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (G_u0) {
    ierr = MatZeroEntries(G_u0);CHKERRQ(ierr);
    ierr = MatShift(G_u0,1.0);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(G_m);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(G_m,&rst,&ren);CHKERRQ(ierr);
  for (r = rst; r < ren; r++) {
    PetscInt    c = 0;
    PetscScalar v = -2.0*(*a);
    ierr = MatSetValues(G_m,1,&r,1,&c,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(G_m,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_m,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_P)*G_MM*X, where L and X are vectors of size N and P respectively, I_P is the identity matrix of size P, \otimes is the Kronecker product,
   and G_MM is the N*P x P matrix with entries

          | G^1_MM |
   G_MM = |   ...  |, G^k has dimension PxP, with {G^k_MM}_ij = \frac{\partial^2 G_k}{\partial m_j \partial m_i}, where G_k is the k-th component of initial condition residual in implicit form, i.e. G(u0,m) = 0
          | G^N_MM |

   with m_j the j-th design variable, N the number of state variables, and P the number of design variables.

   The output should be computed as:  Y = (\sum_k L_k*G^k_MM)*X, with L_k the k-th entry of the adjoint variable L.

                             | -2 0 0 0 |
   In this example, G^k_MM = |  0 0 0 0 |.
                             |  0 0 0 0 |
                             |  0 0 0 0 |
*/
static PetscErrorCode EvalHessianIC_MM(TS ts, PetscReal time, Vec U, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *arr;
  PetscScalar    x,l;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(user->Msct,X,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(user->Msct,X,user->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  x    = arr[0];
  ierr = VecRestoreArrayRead(user->M,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  l    = arr[0];
  ierr = VecRestoreArrayRead(L,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = VecSetValue(Y,0,-l*2.0*x,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction(TS ts,PetscReal time, Vec U, Vec Udot, Vec F,void* ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *aU,*aUdot,*aF;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&aF);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,(const PetscScalar**)&aUdot);CHKERRQ(ierr);
  aF[0] = (user->mm*user->mm+1.0)*aUdot[0] - user->b*PetscPowScalarReal(aU[0],user->p);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,(const PetscScalar**)&aUdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIJacobian(TS ts,PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat P, void* ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscInt       i;
  PetscScalar    *aU,v,shfm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  shfm = shift*(user->mm*user->mm+1.0);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatShift(A,shfm);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  if (user->p != 1.0) v = -user->b*user->p*PetscPowScalarReal(aU[0],user->p - 1.0);
  else v = -user->b*user->p;
  ierr = MatGetOwnershipRange(A,&i,NULL);CHKERRQ(ierr);
  ierr = MatSetValues(A,1,&i,1,&i,&v,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (P && P != A) { /* just to make it different */
    v   *= 1.01;
    ierr = MatZeroEntries(P);CHKERRQ(ierr);
    ierr = MatShift(P,shfm*1.01);CHKERRQ(ierr);
    ierr = MatSetValues(P,1,&i,1,&i,&v,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction_mix(TS ts,PetscReal time, Vec U, Vec Udot, Vec F,void* ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *aUdot,*aF;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&aF);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,(const PetscScalar**)&aUdot);CHKERRQ(ierr);
  aF[0] = (user->mm*user->mm+1.0)*aUdot[0];
  ierr = VecRestoreArrayRead(Udot,(const PetscScalar**)&aUdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIJacobian_mix(TS ts,PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat P, void* ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    shfm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  shfm = shift*(user->mm*user->mm+1.0);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(A,shfm);CHKERRQ(ierr);
  if (P && P != A) { /* just to make it different */
    ierr = MatZeroEntries(P);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatShift(P,shfm*1.01);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal time, Vec U, Vec F,void* ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscScalar    *aU,*aF;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&aF);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  aF[0] = user->b*PetscPowScalarReal(aU[0],user->p);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian(TS ts,PetscReal time, Vec U, Mat A, Mat P, void* ctx)
{
  UserDAE        *user = (UserDAE*)ctx;
  PetscInt       i;
  PetscScalar    *aU,v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  if (user->p != 1.0) v = user->b*user->p*PetscPowScalarReal(aU[0],user->p - 1.0); /* x^0 gives error on my Mac */
  else v = user->b*user->p;
  ierr = MatGetOwnershipRange(A,&i,NULL);CHKERRQ(ierr);
  ierr = MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (P && P != A) { /* just to make it different */
    v   *= 1.01;
    ierr = MatZeroEntries(P);CHKERRQ(ierr);
    ierr = MatSetValues(P,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MyTSSetUpFromDesign(TS ts, Vec x0, Vec M, void *ctx)
{
  UserDAE           *userdae = (UserDAE*)ctx;
  const PetscScalar *a;
  PetscErrorCode    ierr;
  PetscInt          ls;
  Mat               J,pJ;
  TSRHSJacobian     rhsjac;
  TSProblemType     ptype;

  PetscFunctionBeginUser;
  ierr = VecScatterBegin(userdae->Msct,M,userdae->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(userdae->Msct,M,userdae->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetLocalSize(userdae->M,&ls);CHKERRQ(ierr);
  ierr = VecGetArrayRead(userdae->M,&a);CHKERRQ(ierr);
  userdae->a = a[0];
  userdae->b = a[1];
  userdae->p = PetscRealPart(a[2]);
  if (ls > 3) userdae->mm = a[3];
  ierr = VecRestoreArrayRead(userdae->M,&a);CHKERRQ(ierr);
  ierr = VecSet(x0,userdae->a*userdae->a);CHKERRQ(ierr);
  ierr = TSGetRHSJacobian(ts,&J,&pJ,&rhsjac,NULL);CHKERRQ(ierr);
  if (rhsjac == TSComputeRHSJacobianConstant) { /* need to update the constant RHS jacobian */
    if (userdae->p != 1.0) {
      ierr = TSSetRHSJacobian(ts,J,pJ,FormRHSJacobian,userdae);CHKERRQ(ierr);
    } else {
      ierr = FormRHSJacobian(ts,0.0,x0,J,pJ,userdae);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(ts,J,pJ,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    }
  }
  ierr = TSGetProblemType(ts,&ptype);CHKERRQ(ierr);
  if (ptype == TS_LINEAR && userdae->p != 1.0) {
    ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestTSEventFunction(TS ts, PetscReal t, Vec U, PetscScalar fvalue[], void *ctx)
{
  PetscReal *tt = (PetscReal *)ctx;

  PetscFunctionBeginUser;
  fvalue[0] = t - *tt;
  PetscFunctionReturn(0);
}

static PetscErrorCode TestTSPostEvent(TS adjts, PetscInt nevents, PetscInt event_list[], PetscReal t, Vec U, PetscBool forwardsolve, void* ctx)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<nevents;i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Termination event %D detected at time %g\n",event_list[i],(double)t);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DummyPostStep(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

int main(int argc, char* argv[])
{
  TS             ts;
  Mat            J,pJ,G_M,F_M,G_X;
  Mat            checkTLM,Phi,PhiExpl,PhiT,PhiTExpl;
  Mat            H_MM,H_UU,H;
  Vec            U,M,Mgrad,sol,tlmsol;
  UserObjective  userobj;
  UserDAE        userdae;
  TSProblemType  problemtype;
  PetscScalar    a, b, one = 1.0, m;
  PetscReal      p, t0 = 0.0, tf = 2.0, dt = 0.1, rtf;
  PetscReal      obj,objtest,err,normPhi;
  PetscMPIInt    np;
  PetscBool      testpjac = PETSC_TRUE;
  PetscBool      testifunc = PETSC_FALSE;
  PetscBool      testmix = PETSC_FALSE;
  PetscBool      testrhsjacconst = PETSC_FALSE;
  PetscBool      testnullgradM = PETSC_FALSE;
  PetscBool      testnulljacIC = PETSC_FALSE;
  PetscBool      testevent = PETSC_FALSE;
  PetscBool      testeventfinal = PETSC_FALSE;
  PetscBool      testeventconst = PETSC_FALSE;
  PetscBool      testgeneral_fixed = PETSC_FALSE;
  PetscBool      testgeneral_final = PETSC_FALSE;
  PetscBool      testgeneral = PETSC_FALSE;
  PetscBool      testgeneral_final_double = PETSC_FALSE;
  PetscBool      testgeneral_double = PETSC_FALSE;
  PetscBool      testfwdevent = PETSC_FALSE;
  PetscBool      testps = PETSC_TRUE;
  PetscBool      userobjective = PETSC_TRUE;
  PetscBool      usefd = PETSC_FALSE, usetaylor = PETSC_FALSE;
  PetscBool      testm = PETSC_TRUE;
  PetscBool      testremove_multadd = PETSC_FALSE;
  PetscBool      flg;
  PetscReal      dx = PETSC_SMALL;
  PetscReal      testfwdeventctx;
  PetscInt       dsize;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&np);CHKERRQ(ierr);

  /* Command line options */
  t0             = 0.0;
  tf             = 1.0;
  a              = 0.5;
  b              = 0.7;
  p              = 1.0;
  m              = 0.0;
  userobj.isnorm = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"PDE-constrained options","");
  ierr = PetscOptionsScalar("-a","Initial condition","",a,&a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-b","Grow rate","",b,&b,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-m","Mass coefficient","",m,&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-p","Nonlinearity","",p,&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t0","Initial time","",t0,&t0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","Final time","",tf,&tf,NULL);CHKERRQ(ierr);
  dt   = (tf-t0)/512.0;
  ierr = PetscOptionsReal("-dt","Initial time step","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_user_objective","Tests user objective","",userobjective,&userobjective,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_objective_norm","Test f(u) = ||u||^2","",userobj.isnorm,&userobj.isnorm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_pjac","Test with Pmat != Amat","",testpjac,&testpjac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_mix","Test mixing IFunction and RHSFunction","",testmix,&testmix,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_ifunc","Test with IFunction interface","",testifunc,&testifunc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_rhsjacconst","Test with TSComputeRHSJacobianConstant","",testrhsjacconst,&testrhsjacconst,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_nullgrad_M","Test with NULL M gradient","",testnullgradM,&testnullgradM,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_nulljac_IC","Test with NULL G_X jacobian","",testnulljacIC,&testnulljacIC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_event_constant","Test constant functional at given time in between the simulation","",testeventconst,&testeventconst,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_event_func","Test functional at given time in between the simulation","",testevent,&testevent,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_event_final","Test functional at final time of the simulation","",testeventfinal,&testeventfinal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_general_final","Test general functional","",testgeneral_final,&testgeneral_final,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_general_final_double","Test general functional (twice)","",testgeneral_final_double,&testgeneral_final_double,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_general_fixed","Test general functional","",testgeneral_fixed,&testgeneral_fixed,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_general","Test general functional","",testgeneral,&testgeneral,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_general_double","Test general functional (twice)","",testgeneral_double,&testgeneral_double,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_forward_event","Test event handling in forward solve","",testfwdevent,&testfwdevent,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_poststep","Test with TS having a poststep method","",testps,&testps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_mass","Test with parameter dependent mass matrix","",testm,&testm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_fd","Use finite differencing to test gradient evaluation","",usefd,&usefd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_taylor","Use Taylor remainders to check gradient evaluation","",usetaylor,&usetaylor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_remove_multadd","Test with removal of MultAdd operations","",testremove_multadd,&testremove_multadd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dx","dx for FD","",dx,&dx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  problemtype = TS_LINEAR;
  if (!testifunc && !testmix) m = 0.0;

  userdae.a  = a;
  userdae.b  = b;
  userdae.p  = p;
  userdae.mm = m;

  dsize = 3;
  if (testm && (testifunc || testmix)) {
    dsize++;
  }
  if (p != 1.0) {
    problemtype = TS_NONLINEAR;
    testrhsjacconst = PETSC_FALSE;
  }
  if (testmix) testifunc = PETSC_TRUE;

  /* state vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);

  /* design vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&M);CHKERRQ(ierr);
  ierr = VecSetSizes(M,PETSC_DECIDE,dsize);CHKERRQ(ierr);
  ierr = VecSetType(M,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetValue(M,0,a,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(M,1,b,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(M,2,p,INSERT_VALUES);CHKERRQ(ierr);
  if (dsize > 3) {
    ierr = VecSetValue(M,3,m,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
  ierr = VecDuplicate(M,&Mgrad);CHKERRQ(ierr);

  /* ---------- Create TS solver  ---------- */

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,problemtype);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&sol);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,sol);CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);

  /* we test different combinations of IFunction/RHSFunction on the same ODE */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,1,1,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (testpjac) {
    ierr = MatDuplicate(J,MAT_DO_NOT_COPY_VALUES,&pJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    pJ   = J;
  }
  if (!testifunc) {
    ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&userdae);CHKERRQ(ierr);
    if (testrhsjacconst) {
      ierr = FormRHSJacobian(ts,t0,U,J,pJ,&userdae);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(ts,J,pJ,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(ts,J,pJ,FormRHSJacobian,&userdae);CHKERRQ(ierr);
    }
  } else {
    if (testmix) {
      ierr = TSSetIFunction(ts,NULL,FormIFunction_mix,&userdae);CHKERRQ(ierr);
      ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&userdae);CHKERRQ(ierr);
      ierr = TSSetIJacobian(ts,NULL,NULL,FormIJacobian_mix,&userdae);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(ts,NULL,NULL,FormRHSJacobian,&userdae);CHKERRQ(ierr);
    } else {
      ierr = TSSetIFunction(ts,NULL,FormIFunction,&userdae);CHKERRQ(ierr);
      ierr = TSSetIJacobian(ts,J,pJ,FormIJacobian,&userdae);CHKERRQ(ierr);
    }
  }
  if (testfwdevent) {
    PetscInt  dir = PETSC_FALSE;
    PetscBool term = PETSC_TRUE;

    testfwdeventctx = t0 + 0.89*(tf-t0);
    ierr = TSSetEventHandler(ts,1,&dir,&term,TestTSEventFunction,TestTSPostEvent,&testfwdeventctx);CHKERRQ(ierr);
  }
  if (testps) {
    ierr = TSSetPostStep(ts,DummyPostStep);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* ---------- Customize TS objectives ---------- */

  /* Objective hessians (we create matrices with the correct layout here
     and we duplicate them for each objective that needs them) */
  ierr = MatCreate(PETSC_COMM_WORLD,&H_UU);CHKERRQ(ierr);
  ierr = MatSetSizes(H_UU,1,1,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(H_UU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&H_MM);CHKERRQ(ierr);
  ierr = MatSetSizes(H_MM,PETSC_DECIDE,PETSC_DECIDE,dsize,dsize);CHKERRQ(ierr);
  ierr = MatSetUp(H_MM);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H_MM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H_MM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Set cost functionals: many can be added, by simply calling TSAddObjective multiple times */
  if (userobjective) {
    Mat H;

    ierr = MatDuplicate(H_UU,MAT_DO_NOT_COPY_VALUES,&H);CHKERRQ(ierr);
    if (testnullgradM) {
      ierr = TSAddObjective(ts,PETSC_MIN_REAL,EvalObjective,EvalObjectiveGradient_U,NULL,
                            H,EvalObjectiveHessian_UU,NULL,NULL,NULL,NULL,&userobj);CHKERRQ(ierr);
    } else {
      ierr = TSAddObjective(ts,PETSC_MIN_REAL,EvalObjective,EvalObjectiveGradient_U,EvalObjectiveGradient_M,
                            H,EvalObjectiveHessian_UU,NULL,NULL,NULL,NULL,&userobj);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&H);CHKERRQ(ierr);
  }

  /* Cost functional at final time */
  if (testeventfinal) {
    Mat H;

    ierr = MatDuplicate(H_UU,MAT_DO_NOT_COPY_VALUES,&H);CHKERRQ(ierr);
    ierr = TSAddObjective(ts,tf,EvalObjective_Event,EvalObjectiveGradient_U_Event,NULL,
                          H,EvalObjectiveHessian_UU_Event,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
  }

  /* Cost functional in between the simulation */
  if (testevent) {
    Mat H;

    ierr = MatDuplicate(H_UU,MAT_DO_NOT_COPY_VALUES,&H);CHKERRQ(ierr);
    ierr = TSAddObjective(ts,t0 + 0.132*(tf-t0),EvalObjective_Event,EvalObjectiveGradient_U_Event,NULL,
                          H,EvalObjectiveHessian_UU_Event,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
  }

  /* Cost functional in between the simulation (constant) */
  if (testeventconst) {
    ierr = TSAddObjective(ts,t0 + 0.44*(tf-t0),EvalObjective_Const,NULL,NULL,
                          NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  }

  /* Cost functional with nonzero gradient wrt the parameters at a given time */
  if (testgeneral_fixed) {
    Mat        HUU,HMM,HUM;
    Gen_UM_ctx *ctx;

    general_fixed = PETSC_TRUE;
    ierr = MatDuplicate(H_UU,MAT_DO_NOT_COPY_VALUES,&HUU);CHKERRQ(ierr);
    ierr = MatDuplicate(H_MM,MAT_DO_NOT_COPY_VALUES,&HMM);CHKERRQ(ierr);
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&ctx->U);CHKERRQ(ierr);
    ierr = VecDuplicate(M,&ctx->M);CHKERRQ(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,1,PETSC_DECIDE,PETSC_DECIDE,dsize,ctx,&HUM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_MULT,(void(*)(void))MatMult_Gen_UM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Gen_UM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_DESTROY,(void(*)(void))MatDestroy_Gen_UM);CHKERRQ(ierr);
    ierr = TSAddObjective(ts,t0 + 0.77*(tf-t0),EvalObjective_Gen,EvalObjective_U_Gen,EvalObjective_M_Gen,
                          HUU,EvalObjective_UU_Gen,HUM,EvalObjective_UM_Gen,HMM,EvalObjective_MM_Gen,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&HUM);CHKERRQ(ierr);
    ierr = MatDestroy(&HMM);CHKERRQ(ierr);
    ierr = MatDestroy(&HUU);CHKERRQ(ierr);
  }

  /* Cost functional with nonzero gradient wrt the parameters at final time */
  if (testgeneral_final) {
    Mat        HUU,HMM,HUM;
    Gen_UM_ctx *ctx;

    general_fixed = PETSC_TRUE;
    ierr = MatDuplicate(H_UU,MAT_DO_NOT_COPY_VALUES,&HUU);CHKERRQ(ierr);
    ierr = MatDuplicate(H_MM,MAT_DO_NOT_COPY_VALUES,&HMM);CHKERRQ(ierr);
    if (testremove_multadd) {
      ierr = MatSetOperation(HUU,MATOP_MULT_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
      ierr = MatSetOperation(HUU,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
      ierr = MatSetOperation(HMM,MATOP_MULT_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
      ierr = MatSetOperation(HMM,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
    }
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&ctx->U);CHKERRQ(ierr);
    ierr = VecDuplicate(M,&ctx->M);CHKERRQ(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,1,PETSC_DECIDE,PETSC_DECIDE,dsize,ctx,&HUM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_MULT,(void(*)(void))MatMult_Gen_UM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Gen_UM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_DESTROY,(void(*)(void))MatDestroy_Gen_UM);CHKERRQ(ierr);
    if (testremove_multadd) {
      ierr = MatShellSetOperation(HUM,MATOP_MULT_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
      ierr = MatShellSetOperation(HUM,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
    }
    ierr = TSAddObjective(ts,tf,EvalObjective_Gen,EvalObjective_U_Gen,EvalObjective_M_Gen,
                          HUU,EvalObjective_UU_Gen,HUM,EvalObjective_UM_Gen,HMM,EvalObjective_MM_Gen,NULL);CHKERRQ(ierr);
    if (testgeneral_final_double) {
      ierr = TSAddObjective(ts,tf,EvalObjective_Gen,EvalObjective_U_Gen,EvalObjective_M_Gen,
                            HUU,EvalObjective_UU_Gen,HUM,EvalObjective_UM_Gen,HMM,EvalObjective_MM_Gen,NULL);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&HUM);CHKERRQ(ierr);
    ierr = MatDestroy(&HMM);CHKERRQ(ierr);
    ierr = MatDestroy(&HUU);CHKERRQ(ierr);
  }

  /* Cost functional with nonzero gradient wrt the parameters (integrand) */
  if (testgeneral) {
    Mat        HUU,HMM,HUM;
    Gen_UM_ctx *ctx;

    general_fixed = PETSC_TRUE;
    ierr = MatDuplicate(H_UU,MAT_DO_NOT_COPY_VALUES,&HUU);CHKERRQ(ierr);
    ierr = MatDuplicate(H_MM,MAT_DO_NOT_COPY_VALUES,&HMM);CHKERRQ(ierr);
    if (testremove_multadd) {
      ierr = MatSetOperation(HUU,MATOP_MULT_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
      ierr = MatSetOperation(HUU,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
      ierr = MatSetOperation(HMM,MATOP_MULT_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
      ierr = MatSetOperation(HMM,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
    }
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&ctx->U);CHKERRQ(ierr);
    ierr = VecDuplicate(M,&ctx->M);CHKERRQ(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,1,PETSC_DECIDE,PETSC_DECIDE,dsize,ctx,&HUM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_MULT,(void(*)(void))MatMult_Gen_UM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Gen_UM);CHKERRQ(ierr);
    ierr = MatShellSetOperation(HUM,MATOP_DESTROY,(void(*)(void))MatDestroy_Gen_UM);CHKERRQ(ierr);
    if (testremove_multadd) {
      ierr = MatShellSetOperation(HUM,MATOP_MULT_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
      ierr = MatShellSetOperation(HUM,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))NULL);CHKERRQ(ierr);
    }
    ierr = TSAddObjective(ts,PETSC_MIN_REAL,EvalObjective_Gen,EvalObjective_U_Gen,EvalObjective_M_Gen,
                          HUU,EvalObjective_UU_Gen,HUM,EvalObjective_UM_Gen,HMM,EvalObjective_MM_Gen,NULL);CHKERRQ(ierr);
    if (testgeneral_double) {
      ierr = TSAddObjective(ts,PETSC_MIN_REAL,EvalObjective_Gen,EvalObjective_U_Gen,EvalObjective_M_Gen,
                            HUU,EvalObjective_UU_Gen,HUM,EvalObjective_UM_Gen,HMM,EvalObjective_MM_Gen,NULL);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&HUM);CHKERRQ(ierr);
    ierr = MatDestroy(&HMM);CHKERRQ(ierr);
    ierr = MatDestroy(&HUU);CHKERRQ(ierr);
  }

  /* ---------- Customize TS parameter dependency ---------- */

  /* Context for gradient and hessian of the ODE */
  userdae.F_UU = NULL;
  userdae.F_UM = NULL;
  userdae.F_MM = NULL;
  ierr = VecScatterCreateToAll(M,&userdae.Msct,&userdae.M);CHKERRQ(ierr);

  /* Callback to setup when the design variables change */
  ierr = TSSetSetUpFromDesign(ts,MyTSSetUpFromDesign,&userdae);CHKERRQ(ierr);

  /* Set dependence of F(Udot,U,t;M) = 0 from the parameters */
  ierr = MatCreate(PETSC_COMM_WORLD,&F_M);CHKERRQ(ierr);
  ierr = MatSetSizes(F_M,1,PETSC_DECIDE,PETSC_DECIDE,dsize);CHKERRQ(ierr);
  ierr = MatSetUp(F_M);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,F_M,EvalGradientDAE,&userdae);CHKERRQ(ierr);
  ierr = TSSetHessianDAE(ts,EvalHessianDAE_UU,NULL,                EvalHessianDAE_UM,
                            NULL,             NULL,                EvalHessianDAE_UdotM,
                            EvalHessianDAE_MU,EvalHessianDAE_MUdot,EvalHessianDAE_MM,
                            &userdae);CHKERRQ(ierr);

  /* Set dependence of initial conditions (in implicit form G(U(0);M) = 0) from the parameters */
  ierr = MatCreate(PETSC_COMM_WORLD,&G_X);CHKERRQ(ierr);
  ierr = MatSetSizes(G_X,1,1,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(G_X);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G_X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&G_M);CHKERRQ(ierr);
  ierr = MatSetSizes(G_M,1,PETSC_DECIDE,PETSC_DECIDE,dsize);CHKERRQ(ierr);
  ierr = MatSetUp(G_M);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (testnulljacIC) {
    ierr = TSSetGradientIC(ts,NULL,G_M,EvalGradientIC,&userdae.a);CHKERRQ(ierr);
  } else {
    ierr = TSSetGradientIC(ts,G_X,G_M,EvalGradientIC,&userdae.a);CHKERRQ(ierr);
  }
  ierr = TSSetHessianIC(ts,NULL,NULL,NULL,EvalHessianIC_MM,&userdae);CHKERRQ(ierr);

  /* Test objective function evaluation */
  ierr = TSComputeObjectiveAndGradient(ts,t0,dt,tf,U,M,NULL,&obj);CHKERRQ(ierr);

  /* due to termination events, we take the real final time rtf here
     However, we keep testing the other calls with tf */
  ierr = TSGetTime(ts,&rtf);CHKERRQ(ierr);
  objtest = 0.0;
  if (userobjective) {
    PetscScalar bb = b/(m*m+1.0);
    PetscScalar aa = a*a;
    if (bb != 0.0) { /* we can compute the analytic solution for the objective function */
      if (p == 1.0) {
        if (userobj.isnorm) {
          objtest = np * PetscRealPart((aa * aa) / (2.0*bb) * (PetscExpScalar(2.0*(rtf-t0)*bb) - one));
        } else {
          objtest = np * PetscRealPart((aa / bb) * (PetscExpScalar((rtf-t0)*bb) - one));
        }
      } else {
        PetscReal   scale = userobj.isnorm ? 2.0 : 1.0;
        PetscScalar alpha = PetscPowScalarReal(aa,1.0-p);
        PetscScalar  beta = bb*(1.0-p), snp = np;
        PetscReal   gamma = scale/(1.0-p);
        objtest = PetscRealPart(snp / ( (gamma + 1.0) * beta )* ( PetscPowScalar(beta*(rtf-t0)+alpha,gamma+1.0) - PetscPowScalar(alpha,gamma+1.0) ));
      }
    }
  }
  if (testeventconst) objtest += 1.0;
  if (!testgeneral && b != 0.0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Objective function: time [%g,%g], val %g (should be %g)\n",(double)t0,(double)rtf,(double)obj,(double)(objtest+store_Event));CHKERRQ(ierr);
  } else { /* too lazy to compute an analytical solution */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Objective function: time [%g,%g], val %g\n",(double)t0,(double)rtf,(double)obj);CHKERRQ(ierr);
  }

  /* Test gradient evaluation */
  ierr = TSComputeObjectiveAndGradient(ts,t0,dt,tf,U,M,Mgrad,NULL);CHKERRQ(ierr);
  ierr = VecView(Mgrad,NULL);CHKERRQ(ierr);

  /* Test the gradient code by finite differencing the objective evaluation */
  if (usefd) {
    PetscInt    i;
    PetscScalar oa = a, ob = b, om = m;
    PetscReal   op = p;

    for (i=0; i<dsize; i++) {
      PetscReal objdx[2];
      PetscInt  j;

      for (j=0; j<2; j++) {
        PetscScalar param[4];
        PetscInt    k;

        param[0]   = oa;
        param[1]   = ob;
        param[2]   = op;
        param[3]   = om;
        param[i]   = (j == 0 ? param[i] + dx : param[i] - dx);
        userdae.a  = param[0];
        userdae.b  = param[1];
        userdae.p  = PetscRealPart(param[2]);
        userdae.mm = param[3];

        store_Event = 0.0;
        ierr = VecZeroEntries(M);CHKERRQ(ierr);
        for (k=0;k<dsize;k++) {
          ierr = VecSetValue(M,k,param[k],INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
        ierr = TSComputeObjectiveAndGradient(ts,t0,dt,tf,U,M,NULL,&objdx[j]);CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%D-th component of gradient should be (approximated) %g\n",i,(double)((objdx[0]-objdx[1])/(2.*dx)));CHKERRQ(ierr);
    }
  }

  /* Test tangent Linear Model */
  ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
  ierr = VecDuplicate(sol,&tlmsol);CHKERRQ(ierr);
  ierr = VecCopy(sol,tlmsol);CHKERRQ(ierr);
  ierr = VecSet(U,a*a);CHKERRQ(ierr); /* XXX IC */
  ierr = TSCreatePropagatorMat(ts,t0,dt,tf,U,M,NULL,&Phi);CHKERRQ(ierr);
  ierr = MatComputeExplicitOperator(Phi,&PhiExpl);CHKERRQ(ierr);
  ierr = MatNorm(PhiExpl,NORM_INFINITY,&normPhi);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)PhiExpl,"Phi");CHKERRQ(ierr);
  ierr = MatViewFromOptions(PhiExpl,NULL,"-phi_view");CHKERRQ(ierr);
  ierr = MatCreateTranspose(Phi,&PhiT);CHKERRQ(ierr);
  ierr = MatComputeExplicitOperator(PhiT,&PhiTExpl);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)PhiTExpl,"PhiT");CHKERRQ(ierr);
  ierr = MatViewFromOptions(PhiTExpl,NULL,"-phiT_view");CHKERRQ(ierr);
  ierr = MatTranspose(PhiTExpl,MAT_INITIAL_MATRIX,&checkTLM);CHKERRQ(ierr);
  ierr = MatAXPY(checkTLM,-1.0,PhiExpl,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatScale(checkTLM,1./normPhi);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)checkTLM,"||Phi - (Phi^T)^T||/||Phi||");CHKERRQ(ierr);
  ierr = MatNorm(checkTLM,NORM_INFINITY,&err);CHKERRQ(ierr);
  ierr = MatViewFromOptions(checkTLM,NULL,"-err_view");CHKERRQ(ierr);
  if (err > 0.01) { /* 1% difference */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Possible error with TLM: ||Phi|| is  %g (%g)\n",(double)normPhi,(double)err);CHKERRQ(ierr);
    ierr = MatView(PhiExpl,NULL);CHKERRQ(ierr);
    ierr = MatView(PhiTExpl,NULL);CHKERRQ(ierr);
    ierr = MatView(checkTLM,NULL);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&checkTLM);CHKERRQ(ierr);

  /* Test Hessian evaluation */
  ierr = MatCreate(PETSC_COMM_WORLD,&H);CHKERRQ(ierr);
  ierr = TSComputeHessian(ts,t0,dt,tf,U,M,H);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-tshessian_view",&flg);CHKERRQ(ierr);
  if (flg) {
    Mat He;

    ierr = MatComputeExplicitOperator(H,&He);CHKERRQ(ierr);
    ierr = MatViewFromOptions(He,NULL,"-tshessian_view");CHKERRQ(ierr);
    ierr = MatDestroy(&He);CHKERRQ(ierr);
  }

  /* Test gradient and Hessian using Taylor series */
  if (usetaylor) {
    ierr = PetscOptionsSetValue(NULL,"-taylor_ts_hessian","1");CHKERRQ(ierr);
    ierr = VecSetValue(M,0,a,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,1,b,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,2,p,INSERT_VALUES);CHKERRQ(ierr);
    if (dsize > 3) {
      ierr = VecSetValue(M,3,m,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
    ierr = TSTaylorTest(ts,t0,dt,tf,NULL,M,NULL);CHKERRQ(ierr);
  }

  ierr = VecScatterDestroy(&userdae.Msct);CHKERRQ(ierr);
  ierr = VecDestroy(&userdae.M);CHKERRQ(ierr);
  ierr = MatDestroy(&userdae.F_UU);CHKERRQ(ierr);
  ierr = MatDestroy(&userdae.F_MM);CHKERRQ(ierr);
  ierr = MatDestroy(&userdae.F_UM);CHKERRQ(ierr);
  ierr = VecDestroy(&tlmsol);CHKERRQ(ierr);
  /* XXX coverage */
  ierr = TSSetGradientIC(ts,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = VecDestroy(&Mgrad);CHKERRQ(ierr);
  ierr = MatDestroy(&PhiExpl);CHKERRQ(ierr);
  ierr = MatDestroy(&Phi);CHKERRQ(ierr);
  ierr = MatDestroy(&PhiTExpl);CHKERRQ(ierr);
  ierr = MatDestroy(&PhiT);CHKERRQ(ierr);
  ierr = MatDestroy(&pJ);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&H_UU);CHKERRQ(ierr);
  ierr = MatDestroy(&H_MM);CHKERRQ(ierr);
  ierr = MatDestroy(&G_M);CHKERRQ(ierr);
  ierr = MatDestroy(&G_X);CHKERRQ(ierr);
  ierr = MatDestroy(&F_M);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    requires: !single
    suffix: 1
    args: -t0 1.1 -tf 1.2 -ts_type rk -ts_adapt_type basic -ts_atol 1.e-9 -ts_rtol 1.e-9 -test_event_final -p 1.3 -ts_trajectory_type memory -use_taylor

  test:
    requires: !single
    suffix: 2
    args: -t0 1.6 -tf 1.7 -ts_type bdf -ts_adapt_type basic -ts_atol 1.e-9 -ts_rtol 1.e-9 -test_event_final -p 1.3 -use_taylor -ts_trajectory_type memory

  test:
    requires: !single
    suffix: 3
    args: -t0 1.6 -tf 1.7 -ts_type bdf -ts_adapt_type basic -ts_atol 1.e-9 -ts_rtol 1.e-9 -test_event_final -p 1.3 -test_ifunc -test_nulljac_IC -test_nullgrad_M -use_taylor -ts_trajectory_type memory

  test:
    requires: !single
    suffix: 4
    args: -t0 1.1 -tf 1.15 -ts_type rk -ts_adapt_type none -test_event_constant -test_rhsjacconst -ts_trajectory_reconstruction_order 3 -use_taylor -dt 0.001 -ts_trajectory_type memory

  test:
    requires: !single
    suffix: 5
    args: -t0 0.7 -tf 0.8 -ts_type cn -test_event_constant -p 0.8 -test_ifunc -ts_trajectory_reconstruction_order 2 -test_pjac 0 -tsgradient_adjoint_ts_adapt_type history -tshessian_tlm_ts_adapt_type history -tshessian_foadjoint_ts_adapt_type history -tshessian_soadjoint_ts_adapt_type {{none history}} -tshessian_tlm_userijacobian -use_taylor -m 0.1 -dt 0.005 -ts_trajectory_type memory -b 0.01

  test:
    requires: !single
    suffix: 6
    args: -t0 0.01 -tf 0.1 -b 0.3 -a 1.7 -p 1 -ts_type rk -dt 0.01 -ts_adapt_type none -test_event_func -tshessian_mffd -use_taylor -ts_trajectory_type memory -tsgradient_adjoint_ts_adapt_type history

  test:
    requires: !single
    suffix: 7
    args: -t0 0 -tf 0.02 -dt 0.001 -b 0.3 -a 1.7 -p 1 -ts_type rosw -test_ifunc -test_event_func -ts_adapt_type none -tshessian_mffd -use_taylor -ts_trajectory_type memory

  test:
    requires: !single
    suffix: 8
    args: -t0 0 -tf 0.07 -b -0.5 -a -1.1 -p 0.4 -ts_type bdf -test_mix -test_pjac 0 -test_event_constant -ts_adapt_type none -use_taylor -ts_trajectory_type memory -dt 0.005 -test_objective_norm -m 0.8

  test:
    requires: !single
    suffix: 9
    nsize: 2
    args: -t0 -0.3 -tf -0.28 -b 1.2 -a 2.1 -p 0.3 -ts_type rk -test_general_fixed -test_general_final -test_general -test_event_func -test_event_constant -test_event_final -ts_rtol 1.e-4 -ts_atol 1.e-4 -tshessian_mffd -use_taylor -ts_trajectory_type memory

  test:
    requires: !single
    suffix: 10
    nsize: 2
    args: -t0 -0.3 -tf -0.28 -b 1.2 -a 2.1 -p 0.3 -ts_type bdf -test_general_fixed -test_general_final -test_general -test_event_func -test_event_constant -test_event_final -ts_rtol 1.e-4 -ts_atol 1.e-4 -tshessian_mffd -use_taylor -ts_trajectory_type memory

  test:
    requires: !single
    suffix: 11
    args: -t0 0.41 -tf 0.44 -b 0.3 -a 1.25 -p 2.3 -ts_type rk -test_general_final -test_general -test_event_constant -test_event_final -ts_rtol 1.e-4 -ts_atol 1.e-4 -ts_trajectory_type memory -test_forward_event -use_taylor

  test:
    requires: !single
    timeoutfactor: 2
    suffix: 12
    args: -t0 0.41 -tf 0.44 -b 0.3 -a 1.25 -p 2.3 -ts_type bdf -test_general_final -test_general -test_event_constant -test_event_final -ts_rtol 1.e-4 -ts_atol 1.e-4 -ts_trajectory_type memory -test_forward_event -use_taylor -test_ifunc -m 1.1

  test:
    requires: !single
    timeoutfactor: 2
    suffix: 13
    args: -t0 0.41 -tf 0.44 -b 0.3 -a 1.25 -p 2.3 -ts_type bdf -test_general_final -test_general -ts_rtol 1.e-4 -ts_atol 1.e-4 -ts_trajectory_type memory -use_taylor -test_ifunc -m 1.1 -test_general_double {{0 1}separate output} -test_general_final_double {{0 1}separate output} -test_remove_multadd {{0 1}separate output}

TEST*/

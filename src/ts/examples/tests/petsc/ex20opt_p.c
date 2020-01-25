#define c21 2.0
#define rescale 10

static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation DAE equivalent
   Concepts: Optimization using adjoint sensitivity analysis
   Processors: 1
*/
/* ------------------------------------------------------------------------

  Notes:
  This code demonstrates how to solve a DAE-constrained optimization problem with TAO, TSAdjoint and TS.
  The nonlinear problem is written in a DAE equivalent form.
  The objective is to minimize the difference between observation and model prediction by finding an optimal value for parameter \mu.
  The gradient is computed with the discrete adjoint of an implicit theta method, see ex20adj.c for details.
  Alternatively, the gradient and the Hessian can be computed by directly solving the adjoint ode, see also ex16opt_p.c and ex16opt_ic.c for details.
  ------------------------------------------------------------------------- */

#include <petscopt.h>
#include <petsctao.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;

  /* Sensitivity analysis support */
  PetscReal ftime,x_ob[2];
  Mat       A;                       /* Jacobian matrix */
  Mat       Jacp;                    /* JacobianP matrix */
  Vec       x,lambda[2],mup[2];  /* adjoint variables */
};

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormFunction_AO(Tao,Vec,PetscReal*,void*);
PetscErrorCode FormGradient_AO(Tao,Vec,Vec,void*);
PetscErrorCode FormFunctionGradient_AO(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian_AO(Tao,Vec,Mat,Mat,void*);

/*
*  User-defined routines
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] - x[1];
  f[1] = c21*(xdot[0]-x[1]) + xdot[1] - user->mu*((1.0-x[0]*x[0])*x[1] - x[0]) ;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  J[0][0] = a;
  J[0][1] = -1.0;
  J[1][0] = c21*a + user->mu*(1.0 + 2.0*x[0]*x[1]);
  J[1][1] = -c21 + a - user->mu*(1.0-x[0]*x[0]);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (B && A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          row[] = {0,1},col[]={0};
  PetscScalar       J[2][1];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  J[0][0] = 0;
  J[1][0] = (1.-x[0]*x[0])*x[1]-x[0];
  ierr = MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedX;

  PetscFunctionBeginUser;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tfinal);CHKERRQ(ierr);

  while (user->next_output <= t && user->next_output <= tfinal) {
    ierr = VecDuplicate(X,&interpolatedX);CHKERRQ(ierr);
    ierr = TSInterpolate(ts,user->next_output,interpolatedX);CHKERRQ(ierr);
    ierr = VecGetArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",
                       user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),
                       (double)PetscRealPart(x[1]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = VecDestroy(&interpolatedX);CHKERRQ(ierr);
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS                 ts;            /* nonlinear solver */
  Vec                p;
  PetscBool          monitor = PETSC_FALSE;
  PetscScalar        *x_ptr;
  const PetscScalar  *y_ptr;
  PetscMPIInt        size;
  struct _n_User     user;
  PetscErrorCode     ierr;
  Tao                tao;
  KSP                ksp;
  PC                 pc;
  PetscBool          adjointode = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0;
  user.ftime       = 1.0;
  ierr             = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr             = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRQ(ierr);
  ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.A);CHKERRQ(ierr);
  ierr = MatSetUp(user.A);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.x,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jacp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user.A,user.A,IJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.03125);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&user.ftime);CHKERRQ(ierr);

  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);

  ierr         = VecGetArrayRead(user.x,&y_ptr);CHKERRQ(ierr);
  user.x_ob[0] = y_ptr[0];
  user.x_ob[1] = y_ptr[1];
  ierr         = VecRestoreArrayRead(user.x,&y_ptr);CHKERRQ(ierr);

  /* Create sensitivity variable */
  ierr = MatCreateVecs(user.A,&user.lambda[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.mup[0],NULL);CHKERRQ(ierr);

  /*
     Optimization starts
  */
  /* Set initial solution guess */
  ierr = MatCreateVecs(user.Jacp,&p,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(p,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.2;
  ierr = VecRestoreArray(p,&x_ptr);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,p);CHKERRQ(ierr);

  /* Set routine for function and gradient evaluation */
  ierr = PetscOptionsGetBool(NULL,NULL,"-adjointode",&adjointode,NULL);CHKERRQ(ierr);
  if (adjointode) { /* use the adjoint ode approach */
    Mat H;

    ierr = TaoSetObjectiveRoutine(tao,FormFunction_AO,(void *)&user);CHKERRQ(ierr);
    ierr = TaoSetGradientRoutine(tao,FormGradient_AO,(void *)&user);CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient_AO,(void *)&user);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&H);CHKERRQ(ierr);
    ierr = TaoSetHessianRoutine(tao,H,H,FormHessian_AO,(void *)&user);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
  } else {
    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user);CHKERRQ(ierr);
  }

  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  ierr = TaoSolve(tao); CHKERRQ(ierr);

  ierr = VecView(p,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&user.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mup[0]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx)
{
  User              user_ptr = (User)ctx;
  TS                ts;
  PetscScalar       *x_ptr;
  const PetscScalar *y_ptr;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(P,&y_ptr);CHKERRQ(ierr);
  user_ptr->mu = y_ptr[0];
  ierr = VecRestoreArrayRead(P,&y_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,user_ptr);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user_ptr->A,user_ptr->A,IJacobian,user_ptr);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(ts,user_ptr->Jacp,RHSJacobianP,user_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.03125);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user_ptr->x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(user_ptr->x,&x_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user_ptr->ftime);CHKERRQ(ierr);

  ierr = TSSolve(ts,user_ptr->x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user_ptr->x,&y_ptr);CHKERRQ(ierr);
  *f   = rescale*(y_ptr[0]-user_ptr->x_ob[0])*(y_ptr[0]-user_ptr->x_ob[0])+rescale*(y_ptr[1]-user_ptr->x_ob[1])*(y_ptr[1]-user_ptr->x_ob[1]);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Observed value y_ob=[%g; %g], ODE solution y=[%g;%g], Cost function f=%g\n",(double)user_ptr->x_ob[0],(double)user_ptr->x_ob[1],(double)y_ptr[0],(double)y_ptr[1],(double)(*f));CHKERRQ(ierr);
  /*   Redet initial conditions for the adjoint integration */
  ierr = VecGetArray(user_ptr->lambda[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = rescale*2.*(y_ptr[0]-user_ptr->x_ob[0]);
  x_ptr[1] = rescale*2.*(y_ptr[1]-user_ptr->x_ob[1]);
  ierr = VecRestoreArrayRead(user_ptr->x,&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user_ptr->lambda[0],&x_ptr);CHKERRQ(ierr);

  ierr = VecGetArray(user_ptr->mup[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(user_ptr->mup[0],&x_ptr);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,user_ptr->lambda,user_ptr->mup);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
  ierr = VecCopy(user_ptr->mup[0],G);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* callbacks for the adjoint ode approach */

static PetscErrorCode EvalObjective_AO(Vec U, Vec P, PetscReal time, PetscReal *val, void *ctx)
{
  const PetscScalar *x;
  User              user = (User)ctx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&x);CHKERRQ(ierr);
  *val = (x[0]-user->x_ob[0])*(x[0]-user->x_ob[0])+(x[1]-user->x_ob[1])*(x[1]-user->x_ob[1]);
  *val *= rescale;
  ierr = VecRestoreArrayRead(U,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalCostGradient_U_AO(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x;
  PetscScalar       *g;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&x);CHKERRQ(ierr);
  ierr = VecGetArray(grad,&g);CHKERRQ(ierr);
  g[0] = 2.*rescale*(x[0]-user->x_ob[0]);
  g[1] = 2.*rescale*(x[1]-user->x_ob[1]);
  ierr = VecRestoreArray(grad,&g);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* gradient term */
/* wrapper for RHSJacobianP: with this approach, the model ODE is assumed to be in the implicit form F(U,Udot,t) = 0 */
static PetscErrorCode EvalGradientDAE_P(TS ts, PetscReal t, Vec X, Vec Xdot, Vec P, Mat J, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = RHSJacobianP(ts,t,X,J,ctx);CHKERRQ(ierr);
  ierr = MatScale(J,-1.0);CHKERRQ(ierr); /* implicit form */
  PetscFunctionReturn(0);
}

/* hessian terms */
/* returns Y = (L^T \otimes I_N)*F_UU*X, where L and X are vectors of size N, I_N is the identity matrix of size N, \otimes is the Kronecker product,
   and F_UU is the N^2 x N matrix with entries

          | F^1_UU |
   F_UU = |   ...  |, F^k has dimension NxN, with {F^k_UU}_ij = \frac{\partial^2 F_k}{\partial u_j \partial u_i} where F_k is the k-th component of the DAE
          | F^N_UU |

   with u_i the i-th state variable, and N the number of state variables.

   The output should be computed as: Y = (\sum_k L_k*F^k_UU)*X, with L_k the k-th entry of the adjoint variable L.
   In this example

                           | 2*mu*u[1]  2*mu*u[0] |
      F^1_UU = 0, F^2_UU = |                      |
                           | 2*mu*u[0]      0     |

*/
static PetscErrorCode EvalHessianDAE_UU(TS ts, PetscReal t, Vec U, Vec Udot, Vec P, Vec L, Vec X, Vec Y, void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *y,mu;
  const PetscScalar *x,*l,*u,*mup;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(P,&mup);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  mu   = mup[0];
  y[0] = l[1]*(2.*mu*u[1]*x[0] + 2.*mu*u[0]*x[1]);
  y[1] = l[1]*(2.*mu*u[0]*x[0]);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(P,&mup);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_N)*F_UM*X, where L and X are vectors of size N and P respectively, I_N is the identity matrix of size N, \otimes is the Kronecker product,
   and F_UM is the N^2 x P matrix with entries

          | F^1_UM |
   F_UM = |   ...  |, F^k has dimension NxP, with {F^k_UM}_ij = \frac{\partial^2 F_k}{\partial m_j \partial u_i}, where F_k is the k-th component of the DAE
          | F^N_UM |

   with m_j the j-th design variable and u_i the i-th state variable, with N the number of state variables and P the number of design variables.

   The output should be computed as:  Y = (\sum_k L_k*F^k_UM)*X, with L_k the k-th entry of the adjoint variable L.
   In this example,

                           | 2*u[0]*u[1]+1 |
      F^1_UM = 0, F^2_UM = |               |
                           | -1+u[0]*u[0]  |

   Note that with the adjointode approach the ODE is assumed to be in the implicit form F(U,Udot,t;M) = 0
*/
static PetscErrorCode EvalHessianDAE_UP(TS ts, PetscReal time, Vec U, Vec Udot, Vec P, Vec L, Vec X, Vec Y, void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *y;
  const PetscScalar *x,*l,*u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  y[0] = l[1]*((2.*u[0]*u[1]+1.)*x[0]);
  y[1] = l[1]*((-1.+u[0]*u[0])*x[0]);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns Y = (L^T \otimes I_P)*F_MU*X, where L and X are vectors of size N, I_P is the identity matrix of size P, \otimes is the Kronecker product,
   and F_MU is the N*P x N matrix with entries

          | F^1_MU |
   F_MU = |   ...  |, F^k has dimension PxN, with {F^k_MU}_ij = \frac{\partial^2 F_k}{\partial u_j \partial m_i}, where F_k is the k-th component of the DAE
          | F^N_MU |

   with u_j the j-th state variable and m_i the i-th design variable, with N the number of state variables and P the number of design variables.

   The output should be computed as:  Y = (\sum_k L_k*F^k_MU)*X = (\sum_k L_k*(F^k_UM)^T)*X, with L_k the k-th entry of the adjoint variable L.
   In this example

      F^1_MU = 0, F^2_MU = | 2*u[0]*u[1]+1, -1+u[0]*u[0] |

   Note that with the adjointode approach the ODE is assumed to be in the implicit form F(U,Udot,t;M) = 0
*/
static PetscErrorCode EvalHessianDAE_PU(TS ts, PetscReal time, Vec U, Vec Udot, Vec P, Vec L, Vec X, Vec Y, void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *y;
  const PetscScalar *x,*l,*u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  y[0] = l[1]*((2.*u[0]*u[1]+1.)*x[0] + (-1.+u[0]*u[0])*x[1]);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* The DAE dependency on the parameters is linear, thus EvalHessianDAE_PP is zero */

/* TSComputeObjectiveAndGradient just computes the objective function if the gradient is NULL */
PetscErrorCode FormFunction_AO(Tao tao,Vec IC,PetscReal *f,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = FormFunctionGradient_AO(tao,IC,f,NULL,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TSComputeObjectiveAndGradient just computes the gradient if the return value of the objective function is NULL */
PetscErrorCode FormGradient_AO(Tao tao,Vec IC,Vec G,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = FormFunctionGradient_AO(tao,IC,NULL,G,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* wrapper for TSComputeObjectiveAndGradient */
PetscErrorCode FormFunctionGradient_AO(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx)
{
  User              user_ptr = (User)ctx;
  TS                ts;
  PetscScalar       *x_ptr;
  const PetscScalar *y_ptr;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(P,&y_ptr);CHKERRQ(ierr);
  user_ptr->mu = y_ptr[0];
  ierr = VecRestoreArrayRead(P,&y_ptr);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.03125);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,user_ptr);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user_ptr->A,user_ptr->A,IJacobian,user_ptr);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user_ptr->ftime);CHKERRQ(ierr);
  ierr = TSAddObjective(ts,user_ptr->ftime,EvalObjective_AO,EvalCostGradient_U_AO,NULL,
                        NULL,NULL,NULL,NULL,NULL,NULL,user_ptr);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,user_ptr->Jacp,EvalGradientDAE_P,user_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(user_ptr->x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;
  x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(user_ptr->x,&x_ptr);CHKERRQ(ierr);
  ierr = TSComputeObjectiveAndGradient(ts,0.0,PETSC_DECIDE,user_ptr->ftime,user_ptr->x,P,G,f);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* this function gets called every time we change the design state */
static PetscErrorCode TSSetUpFromDesign_Private(TS ts, Vec X0, Vec P, void *ctx)
{
  User user_ptr = (User)ctx;
  PetscScalar       *x_ptr;
  const PetscScalar *y_ptr;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSSetTimeStep(ts,0.03125);CHKERRQ(ierr);
  ierr = VecGetArrayRead(P,&y_ptr);CHKERRQ(ierr);
  user_ptr->mu = y_ptr[0];
  ierr = VecRestoreArrayRead(P,&y_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(X0,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;
  x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(X0,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user_ptr->ftime);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* wrapper for TSComputeHessian */
PetscErrorCode FormHessian_AO(Tao tao,Vec P,Mat H,Mat Hp,void *ctx)
{
  User              user_ptr = (User)ctx;
  TS                ts;
  PetscScalar       *x_ptr;
  const PetscScalar *y_ptr;
  Mat               H_UU;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(P,&y_ptr);CHKERRQ(ierr);
  user_ptr->mu = y_ptr[0];
  ierr = VecRestoreArrayRead(P,&y_ptr);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.03125);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,user_ptr);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user_ptr->A,user_ptr->A,IJacobian,user_ptr);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user_ptr->ftime);CHKERRQ(ierr);
  /* Hessian (wrt the state) of objective function (all other Hessian terms of the objective are zero) */
  ierr = MatDuplicate(user_ptr->A,MAT_DO_NOT_COPY_VALUES,&H_UU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(H_UU,2.0*rescale);CHKERRQ(ierr);
  ierr = TSAddObjective(ts,user_ptr->ftime,EvalObjective_AO,EvalCostGradient_U_AO,NULL,
                        H_UU,NULL,NULL,NULL,NULL,NULL,user_ptr);CHKERRQ(ierr);
  ierr = MatDestroy(&H_UU);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,user_ptr->Jacp,EvalGradientDAE_P,user_ptr);CHKERRQ(ierr);
  /* Hessian terms of the DAE */
  ierr = TSSetHessianDAE(ts,EvalHessianDAE_UU,NULL,EvalHessianDAE_UP, /* NULL term since there's no dependency on Udot */
                            NULL,NULL,NULL,                           /* Udot dependency is NULL */
                            EvalHessianDAE_PU,NULL,NULL,              /* NULL term since there's no dependency on Udot */
                            NULL);CHKERRQ(ierr);

  /* Add TSSetUp from design (needed if we want to use -tshessian_mffd) */
  ierr = TSSetSetUpFromDesign(ts,TSSetUpFromDesign_Private,user_ptr);CHKERRQ(ierr);

  ierr = VecGetArray(user_ptr->x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;
  x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(user_ptr->x,&x_ptr);CHKERRQ(ierr);
  ierr = TSComputeHessian(ts,0.0,PETSC_DECIDE,user_ptr->ftime,user_ptr->x,P,H);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
#if 0
  {
    Mat He;
    ierr = MatComputeOperator(H,NULL,&He);CHKERRQ(ierr);
    ierr = MatView(He,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&He);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*TEST
    build:
      requires: !complex !single

    test:
      args:  -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 -viewer_binary_skip_info -tao_view  -ts_trajectory_dirname ex20opt_pdir -tao_blmvm_mat_lmvm_scale_type none
      output_file: output/ex20opt_p_1.out

    test:
      suffix: ao
      args: -adjointode -tao_monitor -monitor 0 -ts_trajectory_type memory -tao_view

    test:
      suffix: ao_discrete
      args: -adjointode -tsgradient_adjoint_discrete -tao_monitor -monitor 0 -ts_trajectory_type memory -tao_view -tao_test_gradient

    test:
      suffix: ao_hessian
      args: -adjointode -tao_monitor -monitor 0 -tao_view -tao_type tron -ts_trajectory_type memory

    test:
      suffix: ao_hessian_discrete
      args: -adjointode -tao_monitor -monitor 0 -tao_view -tao_type tron -ts_trajectory_type memory -tsgradient_adjoint_discrete  -tshessian_foadjoint_discrete -tshessian_tlm_discrete  -tshessian_soadjoint_discrete -tao_test_hessian

    test:
      suffix: ao_hessian_mf
      args: -adjointode -tao_monitor -monitor 0 -tao_view -tao_type tron -tao_mf_hessian -ts_trajectory_type memory

    test:
      suffix: ao_hessian_tsmf
      args: -adjointode -tao_monitor -monitor 0 -tao_view -tao_type tron -tshessian_mffd -ts_trajectory_type memory

TEST*/

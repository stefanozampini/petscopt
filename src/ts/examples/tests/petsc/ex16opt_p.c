static char help[] = "Solves an ODE-constrained optimization problem -- finding the optimal stiffness parameter for the van der Pol equation.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation
   Concepts: Optimization using adjoint sensitivities
   Processors: 1
*/
/* ------------------------------------------------------------------------

   Notes:
   This code demonstrates how to solve an ODE-constrained optimization problem with TAO, TSAdjoint and TS.
   The objective is to minimize the difference between observation and model prediction by finding an optimal value for parameter \mu.
   The gradient is either computed with the discrete adjoint of an explicit Runge-Kutta method (see ex16adj.c for details), or by solving the adjoint ODE.
   With the adjoint ode approach, we can also compute the Hessian via TSComputeHessian().
  ------------------------------------------------------------------------- */

#include <petscopt.h>
#include <petsctao.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
  PetscReal ftime,x_ob[2];
  Mat       A;                  /* Jacobian matrix */
  Mat       Jacp;               /* JacobianP matrix */
  Vec       x,lambda[2],mup[2]; /* adjoint variables */
};

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormFunction_AO(Tao,Vec,PetscReal*,void*);
PetscErrorCode FormGradient_AO(Tao,Vec,Vec,void*);
PetscErrorCode FormFunctionGradient_AO(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian_AO(Tao,Vec,Mat,Mat,void*);

/*
*  User-defined routines
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = x[1];
  f[1] = user->mu*(1.-x[0]*x[0])*x[1]-x[0];
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscReal         mu   = user->mu;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  J[0][0] = 0;
  J[1][0] = -2.*mu*x[1]*x[0]-1.;
  J[0][1] = 1.0;
  J[1][1] = mu*(1.0-x[0]*x[0]);
  ierr = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (B && A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          row[] = {0,1},col[]={0};
  PetscScalar       J[2][1];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  J[0][0] = 0;
  J[1][0] = (1.-x[0]*x[0])*x[1];
  ierr = MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscReal         tfinal, dt, tprev;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tfinal);CHKERRQ(ierr);
  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",(double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1]));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"t %.6f (tprev = %.6f) \n",(double)t,(double)tprev);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS                 ts;          /* nonlinear solver */
  Vec                p;
  PetscBool          monitor = PETSC_FALSE;
  PetscScalar        *x_ptr;
  PetscMPIInt        size;
  struct _n_User     user;
  PetscErrorCode     ierr;
  Tao                tao;
  Vec                lowerb,upperb;
  KSP                ksp;
  PC                 pc;
  PetscBool          adjointode = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.mu          = 1.0;
  user.next_output = 0.0;
  user.ftime       = 0.5;

  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);

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
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,user.A,user.A,RHSJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.ftime);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = 0.66666654321;
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&(user.ftime));CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  user.x_ob[0] = x_ptr[0];
  user.x_ob[1] = x_ptr[1];
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);

  ierr = MatCreateVecs(user.A,&user.lambda[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.lambda[1],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.mup[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.mup[1],NULL);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOCG);CHKERRQ(ierr);

  /* Set initial solution guess */
  ierr = MatCreateVecs(user.Jacp,&p,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(p,&x_ptr);CHKERRQ(ierr);
  x_ptr[0]   = 6.0;
  ierr = VecRestoreArray(p,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,p);CHKERRQ(ierr);

  /* Set routine for function and gradient evaluation */
  ierr = PetscOptionsGetBool(NULL,NULL,"-adjointode",&adjointode,NULL);CHKERRQ(ierr);
  if (adjointode) { /* use adjoint ode approach */
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
  ierr = VecDuplicate(p,&lowerb);CHKERRQ(ierr);
  ierr = VecDuplicate(p,&upperb);CHKERRQ(ierr);
  ierr = VecGetArray(lowerb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(lowerb,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(upperb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 100.0;
  ierr = VecRestoreArray(upperb,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetVariableBounds(tao,lowerb,upperb);CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  ierr = TaoSetTolerances(tao,1e-13,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
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
  ierr = VecDestroy(&user.lambda[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mup[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mup[1]);CHKERRQ(ierr);

  ierr = VecDestroy(&lowerb);CHKERRQ(ierr);
  ierr = VecDestroy(&upperb);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  ierr = PetscOptFinalize();
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
  User              user = (User)ctx;
  TS                ts;
  PetscScalar       *x_ptr,*y_ptr;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  user->mu = x_ptr[0];
  ierr = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,user->A,user->A,RHSJacobian,user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user->x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2;
  x_ptr[1] = 0.66666654321;
  ierr = VecRestoreArray(user->x,&x_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user->ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = TSSolve(ts,user->x);CHKERRQ(ierr);

  ierr = VecGetArray(user->x,&x_ptr);CHKERRQ(ierr);
  *f   = (x_ptr[0]-user->x_ob[0])*(x_ptr[0]-user->x_ob[0])+(x_ptr[1]-user->x_ob[1])*(x_ptr[1]-user->x_ob[1]);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*   Redet initial conditions for the adjoint integration */
  ierr = VecGetArray(user->lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.*(x_ptr[0]-user->x_ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-user->x_ob[1]);
  ierr = VecRestoreArray(user->x,&x_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->x,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(user->lambda[1],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  x_ptr[1] = 1.0;
  ierr = VecRestoreArray(user->lambda[1],&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(user->mup[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(user->mup[0],&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(user->mup[1],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(user->mup[1],&x_ptr);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,user->lambda,user->mup);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(ts,user->Jacp,RHSJacobianP,user);CHKERRQ(ierr);


  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = VecCopy(user->mup[0],G);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* callbacks for the adjoint ode (AO) approach
   Note that with the adjointode approach the ODE is assumed to be in the implicit form F(U,Udot,t;M) = 0 */

/* the cost functional interface: returns ||u - u_obs||^2 */
static PetscErrorCode EvalObjective_AO(Vec U, Vec P, PetscReal time, PetscReal *val, void *ctx)
{
  const PetscScalar *x;
  User              user = (User)ctx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&x);CHKERRQ(ierr);
  *val = (x[0]-user->x_ob[0])*(x[0]-user->x_ob[0])+(x[1]-user->x_ob[1])*(x[1]-user->x_ob[1]);
  ierr = VecRestoreArrayRead(U,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* the gradient of the cost functional with respect to the state variables */
static PetscErrorCode EvalObjectiveGradient_U_AO(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *x;
  PetscScalar       *g;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&x);CHKERRQ(ierr);
  ierr = VecGetArray(grad,&g);CHKERRQ(ierr);
  g[0] = 2*(x[0]-user->x_ob[0]);
  g[1] = 2*(x[1]-user->x_ob[1]);
  ierr = VecRestoreArray(grad,&g);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* gradient term of the DAE: we can handle X,Xdot and P dependency, even if in this case the dependency is on X only */

/* wrapper for RHSJacobianP */
static PetscErrorCode EvalGradientDAE_P(TS ts, PetscReal t, Vec X, Vec Xdot, Vec P, Mat J, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = RHSJacobianP(ts,t,X,J,ctx);CHKERRQ(ierr);
  ierr = MatScale(J,-1.0);CHKERRQ(ierr); /* implicit form */
  PetscFunctionReturn(0);
}

/* hessian terms of the DAE */

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

                           | 2*u[0]*u[1]  |
      F^1_UM = 0, F^2_UM = |              |
                           | -1+u[0]*u[0] |
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
  y[0] = l[1]*(2.*u[0]*u[1]*x[0]);
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

      F^1_MU = 0, F^2_MU = | 2*u[0]*u[1] , -1+u[0]*u[0] |
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
  y[0] = l[1]*(2.*u[0]*u[1]*x[0] + (-1.+u[0]*u[0])*x[1]);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  User              user = (User)ctx;
  TS                ts;
  PetscScalar       *x_ptr;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  user->mu = x_ptr[0];
  ierr = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,user->A,user->A,RHSJacobian,user);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user->ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = VecGetArray(user->x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2;
  x_ptr[1] = 0.66666654321;
  ierr = VecRestoreArray(user->x,&x_ptr);CHKERRQ(ierr);
  /* the cost functional needs to be evaluated at final time */
  ierr = TSAddObjective(ts,user->ftime,EvalObjective_AO,EvalObjectiveGradient_U_AO,NULL,
                        NULL,NULL,NULL,NULL,NULL,NULL,user);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,user->Jacp,EvalGradientDAE_P,user);CHKERRQ(ierr);
  ierr = TSComputeObjectiveAndGradient(ts,0.0,PETSC_DECIDE,user->ftime,user->x,P,G,f);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* this function gets called every time we change the design state */
static PetscErrorCode TSSetUpFromDesign_Private(TS ts, Vec X0, Vec P, void *ctx)
{
  User           user = (User)ctx;
  PetscScalar    *x_ptr;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  user->mu = x_ptr[0];
  ierr = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(X0,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;
  x_ptr[1] = 0.66666654321;
  ierr = VecRestoreArray(X0,&x_ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* wrapper for TSComputeHessian */
PetscErrorCode FormHessian_AO(Tao tao,Vec P,Mat H,Mat Hp,void *ctx)
{
  User              user = (User)ctx;
  TS                ts;
  Mat               H_UU;
  PetscScalar       *x_ptr;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  user->mu = x_ptr[0];
  ierr = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,user->A,user->A,RHSJacobian,user);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user->ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = VecGetArray(user->x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2;
  x_ptr[1] = 0.66666654321;
  ierr = VecRestoreArray(user->x,&x_ptr);CHKERRQ(ierr);
  /* Hessian (wrt the state) of objective function (all other Hessian terms of the objective are zero) */
  ierr = MatDuplicate(user->A,MAT_DO_NOT_COPY_VALUES,&H_UU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(H_UU,2.0);CHKERRQ(ierr);
  /* the cost functional needs to be evaluated at final time */
  ierr = TSAddObjective(ts,user->ftime,EvalObjective_AO,EvalObjectiveGradient_U_AO,NULL,
                        H_UU,NULL,NULL,NULL,NULL,NULL,user);CHKERRQ(ierr);
  ierr = MatDestroy(&H_UU);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,user->Jacp,EvalGradientDAE_P,user);CHKERRQ(ierr);
  /* The DAE dependency on the parameters is linear, thus EvalHessianDAE_PP is zero */
  ierr = TSSetHessianDAE(ts,EvalHessianDAE_UU,NULL,EvalHessianDAE_UP, /* NULL term since there's no dependency on Udot */
                            NULL,NULL,NULL,                           /* Udot dependency is NULL */
                            EvalHessianDAE_PU,NULL,NULL,              /* NULL term since there's no dependency on Udot */
                            NULL);CHKERRQ(ierr);
  /* Add TSSetUp from design (needed if we want to use -tshessian_mffd) */
  ierr = TSSetSetUpFromDesign(ts,TSSetUpFromDesign_Private,user);CHKERRQ(ierr);

  ierr = TSComputeHessian(ts,0.0,PETSC_DECIDE,user->ftime,user->x,P,H);CHKERRQ(ierr);
#if 0
  {
    Mat He;
    ierr = MatComputeOperator(H,NULL,&He);CHKERRQ(ierr);
    ierr = MatView(He,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&He);CHKERRQ(ierr);
  }
#endif
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

    build:
      requires: !complex !single

    test:
      suffix: 1
      args: -monitor 0 -viewer_binary_skip_info -tao_view -tao_monitor  -tao_gttol 1.e-5 -ts_trajectory_dirname ex16opt_pdir

    test:
      suffix: ao
      args: -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -tsgradient_adjoint_ts_atol 1.e-9

    test:
      suffix: ao_discrete
      args: -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -tao_type tron -tao_mf_hessian -ts_type {{rk cn}separate output} -ts_adapt_type none -tao_test_gradient -tsgradient_adjoint_discrete

    test:
      suffix: ao_hessian_mf
      args: -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -tsgradient_adjoint_ts_atol 1.e-10 -tao_type tron -tao_mf_hessian

    test:
      suffix: ao_hessian_tsmf
      args: -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -tsgradient_adjoint_ts_atol 1.e-10 -tao_type tron -tshessian_mffd

    test:
      suffix: ao_hessian
      args: -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -tsgradient_adjoint_ts_atol 1.e-10 -tshessian_soadjoint_ts_atol 1.e-10 -tshessian_foadjoint_ts_atol 1.e-10 -tshessian_tlm_ts_atol 1.e-10 -tao_type tron

    test:
      suffix: ao_hessian_discrete
      args: -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -ts_adapt_type none -ts_type {{rk cn}separate output} -tsgradient_adjoint_discrete -tshessian_foadjoint_discrete -tshessian_tlm_discrete  -tshessian_soadjoint_discrete -tao_test_hessian -tao_type tron

TEST*/

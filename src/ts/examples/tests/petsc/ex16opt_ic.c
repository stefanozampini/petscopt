static char help[] = "Solves an ODE-constrained optimization problem -- finding the optimal initial conditions for the van der Pol equation.\n\
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
   The objective is to minimize the difference between observation and model prediction by finding optimal values for initial conditions.
   The gradient is computed either with the discrete adjoint of an explicit Runge-Kutta method (see ex16adj.c for details) or by solving the adjoint ode.
   With the adjoint ode approach, we can also compute the Hessian via TSComputeHessian().
  ------------------------------------------------------------------------- */

#include <petscopt.h>
#include <petsctao.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;

  PetscInt  steps;
  PetscReal ftime,x_ob[2];
  Mat       A;             /* Jacobian matrix */
  Vec       x,lambda[2];   /* adjoint variables */
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
  J[1][0] = -2.*mu*x[1]*x[0]-1;
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
  Vec                ic;
  PetscBool          monitor = PETSC_FALSE;
  PetscScalar        *x_ptr;
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.mu          = 1.0;
  user.next_output = 0.0;
  user.steps       = 0;
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.001);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,user.A,user.A,RHSJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = 0.66666654321;
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)(user.ftime));CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&(user.ftime));CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&user.steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)user.ftime);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  user.x_ob[0] = x_ptr[0];
  user.x_ob[1] = x_ptr[1];
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);

  ierr = MatCreateVecs(user.A,&user.lambda[0],NULL);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOCG);CHKERRQ(ierr);

  /* Set initial solution guess */
  ierr = MatCreateVecs(user.A,&ic,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(ic,&x_ptr);CHKERRQ(ierr);
  x_ptr[0]  = 2.1;
  x_ptr[1]  = 0.7;
  ierr = VecRestoreArray(ic,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,ic);CHKERRQ(ierr);

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
  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  ierr = TaoSetTolerances(tao,1e-10,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = VecDestroy(&user.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[0]);CHKERRQ(ierr);

  ierr = VecDestroy(&ic);CHKERRQ(ierr);
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
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  User              user = (User)ctx;
  TS                ts;
  PetscScalar       *y_ptr;
  const PetscScalar *x_ptr;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(IC,user->x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);
  /*   Set RHS Jacobian  for the adjoint integration */
  ierr = TSSetRHSJacobian(ts,user->A,user->A,RHSJacobian,user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.001);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,0.5);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = TSSetTolerances(ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,user->x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&user->ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&user->steps);CHKERRQ(ierr);
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %.6f, steps %D, ftime %g\n",(double)user->mu,user->steps,(double)user->ftime);CHKERRQ(ierr); */

  ierr = VecGetArrayRead(user->x,&x_ptr);CHKERRQ(ierr);
  *f   = (x_ptr[0]-user->x_ob[0])*(x_ptr[0]-user->x_ob[0])+(x_ptr[1]-user->x_ob[1])*(x_ptr[1]-user->x_ob[1]);
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"Observed value y_ob=[%f; %f], ODE solution y=[%f;%f], Cost function f=%f\n",(double)user->x_ob[0],(double)user->x_ob[1],(double)x_ptr[0],(double)x_ptr[1],(double)(*f));CHKERRQ(ierr); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*   Redet initial conditions for the adjoint integration */
  ierr = VecGetArray(user->lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.*(x_ptr[0]-user->x_ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-user->x_ob[1]);
  ierr = VecRestoreArray(user->lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,user->lambda,NULL);CHKERRQ(ierr);


  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = VecCopy(user->lambda[0],G);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* callbacks for the adjoint ode approach
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
PetscErrorCode FormFunctionGradient_AO(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  User           user = (User)ctx;
  TS             ts;
  PetscErrorCode ierr;
  Mat            Id;

  PetscFunctionBeginUser;
  ierr = VecCopy(IC,user->x);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.001);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);
  ierr = TSSetTolerances(ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,user->A,user->A,RHSJacobian,user);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  /* the cost functional needs to be evaluated at time 0.5 */
  ierr = TSAddObjective(ts,0.5,EvalObjective_AO,EvalObjectiveGradient_U_AO,NULL,
                        NULL,NULL,NULL,NULL,NULL,NULL,user);CHKERRQ(ierr);
  /* set Jacobian G_p for the ODE dependence from initial conditions in implicit form: G(u(0),p) = 0 */
  ierr = MatDuplicate(user->A,MAT_DO_NOT_COPY_VALUES,&Id);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatZeroEntries(Id);CHKERRQ(ierr);
  ierr = MatShift(Id,-1.0);CHKERRQ(ierr);
  ierr = TSSetGradientIC(ts,NULL,Id,NULL,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&Id);CHKERRQ(ierr);
  ierr = TSComputeObjectiveAndGradient(ts,0.0,PETSC_DECIDE,0.5,user->x,IC,G,f);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* this function gets called every time we change the design state */
static PetscErrorCode TSSetUpFromDesign_Private(TS ts, Vec X0, Vec IC, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(IC,X0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.001);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* wrapper for TSComputeHessian */
PetscErrorCode FormHessian_AO(Tao tao,Vec IC,Mat H,Mat Hp,void *ctx)
{
  User           user = (User)ctx;
  TS             ts;
  PetscErrorCode ierr;
  Mat            Id,H_UU;

  PetscFunctionBeginUser;
  ierr = VecCopy(IC,user->x);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.001);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);
  ierr = TSSetTolerances(ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,user->A,user->A,RHSJacobian,user);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  /* Hessian (wrt the state) of objective function (all other Hessian terms are zero) */
  ierr = MatDuplicate(user->A,MAT_DO_NOT_COPY_VALUES,&H_UU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H_UU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(H_UU,2.0);CHKERRQ(ierr);
  /* the cost functional needs to be evaluated at time 0.5 */
  ierr = TSAddObjective(ts,0.5,EvalObjective_AO,EvalObjectiveGradient_U_AO,NULL,
                        H_UU,NULL,NULL,NULL,NULL,NULL,user);CHKERRQ(ierr);
  ierr = MatDestroy(&H_UU);CHKERRQ(ierr);
  /* set Jacobian G_p for the ODE dependence from initial conditions in implicit form: G(u(0),p) = 0 */
  ierr = MatDuplicate(user->A,MAT_DO_NOT_COPY_VALUES,&Id);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatZeroEntries(Id);CHKERRQ(ierr);
  ierr = MatShift(Id,-1.0);CHKERRQ(ierr);
  ierr = TSSetGradientIC(ts,NULL,Id,NULL,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&Id);CHKERRQ(ierr);

  /* Add TSSetUp from design (needed if we want to use -tshessian_mffd) */
  ierr = TSSetSetUpFromDesign(ts,TSSetUpFromDesign_Private,NULL);CHKERRQ(ierr);

  ierr = TSComputeHessian(ts,0.0,PETSC_DECIDE,0.5,user->x,IC,H);CHKERRQ(ierr);
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
      requires: !single !complex

    test:
      suffix: 1
      args: -monitor 0 -viewer_binary_skip_info -tao_view -tao_monitor  -tao_gttol 1.e-5 -ts_trajectory_dirname ex16opt_icdir
    test:
      suffix: ao
      args: -ts_trajectory_type memory -adjointode -monitor 0 -tao_view -tao_monitor -tao_gttol 1.e-5
    test:
      suffix: ao_hessian
      args: -ts_trajectory_type memory -adjointode -monitor 0 -tao_view -tao_monitor -tao_gttol 1.e-5 -tao_type tron
    test:
      suffix: ao_hessian_mf
      args: -ts_trajectory_type memory -adjointode -monitor 0 -tao_view -tao_monitor -tao_gttol 1.e-5 -tao_type tron -tao_mf_hessian
    test:
      suffix: ao_hessian_tsmf
      args: -ts_trajectory_type memory -adjointode -monitor 0 -tao_view -tao_monitor -tao_gttol 1.e-5 -tao_type tron -tshessian_mffd
    test:
      suffix: 2
      args: -ts_rhs_jacobian_test_mult_transpose -mat_shell_test_mult_transpose_view -tao_max_it 2 -ts_rhs_jacobian_test_mult -mat_shell_test_mult_view

TEST*/

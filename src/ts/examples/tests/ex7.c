static const char help[] = "Reproduces cvsHessian_ASA_FSA from CVODE testuite.\n";
/*
  This example from https://github.com/LLNL/sundials/blob/master/examples/cvodes/serial/cvsHessian_ASA_FSA.c
  Output can be checked against https://github.com/LLNL/sundials/blob/master/examples/cvodes/serial/cvsHessian_ASA_FSA.out
  Computes the gradient and the Hessian of

  Obj(y) = \int^2_0  0.5 * ( y1^2 + y2^2 + y3^2 ) dt

  where y = [y1,y2,y3] obeys

       [ - p1 * y1^2 - y3 ]           [ 1 ]
  y' = [    - y2          ]    y(0) = [ 1 ]
       [ -p2^2 * y2 * y3  ]           [ 1 ]

  p1 = 1.0
  p2 = 2.0


*/
#include <petscopt.h>
#include <petsctao.h>

static PetscErrorCode EvalObjective(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      v;

  PetscFunctionBeginUser;
  ierr = VecNorm(U,NORM_2,&v);CHKERRQ(ierr);
  *val = 0.5*v*v;
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveGradient_U(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(U,grad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveHessian_UU(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  ierr = MatShift(H,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Uses explicit interface udot = G(u,t) */
static PetscErrorCode FormRHSFunction(TS ts,PetscReal time,Vec U,Vec F,void* ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;
  PetscReal         *m;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSGetApplicationContext(ts,(void*)&m);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(F,&f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  f[0] = - m[0] * u[0] * u[0] - u[2];
  f[1] = - u[1];
  f[2] = - m[1] * m[1] * u[1] * u[2];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian(TS ts,PetscReal time,Vec U,Mat A,Mat P,void* ctx)
{
  const PetscScalar *u;
  PetscReal         *m;
  PetscErrorCode    ierr;
  PetscInt          i[3] = {0, 1, 2};
  PetscScalar       v[3][3];

  PetscFunctionBeginUser;
  ierr = TSGetApplicationContext(ts,(void*)&m);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  v[0][0] = - 2.0 * m[0] * u[0];
  v[0][1] =   0.0;
  v[0][2] = - 1.0;
  v[1][0] =   0.0;
  v[1][1] = - 1.0;
  v[1][2] =   0.0;
  v[2][0] =   0.0;
  v[2][1] = - m[1] * m[1] * u[2];
  v[2][2] = - m[1] * m[1] * u[1];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = MatZeroEntries(P);CHKERRQ(ierr);
  ierr = MatSetValues(P,3,i,3,i,(PetscScalar*)v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A && A != P) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* note: these callbacks are always given in implicit form, that's why the minus sign disappears */
static PetscErrorCode EvalGradientDAE_M(TS ts, PetscReal t, Vec U, Vec Udot, Vec M, Mat F_M, void *ctx)
{
  const PetscScalar *u;
  const PetscScalar *m;
  PetscInt          i[3] = {0, 1, 2};
  PetscScalar       v[3][2];
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  v[0][0] = u[0] * u[0];
  v[0][1] = 0.0;
  v[1][0] = 0.0;
  v[1][1] = 0.0;
  v[2][0] = 0.0;
  v[2][1] = 2.0 * m[1] * u[1] * u[2];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  ierr = MatZeroEntries(F_M);CHKERRQ(ierr);
  ierr = MatSetValues(F_M,3,i,2,i,(PetscScalar*)v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_MM(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *u;
  PetscScalar       v[3][2][2];
  PetscErrorCode    ierr;
  const PetscScalar *x,*l;
  PetscScalar       *y;
  PetscInt          i,j,k;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  v[0][0][0] = 0.0;
  v[0][0][1] = 0.0;
  v[0][1][0] = 0.0;
  v[0][1][1] = 0.0;
  v[1][0][0] = 0.0;
  v[1][0][1] = 0.0;
  v[1][1][0] = 0.0;
  v[1][1][1] = 0.0;
  v[2][0][0] = 0.0;
  v[2][0][1] = 0.0;
  v[2][1][0] = 0.0;
  v[2][1][1] = 2.0 * u[1] * u[2];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

  ierr = VecGetArrayWrite(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  for (j=0;j<2;j++) {
    y[j] = 0.0;
    for (k=0;k<3;k++)
      for (i=0;i<2;i++)
        y[j] += l[k] * v[k][j][i] * x[i];
  }
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_MU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *u;
  const PetscScalar *m;
  PetscScalar       v[3][2][3];
  PetscErrorCode    ierr;
  const PetscScalar *x,*l;
  PetscScalar       *y;
  PetscInt          i,j,k;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  v[0][0][0] = 2 * u[0];
  v[0][0][1] = 0.0;
  v[0][0][2] = 0.0;
  v[0][1][0] = 0.0;
  v[0][1][1] = 0.0;
  v[0][1][2] = 0.0;
  v[1][0][0] = 0.0;
  v[1][0][1] = 0.0;
  v[1][0][2] = 0.0;
  v[1][1][0] = 0.0;
  v[1][1][1] = 0.0;
  v[1][1][2] = 0.0;
  v[2][0][0] = 0.0;
  v[2][0][1] = 0.0;
  v[2][0][2] = 0.0;
  v[2][1][0] = 0.0;
  v[2][1][1] = 2.0 * m[1] * u[2];
  v[2][1][2] = 2.0 * m[1] * u[1];
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

  ierr = VecGetArrayWrite(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  for (j=0;j<2;j++) {
    y[j] = 0.0;
    for (k=0;k<3;k++)
      for (i=0;i<3;i++)
        y[j] += l[k] * v[k][j][i] * x[i];
  }
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_UM(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *u;
  const PetscScalar *m;
  PetscScalar       v[3][3][2];
  PetscErrorCode    ierr;
  const PetscScalar *x,*l;
  PetscScalar       *y;
  PetscInt          i,j,k;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  v[0][0][0] = 2 * u[0];
  v[0][0][1] = 0.0;
  v[0][1][0] = 0.0;
  v[0][1][1] = 0.0;
  v[0][2][0] = 0.0;
  v[0][2][1] = 0.0;
  v[1][0][0] = 0.0;
  v[1][0][1] = 0.0;
  v[1][1][0] = 0.0;
  v[1][1][1] = 0.0;
  v[1][2][0] = 0.0;
  v[1][2][1] = 0.0;
  v[2][0][0] = 0.0;
  v[2][0][1] = 0.0;
  v[2][1][0] = 0.0;
  v[2][1][1] = 2.0 * m[1] * u[2];
  v[2][2][0] = 0.0;
  v[2][2][1] = 2.0 * m[1] * u[1];
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

  ierr = VecGetArrayWrite(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  for (j=0;j<3;j++) {
    y[j] = 0.0;
    for (k=0;k<3;k++)
      for (i=0;i<2;i++)
        y[j] += l[k] * v[k][j][i] * x[i];
  }
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_UU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *x,*l,*m;
  PetscScalar       *y;
  PetscScalar       v[3][3][3];
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  v[0][0][0] = 2.0 * m[0];
  v[0][0][1] = 0.0;
  v[0][0][2] = 0.0;
  v[0][1][0] = 0.0;
  v[0][1][1] = 0.0;
  v[0][1][2] = 0.0;
  v[0][2][0] = 0.0;
  v[0][2][1] = 0.0;
  v[0][2][2] = 0.0;
  v[1][0][0] = 0.0;
  v[1][0][1] = 0.0;
  v[1][0][2] = 0.0;
  v[1][1][0] = 0.0;
  v[1][1][1] = 0.0;
  v[1][1][2] = 0.0;
  v[1][2][0] = 0.0;
  v[1][2][1] = 0.0;
  v[1][2][2] = 0.0;
  v[2][0][0] = 0.0;
  v[2][0][1] = 0.0;
  v[2][0][2] = 0.0;
  v[2][1][0] = 0.0;
  v[2][1][1] = 0.0;
  v[2][1][2] = m[1] * m[1];
  v[2][2][0] = 0.0;
  v[2][2][1] = m[1] * m[1];
  v[2][2][2] = 0.0;
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);

  ierr = VecGetArrayWrite(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  for (j=0;j<3;j++) {
    y[j] = 0.0;
    for (k=0;k<3;k++)
      for (i=0;i<3;i++)
        y[j] += l[k] * v[k][j][i] * x[i];
  }
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* store parameter value in application context for use in TSRHSFunction/Jacobian callbacks */
static PetscErrorCode MyTSSetUpFromDesign(TS ts, Vec u0, Vec M, void *ctx)
{
  PetscReal         *p;
  const PetscScalar *m;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSGetApplicationContext(ts,(void*)&p);CHKERRQ(ierr);
  ierr = VecSet(u0,1.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  p[0] = PetscRealPart(m[0]);
  p[1] = PetscRealPart(m[1]);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* example of Tao callbacks */
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
  OptCtx         opt;
  TS             ts;
  Mat            J,H,He;
  Mat            Phi,Phie;
  Vec            U,M,G;
  PetscScalar    *g;
  PetscReal      t0,tf,dt,obj;
  PetscBool      testtao = PETSC_FALSE, testtaylor = PETSC_FALSE, testtlm = PETSC_FALSE;
  PetscBool      check_dae = PETSC_FALSE, testhessian = PETSC_FALSE;
  PetscReal      AppCtx[2];
  PetscErrorCode ierr;

  ierr = PetscOptInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Command line options */
  t0   = 0.0;
  tf   = 2.0;
  dt   = 1.0/128.0;
  ierr = PetscOptionsBegin(PETSC_COMM_SELF,NULL,"","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t0","Initial time","",t0,&t0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","Final time","",tf,&tf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","Initial time step","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check","Check Hessian DAE terms","",check_dae,&check_dae,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tao","Solve the optimization problem","",testtao,&testtao,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_taylor","Run Taylor test","",testtaylor,&testtaylor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tlm","Test Tangent Linear Model to compute the gradient","",testtlm,&testtlm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_hessian","Test Hessian symmetry","",testhessian,&testhessian,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* state vectors */
  ierr = VecCreate(PETSC_COMM_SELF,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,3,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);

  /* ODE solver */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetTolerances(ts,1.e-8,NULL,1.e-8,NULL);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBDF);CHKERRQ(ierr);
  ierr = TSBDFSetOrder(ts,4);CHKERRQ(ierr); /* PETSc does not have variable order BDF */
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,3,NULL,&J);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,AppCtx);CHKERRQ(ierr);

  /* design vector */
  ierr = VecCreate(PETSC_COMM_SELF,&M);CHKERRQ(ierr);
  ierr = VecSetSizes(M,2,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(M,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetValue(M,0,1.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(M,1,2.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(M);CHKERRQ(ierr);

  /* sensitivity callbacks */

  /* objective function as quadrature term */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,3,NULL,&H);CHKERRQ(ierr);
  ierr = TSAddObjective(ts,PETSC_MIN_REAL,EvalObjective,EvalObjectiveGradient_U,NULL,
                                          H,EvalObjectiveHessian_UU,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);

  /* set callback to setup model solver after a design vector changes */
  ierr = TSSetSetUpFromDesign(ts,MyTSSetUpFromDesign,NULL);CHKERRQ(ierr);

  /* set callback to setup matrix mult for first order ode dependence on parameters */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,2,NULL,&J);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,J,EvalGradientDAE_M,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  /* callbacks for Hessian terms of residual function */
  ierr = TSSetHessianDAE(ts,EvalHessianDAE_UU, NULL, EvalHessianDAE_UM,
                            NULL,              NULL, NULL,
                            EvalHessianDAE_MU, NULL, EvalHessianDAE_MM,
                            NULL);CHKERRQ(ierr);

  /* check DAE callbacks terms */
  if (check_dae) {
    PetscRandom r;
    Vec         L,Udot;

    ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
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

  /* Evaluate objective and gradient */
  opt.t0 = t0;
  opt.dt = dt;
  opt.tf = tf;
  opt.ts = ts;
  ierr = TSComputeObjectiveAndGradient(ts,opt.t0,opt.dt,opt.tf,NULL,M,NULL,&obj);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nObjective: %14.6e\n",(double)obj);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nFinal state:\n");CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&g);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[0]),(double)PetscRealPart(g[1]),(double)PetscRealPart(g[2]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&g);CHKERRQ(ierr);

  /* Compute sensitivity matrix */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecSet(U,1.0);CHKERRQ(ierr);
  ierr = TSCreatePropagatorMat(ts,opt.t0,opt.dt,opt.tf,U,M,NULL,&Phi);CHKERRQ(ierr);

  ierr = MatComputeOperator(Phi,NULL,&Phie);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSensitivity matrix:\n");CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(Phie,(const PetscScalar**)&g);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[0]),(double)PetscRealPart(g[1]),(double)PetscRealPart(g[2]));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[3]),(double)PetscRealPart(g[4]),(double)PetscRealPart(g[5]));CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(Phie,(const PetscScalar**)&g);CHKERRQ(ierr);

  /* Compute gradient */
  ierr = VecDuplicate(M,&G);CHKERRQ(ierr);
  ierr = TSComputeObjectiveAndGradient(ts,opt.t0,opt.dt,opt.tf,NULL,M,G,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGradient:\n");CHKERRQ(ierr);
  ierr = VecGetArrayRead(G,(const PetscScalar**)&g);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e\n",(double)PetscRealPart(g[0]),(double)PetscRealPart(g[1]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(G,(const PetscScalar**)&g);CHKERRQ(ierr);

  /* check tangent linear model */
  if (testtlm) {
    Mat PhiT,PhiTe,TLMe;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTLM matrices (Phi, Phi^T and Phi-(Phi^T)^T)\n");
    ierr = MatView(Phie,NULL);CHKERRQ(ierr);

    ierr = MatCreateTranspose(Phi,&PhiT);CHKERRQ(ierr);
    ierr = MatComputeOperator(PhiT,NULL,&PhiTe);CHKERRQ(ierr);
    ierr = MatView(PhiTe,NULL);CHKERRQ(ierr);

    ierr = MatTranspose(PhiTe,MAT_INITIAL_MATRIX,&TLMe);CHKERRQ(ierr);
    ierr = MatAXPY(TLMe,-1.0,Phie,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(TLMe,NULL);CHKERRQ(ierr);

    ierr = MatDestroy(&TLMe);CHKERRQ(ierr);
    ierr = MatDestroy(&PhiTe);CHKERRQ(ierr);
    ierr = MatDestroy(&PhiT);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&Phie);CHKERRQ(ierr);
  ierr = MatDestroy(&Phi);CHKERRQ(ierr);

  /* check gradient and hessian with Tao */
  if (testtao) {
    Tao tao;
    Vec X;

    ierr = VecDuplicate(M,&X);CHKERRQ(ierr);
    ierr = VecSetRandom(X,NULL);CHKERRQ(ierr);
    ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
    ierr = TaoSetInitialVector(tao,X);CHKERRQ(ierr);
    ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);
    ierr = TaoSetObjectiveRoutine(tao,FormFunction,&opt);CHKERRQ(ierr);
    ierr = TaoSetGradientRoutine(tao,FormGradient,&opt);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&H);CHKERRQ(ierr);
    ierr = TaoSetHessianRoutine(tao,H,H,FormFunctionHessian,&opt);CHKERRQ(ierr);
    ierr = TaoComputeGradient(tao,X,G);CHKERRQ(ierr);
    ierr = TaoComputeHessian(tao,X,H,H);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
    ierr = TaoDestroy(&tao);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
  }

  /* view hessian */
  ierr = MatCreate(PETSC_COMM_SELF,&H);CHKERRQ(ierr);
  ierr = TSComputeHessian(ts,opt.t0,opt.dt,opt.tf,NULL,M,H);CHKERRQ(ierr);
  ierr = MatComputeOperator(H,NULL,&He);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nHessian:\n");CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(He,(const PetscScalar**)&g);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e\n",(double)PetscRealPart(g[0]),(double)PetscRealPart(g[2]));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e\n",(double)PetscRealPart(g[1]),(double)PetscRealPart(g[3]));CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(He,(const PetscScalar**)&g);CHKERRQ(ierr);
  if (testhessian) { /* Check symmetricity of Hessian */
    Mat HeT;

    /* dump H - H^T */
    ierr = MatTranspose(He,MAT_INITIAL_MATRIX,&HeT);CHKERRQ(ierr);
    ierr = MatAXPY(HeT,-1.0,He,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(HeT,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&HeT);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&He);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);

  /* run taylor test */
  if (testtaylor) {
    ierr = PetscOptionsSetValue(NULL,"-taylor_ts_hessian","1");CHKERRQ(ierr);
    ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);
    ierr = TSTaylorTest(ts,opt.t0,opt.dt,opt.tf,NULL,M,NULL);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&G);CHKERRQ(ierr);
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscOptFinalize();
  return ierr;
}

/*TEST

  test:
    requires: !complex !single
    suffix: 1
    args: -ts_adapt_type none -ts_type {{theta rk}separate output} -ts_trajectory_type memory -check -test_tao -tsgradient_adjoint_discrete -tao_test_gradient -tshessian_foadjoint_discrete -tshessian_tlm_discrete -tshessian_soadjoint_discrete -test_taylor -tlm_discrete -adjoint_tlm_discrete -test_tlm -test_hessian

  test:
    requires: !complex !single
    suffix: 2
    args: -ts_adapt_type none -ts_type {{theta rk}separate output} -ts_trajectory_type memory -check -test_tao -tsgradient_adjoint_discrete -tao_test_hessian -tshessian_foadjoint_discrete -tshessian_tlm_discrete -tshessian_soadjoint_discrete -test_taylor -tlm_discrete -adjoint_tlm_discrete -test_tlm -test_hessian

  test:
    requires: !complex !single
    suffix: cvode

TEST*/


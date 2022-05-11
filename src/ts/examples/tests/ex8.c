static const char help[] = "Reproduces cvsRoberts_*_dns from CVODE testuite.\n";
/*
  This example reproduces the tests in
    - https://github.com/LLNL/sundials/blob/master/examples/cvodes/serial/cvsRoberts_FSA_dns.c
    - https://github.com/LLNL/sundials/blob/master/examples/cvodes/serial/cvsRoberts_ASAi_dns.c

  Computes the sensitivity matrix (FSA test) and the gradient (ASAi) of
    dy1/dt = -p1*y1 + p2*y2*y3
    dy2/dt =  p1*y1 - p2*y2*y3 - p3*(y2)^2
    dy3/dt =  p3*(y2)^2
  with initial conditions y1 = 1.0, y2 = y3 = 0.
  The reaction rates are: p1 = 0.04, p2 = 1e4, and p3 = 3e7.

  Gradient (ASAi) of the below objective function
     int_t0^tB0 g(t,p,y) dt
     with g(t,p,y) = y3

*/
#include <petscopt.h>

/* these callbacks are used to test TLM computations for the FSA tests */
static PetscErrorCode EvalObjective_TLMTEST(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      v;

  PetscFunctionBeginUser;
  ierr = VecNorm(U,NORM_2,&v);CHKERRQ(ierr);
  *val = 0.5*v*v;
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveGradient_TLMTEST_U(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(U,grad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* these callbacks are used for the ASAi test */
static PetscErrorCode EvalObjective_ASAi(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  *val = PetscRealPart(u[2]);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveGradient_ASAi_U(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    *g;

  PetscFunctionBeginUser;
  ierr = VecGetArrayWrite(grad,&g);CHKERRQ(ierr);
  g[0] = 0.0;
  g[1] = 0.0;
  g[2] = 1.0;
  ierr = VecRestoreArrayWrite(grad,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The callbacks for the ODE using the explicit (RHS) interface u' = F(u,t) */
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
  f[0] = - m[0] * u[0] + m[1] * u[1] * u[2];
  f[1] = m[0] * u[0] - m[1] * u[1] * u[2] - m[2] * u[1] * u[1];
  f[2] = m[2] * u[1] * u[1];
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
  v[0][0] = - m[0];
  v[0][1] = m[1] * u[2];
  v[0][2] = m[1] * u[1];
  v[1][0] = m[0];
  v[1][1] = - m[1] * u[2] - 2.0 * m[2] * u[1];
  v[1][2] = - m[1] * u[1];
  v[2][0] = 0.0;
  v[2][1] = 2.0 * m[2] * u[1];
  v[2][2] = 0.0;
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

/* The callback for the parameter dependency of the ODE
   note: this callback must be always supplied in implicit form  F(u',u,t;m) = 0, that's why the signs are swapped  */
static PetscErrorCode EvalGradientDAE_M(TS ts, PetscReal t, Vec U, Vec Udot, Vec M, Mat F_M, void *ctx)
{
  const PetscScalar *u;
  const PetscScalar *m;
  PetscInt          i[3] = {0, 1, 2};
  PetscScalar       v[3][3];
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  v[0][0] = u[0];
  v[0][1] = - u[1] * u[2];
  v[0][2] = 0.0;
  v[1][0] = - u[0];
  v[1][1] = u[1] * u[2];
  v[1][2] = u[1] * u[1];
  v[2][0] = 0.0;
  v[2][1] = 0.0;
  v[2][2] = - u[1] * u[1];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  ierr = MatSetValues(F_M,3,i,3,i,(PetscScalar*)v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This is called every time the parameter space changes */
static PetscErrorCode MyTSSetUpFromDesign(TS ts, Vec u0, Vec M, void *ctx)
{
  PetscReal         *p;
  const PetscScalar *m;
  PetscScalar       *u;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayWrite(u0,&u);CHKERRQ(ierr);
  u[0] = 1.0;
  u[1] = 0.0;
  u[2] = 0.0;
  ierr = VecRestoreArrayWrite(u0,&u);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts,(void*)&p);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  p[0] = PetscRealPart(m[0]);
  p[1] = PetscRealPart(m[1]);
  p[2] = PetscRealPart(m[2]);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char* argv[])
{
  TS             ts;
  Mat            J;
  Vec            U,M,G;
  PetscScalar    *g;
  PetscReal      fstf,fatf,dt,obj;
  PetscBool      testtaylor = PETSC_FALSE, testtlm = PETSC_FALSE;
  PetscBool      check_dae = PETSC_FALSE;
  PetscReal      AppCtx[3];
  PetscReal      tlmtf = 0.4;
  PetscInt       tlmi = 0;
  PetscErrorCode ierr;

  ierr = PetscOptInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* The reaction rates */
  AppCtx[0] = 0.04;
  AppCtx[1] = 1.e4;
  AppCtx[2] = 3.e7;

  /* Command line options */
  fatf = 4.e7;
  fstf = 4.e10;
  dt   = 1.0/128.0; /* just a guess for initial time step: we use adaptive time stepping anyway */
  PetscOptionsBegin(PETSC_COMM_SELF,NULL,"","");
  ierr = PetscOptionsReal("-fstf","Final time for forward sensitivity analysis","",fstf,&fstf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-fatf","Final time for adjoint sensitivity analysis","",fatf,&fatf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","Initial time step","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check","Check Hessian DAE terms","",check_dae,&check_dae,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_taylor","Run Taylor test","",testtaylor,&testtaylor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tlm","Test Tangent Linear Model to compute the gradient","",testtlm,&testtlm,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();

  /* state vector */
  ierr = VecCreate(PETSC_COMM_SELF,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,3,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetValue(U,0,1,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(U,1,0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(U,2,0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(U);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(U);CHKERRQ(ierr);

  /* ODE solver */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,AppCtx);CHKERRQ(ierr);
  ierr = TSSetTolerances(ts,1.e-8,NULL,1.e-8,NULL);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBDF);CHKERRQ(ierr);
  ierr = TSBDFSetOrder(ts,4);CHKERRQ(ierr); /* PETSc does not have variable order BDF */
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,3,NULL,&J);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* design vector (defaults to CVODES reaction rates) */
  ierr = VecCreate(PETSC_COMM_SELF,&M);CHKERRQ(ierr);
  ierr = VecSetSizes(M,3,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(M,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetValue(M,0,AppCtx[0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(M,1,AppCtx[1],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(M,2,AppCtx[2],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
  ierr = VecDuplicate(M,&G);CHKERRQ(ierr);

  /* Supply sensitivity callbacks */

  /* set callback to setup matrix for first order ode dependence on parameters */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,3,NULL,&J);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,J,EvalGradientDAE_M,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  /* set callback to setup model solver after a design vector changes */
  ierr = TSSetSetUpFromDesign(ts,MyTSSetUpFromDesign,NULL);CHKERRQ(ierr);

  /* check DAE callbacks terms (runs FD tests on them) */
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
    ierr = VecDestroy(&Udot);CHKERRQ(ierr);
    ierr = VecDestroy(&L);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  }

  /* Evaluate objective and gradient (see https://github.com/LLNL/sundials/blob/master/examples/cvodes/serial/cvsRoberts_ASAi_dns.c) */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nRunning sensitivity tests from cvsRoberts_ASAi_dns.c\n");CHKERRQ(ierr);
  ierr = TSAddObjective(ts,PETSC_MIN_REAL,EvalObjective_ASAi,EvalObjectiveGradient_ASAi_U,NULL,
                        NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = TSComputeObjectiveAndGradient(ts,0.0,dt,fatf,NULL,M,G,&obj);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nObjective at tB=%14.6e: %14.6e\n",(double)fatf,(double)obj);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGradient (tB=%14.6e)\n",(double)fatf);CHKERRQ(ierr);
  ierr = VecGetArrayRead(G,(const PetscScalar**)&g);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[0]),(double)PetscRealPart(g[1]),(double)PetscRealPart(g[2]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(G,(const PetscScalar**)&g);CHKERRQ(ierr);
  ierr = TSComputeObjectiveAndGradient(ts,0.0,PETSC_DECIDE,5.e1,NULL,M,G,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGradient (tB=%14.6e)\n",5.e1);CHKERRQ(ierr);
  ierr = VecGetArrayRead(G,(const PetscScalar**)&g);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[0]),(double)PetscRealPart(g[1]),(double)PetscRealPart(g[2]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(G,(const PetscScalar**)&g);CHKERRQ(ierr);

  /* run taylor test if you want to check correctness of the computed gradient */
  if (testtaylor) {
    ierr = TSTaylorTest(ts,0.0,dt,fatf,NULL,M,NULL);CHKERRQ(ierr);
  }

  /* check tangent linear model, see https://github.com/LLNL/sundials/blob/master/examples/cvodes/serial/cvsRoberts_FSA_dns.c */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nRunning sensitivity tests from cvsRoberts_FSA_dns.c\n");CHKERRQ(ierr);
  while (tlmtf <= fstf) {
    Mat Phi,Phie,PhiT,PhiTe,TLMe;
    Vec T,G2;

    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    if (testtlm) { /* objective function as final state evaluation to check TLM computations correctness */
      ierr = TSResetObjective(ts);CHKERRQ(ierr);
      ierr = TSAddObjective(ts,tlmtf,EvalObjective_TLMTEST,EvalObjectiveGradient_TLMTEST_U,NULL,
                            NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

      ierr = TSComputeObjectiveAndGradient(ts,0.0,dt,tlmtf,U,M,NULL,&obj);CHKERRQ(ierr);
      ierr = VecDuplicate(G,&T);CHKERRQ(ierr);
      ierr = EvalObjectiveGradient_TLMTEST_U(U,M,tlmtf,T,NULL);CHKERRQ(ierr);
      ierr = TSComputeObjectiveAndGradient(ts,0.0,dt,tlmtf,U,M,G,NULL);CHKERRQ(ierr);
    }
    ierr = VecSetValue(U,0,1,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(U,1,0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(U,2,0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(U);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(U);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSolution and sensitivity matrix step %D tf=%14.6e\n",tlmi,(double)tlmtf);CHKERRQ(ierr);
    ierr = TSCreatePropagatorMat(ts,0.0,dt,tlmtf,U,M,NULL,&Phi);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSolution at step %D tf=%14.6e\n",tlmi,(double)tlmtf);CHKERRQ(ierr);
    ierr = VecGetArrayRead(U,(const PetscScalar**)&g);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[0]),(double)PetscRealPart(g[1]),(double)PetscRealPart(g[2]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(U,(const PetscScalar**)&g);CHKERRQ(ierr);
    ierr = MatComputeOperator(Phi,NULL,&Phie);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSensitivity matrix at step %D tf=%14.6e\n",tlmi,(double)tlmtf);CHKERRQ(ierr);
    ierr = MatDenseGetArrayRead(Phie,(const PetscScalar**)&g);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[0]),(double)PetscRealPart(g[1]),(double)PetscRealPart(g[2]));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[3]),(double)PetscRealPart(g[4]),(double)PetscRealPart(g[5]));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%14.6e %14.6e %14.6e\n",(double)PetscRealPart(g[6]),(double)PetscRealPart(g[7]),(double)PetscRealPart(g[8]));CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(Phie,(const PetscScalar**)&g);CHKERRQ(ierr);

    if (testtlm) { /* checks correctness of the TLM matrices */
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTLM matrices (Phi, Phi^T and Phi-(Phi^T)^T)\n");
      ierr = MatCreateTranspose(Phi,&PhiT);CHKERRQ(ierr);
      ierr = MatComputeOperator(PhiT,NULL,&PhiTe);CHKERRQ(ierr);
      ierr = MatComputeOperator(Phi,NULL,&Phie);CHKERRQ(ierr);
      ierr = MatView(Phie,NULL);CHKERRQ(ierr);
      ierr = MatView(PhiTe,NULL);CHKERRQ(ierr);

      ierr = MatTranspose(PhiTe,MAT_INITIAL_MATRIX,&TLMe);CHKERRQ(ierr);
      ierr = MatAXPY(TLMe,-1.0,Phie,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(TLMe,NULL);CHKERRQ(ierr);
      ierr = VecDuplicate(G,&G2);CHKERRQ(ierr);
      ierr = MatMultTranspose(Phie,T,G2);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGradient via TLM (explicit fwd via multtrans) and error \n");
      ierr = VecView(G2,NULL);CHKERRQ(ierr);
      ierr = VecAXPY(G2,-1.0,G);CHKERRQ(ierr);
      ierr = VecView(G2,NULL);CHKERRQ(ierr);

      ierr = MatMult(PhiTe,T,G2);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGradient via TLM (explicit adj via mult) and error \n");
      ierr = VecView(G2,NULL);CHKERRQ(ierr);
      ierr = VecAXPY(G2,-1.0,G);CHKERRQ(ierr);
      ierr = VecView(G2,NULL);CHKERRQ(ierr);

      ierr = MatMult(PhiT,T,G2);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGradient via TLM (via adj mult) and error \n");
      ierr = VecView(G2,NULL);CHKERRQ(ierr);
      ierr = VecAXPY(G2,-1.0,G);CHKERRQ(ierr);
      ierr = VecView(G2,NULL);CHKERRQ(ierr);

      ierr = VecDestroy(&T);CHKERRQ(ierr);
      ierr = VecDestroy(&G2);CHKERRQ(ierr);

      ierr = MatDestroy(&TLMe);CHKERRQ(ierr);
      ierr = MatDestroy(&PhiTe);CHKERRQ(ierr);
      ierr = MatDestroy(&PhiT);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Phie);CHKERRQ(ierr);
    ierr = MatDestroy(&Phi);CHKERRQ(ierr);

    tlmi++;
    tlmtf *= 10;
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
    suffix: cvode
    args: -ts_trajectory_type memory

TEST*/


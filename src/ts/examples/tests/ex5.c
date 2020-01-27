static const char help[] = "Tests state dependent mass matrix.";
/*
  This is example 6.1.2 in http://people.cs.vt.edu/~ycao/publication/adj_part2.pdf
  Computes the gradient of

    Obj(u,m) = u0 + u1

  where u = [u0,u1] obeys

     |  u0   u1 | | u0_dot |   |       0      |
     |          | |        | = |              |
     | -u1   u0 | | u1_dot |   | -u0^2 - u1^2 |
*/
#include <petscopt.h>
#include <petsctao.h>
#include <petsc/private/vecimpl.h>

static PetscErrorCode EvalObjective(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  PetscBool      paper = *((PetscBool*)ctx);
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (paper) {
    PetscScalar s;

    ierr = VecSum(U,&s);CHKERRQ(ierr);
    *val = PetscRealPart(s);
  } else {
    PetscReal v;

    ierr = VecNorm(U,NORM_2,&v);CHKERRQ(ierr);
    *val = v*v;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveGradient_U(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscBool      paper = *((PetscBool*)ctx);
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (paper) {
    ierr = VecSet(grad,1.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(U,grad);CHKERRQ(ierr);
    ierr = VecScale(grad,2.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveHessian_UU(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  PetscBool      paper = *((PetscBool*)ctx);
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  if (!paper) {
    ierr = MatShift(H,2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction(TS ts,PetscReal time,Vec U,Vec Udot,Vec F,void* ctx)
{
  const PetscScalar *u;
  const PetscScalar *udot;
  PetscScalar       *f;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  f[0] = ( u[0] * udot[0] + u[1] * udot[1]);
  f[1] = (-u[1] * udot[0] + u[0] * udot[1]) + u[0]*u[0] + u[1]*u[1];
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIJacobian(TS ts,PetscReal time,Vec U,Vec Udot,PetscReal s,Mat A,Mat P,void* ctx)
{
  const PetscScalar *u;
  const PetscScalar *udot;
  PetscErrorCode    ierr;
  PetscInt          st,i[2];
  PetscScalar       v[2][2];

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
          /* s * F_Udot */        /* F_U */
  v[0][0] = s * (  u[0]) + (  udot[0] +    0.0);
  v[0][1] = s * (  u[1]) + (  udot[1] +    0.0);
  v[1][0] = s * (- u[1]) + (  udot[1] + 2*u[0]);
  v[1][1] = s * (  u[0]) + ( -udot[0] + 2*u[1]);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(P,&st,NULL);CHKERRQ(ierr);
  i[0] = st;
  i[1] = st+1;
  ierr = MatSetValues(A,2,i,2,i,(PetscScalar*)v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (P && A != P) { /* prevent from null pivots */
    ierr = MatZeroEntries(P);CHKERRQ(ierr);
    ierr = MatSetValues(P,2,i,2,i,(PetscScalar*)v,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatShift(P,PETSC_SMALL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_UU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *x,*l;
  PetscScalar       *y;
  PetscScalar       v[2][2][2];
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  v[0][0][0] = 0.0;
  v[0][0][1] = 0.0;
  v[0][1][0] = 0.0;
  v[0][1][1] = 0.0;
  v[1][0][0] = 2.0;
  v[1][0][1] = 0.0;
  v[1][1][0] = 0.0;
  v[1][1][1] = 2.0;
  for (k=0;k<2;k++)
    for (j=0;j<2;j++)
      for (i=0;i<2;i++)
        y[j] += l[k] * v[k][j][i] * x[i];
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_UUdot(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *x,*l;
  PetscScalar       *y;
  PetscScalar       v[2][2][2];
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  v[0][0][0] =  1.0;
  v[0][0][1] =  0.0;
  v[0][1][0] =  0.0;
  v[0][1][1] =  1.0;
  v[1][0][0] =  0.0;
  v[1][0][1] =  1.0;
  v[1][1][0] = -1.0;
  v[1][1][1] =  0.0;
  for (k=0;k<2;k++)
    for (j=0;j<2;j++)
      for (i=0;i<2;i++)
        y[j] += l[k] * v[k][j][i] * x[i];
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianDAE_UdotU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *x,*l;
  PetscScalar       *y;
  PetscScalar       v[2][2][2];
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  v[0][0][0] =  1.0;
  v[0][0][1] =  0.0;
  v[0][1][0] =  0.0;
  v[0][1][1] =  1.0;
  v[1][0][0] =  0.0;
  v[1][0][1] = -1.0;
  v[1][1][0] =  1.0;
  v[1][1][1] =  0.0;
  for (k=0;k<2;k++)
    for (j=0;j<2;j++)
      for (i=0;i<2;i++)
        y[j] += l[k] * v[k][j][i] * x[i];
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This term is zero, we can also avoid setting the callback in TSSetHessianDAE */
static PetscErrorCode EvalHessianDAE_UdotUdot(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(Y,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscBool      testhistory = PETSC_FALSE, flg, check_hessian_dae = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Command line options */
  t0   = 0.0;
  tf   = 1.57;
  dt   = 1.e-3;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t0","Initial time","",t0,&t0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","Final time","",tf,&tf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","Initial time step","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check","Check Hessian DAE terms","",check_hessian_dae,&check_hessian_dae,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tao","Solve the optimization problem","",testtao,&testtao,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tlm","Test Tangent Linear Model to compute the gradient","",testtlm,&testtlm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_taylor","Run Taylor test","",testtaylor,&testtaylor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_history","Run objective using the initially generated history","",testhistory,&testhistory,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-paper","Use objective from the paper","",paper,&paper,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* state vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,2,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);

  /* ODE solver */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetTolerances(ts,1.e-8,NULL,1.e-8,NULL);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetType(J,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(J,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,2,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatDuplicate(J,MAT_DO_NOT_COPY_VALUES,&Jp);CHKERRQ(ierr);
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
  ierr = EvalObjective(Uobj,M,opt.tf,&objnull,&paper);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nFinal state\n");CHKERRQ(ierr);
  ierr = VecView(Uobj,NULL);CHKERRQ(ierr);

  /* sensitivity callbacks */
  ierr = MatCreate(PETSC_COMM_WORLD,&H);CHKERRQ(ierr);
  ierr = MatSetSizes(H,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(H);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* objective function as final state sampling */
  ierr = TSAddObjective(ts,opt.tf,EvalObjective,EvalObjectiveGradient_U,NULL,
                                H,EvalObjectiveHessian_UU,NULL,NULL,NULL,NULL,(void*)&paper);CHKERRQ(ierr);
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
  if (check_hessian_dae) {
    Vec L,Udot;

    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&L);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&Udot);CHKERRQ(ierr);
    ierr = VecSetRandom(U,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(Udot,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(L,NULL);CHKERRQ(ierr);
    ierr = TSCheckHessianDAE(ts,0.0,U,Udot,M,L);CHKERRQ(ierr);
    ierr = VecDestroy(&Udot);CHKERRQ(ierr);
    ierr = VecDestroy(&L);CHKERRQ(ierr);
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
    ierr = EvalObjectiveGradient_U(Uobj,M,opt.tf,T,&paper);CHKERRQ(ierr);
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
    ierr = TaoSetInitialVector(tao,X);CHKERRQ(ierr);
    ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);
    ierr = TaoSetObjectiveRoutine(tao,FormFunction,&opt);CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&opt);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&H);CHKERRQ(ierr);
    ierr = TaoSetHessianRoutine(tao,H,H,FormFunctionHessian,&opt);CHKERRQ(ierr);
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
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    requires: !complex !single
    suffix: paper_discrete
    args: -paper -ts_trajectory_type memory -check -test_tao -test_tlm -tsgradient_adjoint_discrete -tao_test_gradient -tlm_discrete -adjoint_tlm_discrete -tshessian_foadjoint_discrete -tshessian_tlm_discrete -tshessian_soadjoint_discrete -tshessian_view

  test:
    requires: !complex !single
    suffix: hessian_discrete
    args: -paper 0 -ts_trajectory_type memory -check -test_tao -test_tlm -tsgradient_adjoint_discrete -tao_test_hessian -tshessian_foadjoint_discrete -tshessian_tlm_discrete -tshessian_soadjoint_discrete -tlm_discrete -adjoint_tlm_discrete -ts_rtol 1.e-4 -ts_atol 1.e-4 -test_history {{0 1}separate output} -tshessian_view

TEST*/


static const char help[] = "Tests second order initial condition dependence.";
/*
  Computes the gradient of

    Obj(u,m) = 1/2 ||u(T)||^2

  where u = [x1,x2] obeys the Lotka-Volterra type of ODEs (see https://pubs.acs.org/doi/pdf/10.1021/j100239a032)

      udot = F(u)

      F(u) = | k1 * x1 - k4 * x1^2 - k2 * x1 * x2 + k5 * x2^2 |
             | k2 * x1 * x2 - k5 * x2^2 - k3 * x2             |

      G(u(0),m) = | e^m1 * x1 - e^m2 * x2^2 |
                  | x2 - e^m3               |

  This code tests second order dependence of initial conditions wrt parameters.
*/
#include <petscopt.h>
#include <petsctao.h>
#include <petsc/private/vecimpl.h>

static PetscErrorCode EvalObjective(Vec U, Vec M, PetscReal time, PetscReal *val, void *ctx)
{
  Vec               Uobj = (Vec)ctx;
  const PetscScalar *u,*uobj;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Uobj,&uobj);CHKERRQ(ierr);
  *val = PetscRealPart(0.5*((u[0]-uobj[0])*(u[0]-uobj[0]) + (u[1]-uobj[1])*(u[1]-uobj[1])));
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Uobj,&uobj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveGradient_U(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  Vec            Uobj = (Vec)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(grad,-1.0,Uobj,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjectiveHessian_UU(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  ierr = MatShift(H,1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscScalar k[5];
} UserDAE;

static PetscErrorCode EvalHessianDAE_UU(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  UserDAE           *user = (UserDAE*)ctx;
  const PetscScalar *x,*l;
  PetscScalar       *y;
  PetscScalar       k2 = user->k[1];
  PetscScalar       k4 = user->k[3];
  PetscScalar       k5 = user->k[4];
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  y[0] = l[0] * (-2.0 * k4 * x[0] -       k2 * x[1]) + l[1] * (0.0 * x[0] +       k2 * x[1]);
  y[1] = l[0] * (      -k2 * x[0] + 2.0 * k5 * x[1]) + l[1] * ( k2 * x[0] - 2.0 * k5 * x[1]);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecScale(Y,-1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalGradientIC(TS ts, PetscReal t0, Vec u0, Vec M, Mat G_u0, Mat G_m, void *ctx)
{
  const PetscScalar *u,*m;
  PetscScalar       v[6];
  PetscInt          ii[3] = {0,1,2};
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(u0,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);

  ierr = MatZeroEntries(G_u0);CHKERRQ(ierr);
  v[0] = PetscExpScalar(m[0]);
  v[1] = -2.0 * u[1] * PetscExpScalar(m[1]);
  v[2] = 0.0;
  v[3] = 1.0;
  ierr = MatSetValues(G_u0,2,ii,2,ii,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G_u0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_u0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatZeroEntries(G_m);CHKERRQ(ierr);
  v[0] = u[0] * PetscExpScalar(m[0]);
  v[1] = - u[1] * u[1] * PetscExpScalar(m[1]);
  v[2] = 0.0;
  v[3] = 0.0;
  v[4] = 0.0;
  v[5] = - PetscExpScalar(m[2]);
  ierr = MatSetValues(G_m,2,ii,3,ii,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G_m,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_m,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(u0,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianIC_UU(TS ts, PetscReal time, Vec U, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *u,*m,*l,*x;
  PetscScalar       *y,em2;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);

  em2   = PetscExpScalar(m[1]);
  y[0]  = l[0] * (0 * x[0] +       0 * x[1]);
  y[1]  = l[0] * (0 * x[0] - 2 * em2 * x[1]);
  y[0] += l[1] * (0 * x[0] +       0 * x[1]);
  y[1] += l[1] * (0 * x[0] +       0 * x[1]);

  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianIC_MM(TS ts, PetscReal time, Vec U, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *u,*m,*l,*x;
  PetscScalar       *y,em1,em2,em3;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);

  em1   = PetscExpScalar(m[0]);
  em2   = PetscExpScalar(m[1]);
  em3   = PetscExpScalar(m[2]);
  y[0]  = l[0] * (em1 * u[0] * x[0] +                 0 * x[1] +   0 * x[2]);
  y[1]  = l[0] * (         0 * x[0] - em2 * u[1] * u[1] * x[1] +   0 * x[2]);
  y[2]  = l[0] * (         0 * x[0] +                 0 * x[1] +   0 * x[2]);
  y[0] += l[1] * (         0 * x[0] +                 0 * x[1] +   0 * x[2]);
  y[1] += l[1] * (         0 * x[0] +                 0 * x[1] +   0 * x[2]);
  y[2] += l[1] * (         0 * x[0] +                 0 * x[1] - em3 * x[2]);

  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalHessianIC_MU(TS ts, PetscReal time, Vec U, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *u,*m,*l,*x;
  PetscScalar       *y,em1,em2;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);

  em1   = PetscExpScalar(m[0]);
  em2   = PetscExpScalar(m[1]);
  y[0]  = l[0] * (em1 * x[0] +                0 * x[1]);
  y[1]  = l[0] * (  0 * x[0] - 2.0 * em2 * u[1] * x[1]);
  y[2]  = l[0] * (  0 * x[0] +                0 * x[1]);
  y[0] += l[1] * (  0 * x[0] +                0 * x[1]);
  y[1] += l[1] * (  0 * x[0] +                0 * x[1]);
  y[2] += l[1] * (  0 * x[0] +                0 * x[1]);

  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode EvalHessianIC_UM(TS ts, PetscReal time, Vec U, Vec M, Vec L, Vec X, Vec Y, void *ctx)
{
  const PetscScalar *u,*m,*l,*x;
  PetscScalar       *y,em1,em2;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);

  em1   = PetscExpScalar(m[0]);
  em2   = PetscExpScalar(m[1]);
  y[0]  = l[0] * (em1 * x[0] +                0 * x[1] + 0 * x[2]);
  y[1]  = l[0] * (  0 * x[0] - 2.0 * em2 * u[1] * x[1] + 0 * x[2]);
  y[0] += l[1] * (  0 * x[0] +                0 * x[1] + 0 * x[2]);
  y[1] += l[1] * (  0 * x[0] +                0 * x[1] + 0 * x[2]);

  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal time, Vec U, Vec F,void* ctx)
{
  UserDAE           *user = (UserDAE*)ctx;
  PetscScalar       k1 = user->k[0];
  PetscScalar       k2 = user->k[1];
  PetscScalar       k3 = user->k[2];
  PetscScalar       k4 = user->k[3];
  PetscScalar       k5 = user->k[4];
  const PetscScalar *u;
  PetscScalar       *f;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  f[0] = k1 * u[0]        - k4 * u[0] * u[0] - k2 * u[0] * u[1] + k5 * u[1] * u[1];
  f[1] = k2 * u[0] * u[1] - k5 * u[1] * u[1] - k3 * u[1];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian(TS ts,PetscReal time, Vec U, Mat A, Mat P, void* ctx)
{
  UserDAE           *user = (UserDAE*)ctx;
  PetscScalar       k1 = user->k[0];
  PetscScalar       k2 = user->k[1];
  PetscScalar       k3 = user->k[2];
  PetscScalar       k4 = user->k[3];
  PetscScalar       k5 = user->k[4];
  PetscScalar       v[4];
  PetscInt          ii[2] = {0,1};
  const PetscScalar *u;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(P);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  v[0] =  k1 - 2.0 * k4 * u[0] -       k2 * u[1];
  v[1] =           - k2 * u[0] + 2.0 * k5 * u[1];
  v[2] =                               k2 * u[1];
  v[3] = -k3 +       k2 * u[0] - 2.0 * k5 * u[1];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = MatSetValues(P,2,ii,2,ii,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A && A != P) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ResidualIC(Vec U, Vec M, Vec F)
{
  const PetscScalar *m;
  const PetscScalar *u;
  PetscScalar       *f;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  f[0] = PetscExpScalar(m[0]) * u[0] - PetscExpScalar(m[1]) * u[1] * u[1];
  f[1] = u[1] - PetscExpScalar(m[2]);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyTSSetUpFromDesign(TS ts, Vec u0, Vec M, void *ctx)
{
  const PetscScalar *m;
  PetscScalar       *u;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(u0,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,&m);CHKERRQ(ierr);
  u[0] = PetscExpScalar(m[1] + 2.0*m[2] - m[0]);
  u[1] = PetscExpScalar(m[2]);
  ierr = VecRestoreArrayRead(M,&m);CHKERRQ(ierr);
  ierr = VecRestoreArray(u0,&u);CHKERRQ(ierr);
#if 0
  Vec F;
  ierr = VecDuplicate(u0,&F);CHKERRQ(ierr);
  ierr = ResidualIC(u0,M,F);CHKERRQ(ierr);
  ierr = VecView(F,NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode TSCheckGradientIC_U(void *ctx,Vec U, Vec F)
{
  Vec            M = (Vec)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = ResidualIC(U,M,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSCheckGradientIC_M(void *ctx,Vec M, Vec F)
{
  Vec            U = (Vec)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = ResidualIC(U,M,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  TS        ts;
  PetscReal t0,dt,tf;
} TaoCtx;

static PetscErrorCode FormFunctionHessian(Tao tao, Vec M, Mat H, Mat Hpre, void *ctx)
{
  TaoCtx         *tctx = (TaoCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSComputeHessian(tctx->ts,tctx->t0,tctx->dt,tctx->tf,NULL,M,H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormFunctionGradient(Tao tao,Vec M,PetscReal *obj,Vec G,void *ctx)
{
  TaoCtx         *tctx = (TaoCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSComputeObjectiveAndGradient(tctx->ts,tctx->t0,tctx->dt,tctx->tf,NULL,M,G,obj);CHKERRQ(ierr);
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
  TS             ts;
  UserDAE        userdae;
  Mat            J,H,G_M,G_U0;
  Vec            U,Uobj,M,Tsol;
  PetscScalar    k[5];
  PetscReal      t0,tf,dt;
  PetscBool      flg, testmffdic = PETSC_FALSE, testtao = PETSC_FALSE, testtlm = PETSC_FALSE, testtaylor = PETSC_FALSE, testtaylorgn = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Command line options */
  t0   =  0.0;
  tf   = 25.0;
  k[0] =  1.0;
  k[1] =  1.0;
  k[2] =  1.0;
  k[3] =  0.0;
  k[4] =  0.0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Lotka-Volterra parameters","");
  ierr = PetscOptionsScalar("-k1","k1","",k[0],&k[0],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k2","k2","",k[1],&k[1],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k3","k3","",k[2],&k[2],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k4","k4","",k[3],&k[3],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k5","k5","",k[4],&k[4],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t0","Initial time","",t0,&t0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","Final time","",tf,&tf,NULL);CHKERRQ(ierr);
  dt   = (tf-t0)/512.0;
  ierr = PetscOptionsReal("-dt","Initial time step","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tao","Solve the optimization problem","",testtao,&testtao,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_tlm","Test Tangent Linear Model to compute the gradient","",testtlm,&testtlm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_mffd_ic","Run MFFD tests on IC callbacks","",testmffdic,&testmffdic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_taylor","Run Taylor test","",testtaylor,&testtaylor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_taylor_gn","Run Taylor test (use tao solution)","",testtaylorgn,&testtaylorgn,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (testtaylorgn) { testtaylor = PETSC_TRUE; testtao = PETSC_TRUE; }

  /* context for residual callbacks */
  userdae.k[0] = k[0];
  userdae.k[1] = k[1];
  userdae.k[2] = k[2];
  userdae.k[3] = k[3];
  userdae.k[4] = k[4];

  /* state vectors */
  ierr = VecCreate(PETSC_COMM_SELF,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,2,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);

  /* ODE solver */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&userdae);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,&userdae);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);

  /* initial condition as for figure 1 in reference
     run with -ts_monitor_draw_solution_phase 0,0,2,2 */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecSet(U,0.5);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nInitial state\n");CHKERRQ(ierr);
  ierr = VecView(U,NULL);CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  /* store final time and state */
  ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Uobj);CHKERRQ(ierr);
  ierr = VecCopy(U,Uobj);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nFinal state\n");CHKERRQ(ierr);
  ierr = VecView(Uobj,NULL);CHKERRQ(ierr);

  /* design vectors */
  ierr = VecCreate(PETSC_COMM_SELF,&M);CHKERRQ(ierr);
  ierr = VecSetSizes(M,3,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(M,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);

  /* sensitivity callbacks */
  ierr = MatCreate(PETSC_COMM_SELF,&H);CHKERRQ(ierr);
  ierr = MatSetSizes(H,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(H);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = TSAddObjective(ts,tf,EvalObjective,EvalObjectiveGradient_U,NULL,
                        H,EvalObjectiveHessian_UU,NULL,NULL,NULL,NULL,Uobj);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);

  ierr = TSSetSetUpFromDesign(ts,MyTSSetUpFromDesign,NULL);CHKERRQ(ierr);

  ierr = TSSetHessianDAE(ts,EvalHessianDAE_UU,NULL,NULL,
                            NULL,             NULL,NULL,
                            NULL,             NULL,NULL,
                            &userdae);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&G_U0);CHKERRQ(ierr);
  ierr = MatSetSizes(G_U0,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(G_U0,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(G_U0);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G_U0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_U0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&G_M);CHKERRQ(ierr);
  ierr = MatSetSizes(G_M,2,3,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(G_M,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(G_M);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = TSSetGradientIC(ts,G_U0,G_M,EvalGradientIC,NULL);CHKERRQ(ierr);
  ierr = TSSetHessianIC(ts,EvalHessianIC_UU,EvalHessianIC_UM,EvalHessianIC_MU,EvalHessianIC_MM,NULL);CHKERRQ(ierr);

  /* test residual for initial condition */
  if (testmffdic) {
    Mat H,He;
    Vec U,L;

    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = VecSetRandom(U,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);
    ierr = EvalGradientIC(ts,0,U,M,G_U0,G_M,NULL);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_SELF,&H);CHKERRQ(ierr);
    ierr = MatSetSizes(H,2,2,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(H,MATMFFD);CHKERRQ(ierr);
    ierr = MatSetUp(H);CHKERRQ(ierr);
    ierr = MatMFFDSetBase(H,U,NULL);CHKERRQ(ierr);
    ierr = MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))TSCheckGradientIC_U,M);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatComputeOperator(H,NULL,&He);CHKERRQ(ierr);
    ierr = MatView(He,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&He);CHKERRQ(ierr);
    ierr = MatComputeOperator(G_U0,NULL,&He);CHKERRQ(ierr);
    ierr = MatView(He,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
    ierr = MatDestroy(&He);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_SELF,&H);CHKERRQ(ierr);
    ierr = MatSetSizes(H,2,3,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(H,MATMFFD);CHKERRQ(ierr);
    ierr = MatSetUp(H);CHKERRQ(ierr);
    ierr = MatMFFDSetBase(H,M,NULL);CHKERRQ(ierr);
    ierr = MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))TSCheckGradientIC_M,U);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatComputeOperator(H,NULL,&He);CHKERRQ(ierr);
    ierr = MatView(He,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&He);CHKERRQ(ierr);
    ierr = MatComputeOperator(G_M,NULL,&He);CHKERRQ(ierr);
    ierr = MatView(He,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
    ierr = MatDestroy(&He);CHKERRQ(ierr);

    ierr = VecDuplicate(U,&L);CHKERRQ(ierr);
    ierr = VecSetRandom(L,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(U,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);
    ierr = TSCheckHessianIC(ts,0.0,U,M,L);CHKERRQ(ierr);
    ierr = TSCheckHessianDAE(ts,0.0,U,U,M,L);CHKERRQ(ierr);
    ierr = VecDestroy(&L);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&G_M);CHKERRQ(ierr);
  ierr = MatDestroy(&G_U0);CHKERRQ(ierr);

  /* Reconstruct solution */
  Tsol = NULL;
  if (testtao) {
    Tao    tao;
    TaoCtx taoctx;
    Vec    X;

    taoctx.ts = ts;
    taoctx.t0 = t0;
    taoctx.tf = tf;
    taoctx.dt = dt;

    ierr = VecDuplicate(M,&X);CHKERRQ(ierr);
    ierr = VecSet(X,0.0);CHKERRQ(ierr);
    ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
    ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);
    ierr = TaoSetObjectiveRoutine(tao,FormFunction,(void *)&taoctx);CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&taoctx);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&H);CHKERRQ(ierr);
    ierr = TaoSetHessianRoutine(tao,H,H,FormFunctionHessian,(void *)&taoctx);CHKERRQ(ierr);
    ierr = TaoSetInitialVector(tao,X);CHKERRQ(ierr);
    ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
    ierr = TaoSolve(tao);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nTao solution\n");CHKERRQ(ierr);
    ierr = VecView(X,NULL);CHKERRQ(ierr);
    ierr = MyTSSetUpFromDesign(NULL,U,X,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nInitial state solution\n");CHKERRQ(ierr);
    ierr = VecView(U,NULL);CHKERRQ(ierr);

    ierr = TaoDestroy(&tao);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
    if (testtaylorgn) {
      ierr = VecDuplicate(X,&Tsol);CHKERRQ(ierr);
      ierr = VecCopy(X,Tsol);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&X);CHKERRQ(ierr);
  }

  /*
     Test Tangent linear model to compute the gradient
     This is a final time objective, so we can compute the gradient as

            Phi^T * d_obj / d_u

     Phi the propagator matrix (i.e. du/dm, the solution of the Tangent linear model)
  */
  if (testtlm) {
    Mat       Phi,Phie,PhiT,PhiTe,TLMe;
    Vec       UU,T;
    PetscReal obj;

    ierr = VecDuplicate(U,&UU);CHKERRQ(ierr);
    ierr = VecDuplicate(M,&T);CHKERRQ(ierr);

    ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);
    ierr = TSComputeObjectiveAndGradient(ts,t0,dt,tf,U,M,T,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nGradient via adjoint\n");
    ierr = VecView(T,NULL);CHKERRQ(ierr);

    /* recompute so that U will contain the state at time tf */
    ierr = TSComputeObjectiveAndGradient(ts,t0,dt,tf,U,M,NULL,&obj);CHKERRQ(ierr);
    ierr = EvalObjectiveGradient_U(U,M,tf,UU,Uobj);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nGradient of objective function\n");
    ierr = VecView(UU,NULL);CHKERRQ(ierr);

    ierr = TSCreatePropagatorMat(ts,t0,dt,tf,U,M,NULL,&Phi);CHKERRQ(ierr);
    ierr = MatMultTranspose(Phi,UU,T);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nGradient via adjoint (TLM)\n");
    ierr = VecView(T,NULL);CHKERRQ(ierr);

    ierr = MatComputeOperator(Phi,NULL,&Phie);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nTLM matrices (Phi, Phi^T and Phi-(Phi^T)^T)\n");
    ierr = MatView(Phie,NULL);CHKERRQ(ierr);

    ierr = MatCreateTranspose(Phi,&PhiT);CHKERRQ(ierr);
    ierr = MatComputeOperator(PhiT,NULL,&PhiTe);CHKERRQ(ierr);
    ierr = MatView(PhiTe,NULL);CHKERRQ(ierr);

    ierr = MatTranspose(PhiTe,MAT_INITIAL_MATRIX,&TLMe);CHKERRQ(ierr);
    ierr = MatAXPY(TLMe,-1.0,Phie,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(TLMe,NULL);CHKERRQ(ierr);

    ierr = MatMultTranspose(Phie,UU,T);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nGradient via TLM (explicit fwd via multtrans)\n");
    ierr = VecView(T,NULL);CHKERRQ(ierr);

    ierr = MatMult(PhiTe,UU,T);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nGradient via TLM (explicit adj via mult)\n");
    ierr = VecView(T,NULL);CHKERRQ(ierr);

    ierr = MatDestroy(&TLMe);CHKERRQ(ierr);
    ierr = MatDestroy(&Phie);CHKERRQ(ierr);
    ierr = MatDestroy(&Phi);CHKERRQ(ierr);
    ierr = MatDestroy(&PhiTe);CHKERRQ(ierr);
    ierr = MatDestroy(&PhiT);CHKERRQ(ierr);
    ierr = VecDestroy(&UU);CHKERRQ(ierr);
    ierr = VecDestroy(&T);CHKERRQ(ierr);
  }

  /* Run taylor test */
  if (testtaylor) {
    if (Tsol) {
      ierr = VecCopy(Tsol,M);CHKERRQ(ierr);
    } else {
      ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsHasName(NULL,NULL,"-tshessian_view",&flg);CHKERRQ(ierr);
    if (flg) {
      Mat He,H;

      ierr = MatCreate(PETSC_COMM_SELF,&H);CHKERRQ(ierr);
      ierr = TSComputeHessian(ts,t0,dt,tf,NULL,M,H);CHKERRQ(ierr);
      ierr = MatComputeOperator(H,MATAIJ,&He);CHKERRQ(ierr);
      ierr = MatConvert(He,MATAIJ,MAT_INPLACE_MATRIX,&He);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)He,"H");CHKERRQ(ierr);
      ierr = MatViewFromOptions(He,NULL,"-tshessian_view");CHKERRQ(ierr);
      ierr = MatDestroy(&He);CHKERRQ(ierr);
      ierr = MatDestroy(&H);CHKERRQ(ierr);
    }
    ierr = TSTaylorTest(ts,t0,dt,tf,NULL,M,NULL);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&Tsol);CHKERRQ(ierr);
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = VecDestroy(&Uobj);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

test:
    requires: !complex !single
    suffix: 1
    args: -ts_rk_type 3bs -ts_adapt_type dsp -ts_atol 1.e-8 -ts_rtol 1.e-8 -ts_trajectory_type memory -tao_monitor -test_tao -tf 1 -test_tlm -tsgradient_adjoint_ts_adapt_type history -test_taylor -taylor_ts_hessian

test:
    requires: !complex !single
    suffix: 2
    args: -test_mffd_ic -ts_type cn -dt 1.e-2 -ts_adapt_type none -ts_trajectory_type memory -tao_monitor -test_tao -test_tlm -tf 1 -tshessian_view -tshessian_mffd {{0 1}separate output} -test_taylor -taylor_ts_hessian

test:
    requires: !complex !single
    suffix: 3
    args: -ts_type theta -dt 1.e-2 -ts_adapt_type none -ts_trajectory_type memory -tao_monitor -test_tao -tao_type nls -tao_nls_pc_type none  -test_tlm 0 -tf 1 -tshessian_gn {{0 1}separate output} -test_taylor_gn -test_taylor -taylor_ts_hessian -tshessian_view

test:
    requires: !complex !single
    suffix: discrete
    args: -ts_type rk -ts_rk_type {{1fe 2a 3 3bs 4 5f 5dp 5bs 6vr 7vr 8vr}separate output} -ts_adapt_type none -ts_trajectory_type memory -tao_monitor -test_tao -test_tlm -tlm_discrete -adjoint_tlm_discrete -t0 0 -tf 1.e-1 -dt 1.e-3  -tsgradient_adjoint_discrete -tshessian_tlm_discrete -tshessian_soadjoint_discrete -tshessian_foadjoint_discrete -jactsic_pc_type lu -test_taylor -taylor_ts_hessian -taylor_ts_steps 8 -tshessian_view -tao_test_gradient

test:
    requires: !complex !single
    suffix: discrete_adapt
    args: -ts_type rk -ts_rk_type {{2a 3bs 5f 5dp 5bs 6vr 7vr 8vr}separate output} -ts_adapt_type {{dsp basic}separate output} -ts_atol 1.e-6 -ts_rtol 1.e-6 -ts_trajectory_type memory -tao_monitor -test_tao -tao_type nls -tao_nls_pc_type none -test_tlm -t0 0 -tf 0.1 -dt 1.e-2  -tsgradient_adjoint_discrete -tshessian_tlm_discrete -tshessian_soadjoint_discrete -tshessian_foadjoint_discrete -jactsic_pc_type lu -test_taylor -taylor_ts_hessian -taylor_ts_steps 6 -tshessian_view -tao_test_hessian

test:
    requires: !complex !single
    suffix: discrete_cn
    args: -ts_type cn -ts_adapt_type none -ts_trajectory_type memory -tao_monitor -test_tao -test_tlm -tlm_discrete -adjoint_tlm_discrete -t0 0 -tf 1.e-1 -dt 1.e-3  -tsgradient_adjoint_discrete -test_taylor -taylor_ts_hessian -taylor_ts_steps 8 -tshessian_view -tao_test_gradient -tshessian_mffd

TEST*/

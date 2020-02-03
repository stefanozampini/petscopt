static const char help[] = "Demonstrates the use of TSCreatePropagatorMat.";

#include <petscopt.h>

/*
  This example tests the creation of the propagator matrix for 2 model problems:

  - Lorentz

    |dxdt|   | -s  s  0 | |x|
    |dydt| = |  r -1 -x | |y|
    |dzdt|   |  y  0 -b | |z|

  - Oscillator (w = omega^2, g = 2*gamma)

    |dxdt|   |  0  1 | |x|
    |dydt| = | -w  g | |y|
*/

typedef struct {
  PetscReal b;
  PetscReal s;
  PetscReal r;
} User_Lorentz;

typedef struct {
  PetscReal omega;
  PetscReal gamma;
} User_Osc;

static PetscErrorCode FormRHSFunction_Osc(TS ts,PetscReal time, Vec U, Vec F,void* ctx)
{
  User_Osc       *user = (User_Osc*)ctx;
  PetscScalar    w = user->omega*user->omega, g = 2*user->gamma;
  PetscScalar    *a,x,y;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,(const PetscScalar**)&a);CHKERRQ(ierr);
  x    = a[0];
  y    = a[1];
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&a);CHKERRQ(ierr);
  ierr = VecGetArray(F,&a);CHKERRQ(ierr);
  a[0] = 0. * x + 1. * y;
  a[1] = -w * x -  g * y;
  ierr = VecRestoreArray(F,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian_Osc(TS ts,PetscReal time, Vec U, Mat A, Mat P, void* ctx)
{
  User_Osc       *user = (User_Osc*)ctx;
  PetscScalar    w = user->omega*user->omega, g = 2*user->gamma;
  PetscScalar    val[2][2];
  PetscInt       idx[2],rst;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  val[0][0] = 0;  val[0][1] = 1;
  val[1][0] = -w; val[1][1] = -g;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rst,NULL);CHKERRQ(ierr);
  idx[0] = rst;
  idx[1] = rst + 1;
  ierr = MatSetValues(A,2,idx,2,idx,(const PetscScalar*)val,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction_Osc(TS ts,PetscReal time, Vec U, Vec Udot, Vec F,void* ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = FormRHSFunction_Osc(ts,time,U,F,ctx);CHKERRQ(ierr);
  ierr = VecAYPX(F,-1.0,Udot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIJacobian_Osc(TS ts,PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat P, void* ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = FormRHSJacobian_Osc(ts,time,U,A,P,ctx);CHKERRQ(ierr);
  ierr = MatScale(A,-1.0);CHKERRQ(ierr);
  ierr = MatShift(A,shift);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction_Lorentz(TS ts,PetscReal time, Vec U, Vec F,void* ctx)
{
  User_Lorentz   *user = (User_Lorentz*)ctx;
  PetscScalar    s = user->s,r = user->r,b = user->b;
  PetscScalar    *a,x,y,z;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,(const PetscScalar**)&a);CHKERRQ(ierr);
  x    = a[0];
  y    = a[1];
  z    = a[2];
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&a);CHKERRQ(ierr);
  ierr = VecGetArray(F,&a);CHKERRQ(ierr);
  a[0] = -s * x +  s * y + 0. * z;
  a[1] =  r * x - 1. * y -  x * z;
  a[2] =  y * x + 0. * y -  b * z;
  ierr = VecRestoreArray(F,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian_Lorentz(TS ts,PetscReal time, Vec U, Mat A, Mat P, void* ctx)
{
  User_Lorentz   *user = (User_Lorentz*)ctx;
  PetscScalar    s = user->s,r = user->r,b = user->b;
  PetscScalar    *a,x,y,z,val[3][3];
  PetscInt       idx[3],rst;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,(const PetscScalar**)&a);CHKERRQ(ierr);
  x    = a[0];
  y    = a[1];
  z    = a[2];
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&a);CHKERRQ(ierr);

  val[0][0] = -s;  val[0][1] = s;  val[0][2] = 0;
  val[1][0] = r-z; val[1][1] = -1; val[1][2] = -x;
  val[2][0] = y;   val[2][1] = x;  val[2][2] = -b;

  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rst,NULL);CHKERRQ(ierr);
  idx[0] = rst;
  idx[1] = rst + 1;
  idx[2] = rst + 2;
  ierr = MatSetValues(A,3,idx,3,idx,(const PetscScalar*)val,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction_Lorentz(TS ts,PetscReal time, Vec U, Vec Udot, Vec F,void* ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = FormRHSFunction_Lorentz(ts,time,U,F,ctx);CHKERRQ(ierr);
  ierr = VecAYPX(F,-1.0,Udot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIJacobian_Lorentz(TS ts,PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat P, void* ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = FormRHSJacobian_Lorentz(ts,time,U,A,P,ctx);CHKERRQ(ierr);
  ierr = MatScale(A,-1.0);CHKERRQ(ierr);
  ierr = MatShift(A,shift);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char* argv[])
{
  TS             ts;
  Mat            J,Phi,PhiExpl,PhiT,PhiTExpl,H,P = NULL;
  Vec            U,U0;
  User_Lorentz   luser;
  User_Osc       ouser;
  PetscBool      ifunc = PETSC_FALSE;
  PetscReal      t0 = 0.0, tf = 0.1, dt = 0.001;
  PetscInt       N = 3;
  PetscReal      err,params[3],normPhi,omega = 3,gamma = 2;
  PetscBool      use_lorentz = PETSC_FALSE, testP = PETSC_FALSE;
  TSProblemType  ptype;
  PetscErrorCode ierr;

  ierr = PetscOptInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  params[0] = 8./3.; 
  params[1] = 10;
  params[2] = 28;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Testing options","");
  ierr = PetscOptionsBool("-lorentz","Use lorentz","",use_lorentz,&use_lorentz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-lorentz_params",NULL,NULL,params,&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-omega","Omega","",omega,&omega,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-gamma","Gamma","",gamma,&gamma,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t0","Initial time","",t0,&t0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","Initial time step","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","Final time","",tf,&tf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ifunc","Use ifunction","",ifunc,&ifunc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-testp","Test projector argument","",testP,&testP,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (use_lorentz) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Using Lorentz parameters: %g %g %g\n",(double)params[0],(double)params[1],(double)params[2]);CHKERRQ(ierr);
    N = 3;
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Using Oscillator with Omega and Gamma: %g %g\n",(double)omega,(double)gamma);CHKERRQ(ierr);
    N = 2;
  }
  luser.b = params[0];
  luser.s = params[1];
  luser.r = params[2];
  ouser.omega = omega;
  ouser.gamma = gamma;

  /* state vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,N,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&U0);CHKERRQ(ierr);

  /* jacobian */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,N,N,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(J,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,N,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,N,NULL,0,NULL);CHKERRQ(ierr);

  /* TS solver */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,PETSC_MAX_INT);CHKERRQ(ierr);
  if (ifunc) {
    ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
    if (use_lorentz) {
      ierr = TSSetIFunction(ts,NULL,FormIFunction_Lorentz,&luser);CHKERRQ(ierr);
      ierr = TSSetIJacobian(ts,J,J,FormIJacobian_Lorentz,&luser);CHKERRQ(ierr);
      ierr = FormIJacobian_Lorentz(ts,0.0,U,U,1.0,J,J,&luser);CHKERRQ(ierr);
    } else {
      ierr = TSSetIFunction(ts,NULL,FormIFunction_Osc,&ouser);CHKERRQ(ierr);
      ierr = TSSetIJacobian(ts,J,J,FormIJacobian_Osc,&ouser);CHKERRQ(ierr);
      ierr = FormIJacobian_Osc(ts,0.0,U,U,1.0,J,J,&ouser);CHKERRQ(ierr);
    }
  } else {
    if (use_lorentz) {
      ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction_Lorentz,&luser);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian_Lorentz,&luser);CHKERRQ(ierr);
      ierr = FormRHSJacobian_Lorentz(ts,0.0,U,J,J,&luser);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction_Osc,&ouser);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian_Osc,&ouser);CHKERRQ(ierr);
      ierr = FormRHSJacobian_Osc(ts,0.0,U,J,J,&ouser);CHKERRQ(ierr);
    }
  }
  if (!use_lorentz) {
    ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* just test the code path */
  if (testP) {
    ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr);
    ierr = MatSetSizes(P,N,N,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(P,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetUp(P);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecSetRandom(U0,NULL);CHKERRQ(ierr);
    ierr = MatDiagonalSet(P,U0,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecSetRandom(U0,NULL);CHKERRQ(ierr);
  ierr = TSCreatePropagatorMat(ts,t0,dt,tf,U0,NULL,P,&Phi);CHKERRQ(ierr);
  ierr = TSGetProblemType(ts,&ptype);CHKERRQ(ierr);
  if (ptype == TS_LINEAR) { /* For linear problems Phi should give the ODE itself */
    Vec W,sol;

    ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
    ierr = VecDuplicate(sol,&W);CHKERRQ(ierr);
    ierr = MatMult(Phi,U0,W);CHKERRQ(ierr);
    if (P) { /* MatMult output should be P*sol */
      Vec y;

      ierr = VecDuplicate(sol,&y);CHKERRQ(ierr);
      ierr = VecCopy(sol,y);CHKERRQ(ierr);
      ierr = MatMult(P,y,sol);CHKERRQ(ierr);
      ierr = VecDestroy(&y);CHKERRQ(ierr);
    }
    ierr = VecAXPY(W,-1.0,sol);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_INFINITY,&err);CHKERRQ(ierr);
    if (err > PETSC_SMALL) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Possible error with TLM: ||Phi u0 - u(t)|| is  %g\n",(double)err);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&W);CHKERRQ(ierr);
  }
  ierr = MatComputeOperator(Phi,NULL,&PhiExpl);CHKERRQ(ierr);
  ierr = MatNorm(PhiExpl,NORM_INFINITY,&normPhi);CHKERRQ(ierr);
  if (P) {
    ierr = MatGetDiagonal(P,U0);CHKERRQ(ierr);
    ierr = VecReciprocal(U0);CHKERRQ(ierr);
    ierr = MatDiagonalScale(PhiExpl,U0,NULL);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)PhiExpl,"Phi");CHKERRQ(ierr);
  ierr = MatViewFromOptions(PhiExpl,NULL,"-prop_view");CHKERRQ(ierr);
  ierr = MatCreateTranspose(Phi,&PhiT);CHKERRQ(ierr);
  ierr = MatComputeOperator(PhiT,NULL,&PhiTExpl);CHKERRQ(ierr);
  if (P) {
    ierr = MatGetDiagonal(P,U0);CHKERRQ(ierr);
    ierr = VecReciprocal(U0);CHKERRQ(ierr);
    ierr = MatDiagonalScale(PhiTExpl,NULL,U0);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)PhiTExpl,"PhiT");CHKERRQ(ierr);
  ierr = MatViewFromOptions(PhiTExpl,NULL,"-propT_view");CHKERRQ(ierr);
  ierr = MatTranspose(PhiTExpl,MAT_INITIAL_MATRIX,&H);CHKERRQ(ierr);
  ierr = MatAXPY(H,-1.0,PhiExpl,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatScale(H,1./normPhi);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)H,"||Phi - (Phi^T)^T||/||Phi||");CHKERRQ(ierr);
  ierr = MatNorm(H,NORM_INFINITY,&err);CHKERRQ(ierr);
  ierr = MatViewFromOptions(H,NULL,"-err_view");CHKERRQ(ierr);
  if (err > 0.01) { /* 1% difference */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Possible error with TLM: ||Phi|| is  %g (%g)\n",(double)normPhi,(double)err);CHKERRQ(ierr);
    ierr = MatView(PhiExpl,NULL);CHKERRQ(ierr);
    ierr = MatView(PhiTExpl,NULL);CHKERRQ(ierr);
    ierr = MatView(H,NULL);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&H);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&U0);CHKERRQ(ierr);
  ierr = MatDestroy(&PhiExpl);CHKERRQ(ierr);
  ierr = MatDestroy(&Phi);CHKERRQ(ierr);
  ierr = MatDestroy(&PhiTExpl);CHKERRQ(ierr);
  ierr = MatDestroy(&PhiT);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = PetscOptFinalize();
  return ierr;
}

/*TEST
    test:
      suffix: lorentz
      args: -ts_type rk -tf 3 -lorentz -ts_rtol 1.e-6 -ts_atol 1.e-6 -ts_trajectory_type memory -testp -prop_view -propT_view

    test:
      suffix: lorentz_i_noifunc
      args: -ts_type bdf -ts_bdf_order 3 -tlm_ts_bdf_order 3 -adjoint_tlm_ts_bdf_order 3 -tf 0.5  -lorentz -ts_rtol 1.e-6 -ts_atol 1.e-6 -ts_trajectory_type memory -prop_view -propT_view
      output_file: output/ex2_lorentz_out.out

    test:
      suffix: lorentz_i
      args: -ts_type bdf -ts_bdf_order 3 -tlm_ts_bdf_order 3 -adjoint_tlm_ts_bdf_order 3 -tf 0.5  -lorentz -ts_rtol 1.e-6 -ts_atol 1.e-6 -ts_trajectory_type memory -ifunc -prop_view -propT_view
      output_file: output/ex2_lorentz_out_i.out

    test:
      suffix: oscillator
      args: -ts_type rk -ts_rk_type 5dp -tlm_ts_rk_type 5dp -adjoint_tlm_ts_rk_type 5dp -tf 10 -ts_trajectory_type memory -testp  -adjoint_tlm_constjacobians -prop_view -propT_view
      output_file: output/ex2_oscillator.out

    test:
      suffix: oscillator_i_noifunc
      args: -ts_type cn -tf 10 -dt 0.1 -ts_trajectory_type memory -tlm_ts_adapt_type none -tlm_constjacobians -adjoint_tlm_reuseksp -prop_view -propT_view
      output_file: output/ex2_oscillator_out.out

    test:
      suffix: oscillator_i
      args: -ts_type cn -tf 10 -dt 0.1 -ts_trajectory_type memory -ifunc -tlm_userijacobian 0 -tlm_reuseksp -tlm_constjacobians -adjoint_tlm_constjacobians -prop_view -propT_view
      output_file: output/ex2_oscillator_out_i.out

    test:
      suffix: propagator_rk_discrete
      args: -ts_type rk -tf 3 -ts_rk_type 5bs -lorentz {{0 1}separate output} -ts_adapt_type basic -ts_rtol 1.e-6 -ts_atol 1.e-6 -ts_trajectory_type memory -testp -prop_view -propT_view -err_view -tlm_discrete -adjoint_tlm_discrete

    test:
      suffix: propagator_cn_discrete
      args: -ts_type cn -tf 0.1 -dt 1.e-2 -lorentz {{0 1}separate output} -ifunc {{0 1}separate output} -ts_adapt_type basic -ts_trajectory_type memory -prop_view -propT_view -err_view -tlm_discrete -adjoint_tlm_discrete

    test:
      suffix: propagator_theta_discrete
      args: -ts_type theta -ts_theta_theta {{0.31 0.5 0.84 1.0}separate output} -tf 0.1 -dt 1.e-2 -lorentz {{0 1}separate output} -ifunc {{0 1}separate output} -ts_adapt_type basic -ts_trajectory_type memory -prop_view -propT_view -err_view -tlm_discrete -adjoint_tlm_discrete

TEST*/

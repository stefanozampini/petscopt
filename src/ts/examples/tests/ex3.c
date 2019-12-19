static char help[] = "Demonstrates PETScOpt capalibities with index-1 DAEs.\n";

/*
   This example has been taken from http://www.cs.ucsb.edu/~cse/Files/SCE001495.pdf.
   The SNES-based computation of initial conditions is slightly different than what described therein.
*/

#include <petscopt.h>
#include <petsctao.h>
#include <petscts.h>
#include <petscdmda.h>
#include <petscdmredundant.h>

/* Species */
typedef struct {
  PetscScalar c1,c2;
} Field;

/* DAE parameters */
typedef struct {
  PetscReal A[2][2];
  PetscReal d1,d2;
  PetscReal alpha;
  PetscReal beta;
  PetscReal fpi;
  /* These are not parameters: we just pass them around */
  PetscReal initdt;
  PetscReal tf;
} AppCtx;

/* Reactions and their partial derivatives */
static PetscScalar f1(PetscScalar c1, PetscScalar c2, PetscScalar b, AppCtx* ctx)
{
  return c1 * (b + ctx->A[0][0] * c1 + ctx->A[0][1] * c2);
}

static PetscScalar f1_1(PetscScalar c1, PetscScalar c2, PetscScalar b, AppCtx* ctx)
{
  return b + 2.0 * ctx->A[0][0] * c1 + ctx->A[0][1] * c2;
}

static PetscScalar f1_2(PetscScalar c1, PetscScalar c2, PetscScalar b, AppCtx* ctx)
{
  return c1 * ctx->A[0][1];
}

#if 0
static PetscScalar f1_b(PetscScalar c1, PetscScalar c2, PetscScalar b, AppCtx* ctx)
{
  return c1;
}
#endif

static PetscScalar f2(PetscScalar c1, PetscScalar c2, PetscScalar b, AppCtx* ctx)
{
  return c2 * (b + ctx->A[1][0] * c1 + ctx->A[1][1] * c2);
}

static PetscScalar f2_1(PetscScalar c1, PetscScalar c2, PetscScalar b, AppCtx* ctx)
{
  return c2 * ctx->A[1][0];
}

static PetscScalar f2_2(PetscScalar c1, PetscScalar c2, PetscScalar b, AppCtx* ctx)
{
  return b + ctx->A[1][0] * c1 + 2.0 * ctx->A[1][1] * c2;
}

#if 0
static PetscScalar f2_b(PetscScalar c1, PetscScalar c2, PetscScalar b, AppCtx* ctx)
{
  return c2;
}
#endif

/* DAE residual F(Udot,U,t) = 0 */
static PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void* ctx)
{
  AppCtx         *appctx;
  DM             da,cda;
  DMDACoor2d     **coords;
  Vec            C,lU,lUdot;
  DMDALocalInfo  info;
  Field          **u,**udot,**f;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetApplicationContext(ts,(void **)&appctx);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da,&lU);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&lUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,lU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,lU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,Udot,INSERT_VALUES,lUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,Udot,INSERT_VALUES,lUdot);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&C);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,C,&coords);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx);
  hy   = 1.0/(PetscReal)(info.my);
  sx   = 1.0/(hx*hx);
  sy   = 1.0/(hy*hy);
  xs   = info.xs;
  xm   = info.xm;
  ys   = info.ys;
  ym   = info.ym;

  ierr = DMDAVecGetArrayRead(da,lU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,lUdot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      PetscScalar c1,c1xx,c1yy;
      PetscScalar c2,c2xx,c2yy;
      PetscScalar b1,b2,x,y,ff1,ff2;

      x          = coords[j][i].x;
      y          = coords[j][i].y;
      c1         = u[j][i].c1;
      c2         = u[j][i].c2;
      c1xx       = (-2.0*c1 + u[j][i-1].c1 + u[j][i+1].c1)*sx;
      c2xx       = (-2.0*c2 + u[j][i-1].c2 + u[j][i+1].c2)*sx;
      c1yy       = (-2.0*c1 + u[j-1][i].c1 + u[j+1][i].c1)*sy;
      c2yy       = (-2.0*c2 + u[j-1][i].c2 + u[j+1][i].c2)*sy;
      b1         = 1.0 + appctx->alpha*x*y + appctx->beta*PetscSinScalar(appctx->fpi*PETSC_PI*x)*PetscSinScalar(appctx->fpi*PETSC_PI*y);
      b2         = -b1;
      ff1        = f1(c1,c2,b1,appctx);
      ff2        = f2(c1,c2,b2,appctx);
      f[j][i].c1 = udot[j][i].c1 - ff1 - appctx->d1*(c1xx + c1yy);
      f[j][i].c2 =               - ff2 - appctx->d2*(c2xx + c2yy);
    }
  }
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,lU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,lUdot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda,C,&coords);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(da,&lU);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&lUdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* DAE Jacobian s * F_Udot + F_U */
static PetscErrorCode FormIJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal s, Mat J, Mat Jp, void *ctx)
{
  AppCtx         *appctx;
  DM             da,cda;
  DMDACoor2d     **coords;
  Vec            lU,C;
  DMDALocalInfo  info;
  Field          **u;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetApplicationContext(ts,(void **)&appctx);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&lU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,lU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,lU);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&C);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx);
  hy   = 1.0/(PetscReal)(info.my);
  sx   = 1.0/(hx*hx);
  sy   = 1.0/(hy*hy);
  xs   = info.xs;
  xm   = info.xm;
  ys   = info.ys;
  ym   = info.ym;

  ierr = MatZeroEntries(Jp);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,C,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,lU,&u);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      MatStencil  row,col[5];
      PetscScalar val[20], dd = -2.0*sx - 2.0*sy;
      PetscScalar c1,c2,b1,b2,x,y,f11,f12,f21,f22;
      PetscInt    c;

      x   = coords[j][i].x;
      y   = coords[j][i].y;
      c1  = u[j][i].c1;
      c2  = u[j][i].c2;
      b1  = 1.0 + appctx->alpha*x*y + appctx->beta*PetscSinScalar(appctx->fpi*PETSC_PI*x)*PetscSinScalar(appctx->fpi*PETSC_PI*y);
      b2  = -b1;
      f11 = f1_1(c1,c2,b1,appctx);
      f12 = f1_2(c1,c2,b1,appctx);
      f21 = f2_1(c1,c2,b2,appctx);
      f22 = f2_2(c1,c2,b2,appctx);

      row.i = i;
      row.j = j;

      col[0].i = i-1; col[0].j = j;   val[ 0] =   - appctx->d1*sx      ; val[ 1] =   0.0; val[ 2] =   0.0; val[ 3] = - appctx->d2*sx;
      col[1].i = i+1; col[1].j = j;   val[ 4] =   - appctx->d1*sx      ; val[ 5] =   0.0; val[ 6] =   0.0; val[ 7] = - appctx->d2*sx;
      col[2].i = i;   col[2].j = j-1; val[ 8] =   - appctx->d1*sy      ; val[ 9] =   0.0; val[10] =   0.0; val[11] = - appctx->d2*sy;
      col[3].i = i;   col[3].j = j+1; val[12] =   - appctx->d1*sy      ; val[13] =   0.0; val[14] =   0.0; val[15] = - appctx->d2*sy;
      col[4].i = i;   col[4].j = j;   val[16] = s - appctx->d1*dd - f11; val[17] = - f12; val[18] = - f21; val[19] = - appctx->d2*dd - f22;

      /* We use ADD_VALUES since we are using DM_BOUNDARY_MIRROR to impose Neumann BC */
      for (c = 0; c < 5; c++) {
        ierr = MatSetValuesBlockedStencil(Jp,1,&row,1,&col[c],val+4*c,ADD_VALUES);CHKERRQ(ierr);
      }
      /* TODO: THIS DOES NOT WORK */
      /* ierr = MatSetValuesBlockedStencil(Jp,1,&row,5,col,val,ADD_VALUES);CHKERRQ(ierr); */
    }
  }
  ierr = DMDAVecRestoreArrayRead(cda,C,&coords);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,lU,&u);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(da,&lU);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Jp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jp) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* DAE Jacobian w.r.t parameters F_M */
static PetscErrorCode FormGradientDAE(TS ts, PetscReal t, Vec U, Vec Udot, Vec M, Mat F_M, void *ctx)
{
  AppCtx         *appctx;
  DM             da,cda;
  DMDACoor2d     **coords;
  Vec            lU,C;
  Vec            F_m[5];
  DMDALocalInfo  info;
  Field          **u;
  PetscScalar    *array;
  Field          **f_m[5];
  PetscInt       np,i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatGetSize(F_M,NULL,&np);CHKERRQ(ierr);
  if (np <= 0) PetscFunctionReturn(0);
  ierr = MatGetLocalSize(F_M,&j,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(F_M,&array);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts,(void **)&appctx);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  for (i=0;i<np;i++) {
    ierr = DMGetGlobalVector(da,&F_m[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(F_m[i],array + j*i);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,F_m[i],&f_m[i]);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(F_M,&array);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&C);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx);
  hy   = 1.0/(PetscReal)(info.my);
  sx   = 1.0/(hx*hx);
  sy   = 1.0/(hy*hy);
  xs   = info.xs;
  xm   = info.xm;
  ys   = info.ys;
  ym   = info.ym;

  ierr = DMGetLocalVector(da,&lU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,lU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,lU);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,C,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,lU,&u);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      PetscScalar c1,c2;
      PetscScalar c1xx,c1yy,c2xx,c2yy;
#if 0
      PetscScalar b1,b2,x,y;
      PetscScalar f1b,f2b,b1alpha,b1beta,b1fpi;
      PetscScalar ff1alpha,ff1beta,ff1fpi;
      PetscScalar ff2alpha,ff2beta,ff2fpi;
#endif

      /*
        Residual
          ff1  = f1(c1,c2,b1,appctx);
          ff2  = f2(c1,c2,b2,appctx);
          f[j][i].c1 = udot[j][i].c1 - ff1 - appctx->d1*(c1xx + c1yy);
          f[j][i].c2 =               - ff2 - appctx->d2*(c2xx + c2yy);
      */
      c1   = u[j][i].c1;
      c2   = u[j][i].c2;
      c1xx = (-2.0*c1 + u[j][i-1].c1 + u[j][i+1].c1)*sx;
      c2xx = (-2.0*c2 + u[j][i-1].c2 + u[j][i+1].c2)*sx;
      c1yy = (-2.0*c1 + u[j-1][i].c1 + u[j+1][i].c1)*sy;
      c2yy = (-2.0*c2 + u[j-1][i].c2 + u[j+1][i].c2)*sy;

#if 0
      x    = coords[j][i].x;
      y    = coords[j][i].y;
      b1   = 1.0 + appctx->alpha*x*y + appctx->beta*PetscSinScalar(appctx->fpi*PETSC_PI*x)*PetscSinScalar(appctx->fpi*PETSC_PI*y);
      b2   = -b1;

      /* coefficient derivatives */
      b1alpha  = x*y;
      b1beta   = PetscSinScalar(appctx->fpi*PETSC_PI*x)*PetscSinScalar(appctx->fpi*PETSC_PI*y);
      b1fpi    = appctx->beta*PETSC_PI*(x*PetscCosScalar(appctx->fpi*PETSC_PI*x)*PetscSinScalar(appctx->fpi*PETSC_PI*y) +
                                        y*PetscSinScalar(appctx->fpi*PETSC_PI*x)*PetscCosScalar(appctx->fpi*PETSC_PI*y));
      f1b      = f1_b(c1,c2,b1,appctx);
      f2b      = f2_b(c1,c2,b2,appctx);
      ff1alpha =   f1b*b1alpha;
      ff1beta  =   f1b*b1beta;
      ff1fpi   =   f1b*b1fpi;
      ff2alpha = - f2b*b1alpha;
      ff2beta  = - f2b*b1beta;
      ff2fpi   = - f2b*b1fpi;
#endif

      /* parameter dependency */
      f_m[0][j][i].c1 = -(c1xx + c1yy);
      f_m[0][j][i].c2 = 0.0;
      f_m[1][j][i].c1 = 0.0;
      f_m[1][j][i].c2 = -(c2xx + c2yy);
    }
  }
  ierr = DMDAVecRestoreArrayRead(cda,C,&coords);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,lU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&lU);CHKERRQ(ierr);

  for (i=0;i<np;i++) {
    ierr = DMDAVecRestoreArray(da,F_m[i],&f_m[i]);CHKERRQ(ierr);
    ierr = VecResetArray(F_m[i]);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&F_m[i]);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Residual for initial condition computation
   - u-u0 on differential variables
   - constraints on algebraic variables
*/
static PetscErrorCode FormICFunction(SNES snes, Vec X, Vec F, void* ctx)
{
  TS             ts;
  DM             da;
  Vec            U0 = (Vec)(ctx), U;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESGetApplicationContext(snes,(void **)&ts);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts,0,X,X,F,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&U);CHKERRQ(ierr);
  ierr = VecCopy(X,U);CHKERRQ(ierr);
  ierr = VecStrideSet(U,1,0.0);CHKERRQ(ierr);
  ierr = VecStrideSet(F,0,0.0);CHKERRQ(ierr);
  ierr = VecAXPY(U,-1.0,U0);CHKERRQ(ierr);
  ierr = VecAXPY(F,1.0,U);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Jacobian for initial condition computation */
static PetscErrorCode FormICJacobian(SNES snes, Vec U, Mat J, Mat Jp, void* ctx)
{
  TS             ts;
  IS             is1;
  PetscInt       st,en;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESGetApplicationContext(snes,(void **)&ts);CHKERRQ(ierr);
  ierr = TSComputeIJacobian(ts,0,U,U,0.0,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U,&st,&en);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)U),(en-st)/2,st,2,&is1);CHKERRQ(ierr);
  ierr = MatSetOption(Jp,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(Jp,is1,1.0,NULL,NULL);CHKERRQ(ierr);
  if (J && J != Jp) {
    ierr = MatSetOption(J,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatZeroRowsIS(J,is1,1.0,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Creates a SNES and solves the initial condition problem */
static PetscErrorCode FormInitialConditions(TS ts, Vec U, PetscBool wsnes)
{
  AppCtx         *appctx;
  SNES           snes;
  KSP            ksp;
  Mat            J;
  DM             da,cda;
  DMDACoor2d     **coords;
  Vec            U0,C;
  Field          **u;
  PetscScalar    x,y,b1,b2;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts,(void **)&appctx);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&C);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,C,&coords);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  if (wsnes) {
    ierr = DMGetGlobalVector(da,&U0);CHKERRQ(ierr);
  } else U0 = U;
  ierr = VecSet(U0,0.0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,U0,&u);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x          = coords[j][i].x;
      y          = coords[j][i].y;
      u[j][i].c1 = 10.0 + (16.0*x*(1.0-x)*y*(1.0-y))*(16.0*x*(1.0-x)*y*(1.0-y));
      b1         = 1.0 + appctx->alpha*x*y + appctx->beta*PetscSinScalar(appctx->fpi*PETSC_PI*x)*PetscSinScalar(appctx->fpi*PETSC_PI*y);
      b2         = -b1;
      u[j][i].c2 = -(b2 + appctx->A[1][0]*u[j][i].c1)/appctx->A[1][1];
    }
  }
  ierr = DMDAVecRestoreArrayRead(cda,C,&coords);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,U0,&u);CHKERRQ(ierr);

  if (wsnes) {
    ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
    ierr = SNESCreate(PetscObjectComm((PetscObject)da),&snes);CHKERRQ(ierr);
    ierr = SNESSetApplicationContext(snes,ts);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(snes,"ic_");CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,NULL,FormICFunction,U0);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormICJacobian,NULL);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = VecStrideSet(U,1,1.e7);CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,U);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&U0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Optimization callbacks */
static PetscErrorCode EvalObjective(Vec U, Vec M, PetscReal time, PetscReal *v, void *ctx)
{
  PetscBool      *test = (PetscBool*)ctx;
  PetscReal      vv[2] = { 0.0, 0.0 };
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (test[0]) {
    ierr = VecStrideNorm(U,0,NORM_2,&vv[0]);CHKERRQ(ierr);
  }
  if (test[1]) {
    ierr = VecStrideNorm(U,1,NORM_2,&vv[1]);CHKERRQ(ierr);
  }
  *v = vv[0]*vv[0] + vv[1]*vv[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjective_U(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
{
  PetscBool      *test = (PetscBool*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(U,grad);CHKERRQ(ierr);
  ierr = VecScale(grad,2.0);CHKERRQ(ierr);
  if (!test[0]) {
    ierr = VecStrideSet(grad,0,0.0);CHKERRQ(ierr);
  }
  if (!test[1]) {
    ierr = VecStrideSet(grad,1,0.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalObjective_UU(Vec U, Vec M, PetscReal time, Mat H, void *ctx)
{
  PetscBool      *test = (PetscBool*)ctx;
  DM             da;
  Vec            D;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(U,&da);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&D);CHKERRQ(ierr);
  ierr = VecSet(D,2.0);CHKERRQ(ierr);
  if (!test[0]) {
    ierr = VecStrideSet(D,0,0.0);CHKERRQ(ierr);
  }
  if (!test[1]) {
    ierr = VecStrideSet(D,1,0.0);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  ierr = MatDiagonalSet(H,D,INSERT_VALUES);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* callback for use with TAO solvers */
static PetscErrorCode FormObjective(Tao tao, Vec M, PetscReal *obj, void* ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSComputeObjectiveAndGradient(ts,0.0,0.0,0.0,NULL,M,NULL,obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Setup the ODE solver when parameters change */
static PetscErrorCode MyTSSetUpFromDesign(TS ts, Vec x0, Vec M, void *ctx)
{
  DM                dmred;
  Vec               lM;
  AppCtx            *appctx;
  PetscBool         *wsnes = (PetscBool*)ctx;
  const PetscScalar *a;
  PetscInt          np;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(M,&dmred);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmred,&lM);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmred,M,INSERT_VALUES,lM);CHKERRQ(ierr);
  ierr = VecGetSize(lM,&np);CHKERRQ(ierr);
  ierr = VecGetArrayRead(lM,&a);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts,(void **)&appctx);CHKERRQ(ierr);
  switch (np) {
  case 2:
    appctx->d2 = PetscRealPart(a[1]);
  case 1:
    appctx->d1 = PetscRealPart(a[0]);
  case 0:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unhandled parameter size %D",np);
  }
  ierr = VecRestoreArrayRead(lM,&a);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmred,&lM);CHKERRQ(ierr);
  ierr = FormInitialConditions(ts,x0,*wsnes);CHKERRQ(ierr);

  /* Set initial timesteps and final time;
     this could have been done (equivalently) from the public API of
     TSComputeObjectiveAndGradient() and TSComputeHessian() too */
  ierr = TSSetTimeStep(ts,appctx->initdt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,appctx->tf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Tao            tao;
  TS             ts;
  DM             da,dmred;
  Mat            H,J;
  Vec            x,G,M;
  AppCtx         appctx;
  PetscReal      Ain[2][2],tf,dt;
  PetscInt       n,np,nc;
  PetscBool      wsnes = PETSC_TRUE, test[2] = { PETSC_TRUE, PETSC_FALSE };
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);
  if (ierr) return ierr;

  appctx.A[0][0] = Ain[0][0] = -1.0;
  appctx.A[0][1] = Ain[0][1] = -0.5e-6;
  appctx.A[1][0] = Ain[1][0] =  1.e4;
  appctx.A[1][1] = Ain[1][1] = -1.0;
  appctx.alpha   = 50;
  appctx.beta    = 100;
  appctx.d1      = 1;
  appctx.d2      = .05;
  appctx.fpi     = 4.0;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"PDE-constrained options","");
  ierr = PetscOptionsBool("-ic_snes","Initial conditions with SNES",__FILE__,wsnes,&wsnes,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Alpha",__FILE__,appctx.alpha,&appctx.alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-beta","Beta",__FILE__,appctx.beta,&appctx.beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-d1","D1",__FILE__,appctx.d1,&appctx.d1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-d2","D2",__FILE__,appctx.d2,&appctx.d2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-fp","FP",__FILE__,appctx.fpi,&appctx.fpi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-A", "A",__FILE__,&Ain[0][0],(n=4,&n),NULL);CHKERRQ(ierr);
  ierr = PetscMemcpy(&appctx.A[0][0],&Ain[0][0],n*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscOptionsBoolArray("-test", "Test component in gradient",__FILE__,test,(n=2,&n),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* structured uniform 2-D grid with Neumann boundary conditions */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_MIRROR,DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,20,20,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"c1");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"c2");CHKERRQ(ierr);

  /* initialize ODE solver with callbacks */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,&appctx);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_DAE_SEMI_EXPLICIT_INDEX1);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FormIFunction,NULL);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,10.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.0625);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  {
    PetscReal satol,srtol;
    Vec       vatol,vrtol;

    ierr = TSGetTolerances(ts,&satol,NULL,&srtol,NULL);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da,&vatol);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da,&vrtol);CHKERRQ(ierr);
    ierr = VecStrideSet(vatol,0,satol);CHKERRQ(ierr);
    ierr = VecStrideSet(vrtol,0,srtol);CHKERRQ(ierr);
    ierr = VecStrideSet(vatol,1,-1);CHKERRQ(ierr);
    ierr = VecStrideSet(vrtol,1,-1);CHKERRQ(ierr);
    ierr = TSSetTolerances(ts,satol,vatol,srtol,vrtol);CHKERRQ(ierr);
    ierr = VecDestroy(&vatol);CHKERRQ(ierr);
    ierr = VecDestroy(&vrtol);CHKERRQ(ierr);
  }

  /* Solve the nonlinear ODE */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = FormInitialConditions(ts,x,wsnes);CHKERRQ(ierr);
  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* Set callbacks for optimization */
  ierr = MatCreate(PETSC_COMM_WORLD,&H);CHKERRQ(ierr);
  ierr = MatSetSizes(H,n,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(H);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = TSAddObjective(ts,tf,EvalObjective,EvalObjective_U,NULL,
                        H,EvalObjective_UU,NULL,NULL,NULL,NULL,test);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);

  np = 2;

  ierr = DMRedundantCreate(PETSC_COMM_WORLD,0,np,&dmred);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmred,&M);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmred,&G);CHKERRQ(ierr);
  ierr = DMDestroy(&dmred);CHKERRQ(ierr);

  ierr = VecGetLocalSize(M,&nc);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,n,nc,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(J,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,J,FormGradientDAE,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSSetSetUpFromDesign(ts,MyTSSetUpFromDesign,&wsnes);CHKERRQ(ierr);

  appctx.initdt = dt;
  appctx.tf = tf;
  ierr = VecSetValue(M,0,appctx.d1,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(M,1,appctx.d2,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(M);CHKERRQ(ierr);

  /* Test gradient with TAO */
  ierr = TSComputeObjectiveAndGradient(ts,0.0,0.0,0.0,NULL,M,G,NULL);CHKERRQ(ierr);
  ierr = VecViewFromOptions(G,NULL,"-gradient_view");CHKERRQ(ierr);
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(tao,FormObjective,ts);CHKERRQ(ierr);
  ierr = TaoTestGradient(tao,M,G);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* Test gradient with Taylor test */
  ierr = TSTaylorTest(ts,0.0,0.0,0.0,NULL,M,NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&G);CHKERRQ(ierr);
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: !complex
      timeoutfactor: 3
      suffix: 1
      filter: sed -e "s/-nan/nan/g" -e "s/coded Hessian/coded Gradient/g"
      nsize: 2
      args: -ts_max_steps 4 -da_grid_x 20 -da_grid_y 20 -ts_trajectory_type memory -ic_snes {{0 1}separate output} -test {{0,0 1,0 0,1 1,1}separate output} -tao_test_gradient -taylor_ts_hessian  -tshessian_mffd

TEST*/

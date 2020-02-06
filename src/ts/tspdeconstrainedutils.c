#include <petscopt/private/tspdeconstrainedutilsimpl.h>
#include <petscopt/private/tsobjimpl.h>
#include <petscopt/private/tsoptimpl.h>
#include <petsc/private/tsimpl.h> /* adapt and poststep dependency */
#include <petsc/private/kspimpl.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dmimpl.h>
#include <petscopt/tlmts.h>
#include <petscopt/augmentedts.h>
#include <petscdm.h>
#include <petscdmcomposite.h>
#include <petscdmshell.h>

#if PETSC_VERSION_LT(3,13,0)
#include <petsc/private/matimpl.h>
typedef struct {
  PetscScalar diag;
} Mat_ConstantDiagonal;

static PetscErrorCode MatGetRow_ConstantDiagonal(Mat A, PetscInt row, PetscInt *ncols, PetscInt *cols[], PetscScalar *vals[])
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)A->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (ncols) *ncols = 1;
  if (cols) {
    ierr = PetscMalloc1(1,cols);CHKERRQ(ierr);
    (*cols)[0] = row;
  }
  if (vals) {
    ierr = PetscMalloc1(1,vals);CHKERRQ(ierr);
    (*vals)[0] = ctx->diag;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRow_ConstantDiagonal(Mat A, PetscInt row, PetscInt *ncols, PetscInt *cols[], PetscScalar *vals[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ncols) *ncols = 0;
  if (cols) {
    ierr = PetscFree(*cols);CHKERRQ(ierr);
    *cols = NULL;
  }
  if (vals) {
    ierr = PetscFree(*vals);CHKERRQ(ierr);
    *vals = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_ConstantDiagonal(Mat A, Vec x, Vec y)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)A->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = VecAXPBY(y,ctx->diag,0.0,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_ConstantDiagonal(Mat mat,Vec v1,Vec v2,Vec v3)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)mat->data;

  PetscFunctionBegin;
  if (v2 == v3) {
    ierr = VecAXPBY(v3,ctx->diag,1.0,v1);CHKERRQ(ierr);
  } else {
    ierr = VecAXPBYPCZ(v3,ctx->diag,1.0,0.0,v1,v2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_ConstantDiagonal(Mat mat,Vec v1,Vec v2,Vec v3)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)mat->data;

  PetscFunctionBegin;
  if (v2 == v3) {
    ierr = VecAXPBY(v3,ctx->diag,1.0,v1);CHKERRQ(ierr);
  } else {
    ierr = VecAXPBYPCZ(v3,ctx->diag,1.0,0.0,v1,v2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode add_missing_cdiag(Mat);

static PetscErrorCode MatDuplicate_ConstantDiagonal(Mat A, MatDuplicateOption op, Mat *B)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *actx = (Mat_ConstantDiagonal*)A->data;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
  ierr = MatSetType(*B,MATCONSTANTDIAGONAL);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->rmap,&(*B)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&(*B)->cmap);CHKERRQ(ierr);
  if (op == MAT_COPY_VALUES) {
    Mat_ConstantDiagonal *bctx = (Mat_ConstantDiagonal*)(*B)->data;
    bctx->diag = actx->diag;
  }
  ierr = add_missing_cdiag(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_ConstantDiagonal(Mat mat,PetscBool *missing,PetscInt *dd)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode add_missing_cdiag(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetOperation(A,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_ConstantDiagonal);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_MULT_TRANSPOSE_ADD,(void (*)(void))MatMultTransposeAdd_ConstantDiagonal);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_DUPLICATE,(void (*)(void))MatDuplicate_ConstantDiagonal);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_MISSING_DIAGONAL,(void (*)(void))MatMissingDiagonal_ConstantDiagonal);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_MULT_ADD,(void (*)(void))MatMultAdd_ConstantDiagonal);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_GET_ROW,(void (*)(void))MatGetRow_ConstantDiagonal);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_RESTORE_ROW,(void (*)(void))MatRestoreRow_ConstantDiagonal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode DMGetLocalToGlobalMapping_Dummy(DM dm)
{
  Vec            g;
  IS             is;
  PetscInt       n,st;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingDestroy(&dm->ltogmap);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&g);CHKERRQ(ierr);
  ierr = VecGetLocalSize(g,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(g,&st,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)dm),n,st,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&dm->ltogmap);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadTSUpdateStates(TS ts, Vec U, Vec Udot)
{
  TSQuadCtx      *qctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,(void*)&qctx);CHKERRQ(ierr);
  qctx->U    = U;
  qctx->Udot = Udot;
  PetscFunctionReturn(0);
}

static PetscErrorCode QuadTSRHSFunction(TS ts, PetscReal time, Vec U, Vec F, void *ctx)
{
  TSQuadCtx      *qctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,(void*)&qctx);CHKERRQ(ierr);
  ierr = (*qctx->evalquad)(qctx->U,qctx->Udot,time,F,qctx->evalquadctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode QuadTSRHSJacobian(TS ts, PetscReal time, Vec U, Mat A, Mat P, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(P);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode QuadTSJacCoupling(TS ts, PetscReal time, Vec Q, Vec Qdot, PetscReal s, Mat A, Mat P, void *ctx)
{
  TSQuadCtx      *qctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,(void*)&qctx);CHKERRQ(ierr);
  ierr = (*qctx->evaljaccoup)(qctx->U,qctx->Udot,time,s,A,P,qctx->evalquadctx);CHKERRQ(ierr);
  if (Qdot) {
    ierr = MatScale(A,-1.0);CHKERRQ(ierr);
    if (A && A != P) { ierr = MatScale(P,-1.0);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSCreateQuadTS(MPI_Comm comm, Vec v, PetscBool diffrhs, TSQuadCtx *ctx, TS *qts)
{
  Mat            A,B;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCreate(comm,qts);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*qts,ctx);CHKERRQ(ierr);
  if (!v) {
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    n    = !rank ? 1 : 0;
    ierr = VecCreateMPI(comm,n,PETSC_DECIDE,&v);CHKERRQ(ierr);
    ierr = TSSetSolution(*qts,v);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  } else {
    ierr = TSSetSolution(*qts,v);CHKERRQ(ierr);
    ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  }
  ierr = MatCreateConstantDiagonal(comm,n,n,PETSC_DECIDE,PETSC_DECIDE,0.0,&A);CHKERRQ(ierr);
  ierr = MatCreateConstantDiagonal(comm,n,n,PETSC_DECIDE,PETSC_DECIDE,0.0,&B);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(*qts,NULL,QuadTSRHSFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(*qts,A,diffrhs ? B : A,QuadTSRHSJacobian,NULL);CHKERRQ(ierr);

#if PETSC_VERSION_LT(3,13,0)
  /* Missing ops for MATCONSTANTDIAGONAL */
  ierr = add_missing_cdiag(A);CHKERRQ(ierr);
  ierr = add_missing_cdiag(B);CHKERRQ(ierr);
#endif
  /* fixes for using DMCOMPOSITE later */
  {
    DM  dm;
    Mat T;

    ierr = TSGetDM(*qts,&dm);CHKERRQ(ierr);
    dm->ops->getlocaltoglobalmapping = DMGetLocalToGlobalMapping_Dummy;
    ierr = VecCreateSeq(PETSC_COMM_SELF,1,&v);CHKERRQ(ierr);
    ierr = DMShellSetLocalVector(dm,v);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = DMSetMatType(dm,MATCONSTANTDIAGONAL);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&T);CHKERRQ(ierr);
    ierr = DMShellSetMatrix(dm,T);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Wrappers for quadrature evaluation
  XXX_FWD are evaluated during the forward run
  XXX_TLM are evaluated during the tangent linear model run within Hessian computations
*/

typedef struct {
  TSObj obj;
  Vec   design;
  Vec   work;
  Vec   f_x;
  Vec   swork;
} FWDEvalQuadCtx;

static PetscErrorCode EvalQuadObj_FWD(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscScalar    *a;
  PetscReal      f;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEval(evalctx->obj,U,evalctx->design,t,&f);CHKERRQ(ierr);
  ierr = VecGetArray(F,&a);CHKERRQ(ierr);
  ierr = VecGetLocalSize(F,&n);CHKERRQ(ierr);
  if (n) { a[0] = f; }
  ierr = VecRestoreArray(F,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadObjFixed_FWD(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscScalar    *a;
  PetscReal      f;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEvalFixed(evalctx->obj,U,evalctx->design,t,&f);CHKERRQ(ierr);
  ierr = VecGetArray(F,&a);CHKERRQ(ierr);
  ierr = VecGetLocalSize(F,&n);CHKERRQ(ierr);
  if (n) { a[0] = f; }
  ierr = VecRestoreArray(F,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrand_FWD(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscBool      has_m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEval_M(evalctx->obj,U,evalctx->design,t,evalctx->work,&has_m,F);CHKERRQ(ierr);
  if (!has_m) {
    ierr = VecSet(F,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrandFixed_FWD(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscBool      has_m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEvalFixed_M(evalctx->obj,U,evalctx->design,t,evalctx->work,&has_m,F);CHKERRQ(ierr);
  if (!has_m) { /* can be called with a non-matching time */
    ierr = VecSet(F,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode JacCoupling_FWD(Vec U, Vec Udot, PetscReal t, PetscReal s, Mat A, Mat B, void* ctx)
{
  FWDEvalQuadCtx *ectx = (FWDEvalQuadCtx*)ctx;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEval_U(ectx->obj,U,ectx->design,t,ectx->swork,&has,ectx->f_x);CHKERRQ(ierr);
  if (!has) { ierr = VecSet(ectx->f_x,0.);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode JacCouplingMult_FWD(Mat A, Vec x, Vec y)
{
  PetscScalar    *a,v;
  FWDEvalQuadCtx *ctx;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = VecTDot(ctx->f_x,x,&v);CHKERRQ(ierr);
  ierr = VecGetArray(y,&a);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
  if (n) { a[0] = v; }
  ierr = VecRestoreArray(y,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  TSObj     obj; /* TODO : access via TSGetTSObj? */
  TS        fwdts;
  TS        adjts;
  PetscReal t0,tf;
  Vec       design;
  Vec       direction;
  Vec       work1;
  Vec       work2;
  PetscReal jac_time;
  PetscReal jac_shift;
} TLMEvalQuadCtx;

/* computes d^2 f / dp^2 direction + d^2 f / dp dx U + (L^T \otimes I_M)(H_MM direction + H_MU U + H_MUdot Udot) during TLM runs */
static PetscErrorCode EvalQuadIntegrand_TLM(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  TLMEvalQuadCtx *q = (TLMEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  TS             adjts = q->adjts;
  TSTrajectory   tj,atj = NULL;
  TSOpt          tsopt;
  Vec            FWDH[2],FOAH = NULL;
  PetscReal      adjt  = q->tf - t + q->t0;
  PetscBool      AXPY;
  PetscBool      Hhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTSOpt(fwdts,&tsopt);CHKERRQ(ierr);
  if (adjts) { /* If not present, Gauss-Newton Hessian */
    ierr = TSGetTrajectory(adjts,&atj);CHKERRQ(ierr);
    ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
  }
  if (!Udot) Hhas[2][1] = PETSC_FALSE;
  ierr = TSGetTrajectory(fwdts,&tj);CHKERRQ(ierr);
  ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,fwdts,t,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  ierr = TSObjEval_MU(q->obj,FWDH[0],q->design,t,U,q->work1,&AXPY,F);CHKERRQ(ierr);
  if (!AXPY) {
    ierr = TSObjEval_MM(q->obj,FWDH[0],q->design,t,q->direction,q->work1,&AXPY,F);CHKERRQ(ierr);
  } else {
    PetscBool has;

    ierr = TSObjEval_MM(q->obj,FWDH[0],q->design,t,q->direction,q->work1,&has,q->work2);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr);
    }
  }
  if (Hhas[2][0] || Hhas[2][1] || Hhas[2][2]) { /* Not for GN */
    ierr = TSTrajectoryGetUpdatedHistoryVecs(atj,adjts,adjt,&FOAH,NULL);CHKERRQ(ierr);
  }
  if (Hhas[2][2]) { /* (L^T \otimes I_M) H_MM direction */
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,2,t,FWDH[0],FWDH[1],q->design,FOAH,q->direction,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,2,t,FWDH[0],FWDH[1],q->design,FOAH,q->direction,F);CHKERRQ(ierr);
    }
    AXPY = PETSC_TRUE;
  }
  if (Hhas[2][0]) { /* (L^T \otimes I_M) H_MX \eta, \eta (=U) the TLM solution */
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,0,t,FWDH[0],FWDH[1],q->design,FOAH,U,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,0,t,FWDH[0],FWDH[1],q->design,FOAH,U,F);CHKERRQ(ierr);
    }
    AXPY = PETSC_TRUE;
  }
  /* this is evaluated only when Udot != 0, i.e. the TLM is solved via the IFunction interface */
  if (Hhas[2][1]) { /* (L^T \otimes I_M) H_MXdot \etadot (=Udot), \eta the TLM solution */
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,1,t,FWDH[0],FWDH[1],q->design,FOAH,Udot,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,1,t,FWDH[0],FWDH[1],q->design,FOAH,Udot,F);CHKERRQ(ierr);
    }
    AXPY = PETSC_TRUE;
  }
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  if (FOAH) {
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(atj,&FOAH,NULL);CHKERRQ(ierr);
  }
  if (!AXPY) { ierr = VecSet(F,0.);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrandFixed_TLM(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  TLMEvalQuadCtx *q = (TLMEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  TSTrajectory   tj;
  PetscBool      has;
  Vec            FWDH;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTrajectory(fwdts,&tj);CHKERRQ(ierr);
  ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,fwdts,t,&FWDH,NULL);CHKERRQ(ierr);
  ierr = TSObjEvalFixed_MU(q->obj,FWDH,q->design,t,U,q->work1,&has,F);CHKERRQ(ierr);
  if (!has) {
    ierr = TSObjEvalFixed_MM(q->obj,FWDH,q->design,t,q->direction,q->work1,&has,F);CHKERRQ(ierr);
    if (!has) { /* can be called with a non-matching time */
      ierr = VecSet(F,0);CHKERRQ(ierr);
    }
  } else {
    ierr = TSObjEvalFixed_MM(q->obj,FWDH,q->design,t,q->direction,q->work1,&has,q->work2);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr);
    }
  }
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&FWDH,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode JacCoupling_TLM(Vec U, Vec Udot, PetscReal t, PetscReal s, Mat A, Mat B, void* ctx)
{
  TLMEvalQuadCtx *q = (TLMEvalQuadCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  q->jac_shift = s;
  q->jac_time  = t;
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode JacCouplingMult_TLM(Mat A, Vec x, Vec y)
{
  TLMEvalQuadCtx *q;
  TS             fwdts,adjts;
  TSTrajectory   tj,atj = NULL;
  TSOpt          tsopt;
  Vec            FWDH[2],FOAH = NULL;
  PetscBool      AXPY = PETSC_FALSE;
  PetscBool      Hhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};
  PetscReal      t,s,adjt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(A,(void**)&q);CHKERRQ(ierr);
  fwdts = q->fwdts;
  adjts = q->adjts;
  t     = q->jac_time;
  s     = q->jac_shift;
  adjt  = q->tf - t + q->t0;

  ierr = TSGetTSOpt(fwdts,&tsopt);CHKERRQ(ierr);
  if (adjts) { /* If not present, Gauss-Newton Hessian */
    ierr = TSGetTrajectory(adjts,&atj);CHKERRQ(ierr);
    ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
  }
  if (s == 0.) Hhas[2][1] = PETSC_FALSE;
  if (!Hhas[2][0] && !Hhas[2][1]) {
    ierr = VecSet(y,0.);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = TSGetTrajectory(fwdts,&tj);CHKERRQ(ierr);
  ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,fwdts,t,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  ierr = TSTrajectoryGetUpdatedHistoryVecs(atj,adjts,adjt,&FOAH,NULL);CHKERRQ(ierr);
  if (Hhas[2][0]) { /* (L^T \otimes I_M) H_MX */
    ierr = TSOptEvalHessianDAE(tsopt,2,0,t,FWDH[0],FWDH[1],q->design,FOAH,x,y);CHKERRQ(ierr);
    AXPY = PETSC_TRUE;
  }
  /* this is evaluated only when Udot != 0, i.e. the TLM is solved via the IFunction interface */
  if (Hhas[2][1]) { /* s * (L^T \otimes I_M) H_MXdot */
    if (!AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,1,t,FWDH[0],FWDH[1],q->design,FOAH,x,y);CHKERRQ(ierr);
      ierr = VecScale(y,s);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,1,t,FWDH[0],FWDH[1],q->design,FOAH,x,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(y,s,q->work1);CHKERRQ(ierr);
    }
  }
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(atj,&FOAH,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Apply "Jacobians" of initial conditions
   if transpose is false : y =   G_x^-1 G_m x or G_x^-1 x if G_m == 0
   if transpose is true  : y = G_m^T G_x^-T x or G_x^-T x if G_m == 0
   (x0,design) are the variables one needs to linearize against to get the partial Jacobians G_x and G_m
   The case for useGm == PETSC_FALSE is a hack to reuse the same code for the second-order adjoint
*/
PetscErrorCode TSLinearizedICApply_Private(TS ts, PetscReal t0, Vec x0, Vec design, Vec x, Vec y, PetscBool transpose, PetscBool useGm)
{
  KSP            ksp = NULL;
  Mat            G_x, G_m = NULL;
  Vec            workvec = NULL;
  TSOpt          tsopt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidHeaderSpecific(x0,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(x,VEC_CLASSID,5);
  PetscValidHeaderSpecific(y,VEC_CLASSID,6);
  PetscValidLogicalCollectiveBool(ts,transpose,7);
  ierr = TSGetTSOpt(ts,&tsopt);CHKERRQ(ierr);
  ierr = TSOptEvalGradientIC(tsopt,t0,x0,design,&G_x,useGm ? &G_m : NULL);CHKERRQ(ierr);
  if (useGm && !G_m) {
    ierr = VecSet(y,0.);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (G_x) { /* this is optional. If not provided, identity is assumed */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradientIC_G",(PetscObject*)&ksp);CHKERRQ(ierr);
    if (!ksp) {
      const char *prefix;
      ierr = KSPCreate(PetscObjectComm((PetscObject)ts),&ksp);CHKERRQ(ierr);
      ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(ksp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"jactsic_");CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,G_x,G_x);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_G",(PetscObject)ksp);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)ksp);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(ksp,G_x,G_x);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradientIC_GW",(PetscObject*)&workvec);CHKERRQ(ierr);
    if (!workvec) {
      ierr = MatCreateVecs(G_x,&workvec,NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_GW",(PetscObject)workvec);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)workvec);CHKERRQ(ierr);
    }
  }
  if (transpose) {
    if (ksp) {
      if (useGm) {
        ierr = KSPSolveTranspose(ksp,x,workvec);CHKERRQ(ierr);
        ierr = MatMultTranspose(G_m,workvec,y);CHKERRQ(ierr);
      } else {
        ierr = KSPSolveTranspose(ksp,x,y);CHKERRQ(ierr);
      }
    } else {
      if (useGm) {
        ierr = MatMultTranspose(G_m,x,y);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(x,y);CHKERRQ(ierr);
      }
    }
  } else {
    if (ksp) {
      if (useGm) {
        ierr = MatMult(G_m,x,workvec);CHKERRQ(ierr);
        ierr = KSPSolve(ksp,workvec,y);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(ksp,x,y);CHKERRQ(ierr);
      }
    } else {
      if (useGm) {
        ierr = MatMult(G_m,x,y);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(x,y);CHKERRQ(ierr);
      }
    }
  }
  if (ksp) { /* destroy inner vectors to avoid ABA issues when destroying the DM */
    ierr = VecDestroy(&ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecDestroy(&ksp->vec_sol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* auxiliary function to solve a forward model with a quadrature */
PetscErrorCode TSSolveWithQuadrature_Private(TS ts, Vec X, Vec design, Vec direction, Vec quadvec, PetscReal *quadscalar, PetscBool *done)
{
  TS                ats,model = NULL;
  TS                qts[] = {NULL,NULL}, aqts[] = {NULL,NULL};
  TSQuadCtx         qctxs[2];
  Vec               U;
  PetscReal         t0,tf,tfup,dt;
  PetscBool         fidt,stop,diffrhs;
  FWDEvalQuadCtx    qfwd;
  TLMEvalQuadCtx    qtlm;
  TSObj             funchead;
  PetscInt          i,nq,anq;
  PetscErrorCode    ierr;
  PetscErrorCode    (*qup[])(TS,Vec,Vec) = {QuadTSUpdateStates,QuadTSUpdateStates};
  Vec               qvec_fixed[]= {NULL,NULL};
  QuadEval          squad,squad_fixed,vquad,vquad_fixed;
  TSIJacobian       jaccoup[] = {NULL,NULL};
  Mat               Acoup[] = {NULL,NULL}, Bcoup[] = {NULL,NULL};
  void              *squad_ctx,*vquad_ctx;
  TSConvergedReason reason = TS_CONVERGED_ITERATING;

  PetscFunctionBegin;
  if (direction) {
    PetscValidHeaderSpecific(direction,VEC_CLASSID,4);
    PetscCheckSameComm(ts,1,direction,4);
  }
  if (quadvec) {
    PetscValidHeaderSpecific(quadvec,VEC_CLASSID,5);
    PetscCheckSameComm(ts,1,quadvec,5);
  }
  PetscValidPointer(done,7);
  *done = PETSC_FALSE;

  if (direction && !quadvec) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Cannot compute Hessian without quadrature vector");
  if (direction) { /* when the direction is present, the ts need to be a TLMTS */
    ierr = TLMTSGetModelTS(ts,&model);CHKERRQ(ierr);
    ierr = TSGetTSObj(model,&funchead);CHKERRQ(ierr);
  } else {
    ierr = TSGetTSObj(ts,&funchead);CHKERRQ(ierr);
  }

  /* quadrature evaluations */
  squad       = direction ? NULL : (quadscalar ? EvalQuadObj_FWD      : NULL);
  squad_fixed = direction ? NULL : (quadscalar ? EvalQuadObjFixed_FWD : NULL);
  vquad       = quadvec ? (direction ? EvalQuadIntegrand_TLM      : EvalQuadIntegrand_FWD)      : NULL;
  vquad_fixed = quadvec ? (direction ? EvalQuadIntegrandFixed_TLM : EvalQuadIntegrandFixed_FWD) : NULL;

  /* solution vector */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  if (!U) {
    ierr = VecDuplicate(X,&U);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)U);CHKERRQ(ierr);
  }
  if (X) {
    ierr = VecCopy(X,U);CHKERRQ(ierr);
  }

  /* init contexts for quadrature evaluations */
  qfwd.obj    = funchead;
  qfwd.design = design;
  qfwd.work   = NULL;
  qfwd.f_x    = NULL;
  qfwd.swork  = NULL;
  qtlm.obj    = funchead;
  qtlm.design = design;
  qtlm.fwdts  = NULL;
  qtlm.adjts  = NULL;
  qtlm.work1  = NULL;
  qtlm.work2  = NULL;
  qtlm.jac_shift = 0.0;
  qtlm.jac_time = PETSC_MIN_REAL;
  squad_ctx   = &qfwd;
  vquad_ctx   = direction ? (void*)&qtlm : (void*)&qfwd;

  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* determine the presence of cost integrals */
  ierr = TSGetMaxTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);

  if (squad) {
    PetscBool has;

    ierr = TSObjHasObjectiveIntegrand(funchead,&has,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    if (!has) squad = NULL;
    ierr = TSObjHasObjectiveFixed(funchead,t0-PETSC_SMALL,tf,&has,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    if (!has) squad_fixed = NULL;
  }

  if (vquad || vquad_fixed) {
    PetscBool has = PETSC_FALSE;

    if (direction) { /* Hessian computations */
      PetscBool flg;

      ierr = TLMTSIsDiscrete(ts,&flg);CHKERRQ(ierr);
      if (flg) has = PETSC_FALSE; /* Quadrature performed during second-order adjoint */
      else {
        TS          adjts;
        TSOpt       tsopt;
        TSIFunction ifunc;
        PetscBool   has1,has2;
        PetscBool   Hhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                                  {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                                  {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};

        ierr = PetscObjectQuery((PetscObject)ts,"_ts_hessian_foats",(PetscObject*)&adjts);CHKERRQ(ierr);
        ierr = TSGetTSOpt(model,&tsopt);CHKERRQ(ierr);
        ierr = TSObjHasObjectiveIntegrand(funchead,NULL,NULL,NULL,NULL,&has1,&has2);CHKERRQ(ierr);
        has  = (PetscBool)(has1 || has2);
        ierr = PetscObjectQuery((PetscObject)model,"_ts_hessian_foats",(PetscObject*)&adjts);CHKERRQ(ierr);
        if (adjts) {
          ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
        }
        ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
        if (!ifunc) Hhas[2][1] = PETSC_FALSE;
        has = (PetscBool)(has || Hhas[2][0] || Hhas[2][1] || Hhas[2][2]);
      }
    }
    if (!has) vquad = NULL;

    if (direction) { /* we use PETSC_SMALL since some of the fixed terms can be at the initial time */
      PetscBool has1,has2;

      ierr = TSObjHasObjectiveFixed(funchead,t0-PETSC_SMALL,tf,NULL,NULL,NULL,NULL,&has1,&has2,NULL);CHKERRQ(ierr);
      has  = (PetscBool)(has1 || has2);
    } else {
      ierr = TSObjHasObjectiveFixed(funchead,t0-PETSC_SMALL,tf,NULL,NULL,&has,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    if (!has) vquad_fixed = NULL;
  }

  /* Finish setup of quadrature contexts.
     For gradient computations, we just need the design vector and one work vector for the function evaluation.
     For Hessian computations, we need extra data */
  if (vquad || vquad_fixed) {
    if (!direction) {
      ierr = VecDuplicate(design,&qfwd.work);CHKERRQ(ierr);
    } else {
      /* qtlm.adjts may be NULL for Gauss-Newton approximations */
      ierr = PetscObjectQuery((PetscObject)model,"_ts_hessian_foats",(PetscObject*)&qtlm.adjts);CHKERRQ(ierr);
      qtlm.fwdts     = model;
      qtlm.t0        = t0;
      qtlm.tf        = tf;
      qtlm.design    = design;
      qtlm.direction = direction;

      ierr = VecDuplicate(design,&qtlm.work1);CHKERRQ(ierr);
      ierr = VecDuplicate(design,&qtlm.work2);CHKERRQ(ierr);
    }
  }

  /* SetUp quadrature TSs */
  nq = 0;
  diffrhs = (ts->Arhs != ts->Brhs) ? PETSC_TRUE : PETSC_FALSE;
  if (squad || squad_fixed) {
    DM  dm;
    Vec v;

    ierr = TSCreateQuadTS(PetscObjectComm((PetscObject)ts),NULL,diffrhs,&qctxs[nq],&qts[nq]);CHKERRQ(ierr);
    ierr = TSSetProblemType(qts[nq],TS_NONLINEAR);CHKERRQ(ierr);

    qctxs[nq].evalquad       = squad;
    qctxs[nq].evalquad_fixed = squad_fixed;
    qctxs[nq].evalquadctx    = squad_ctx;

    /* Coupling Jacobian */
    if (squad) {
      Vec      sol;
      PetscInt n,N;

      ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&qfwd.f_x);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&qfwd.swork);CHKERRQ(ierr);
      jaccoup[nq] = QuadTSJacCoupling;
      qctxs[nq].evaljaccoup = JacCoupling_FWD;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      ierr = VecGetSize(sol,&N);CHKERRQ(ierr);
      ierr = VecGetLocalSize(sol,&n);CHKERRQ(ierr);
      ierr = MatCreateShell(PetscObjectComm((PetscObject)ts),PETSC_DECIDE,n,1,N,qctxs[nq].evalquadctx,&Acoup[nq]);CHKERRQ(ierr);
      if (diffrhs) {
        ierr = MatCreateShell(PetscObjectComm((PetscObject)ts),PETSC_DECIDE,n,1,N,qctxs[nq].evalquadctx,&Bcoup[nq]);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)Acoup[nq]);CHKERRQ(ierr);
        Bcoup[nq] = Acoup[nq];
      }
      ierr = MatShellSetOperation(Acoup[nq],MATOP_MULT,(void(*)(void))JacCouplingMult_FWD);CHKERRQ(ierr);
    }

    /* Init */
    ierr = TSGetDM(qts[nq],&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&qvec_fixed[nq]);CHKERRQ(ierr);
    ierr = TSGetSolution(qts[nq],&v);CHKERRQ(ierr);
    ierr = VecSet(v,0.0);CHKERRQ(ierr);
    ierr = VecSet(qvec_fixed[nq],0.0);CHKERRQ(ierr);
    if (qctxs[nq].evalquad_fixed) {
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      ierr = (*qctxs[nq].evalquad_fixed)(sol,NULL,t0,qvec_fixed[nq],qctxs[nq].evalquadctx);CHKERRQ(ierr);
    }
    nq++;
  }
  if (vquad || vquad_fixed) {
    DM dm;

    qctxs[nq].evalquad       = vquad;
    qctxs[nq].evalquad_fixed = vquad_fixed;
    qctxs[nq].evalquadctx    = vquad_ctx;

    ierr = TSCreateQuadTS(PetscObjectComm((PetscObject)ts),quadvec,diffrhs,&qctxs[nq],&qts[nq]);CHKERRQ(ierr);
    ierr = TSSetProblemType(qts[nq],direction ? TS_LINEAR : TS_NONLINEAR);CHKERRQ(ierr);

    /* Coupling Jacobian */
    if (direction && vquad && qtlm.adjts) {
      TSOpt     tsopt;
      TSIFunction ifunc;
      PetscBool Hhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                              {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                              {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};

      ierr = TSGetTSOpt(model,&tsopt);CHKERRQ(ierr);
      ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
      ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
      if (!ifunc) Hhas[2][1] = PETSC_FALSE;
      if (Hhas[2][0] || Hhas[2][1]) {
        Vec      sol;
        PetscInt n,N,m,M;

        jaccoup[nq] = QuadTSJacCoupling;
        qctxs[nq].evaljaccoup = JacCoupling_TLM;

        ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
        ierr = VecGetSize(sol,&N);CHKERRQ(ierr);
        ierr = VecGetLocalSize(sol,&n);CHKERRQ(ierr);
        ierr = VecGetSize(design,&M);CHKERRQ(ierr);
        ierr = VecGetLocalSize(design,&m);CHKERRQ(ierr);
        ierr = MatCreateShell(PetscObjectComm((PetscObject)ts),m,n,M,N,qctxs[nq].evalquadctx,&Acoup[nq]);CHKERRQ(ierr);
        if (diffrhs) {
          ierr = MatCreateShell(PetscObjectComm((PetscObject)ts),m,n,M,N,qctxs[nq].evalquadctx,&Bcoup[nq]);CHKERRQ(ierr);
        } else {
          ierr = PetscObjectReference((PetscObject)Acoup[nq]);CHKERRQ(ierr);
          Bcoup[nq] = Acoup[nq];
        }
        ierr = MatShellSetOperation(Acoup[nq],MATOP_MULT,(void(*)(void))JacCouplingMult_TLM);CHKERRQ(ierr);
      }
    }

    /* Init (add quadvec to qvec_fixed and start from 0) */
    ierr = TSGetDM(qts[nq],&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&qvec_fixed[nq]);CHKERRQ(ierr);
    ierr = VecSet(qvec_fixed[nq],0.0);CHKERRQ(ierr);
    if (qctxs[nq].evalquad_fixed) {
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      ierr = (*qctxs[nq].evalquad_fixed)(sol,NULL,t0,qvec_fixed[nq],qctxs[nq].evalquadctx);CHKERRQ(ierr);
    }
    ierr = VecAXPY(qvec_fixed[nq],1.0,quadvec);CHKERRQ(ierr);
    ierr = VecSet(quadvec,0.0);CHKERRQ(ierr);
    nq++;
  }

  /* Augmented TS (only for active quadratures) */
  for (i=0,anq=0;i<nq;i++) {
    if (qctxs[i].evalquad == NULL) {
      ierr = MatDestroy(&Acoup[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&Bcoup[i]);CHKERRQ(ierr);
    } else {
      Acoup[anq]   = Acoup[i];
      Bcoup[anq]   = Bcoup[i];
      aqts[anq]    = qts[i];
      qup[anq]     = qup[i];
      jaccoup[anq] = jaccoup[i];
      anq++;
    }
  }
  ierr = TSCreateAugmentedTS(ts,anq,aqts,NULL,qup,jaccoup,Acoup,Bcoup,PETSC_FALSE,&ats);CHKERRQ(ierr);
  ierr = MatDestroy(&Acoup[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&Acoup[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&Bcoup[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&Bcoup[1]);CHKERRQ(ierr);
  ierr = AugmentedTSInitialize(ats);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ats);CHKERRQ(ierr);

  /* forward solve */
  fidt = PETSC_TRUE;
  ierr = TSGetTimeStep(ats,&dt);CHKERRQ(ierr);
  if (ats->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)ats->adapt,TSADAPTNONE,&fidt);CHKERRQ(ierr);
  }

  /* Time loop */
  tfup = tf;
  do {
    PetscReal tt;
    PetscInt  q;

    ierr = TSGetTime(ats,&t0);CHKERRQ(ierr);
    if (direction) { /* we always stop at the selected times */
      PetscBool has1,has2;

      ierr  = TSObjHasObjectiveFixed(funchead,t0,tf,NULL,NULL,NULL,NULL,&has1,&has2,&tfup);CHKERRQ(ierr);
    } else {
      PetscBool has1,has2;

      ierr = TSObjHasObjectiveFixed(funchead,t0,tf,&has1,NULL,&has2,NULL,NULL,NULL,&tfup);CHKERRQ(ierr);
    }
    tt   = tfup;
    ierr = TSSetMaxTime(ats,tfup);CHKERRQ(ierr);
    ierr = TSSolve(ats,NULL);CHKERRQ(ierr);

    /* determine if something went wrong and attempt recovery */
    ierr = TSGetConvergedReason(ats,&reason);CHKERRQ(ierr);
    if (reason <= TS_CONVERGED_ITERATING) goto cleanup;

    /* determine if TS finished before the max time requested */
    ierr = TSGetTime(ats,&tfup);CHKERRQ(ierr);
    stop = (PetscAbsReal(tt-tfup) < PETSC_SMALL) ? PETSC_FALSE : PETSC_TRUE;
    for (q=0;q<nq;q++) {
      DM  dm;
      Vec sol,v;

      if (!qctxs[q].evalquad_fixed) continue;
      ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
      ierr = TSGetDM(qts[q],&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&v);CHKERRQ(ierr);
      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      ierr = (*qctxs[q].evalquad_fixed)(sol,NULL,tfup,v,qctxs[q].evalquadctx);CHKERRQ(ierr);
      ierr = VecAXPY(qvec_fixed[q],1.0,v);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&v);CHKERRQ(ierr);
    }
    if (fidt) { /* restore fixed time step */
      ierr = TSSetTimeStep(ats,dt);CHKERRQ(ierr);
    }
  } while (PetscAbsReal(tfup - tf) > PETSC_SMALL && !stop);

cleanup:
  ierr = AugmentedTSFinalize(ats);CHKERRQ(ierr);
  if (reason > TS_CONVERGED_ITERATING) {
    for (i=0;i<nq;i++) {
      Vec v;

      ierr = TSGetSolution(qts[i],&v);CHKERRQ(ierr);
      ierr = VecAXPY(v,1.0,qvec_fixed[i]);CHKERRQ(ierr);
    }
    if (quadscalar && (squad || squad_fixed)) {
      Vec         v;
      PetscScalar *a;
      PetscInt    n;

      ierr = TSGetSolution(qts[0],&v);CHKERRQ(ierr);
      ierr = VecGetArray(v,&a);CHKERRQ(ierr);
      ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
      if (n) *quadscalar = PetscRealPart(a[0]);
      ierr = MPI_Bcast(quadscalar,1,MPIU_REAL,0,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
      ierr = VecRestoreArray(v,&a);CHKERRQ(ierr);
    }
    *done = PETSC_TRUE;
  } else {
    PetscReal zero=0.0,one=1.0,inf=one/zero;
    if (quadscalar) *quadscalar = inf;
    if (quadvec) { ierr = VecSetInf(quadvec);CHKERRQ(ierr); }
  }
  for (i=0;i<nq;i++) {
    DM dm;

    ierr = TSGetDM(qts[i],&dm);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&qvec_fixed[i]);CHKERRQ(ierr);
  }
  ierr = TSDestroy(&qts[0]);CHKERRQ(ierr);
  ierr = TSDestroy(&qts[1]);CHKERRQ(ierr);
  ierr = TSDestroy(&ats);CHKERRQ(ierr);

  ierr = VecDestroy(&qfwd.work);CHKERRQ(ierr);
  if (qfwd.f_x) {
    DM dm;

    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&qfwd.f_x);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&qfwd.swork);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&qtlm.work1);CHKERRQ(ierr);
  ierr = VecDestroy(&qtlm.work2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

PetscErrorCode TSCreateQuadTS(MPI_Comm comm, Vec v, PetscBool diffrhs, TSQuadCtx *ctx, TS *qts)
{
  Mat            A,B;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCreate(comm,qts);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*qts,ctx);CHKERRQ(ierr);
  if (!v) {
    n = 1;
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

  /* Missing ops for MATCONSTANTDIAGONAL */
  ierr = add_missing_cdiag(A);CHKERRQ(ierr);
  ierr = add_missing_cdiag(B);CHKERRQ(ierr);

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
/* ------------------ Wrappers for quadrature evaluation ----------------------- */

PetscErrorCode TSQuadratureCtxDestroy_Private(void *ptr)
{
  TSQuadratureCtx* q = (TSQuadratureCtx*)ptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(5,&q->wquad);CHKERRQ(ierr);
  ierr = PetscFree(q->veval_ctx);CHKERRQ(ierr);
  ierr = PetscFree(q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  XXX_FWD are evaluated during the forward run
  XXX_TLM are evaluated during the tangent linear model run within Hessian computations
  XXX_ADJ are evaluated during the adjoint run
*/

typedef struct {
  TSObj obj;
  Vec   design;
  Vec   work;
} FWDEvalQuadCtx;

static PetscErrorCode EvalQuadObj_FWD(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscErrorCode ierr;
  PetscScalar    *a;
  PetscReal      f;

  PetscFunctionBegin;
  ierr = TSObjEval(evalctx->obj,U,evalctx->design,t,&f);CHKERRQ(ierr);
  ierr = VecGetArray(F,&a);CHKERRQ(ierr);
  a[0] = f;
  ierr = VecRestoreArray(F,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadObjFixed_FWD(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  FWDEvalQuadCtx *evalctx = (FWDEvalQuadCtx*)ctx;
  PetscErrorCode ierr;
  PetscScalar    *a;
  PetscReal      f;

  PetscFunctionBegin;
  ierr = TSObjEvalFixed(evalctx->obj,U,evalctx->design,t,&f);CHKERRQ(ierr);
  ierr = VecGetArray(F,&a);CHKERRQ(ierr);
  a[0] = f;
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

typedef struct {
  TSObj     obj; /* TODO : access via TSGetTSObj? */
  TS        fwdts;
  TS        adjts;
  TS        tlmts;
  PetscReal t0,tf;
  Vec       design;
  Vec       direction;
  Vec       work1;
  Vec       work2;
  PetscBool init;
} TLMEvalQuadCtx;

/* computes d^2 f / dp^2 direction + d^2 f / dp dx U + (L^T \otimes I_M)(H_MM direction + H_MU U + H_MUdot Udot) during TLM runs */
static PetscErrorCode EvalQuadIntegrand_TLM(Vec U, Vec Udot, PetscReal t, Vec F, void* ctx)
{
  TLMEvalQuadCtx *q = (TLMEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  TS             adjts = q->adjts;
  TS             tlmts = q->tlmts;
  TSTrajectory   tj,atj = NULL,ltj;
  TSOpt          tsopt;
  Vec            FWDH[2],FOAH = NULL;
  PetscReal      adjt  = q->tf - t + q->t0;
  PetscBool      AXPY, rest = PETSC_FALSE;
  PetscBool      Hhas[3][3] = {{PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE},
                               {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTrajectory(fwdts,&tj);CHKERRQ(ierr);
  if (adjts) { /* If not present, Gauss-Newton Hessian */
    ierr = TSGetTrajectory(adjts,&atj);CHKERRQ(ierr);
  }
  ierr = TSGetTrajectory(tlmts,&ltj);CHKERRQ(ierr);
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
  ierr = TSGetTSOpt(fwdts,&tsopt);CHKERRQ(ierr);
  if (adjts) {
    ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
  }
  if (Hhas[2][0] || (Hhas[2][1] && q->init) || Hhas[2][2]) { /* Not for GN */
    rest = PETSC_TRUE;
    ierr = TSTrajectoryGetUpdatedHistoryVecs(atj,adjts,adjt,&FOAH,NULL);CHKERRQ(ierr);
  }
  if (Hhas[2][2]) { /* (L^T \otimes I_M) H_MM direction */
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,2,t,FWDH[0],FWDH[1],q->design,FOAH,q->direction,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,2,t,FWDH[0],FWDH[1],q->design,FOAH,q->direction,F);CHKERRQ(ierr);
      AXPY = PETSC_TRUE;
    }
  }
  if (Hhas[2][0]) { /* (L^T \otimes I_M) H_MX \eta, \eta (=U) the TLM solution */
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,0,t,FWDH[0],FWDH[1],q->design,FOAH,U,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,0,t,FWDH[0],FWDH[1],q->design,FOAH,U,F);CHKERRQ(ierr);
      AXPY = PETSC_TRUE;
    }
  }
  if (Hhas[2][1] && q->init) { /* (L^T \otimes I_M) H_MXdot \etadot, \eta the TLM solution */
    Vec TLMHdot = Udot;
    DM  dm;

    if (!Udot) { /* FIXME */
    ierr = TSGetDM(tlmts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&TLMHdot);CHKERRQ(ierr);
    ierr = TSTrajectoryGetVecs(ltj,NULL,PETSC_DECIDE,&t,NULL,TLMHdot);CHKERRQ(ierr);
    }
    if (AXPY) {
      ierr = TSOptEvalHessianDAE(tsopt,2,1,t,FWDH[0],FWDH[1],q->design,FOAH,TLMHdot,q->work1);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = TSOptEvalHessianDAE(tsopt,2,1,t,FWDH[0],FWDH[1],q->design,FOAH,TLMHdot,F);CHKERRQ(ierr);
    }
    if (!Udot) { /* FIXME */
    ierr = DMRestoreGlobalVector(dm,&TLMHdot);CHKERRQ(ierr);
    }
  }
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  if (rest) {
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(atj,&FOAH,NULL);CHKERRQ(ierr);
  }
  q->init = PETSC_TRUE; /* FIXME */
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

PetscErrorCode TSQuadraturePostStep_Private(TS ts)
{
  PetscContainer    container;
  Vec               solution;
  TSQuadratureCtx   *qeval_ctx;
  PetscReal         dt,time,ptime;
  TSConvergedReason reason;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing evaluate_quadrature container");
  ierr = PetscContainerGetPointer(container,(void**)&qeval_ctx);CHKERRQ(ierr);
  if (qeval_ctx->user && !qeval_ctx->userafter) {
    PetscStackPush("User post-step function");
    ierr = (*qeval_ctx->user)(ts);CHKERRQ(ierr);
    PetscStackPop;
  }

  ierr = TSGetSolution(ts,&solution);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  if (reason == TS_CONVERGED_TIME) {
    ierr = TSGetMaxTime(ts,&time);CHKERRQ(ierr);
  }

  /* time step used */
  ierr = TSGetPrevTime(ts,&ptime);CHKERRQ(ierr);
  dt   = time - ptime;

  /* scalar quadrature (psquad have been initialized with the first function evaluation) */
  if (qeval_ctx->seval) {
    Vec dummy; /* FIXME */
    PetscScalar *a;

    ierr = VecCreateSeq(PETSC_COMM_SELF,1,&dummy);CHKERRQ(ierr);
    PetscStackPush("TS scalar quadrature function");
    ierr = (*qeval_ctx->seval)(solution,NULL,time,dummy,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
    ierr = VecGetArray(dummy,&a);CHKERRQ(ierr);
    qeval_ctx->squad += dt*(*a+qeval_ctx->psquad)/2.0;
    qeval_ctx->psquad = *a;
    ierr = VecRestoreArray(dummy,&a);CHKERRQ(ierr);
    ierr = VecDestroy(&dummy);CHKERRQ(ierr);
  }

  /* scalar quadrature (qeval_ctx->wquad[qeval_ctx->old] have been initialized with the first function evaluation) */
  if (qeval_ctx->veval) {
    PetscScalar t[2];
    PetscInt    tmp;

    PetscStackPush("TS vector quadrature function");
    ierr = (*qeval_ctx->veval)(solution,NULL,time,qeval_ctx->wquad[qeval_ctx->cur],qeval_ctx->veval_ctx);CHKERRQ(ierr);
    PetscStackPop;

    /* trapezoidal rule */
    t[0] = dt/2.0;
    t[1] = dt/2.0;
    ierr = VecMAXPY(qeval_ctx->vquad,2,t,qeval_ctx->wquad);CHKERRQ(ierr);

    /* swap pointers */
    tmp            = qeval_ctx->cur;
    qeval_ctx->cur = qeval_ctx->old;
    qeval_ctx->old = tmp;
  }
  if (qeval_ctx->user && qeval_ctx->userafter) {
    PetscStackPush("User post-step function");
    ierr = (*qeval_ctx->user)(ts);CHKERRQ(ierr);
    PetscStackPop;
  }
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
PetscErrorCode TSSolveWithQuadrature_Private(TS ts, Vec X, Vec design, Vec direction, Vec quadvec, PetscReal *quadscalar)
{
  TS              model = NULL;
  Vec             U;
  PetscContainer  container;
  TSQuadratureCtx *qeval_ctx;
  PetscReal       t0,tf,tfup,dt;
  PetscBool       fidt,stop;
  SQuadEval       seval_fixed, seval;
  VQuadEval       veval_fixed, veval;
  FWDEvalQuadCtx  qfwd;
  TLMEvalQuadCtx  qtlm;
  TSObj           funchead;
  Vec             dummy;
  PetscErrorCode  ierr;

  QuadEval squad,squad_fixed,vquad,vquad_fixed;
  void     *squad_ctx,*vquad_ctx;

  PetscFunctionBegin;
  if (direction) {
    PetscValidHeaderSpecific(direction,VEC_CLASSID,4);
    PetscCheckSameComm(ts,1,direction,4);
  }
  if (quadvec) {
    PetscValidHeaderSpecific(quadvec,VEC_CLASSID,5);
    PetscCheckSameComm(ts,1,quadvec,5);
  }
  if (direction && !quadvec) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Cannot compute Hessian without quadrature vector");
  if (direction) { /* when the direction is present, the ts need to be a TLMTS */
    ierr = TLMTSGetModelTS(ts,&model);CHKERRQ(ierr);
    ierr = TSGetTSObj(model,&funchead);CHKERRQ(ierr);
  } else {
    ierr = TSGetTSObj(ts,&funchead);CHKERRQ(ierr);
  }

  /* quadrature evaluations */
  seval       = direction ? NULL : (quadscalar ? EvalQuadObj_FWD      : NULL);
  seval_fixed = direction ? NULL : (quadscalar ? EvalQuadObjFixed_FWD : NULL);
  veval       = quadvec ? (direction ? EvalQuadIntegrand_TLM      : EvalQuadIntegrand_FWD)      : NULL;
  veval_fixed = quadvec ? (direction ? EvalQuadIntegrandFixed_TLM : EvalQuadIntegrandFixed_FWD) : NULL;

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
  qtlm.obj    = funchead;
  qtlm.design = design;

  /* set special purpose post step method for quadrature evaluation */
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) {
    ierr = PetscNew(&qeval_ctx);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,(void *)qeval_ctx);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,TSQuadratureCtxDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)container);CHKERRQ(ierr);
  } else {
    ierr = PetscContainerGetPointer(container,(void**)&qeval_ctx);CHKERRQ(ierr);
  }

  qeval_ctx->user      = ts->poststep;
  qeval_ctx->userafter = PETSC_FALSE;
  qeval_ctx->seval     = seval;
  qeval_ctx->seval_ctx = &qfwd;
  qeval_ctx->squad     = 0.0;
  qeval_ctx->psquad    = 0.0;
  qeval_ctx->veval     = veval;
  qeval_ctx->vquad     = quadvec;
  qeval_ctx->cur       = 0;
  qeval_ctx->old       = 1;

  ierr = TSSetUp(ts);CHKERRQ(ierr);

  ierr = TSGetMaxTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  /* fixed term at t0 for scalar quadrature */
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&dummy);CHKERRQ(ierr);
  if (seval_fixed) {
    Vec sol;
    PetscScalar *a;

    ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
    PetscStackPush("TS scalar quadrature function (fixed time)");
    ierr = (*seval_fixed)(sol,NULL,t0,dummy,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
    ierr = VecGetArray(dummy,&a);CHKERRQ(ierr);
    qeval_ctx->squad = *a;
    ierr = VecRestoreArray(dummy,&a);CHKERRQ(ierr);
  }

  /* initialize trapz rule for scalar quadrature */
  if (qeval_ctx->seval) {
    Vec sol;
    PetscScalar *a;

    ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
    PetscStackPush("TS scalar quadrature function");
    ierr = (*qeval_ctx->seval)(sol,NULL,t0,dummy,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
    ierr = VecGetArray(dummy,&a);CHKERRQ(ierr);
    qeval_ctx->psquad = *a;
    ierr = VecRestoreArray(dummy,&a);CHKERRQ(ierr);
  }

  squad = seval;
  squad_fixed = seval_fixed;
  squad_ctx = &qfwd;
  if (veval) {
    PetscBool has;

    if (direction) { /* Hessian computations */
      PetscBool has1,has2,Hhas[3][3];
      TSOpt     tsopt;

      ierr = TSGetTSOpt(model,&tsopt);CHKERRQ(ierr);
      ierr = TSObjHasObjectiveIntegrand(funchead,NULL,NULL,NULL,NULL,&has1,&has2);CHKERRQ(ierr);
      has  = (PetscBool)(has1 || has2);
      ierr = TSOptHasHessianDAE(tsopt,Hhas);CHKERRQ(ierr);
      if (Hhas[2][0] || Hhas[2][1] || Hhas[2][2]) has = PETSC_TRUE;
    } else {
      ierr = TSObjHasObjectiveIntegrand(funchead,NULL,NULL,&has,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    if (!has) { /* cost integrands not present */
      qeval_ctx->veval = NULL;
      veval = NULL;
    }
    if (!qeval_ctx->wquad) {
      ierr = VecDuplicateVecs(qeval_ctx->vquad,5,&qeval_ctx->wquad);CHKERRQ(ierr);
    }
    if (direction) { /* we use PETSC_SMALL since some of the fixed terms can be at the initial time */
      PetscBool has1,has2;

      ierr = TSObjHasObjectiveFixed(funchead,t0-PETSC_SMALL,tf,NULL,NULL,NULL,NULL,&has1,&has2,NULL);CHKERRQ(ierr);
      has  = (PetscBool)(has1 || has2);
    } else {
      ierr = TSObjHasObjectiveFixed(funchead,t0-PETSC_SMALL,tf,NULL,NULL,&has,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    if (!has) veval_fixed = NULL;
  }
  vquad = veval;
  vquad_fixed = veval_fixed;
  vquad_ctx = direction ? (void*)&qtlm : (void*)&qfwd;

  /* for gradient computations, we just need the design vector and one work vector for the function evaluation
     for Hessian computations, we need extra data */
  if (qeval_ctx->veval || veval_fixed) {
    if (!direction) {
      qfwd.work = qeval_ctx->wquad[2];
      qeval_ctx->veval_ctx = &qfwd;
    } else {
      /* qtlm.adjts may be NULL for Gauss-Newton approximations */
      ierr = PetscObjectQuery((PetscObject)model,"_ts_hessian_foats",(PetscObject*)&qtlm.adjts);CHKERRQ(ierr);
      qtlm.fwdts     = model;
      qtlm.tlmts     = ts;
      qtlm.t0        = t0;
      qtlm.tf        = tf;
      qtlm.design    = design;
      qtlm.direction = direction;
      qtlm.work1     = qeval_ctx->wquad[2];
      qtlm.work2     = qeval_ctx->wquad[3];
      qtlm.init      = PETSC_FALSE; /* skip etadot terms when initializing the quadrature */
      qeval_ctx->veval_ctx = &qtlm;
    }

    if (veval_fixed) { /* Fixed term at t0: we use wquad[4] since wquad[3] can be used by the TLM quadrature */
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS vector quadrature function (fixed time)");
      ierr = (*veval_fixed)(sol,NULL,t0,qeval_ctx->wquad[4],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
      ierr = VecAXPY(qeval_ctx->vquad,1.0,qeval_ctx->wquad[4]);CHKERRQ(ierr);
    }

    /* initialize trapz rule for vector quadrature */
    if (qeval_ctx->veval) {
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS vector quadrature function");
      ierr = (*qeval_ctx->veval)(sol,NULL,t0,qeval_ctx->wquad[qeval_ctx->old],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
    }
  }

  /* FIXME */
  PetscBool diffrhs = (ts->Arhs != ts->Brhs) ? PETSC_TRUE : PETSC_FALSE;
  TS        ats = NULL;
  PetscInt  nq = 0;
  TS        qts[] = {NULL,NULL};
  TSQuadCtx qctxs[2];
  PetscErrorCode (*qup[])(TS,Vec,Vec) = {QuadTSUpdateStates,QuadTSUpdateStates};
  Vec       qvec_fixed[]= {NULL,NULL};
  if (squad) {
    DM  dm;
    Vec v;

    qctxs[nq].evalquad       = squad;
    qctxs[nq].evalquad_fixed = squad_fixed;
    qctxs[nq].evalquadctx    = squad_ctx;

    ierr = TSCreateQuadTS(PetscObjectComm((PetscObject)ts),NULL,diffrhs,&qctxs[nq],&qts[nq]);CHKERRQ(ierr);

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

  if (vquad) {
    DM dm;

    qctxs[nq].evalquad       = vquad;
    qctxs[nq].evalquad_fixed = vquad_fixed;
    qctxs[nq].evalquadctx    = vquad_ctx;

    ierr = TSCreateQuadTS(PetscObjectComm((PetscObject)ts),quadvec,diffrhs,&qctxs[nq],&qts[nq]);CHKERRQ(ierr);

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

  /* FIXME: forward quadrature always works */
  PetscBool new = PETSC_TRUE;
  PetscOptionsGetBool(NULL,NULL,"-new",&new,NULL);
  if (nq) {
    ierr = TSCreateAugmentedTS(ts,nq,qts,NULL,qup,NULL,NULL,NULL,PETSC_FALSE,&ats);CHKERRQ(ierr);
    ierr = AugmentedTSInitialize(ats);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ats);CHKERRQ(ierr);
  }
  TS usedts = new ? ats : ts;
  if (!usedts) usedts = ts;
  if (usedts == ats) qtlm.init = PETSC_TRUE;
  if (usedts != ats)  { ierr = TSSetPostStep(ts,TSQuadraturePostStep_Private);CHKERRQ(ierr); }
  /* forward solve */
  fidt = PETSC_TRUE;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (ts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&fidt);CHKERRQ(ierr);
  }

  /* determine if there are functionals, gradients or Hessians wrt parameters of the type f(U,M,t=fixed) to be evaluated */
  /* we don't use events since there's no API to add new events to a pre-existing set */
  tfup = tf;
  do {
    PetscBool has_f = PETSC_FALSE, has_m = PETSC_FALSE;
    PetscReal tt;

    ierr = TSGetTime(usedts,&t0);CHKERRQ(ierr);
    if (direction) { /* we always stop at the selected times */
      PetscBool has1,has2;

      ierr  = TSObjHasObjectiveFixed(funchead,t0,tf,NULL,NULL,NULL,NULL,&has1,&has2,&tfup);CHKERRQ(ierr);
      has_m = (PetscBool)(has1 || has2);
    } else {
      ierr = TSObjHasObjectiveFixed(funchead,t0,tf,&has_f,NULL,&has_m,NULL,NULL,NULL,&tfup);CHKERRQ(ierr);
    }
    tt   = tfup;
    ierr = TSSetMaxTime(usedts,tfup);CHKERRQ(ierr);
    ierr = TSSolve(usedts,NULL);CHKERRQ(ierr);

    /* determine if TS finished before the max time requested */
    ierr = TSGetTime(usedts,&tfup);CHKERRQ(ierr);
    stop = (PetscAbsReal(tt-tfup) < PETSC_SMALL) ? PETSC_FALSE : PETSC_TRUE;
    if (has_f && squad_fixed) {
      if (usedts == ats) {
        DM  dm;
        Vec sol,v;
        PetscInt q;

        q    = 0;
        ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
        ierr = TSGetDM(qts[q],&dm);CHKERRQ(ierr);
        ierr = DMGetGlobalVector(dm,&v);CHKERRQ(ierr);
        ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
        ierr = (*qctxs[q].evalquad_fixed)(sol,NULL,tfup,v,qctxs[q].evalquadctx);CHKERRQ(ierr);
        ierr = VecAXPY(qvec_fixed[q],1.0,v);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(dm,&v);CHKERRQ(ierr);
      } else {
        Vec         sol;
        PetscScalar *a;

        ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
        PetscStackPush("TS scalar quadrature function (fixed time)");
        ierr = (*seval_fixed)(sol,NULL,tfup,dummy,qeval_ctx->seval_ctx);CHKERRQ(ierr);
        PetscStackPop;
        ierr = VecGetArray(dummy,&a);CHKERRQ(ierr);
        qeval_ctx->squad += *a;
        ierr = VecRestoreArray(dummy,&a);CHKERRQ(ierr);
      }
    }
    if (has_m && vquad_fixed) { /* we use wquad[4] since wquad[3] can be used by the TLM quadrature */

      if (usedts == ats) {
        DM  dm;
        Vec sol,v;
        PetscInt q;

        q    = nq-1;
        ierr = AugmentedTSUpdateModelSolution(ats);CHKERRQ(ierr);
        ierr = TSGetDM(qts[q],&dm);CHKERRQ(ierr);
        ierr = DMGetGlobalVector(dm,&v);CHKERRQ(ierr);
        ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
        ierr = (*qctxs[q].evalquad_fixed)(sol,NULL,tfup,v,qctxs[q].evalquadctx);CHKERRQ(ierr);
        ierr = VecAXPY(qvec_fixed[q],1.0,v);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(dm,&v);CHKERRQ(ierr);
      } else {
        Vec sol;

        ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
        PetscStackPush("TS vector quadrature function (fixed time)");
        ierr = (*veval_fixed)(sol,NULL,tfup,qeval_ctx->wquad[4],qeval_ctx->veval_ctx);CHKERRQ(ierr);
        PetscStackPop;
        ierr = VecAXPY(qeval_ctx->vquad,1.0,qeval_ctx->wquad[4]);CHKERRQ(ierr);
      }
    }
    if (fidt) { /* restore fixed time step */
      ierr = TSSetTimeStep(usedts,dt);CHKERRQ(ierr);
    }
  } while (PetscAbsReal(tfup - tf) > PETSC_SMALL && !stop);

  if (usedts == ats) {
    PetscInt i;

    ierr = AugmentedTSFinalize(ats);CHKERRQ(ierr);
    for (i=0;i<nq;i++) {
      DM  dm;
      Vec v;

      ierr = TSGetSolution(qts[i],&v);CHKERRQ(ierr);
      ierr = VecAXPY(v,1.0,qvec_fixed[i]);CHKERRQ(ierr);
      ierr = TSGetDM(qts[i],&dm);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&qvec_fixed[i]);CHKERRQ(ierr);
    }
  }
  if (usedts != ats) { /* restore user PostStep */
    ierr = TSSetPostStep(ts,qeval_ctx->user);CHKERRQ(ierr);
  }
  if (quadscalar) {
    if (usedts == ts) *quadscalar = qeval_ctx->squad;
    else {
      Vec         v;
      PetscScalar *a;

      ierr = TSGetSolution(qts[0],&v);CHKERRQ(ierr);
      ierr = VecGetArray(v,&a);CHKERRQ(ierr);
      *quadscalar = PetscRealPart(a[0]);
      ierr = VecRestoreArray(v,&a);CHKERRQ(ierr);
    }
  }
  ierr = TSDestroy(&qts[0]);CHKERRQ(ierr);
  ierr = TSDestroy(&qts[1]);CHKERRQ(ierr);
  ierr = TSDestroy(&ats);CHKERRQ(ierr);
  /* zero contexts for quadrature */
  qeval_ctx->seval_ctx = NULL;
  qeval_ctx->veval_ctx = NULL;

  ierr = VecDestroy(&dummy);CHKERRQ(ierr);
  /* get back scalar value */
  PetscFunctionReturn(0);
}

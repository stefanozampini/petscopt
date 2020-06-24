#include <petscopt/petscopt_types.h>
#include <petscopt/ksp.h>
#include <petsc/private/kspimpl.h>
#include <petscblaslapack.h>

typedef struct {
  PetscErrorCode (*inner)(Vec,Vec,PetscScalar*,void*);
  PetscErrorCode (*riesz)(Vec,Vec,void*);
  void           *rctx;
  /* eigenvalues support */
  PetscInt       emax,ne;
  PetscReal      *ed,*eu,*wed,*weu;
} KSP_HilbertCG;

static PetscErrorCode KSPHilbertCGRiesz(KSP ksp, Vec x, Vec y)
{
  KSP_HilbertCG  *hcg = (KSP_HilbertCG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (hcg->riesz) {
    ierr = (*hcg->riesz)(x,y,hcg->rctx);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHilbertCGDot(KSP ksp, Vec x, Vec y, PetscScalar *d)
{
  KSP_HilbertCG  *hcg = (KSP_HilbertCG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (hcg->inner) {
    ierr = (*hcg->inner)(x,y,d,hcg->rctx);CHKERRQ(ierr);
  } else {
    ierr = VecDot(x,y,d);CHKERRQ(ierr);
  }
  KSPCheckDot(ksp,*d);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHilbertCGNorm(KSP ksp, Vec x, PetscReal *d)
{
  PetscErrorCode ierr;
  PetscScalar    v;

  PetscFunctionBegin;
  *d   = 0.0;
  ierr = KSPHilbertCGDot(ksp,x,x,&v);CHKERRQ(ierr);
  *d   = PetscSqrtReal(PetscRealPart(v));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_HilbertCG(KSP ksp)
{
  KSP_HilbertCG  *hcg = (KSP_HilbertCG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree4(hcg->ed,hcg->eu,hcg->wed,hcg->weu);CHKERRQ(ierr);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPComputeEigenvalues_HilbertCG(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
  KSP_HilbertCG  *hcg = (KSP_HilbertCG*)ksp->data;
  char           jobz = 'N',range = 'A';
  PetscReal      *work,dummyr = 0.;
  PetscScalar    dummys[1];
  PetscBLASInt   dummyi = 0;
  PetscBLASInt   lwork,liwork,*iwork,nb = hcg->ne,no;
  PetscBLASInt   info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nmax < hcg->ne) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  ierr = PetscArrayzero(c,nmax);CHKERRQ(ierr);
  ierr = PetscArrayzero(r,nmax);CHKERRQ(ierr);
  *neig = nb;
  if (!hcg->ne) PetscFunctionReturn(0);
  ierr = PetscArraycpy(hcg->wed,hcg->ed,nb);CHKERRQ(ierr);
  ierr = PetscArraycpy(hcg->weu,hcg->eu,nb-1);CHKERRQ(ierr);
  lwork = 12*hcg->ne;
  liwork = 8*hcg->ne;
  ierr = PetscMalloc2(lwork,&work,liwork,&iwork);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKstegr",LAPACKstegr_(&jobz,&range,&nb,hcg->wed,hcg->weu,&dummyr,&dummyr,&dummyi,&dummyi,&dummyr,&no,r,dummys,&nb,&dummyi,work,&lwork,iwork,&liwork,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"xSTEGR error %d",(int)info);
  if (no != nb) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"xSTEGR error, returned %d",(int)no);
  ierr = PetscFree2(work,iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPComputeExtremeSingularValues_HilbertCG(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  PetscReal      *r,*c;
  PetscInt       n,neig;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (emin) *emin = 1.0;
  if (emax) *emax = 1.0;
  ierr = KSPGetIterationNumber(ksp,&n);CHKERRQ(ierr);
  if (!n) PetscFunctionReturn(0);
  ierr = PetscMalloc2(n+1,&r,n+1,&c);CHKERRQ(ierr);
  ierr = KSPComputeEigenvalues(ksp,n+1,r,c,&neig);CHKERRQ(ierr);
  if (emin) *emin = r[0];
  if (emax) *emax = r[neig-1];
  ierr = PetscFree2(r,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_HilbertCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,4);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_HilbertCG(KSP ksp, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHilbertCGConverged(KSP ksp, PetscInt it, PetscReal r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ksp->reason = KSP_CONVERGED_ITERATING;
  if (!ksp->converged) PetscFunctionReturn(0);
  if (it == 0 && ksp->converged == KSPConvergedDefault && !ksp->guess_zero) {
    KSPConvergedDefaultCtx *cctx = (KSPConvergedDefaultCtx*)ksp->cnvP;

    if (!cctx->initialrtol) { /* hack to prevent wrong initial norm computation of B */
      PetscReal snorm;

      ierr = PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of RHS\n");CHKERRQ(ierr);
      ierr = KSPHilbertCGNorm(ksp,ksp->vec_rhs,&snorm);CHKERRQ(ierr);
      /* handle special case of zero RHS and nonzero guess */
      if (!snorm) {
        ierr  = PetscInfo(ksp,"Special case, user has provided nonzero initial guess and zero RHS\n");CHKERRQ(ierr);
        snorm = r;
      }
      if (cctx->mininitialrtol) ksp->rnorm0 = PetscMin(snorm,r);
      else ksp->rnorm0 = snorm;
      ksp->guess_zero = PETSC_TRUE;
      ierr = (*ksp->converged)(ksp,0,r,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      ksp->guess_zero = PETSC_FALSE;
    } else {
      ierr = (*ksp->converged)(ksp,0,r,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    }
  } else {
    ierr = (*ksp->converged)(ksp,it,r,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_HilbertCG(KSP ksp)
{
  KSP_HilbertCG  *hcg = (KSP_HilbertCG*)ksp->data;
  Mat            H;
  Vec            X,B,R,D,W,rW;
  PetscReal      rold,r,h,alpha=1.0,beta=0.0,alphaold;
  PetscScalar    v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  D  = ksp->work[1];
  W  = ksp->work[2];
  rW = ksp->work[3];
  ksp->its = 0;
  ierr = PCGetOperators(ksp->pc,&H,NULL);CHKERRQ(ierr);
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,H,X,W);CHKERRQ(ierr);
    ierr = VecAYPX(W,-1.0,B);CHKERRQ(ierr);
    ierr = KSPHilbertCGRiesz(ksp,W,R);CHKERRQ(ierr);
  } else {
    ierr = VecSet(X,0.0);CHKERRQ(ierr);
    ierr = KSPHilbertCGRiesz(ksp,B,R);CHKERRQ(ierr);
  }
  ierr = VecCopy(R,D);CHKERRQ(ierr);
  if (ksp->calc_sings) {
    if (hcg->emax < ksp->max_it) {
      ierr = PetscFree4(hcg->ed,hcg->eu,hcg->wed,hcg->weu);CHKERRQ(ierr);
      hcg->emax = ksp->max_it;
      ierr = PetscMalloc4(hcg->emax+1,&hcg->ed,hcg->emax+1,&hcg->eu,
                          hcg->emax+1,&hcg->wed,hcg->emax+1,&hcg->weu);CHKERRQ(ierr);
    }
  }
  hcg->ne = 0;

  ierr = KSPHilbertCGNorm(ksp,R,&r);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,r);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,r);CHKERRQ(ierr);
  ksp->rnorm = r;
  ierr = KSPHilbertCGConverged(ksp,0,r);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* check for negative direction */
  ierr = MatMult(H,D,W);CHKERRQ(ierr);
  ierr = VecDot(W,D,&v);CHKERRQ(ierr);
  h    = PetscRealPart(v);
  if (h <= 0) {
    ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
    ierr = PetscInfo(ksp,"Negative curvature for initial residual\n");CHKERRQ(ierr);
    ierr = VecCopy(D,X);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  do {
    ksp->its++;
    if (ksp->calc_sings) hcg->ne++;

    alphaold = alpha;
    alpha = r*r/h; /* alpha for CG, beta in the book */
    if (ksp->calc_sings) { hcg->ed[ksp->its-1] = 1.0/alpha + beta/alphaold; }

    ierr = VecAXPY(X,alpha,D);CHKERRQ(ierr);
    ierr = KSPHilbertCGRiesz(ksp,W,rW);CHKERRQ(ierr);
    ierr = VecAXPY(R,-alpha,rW);CHKERRQ(ierr);
    rold = r;
    ierr = KSPHilbertCGNorm(ksp,R,&r);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,r);CHKERRQ(ierr);

    ierr = KSPMonitor(ksp,ksp->its,r);CHKERRQ(ierr);
    ksp->rnorm = r;
    ierr = KSPHilbertCGConverged(ksp,ksp->its,r);CHKERRQ(ierr);
    if (ksp->reason) PetscFunctionReturn(0);
    if (ksp->its < ksp->max_it) {
      beta = (r*r)/(rold*rold); /* beta for cg, gamma in the book */

      ierr = VecAYPX(D,beta,R);CHKERRQ(ierr);

      if (ksp->calc_sings) { hcg->eu[ksp->its-1] = PetscSqrtReal(beta)/alpha; }

      /* check for negative direction */
      ierr = MatMult(H,D,W);CHKERRQ(ierr);
      ierr = VecDot(W,D,&v);CHKERRQ(ierr);
      h    = PetscRealPart(v);
      if (h <= 0) {
        ksp->reason = KSP_CONVERGED_CG_NEG_CURVE;
        ierr = PetscInfo1(ksp,"Negative curvature at step %D\n",ksp->its);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
    }
  } while (ksp->its < ksp->max_it);
  /* always return converged */
  if (ksp->its >= ksp->max_it) ksp->reason = KSP_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPHilbertCGSetMaps(KSP ksp, PetscErrorCode (*inner)(Vec,Vec,PetscScalar*,void*), PetscErrorCode (*riesz)(Vec,Vec,void*), void* ctx)
{
  PetscErrorCode ierr;
  KSP_HilbertCG  *hcg;
  PetscBool      ishcg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPHILBERTCG,&ishcg);CHKERRQ(ierr);
  if (!ishcg) PetscFunctionReturn(0);
  hcg = (KSP_HilbertCG*)ksp->data;
  hcg->inner = inner;
  hcg->riesz = riesz;
  hcg->rctx  = ctx;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode KSPCreate_HilbertCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_HilbertCG  *hcg;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);

  ierr = PetscNew(&hcg);CHKERRQ(ierr);

  ksp->data                              = hcg;
  ksp->ops->solve                        = KSPSolve_HilbertCG;
  ksp->ops->view                         = KSPView_HilbertCG;
  ksp->ops->setup                        = KSPSetUp_HilbertCG;
  ksp->ops->reset                        = NULL;
  ksp->ops->destroy                      = KSPDestroy_HilbertCG;
  ksp->ops->buildsolution                = KSPBuildSolutionDefault;
  ksp->ops->buildresidual                = KSPBuildResidualDefault;
  ksp->ops->setfromoptions               = NULL;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_HilbertCG;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_HilbertCG;
  PetscFunctionReturn(0);
}

static char help[] = "Heat Equation with finite elements.\n";

#include <petscopt.h>
#include <petscds.h>
#include <petscdmplex.h>

/* no parameter dependency */
static void f0_null(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u_t[0] - sin(PETSC_PI*t*10);
}

static void f1_null(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    f1[d] = u_x[d];
  }
}

static void g00_null(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g00[])
{
  g00[0] = u_tShift;
}

static void g11_null(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g11[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    g11[d*dim+d] = 1.0;
  }
}


/* udot term with parameter dependency */
static void f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = (a[0]+1.0)*u_t[0] - sin(PETSC_PI*t*10);
}

static void g00(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g00[])
{
  g00[0] = u_tShift*(a[0]+1.0);
}

/* udot term parameter dependency callback */
static void uf0_a(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscInt numConstantsE, const PetscScalar constants[],
                  PetscBool a0_h[], PetscBool a1_h[],
                  PetscScalar g_a0[], PetscScalar g_a1[])
{
  PetscInt af;
  for (af = 0; af < NfAux; af++) a0_h[af] = PETSC_FALSE; /* TODO move outside */
  for (af = 0; af < NfAux; af++) a1_h[af] = PETSC_FALSE; /* TODO move outside */
  a0_h[0] = PETSC_TRUE;
  g_a0[0] = u_t[0];
}

static PetscBool   f0_a0_h[1024];
static PetscBool   f0_a1_h[1024];
static PetscScalar g0_a0[1024];
static PetscScalar g0_a1[1024];

static PetscBool   f1_a0_h[1024];
static PetscBool   f1_a1_h[1024];
static PetscScalar g1_a0[1024];
static PetscScalar g1_a1[1024];

/* proposal for general callback */
static void f0_a(PetscInt dim, PetscInt Nf, PetscInt NfAuxE,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt NfAux = NfAuxE/2; /* one auxiliary field for a, one for da (perturbation) */
  PetscInt uf = 0; /* TODO: need context for this for PetscDSResidual function */
  PetscInt uNc = uOff[uf+1]-uOff[uf];
  PetscInt d,af,uc,ac;

  const PetscScalar* da   = a + aOff[NfAux];
  const PetscScalar* da_x = a_x + aOff_x[NfAux];
  const PetscScalar* M0   = g0_a0;
  const PetscScalar* M1   = g0_a1;

  uf0_a(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, x, numConstants, constants, f0_a0_h, f0_a1_h, g0_a0, g0_a1);
  /* f0 is zeroed outside */
  for (af = 0; af < NfAux; af++) {
    const PetscInt aNc = aOff[af + 1] - aOff[af];
    if (f0_a0_h[af]) {
      for (uc = 0; uc < uNc; uc++) {
        for (ac = 0; ac < aNc; ac++) {
          f0[uc] += M0[uc*aNc + ac]*da[ac];
        }
      }
    }
    if (f0_a1_h[af]) {
      for (uc = 0; uc < uNc; uc++) {
        for (ac = 0; ac < aNc; ac++) {
          for (d = 0; d < dim; d++) {
            f0[uc] += M1[uc*aNc*dim + ac*dim + d]*da_x[ac*dim + d];
          }
        }
      }
    }
    M0   += uNc*aNc;
    M1   += uNc*aNc*dim;
    da   += aNc;
    da_x += aNc*dim;
  }
}

/* u term with parameter dependency */
static void f1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    f1[d] = a[aOff[1] + d]*u_x[d];
  }
}

static void uf1_a(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscInt numConstantsE, const PetscScalar constants[],
                  PetscBool a0_h[], PetscBool a1_h[],
                  PetscScalar g_a0[], PetscScalar g_a1[])
{
  PetscInt af;
  for (af = 0; af < NfAux; af++) a0_h[af] = PETSC_FALSE; /* TODO move outside */
  for (af = 0; af < NfAux; af++) a1_h[af] = PETSC_FALSE; /* TODO move outside */
  a0_h[1] = PETSC_TRUE;
  PetscInt d;
  PetscInt sh = (aOff[1] - aOff[0])*dim;
  for (d = 0; d < dim; ++d) {
    g_a0[sh + dim*d +d] = u_x[d];
  }
}

static void f1_a(PetscInt dim, PetscInt Nf, PetscInt NfAuxE,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt NfAux = NfAuxE/2; /* one field for a, one for da (perturbation) */
  PetscInt uf = 0; /* TODO: need context for this */
  PetscInt uNc = uOff[uf+1]-uOff[uf];
  PetscInt d,d2,af,uc,ac;

  const PetscScalar* da   = a + aOff[NfAux];
  const PetscScalar* da_x = a_x + aOff_x[NfAux];
  const PetscScalar* M0   = g1_a0;
  const PetscScalar* M1   = g1_a1;
//printf("ok\n");
  uf1_a(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, x, numConstants, constants, f1_a0_h, f1_a1_h, g1_a0, g1_a1);
  /* f1 is zeroed outside */
  for (af = 0; af < NfAux; af++) {
    const PetscInt aNc = aOff[af + 1] - aOff[af];
//printf("f1_a0: %d / %d ->  %d (aNc %d uNc %d)\n",af,NfAux,f1_a0_h[af],aNc,uNc);
    if (f1_a0_h[af]) {
      for (uc = 0; uc < uNc; uc++) {
        for (d = 0; d < dim; d++) {
          for (ac = 0; ac < aNc; ac++) {
            //printf("f1[%d] += %g*%g\n",uc*dim+d,M0[uc*aNc*dim + ac*dim + d],da[ac]);
            f1[uc*dim+d] += M0[uc*aNc*dim + ac*dim + d]*da[ac];
          }
        }
      }
    }
//printf("f1_a1: %d / %d ->  %d (aNc %d uNc %d)\n",af,NfAux,f1_a0_h[af],aNc,uNc);
    if (f1_a1_h[af]) {
      for (uc = 0; uc < uNc; uc++) {
        for (d = 0; d < dim; d++) {
          for (ac = 0; ac < aNc; ac++) {
            for (d2 = 0; d2 < dim; d2++) {
              f1[uc*dim+d] += M1[uc*aNc*dim*dim + ac*dim*dim + d*dim+d2]*da_x[ac*dim + d2];
            }
          }
        }
      }
    }
    M0   += uNc*aNc*dim;
    M1   += uNc*aNc*dim*dim;
    da   += aNc;
    da_x += aNc*dim;
  }
}

static void g11(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g11[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    g11[d*dim+d] = a[aOff[1] + d];
  }
}

static PetscErrorCode identity(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 1.0;
  return 0;
}

typedef void (*PetscDSResidual)(PetscInt,PetscInt,PetscInt,
                                const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],
                                const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],
                                PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]);
typedef void (*PetscDSJacobian)(PetscInt,PetscInt,PetscInt,
                                const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],
                                const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],
                                PetscReal,PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]);

typedef PetscErrorCode (*DMFunction)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar*,void *);

#define MAX_SFIELDS 16
#define MAX_AFIELDS 16
typedef struct {
  PetscInt        dim;
  PetscBool       simplex;
  PetscInt        Nfs;
  PetscDSResidual f0[MAX_SFIELDS];
  PetscDSResidual f1[MAX_SFIELDS];
  PetscBool       f0pd[MAX_SFIELDS];
  PetscBool       f1pd[MAX_SFIELDS];
  PetscDSJacobian g00[MAX_SFIELDS][MAX_SFIELDS];
  PetscDSJacobian g01[MAX_SFIELDS][MAX_SFIELDS];
  PetscDSJacobian g10[MAX_SFIELDS][MAX_SFIELDS];
  PetscDSJacobian g11[MAX_SFIELDS][MAX_SFIELDS];
  PetscObject     sdisc[MAX_SFIELDS];
  PetscInt        Nfa;
  PetscObject     adisc[MAX_AFIELDS];
  DMFunction      a0[MAX_AFIELDS];
} AppCtx;

static PetscBool   f0pd[1024]; /* XXX */
static PetscBool   f1pd[1024]; /* XXX */

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscErrorCode ierr;
  PetscInt       testcase = 1; /* TODO Enum option */

  PetscFunctionBeginUser;
  ctx->dim     = 2;
  ctx->simplex = PETSC_FALSE;
  ctx->Nfs     = 0;
  ctx->Nfa     = 0;

  ierr = PetscOptionsBegin(comm, "", "XXX", "YYY");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", __FILE__, ctx->dim, &ctx->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-testcase", "The test case", __FILE__, testcase, &testcase, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", __FILE__, ctx->simplex, &ctx->simplex, NULL);CHKERRQ(ierr);
  switch (testcase) {
  case 0: /* heat eq, no parameter dependency */
    ctx->Nfs = 1;
    ctx->f0[0] = f0_null;
    ctx->f1[0] = f1_null;
    ctx->f0pd[0] = PETSC_FALSE;
    ctx->f1pd[0] = PETSC_FALSE;
    ctx->g00[0][0] = g00_null;
    ctx->g01[0][0] = NULL;
    ctx->g10[0][0] = NULL;
    ctx->g11[0][0] = g11_null;
    ierr = PetscFECreateDefault(comm,ctx->dim,1,ctx->simplex,"s0_",-1,(PetscFE*)&ctx->sdisc[0]);CHKERRQ(ierr);

    ctx->Nfa = 0;
    break;
  case 1: /* heat eq, parameter dependency only on udot term */
    ctx->Nfs = 1;
    ctx->f0[0] = f0;
    ctx->f1[0] = f1_null;
    ctx->f0pd[0] = PETSC_TRUE;
    ctx->f1pd[0] = PETSC_FALSE;
    ctx->g00[0][0] = g00;
    ctx->g01[0][0] = NULL;
    ctx->g10[0][0] = NULL;
    ctx->g11[0][0] = g11_null;
    ierr = PetscFECreateDefault(comm,ctx->dim,1,ctx->simplex,"s0_",-1,(PetscFE*)&ctx->sdisc[0]);CHKERRQ(ierr);

    ctx->Nfa = 1;
    ctx->a0[0] = identity;
    ierr = PetscFECreateDefault(comm,ctx->dim,1,ctx->simplex,"a0_",-1,(PetscFE*)&ctx->adisc[0]);CHKERRQ(ierr);

    /* XXX */
    ierr = PetscFECopyQuadrature((PetscFE)ctx->sdisc[0],(PetscFE)ctx->adisc[0]);CHKERRQ(ierr);
    break;
  case 2: /* heat eq, parameter dependency only on u term (via u_x) */
    ctx->Nfs = 1;
    ctx->f0[0] = f0_null;
    ctx->f1[0] = f1;
    ctx->f0pd[0] = PETSC_FALSE;
    ctx->f1pd[0] = PETSC_TRUE;
    ctx->g00[0][0] = g00_null;
    ctx->g01[0][0] = NULL;
    ctx->g10[0][0] = NULL;
    ctx->g11[0][0] = g11;
    ierr = PetscFECreateDefault(comm,ctx->dim,1,ctx->simplex,"s0_",-1,(PetscFE*)&ctx->sdisc[0]);CHKERRQ(ierr);

    ctx->Nfa = 2;
    ctx->a0[0] = identity;
    ctx->a0[1] = identity;
    ierr = PetscFECreateDefault(comm,ctx->dim,1,ctx->simplex,"a0_",-1,(PetscFE*)&ctx->adisc[0]);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(comm,ctx->dim,ctx->dim,ctx->simplex,"a1_",-1,(PetscFE*)&ctx->adisc[1]);CHKERRQ(ierr);

    /* XXX */
    ierr = PetscFECopyQuadrature((PetscFE)ctx->sdisc[0],(PetscFE)ctx->adisc[0]);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature((PetscFE)ctx->sdisc[0],(PetscFE)ctx->adisc[1]);CHKERRQ(ierr);
    break;
  case 3:
    ctx->Nfs = 1;
    ctx->f0[0] = f0;
    ctx->f1[0] = f1;
    ctx->f0pd[0] = PETSC_TRUE;
    ctx->f1pd[0] = PETSC_TRUE;
    ctx->g00[0][0] = g00;
    ctx->g01[0][0] = NULL;
    ctx->g10[0][0] = NULL;
    ctx->g11[0][0] = g11;
    ierr = PetscFECreateDefault(comm,ctx->dim,1,ctx->simplex,"s0_",-1,(PetscFE*)&ctx->sdisc[0]);CHKERRQ(ierr);

    ctx->Nfa = 2;
    ctx->a0[0] = identity;
    ctx->a0[1] = identity;
    ierr = PetscFECreateDefault(comm,ctx->dim,1,ctx->simplex,"a0_",-1,(PetscFE*)&ctx->adisc[0]);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(comm,ctx->dim,ctx->dim,ctx->simplex,"a1_",-1,(PetscFE*)&ctx->adisc[1]);CHKERRQ(ierr);

    /* XXX */
    ierr = PetscFECopyQuadrature((PetscFE)ctx->sdisc[0],(PetscFE)ctx->adisc[0]);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature((PetscFE)ctx->sdisc[0],(PetscFE)ctx->adisc[1]);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown test case %d\n",(int)testcase);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* XXX */
  {
    PetscInt f;

    for (f = 0; f < ctx->Nfs; f++) f0pd[f] = ctx->f0pd[f];
    for (f = 0; f < ctx->Nfs; f++) f1pd[f] = ctx->f1pd[f];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *ctx)
{
  DM             pdm = NULL;
  const PetscInt dim = ctx->dim;
  PetscBool      hasLabel;
  PetscInt       faces[] = {1,1,1};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, dim, ctx->simplex, faces, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMHasLabel(*dm, "marker", &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) {ierr = CreateBCLabel(*dm, "marker");CHKERRQ(ierr);}
  ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
  if (pdm) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = pdm;
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *ctx)
{
  PetscDS        prob;
  PetscInt       f,g;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  for (f = 0; f < ctx->Nfs; f++) {
    ierr = PetscDSSetResidual(prob,f,ctx->f0[f],ctx->f1[f]);CHKERRQ(ierr);
    for (g = 0; g < ctx->Nfs; g++) {
      ierr = PetscDSSetJacobian(prob,f,g,ctx->g00[f][g],ctx->g01[f][g],ctx->g10[f][g],ctx->g11[f][g]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx* ctx)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (f = 0; f < ctx->Nfs; f++) {
    ierr = DMSetField(dm, f, NULL, ctx->sdisc[f]);CHKERRQ(ierr);
  }
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetUpMaterial(DM dm, DM dmAux, AppCtx *ctx)
{
  Vec            nu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dmAux, &nu);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, ctx->a0, NULL, INSERT_ALL_VALUES, nu);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) nu);CHKERRQ(ierr);
  ierr = VecViewFromOptions(nu, NULL, "-lm_view");CHKERRQ(ierr);
  ierr = VecDestroy(&nu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, AppCtx *ctx)
{
  DM             dmAux, coordDM;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ctx->Nfa) {
    ierr = PetscObjectCompose((PetscObject)dm,"dmAux",(PetscObject)NULL);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
  ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
  for (f = 0; f < ctx->Nfa; f++) {
    ierr = DMSetField(dmAux, f, NULL, ctx->adisc[f]);CHKERRQ(ierr);
  }
  ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
  ierr = SetUpMaterial(dm,dmAux,ctx);CHKERRQ(ierr);
  ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyAppCtx(AppCtx *ctx)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (f = 0; f < ctx->Nfs; f++) {
    ierr = PetscObjectDestroy(&ctx->sdisc[f]);CHKERRQ(ierr);
  }
  for (f = 0; f < ctx->Nfa; f++) {
    ierr = PetscObjectDestroy(&ctx->adisc[f]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct {
  DM dmAux;
  VecScatter sctM;
  VecScatter sctdM;

  /* not refcounted */
  TS  ts;
  PetscReal t;
  Vec U;
  Vec Udot;
  Vec M;
} DMLocalPD;

static PetscErrorCode DMLocalPDDestroy_Private(void *ptr)
{
  DMLocalPD*     dml = (DMLocalPD*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterDestroy(&dml->sctM);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&dml->sctdM);CHKERRQ(ierr);
  ierr = VecDestroy(&dml->M);CHKERRQ(ierr);
  ierr = DMDestroy(&dml->dmAux);CHKERRQ(ierr);
  ierr = PetscFree(ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyGradientDAEMult(Mat J, Vec x, Vec y)
{
  DM              dm;
  PetscDS         ds;
  Vec             lM;
  DMLocalPD       *dml;
  PetscObject     oldA,olddmAux;
  PetscDSResidual *f0,*f1;
  PetscInt        f,Nf;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,(void *)&dml);CHKERRQ(ierr);
  ierr = VecScatterBegin(dml->sctdM,x,dml->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(dml->sctdM,x,dml->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dml->dmAux,&lM);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dml->dmAux,dml->M,INSERT_VALUES,lM);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dml->dmAux,dml->M,INSERT_VALUES,lM);CHKERRQ(ierr);
  ierr = TSGetDM(dml->ts,&dm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm,"A",&oldA);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm,"dmAux",&olddmAux);CHKERRQ(ierr);
  ierr = PetscObjectReference(oldA);CHKERRQ(ierr);
  ierr = PetscObjectReference(olddmAux);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dm,"A",(PetscObject)lM);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dm,"dmAux",(PetscObject)dml->dmAux);CHKERRQ(ierr);

  ierr = DMGetDS(dm,&ds);CHKERRQ(ierr); 
  ierr = PetscDSGetNumFields(ds,&Nf);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf,&f0,Nf,&f1);CHKERRQ(ierr);
  for (f = 0; f < Nf; f++) {
    ierr = PetscDSGetResidual(ds,f,&f0[f],&f1[f]);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(ds,f,f0pd[f] ? f0_a : NULL,f1pd[f] ? f1_a : NULL);CHKERRQ(ierr); /* XXX */
  }
  ierr = TSComputeIFunction(dml->ts,dml->t,dml->U,dml->Udot,y,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dm,"A",(PetscObject)oldA);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dm,"dmAux",(PetscObject)olddmAux);CHKERRQ(ierr);
  ierr = PetscObjectDereference(oldA);CHKERRQ(ierr);
  ierr = PetscObjectDereference(olddmAux);CHKERRQ(ierr);
  for (f = 0; f < Nf; f++) {
    ierr = PetscDSSetResidual(ds,f,f0[f],f1[f]);CHKERRQ(ierr); /* XXX */
  }
  ierr = PetscFree2(f0,f1);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dml->dmAux,&lM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyGradientDAEMultTranspose(Mat J, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyEvalGradientDAE(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Mat J, void *ctx)
{
  DMLocalPD      *dml = (DMLocalPD*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(dml->M,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(dml->sctM,M,dml->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(dml->sctM,M,dml->M,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  dml->t = time;
  dml->U = U;
  dml->Udot = Udot;
  PetscFunctionReturn(0);
}

static PetscErrorCode MyTSSetUpFromDesign(TS ts,Vec U0,Vec M,void *ctx)
{
  DM             dm, dmAux;
  Vec            lM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(U0,0.0);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm,"dmAux",(PetscObject*)&dmAux);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMCreateLocalVector(dmAux,&lM);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmAux,M,INSERT_VALUES,lM);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmAux,M,INSERT_VALUES,lM);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"A",(PetscObject)lM);CHKERRQ(ierr);
    ierr = VecDestroy(&lM);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetDMLocalPDCallbacks(TS ts)
{
  DM             dm, dmAux;
  Mat            J;
  Vec            sv,pv;
  IS             *isM,*isM2,isMIn,isMOut,isdMOut;
  PetscInt       m,n,M,N,f,Nfa;
  DMLocalPD      *ctx;
  PetscContainer c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm,"dmAux",(PetscObject*)&dmAux);CHKERRQ(ierr);
  if (!dmAux) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)ts,"dmLocalPD",(PetscObject*)&c);CHKERRQ(ierr);
  if (c) PetscFunctionReturn(0);

  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->ts = ts;
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(ts)),&c);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,ctx);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(c,DMLocalPDDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"dmLocalPD",(PetscObject)c);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  ierr = DMClone(dmAux, &ctx->dmAux);CHKERRQ(ierr);
  ierr = DMGetNumFields(dmAux, &Nfa);CHKERRQ(ierr);
  for (f = 0; f < Nfa; f++) {
    DMLabel     labf;
    PetscObject discf;

    ierr = DMGetField(dmAux, f, &labf, &discf);CHKERRQ(ierr);
    ierr = DMSetField(ctx->dmAux, f, labf, discf);CHKERRQ(ierr);
  }
  for (f = 0; f < Nfa; f++) {
    DMLabel     labf;
    PetscObject discf;

    ierr = DMGetField(dmAux, f, &labf, &discf);CHKERRQ(ierr);
    ierr = DMSetField(ctx->dmAux, f + Nfa, labf, discf);CHKERRQ(ierr);
  }
  ierr = DMCreateDS(ctx->dmAux);CHKERRQ(ierr);

  ierr = DMCreateFieldIS(dmAux,NULL,NULL,&isM);CHKERRQ(ierr);
  ierr = DMCreateFieldIS(ctx->dmAux,NULL,NULL,&isM2);CHKERRQ(ierr);
  ierr = ISConcatenate(PetscObjectComm((PetscObject)ts),Nfa,isM,&isMIn);CHKERRQ(ierr);
  ierr = ISConcatenate(PetscObjectComm((PetscObject)ts),Nfa,isM2,&isMOut);CHKERRQ(ierr);
  ierr = ISConcatenate(PetscObjectComm((PetscObject)ts),Nfa,isM2+Nfa,&isdMOut);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(ctx->dmAux,&ctx->M);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmAux,&pv);CHKERRQ(ierr);
  ierr = VecScatterCreate(pv,isMIn,ctx->M,isMOut,&ctx->sctM);CHKERRQ(ierr);
  ierr = VecScatterCreate(pv,isMIn,ctx->M,isdMOut,&ctx->sctdM);CHKERRQ(ierr);
  ierr = ISDestroy(&isMIn);CHKERRQ(ierr);
  ierr = ISDestroy(&isMOut);CHKERRQ(ierr);
  ierr = ISDestroy(&isdMOut);CHKERRQ(ierr);
  for (f = 0; f < Nfa; f++) {
    ierr = ISDestroy(&isM[f]);CHKERRQ(ierr);
    ierr = ISDestroy(&isM2[f]);CHKERRQ(ierr);
    ierr = ISDestroy(&isM2[f+Nfa]);CHKERRQ(ierr);
  }
  ierr = PetscFree(isM);CHKERRQ(ierr);
  ierr = PetscFree(isM2);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dm,&sv);CHKERRQ(ierr);
  ierr = VecGetLocalSize(sv,&m);CHKERRQ(ierr);
  ierr = VecGetSize(sv,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(pv,&n);CHKERRQ(ierr);
  ierr = VecGetSize(pv,&N);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)ts),m,n,M,N,ctx,&J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J,MATOP_MULT,(void(*)(void))MyGradientDAEMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J,MATOP_MULT_TRANSPOSE,(void(*)(void))MyGradientDAEMultTranspose);CHKERRQ(ierr);
  ierr = TSSetGradientDAE(ts,J,MyEvalGradientDAE,ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSSetSetUpFromDesign(ts,MyTSSetUpFromDesign,ctx);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&sv);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmAux,&pv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode RunTest(DM dm)
{
  Vec            M,U,Udot,F;
  Vec            lM,lU,lUdot,lF;
  PetscRandom    r;
  DM             coordDM, dmAux;
  PetscDS        sds, ads;
  PetscErrorCode ierr;
  PetscReal      t = 0.0;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm,&sds);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)sds),"State DS\n");CHKERRQ(ierr);
  ierr = PetscDSView(sds,NULL);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm,"dmAux",(PetscObject*)&dmAux);CHKERRQ(ierr);
  ierr = DMGetDS(dmAux,&ads);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ads),"Param DS\n");CHKERRQ(ierr);
  ierr = PetscDSView(ads,NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)dm),&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

#if 0
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);

  ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
  ierr = DMClearDS(dmAux);CHKERRQ(ierr);
  ierr = DMSetRegionDS(dmAux, NULL, NULL, pds);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmAux, &M);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmAux, &lM);CHKERRQ(ierr);
  ierr = VecSetRandom(M,r);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmAux, M, INSERT_VALUES, lM);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmAux, M, INSERT_VALUES, lM);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) lM);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmAux, &M);CHKERRQ(ierr);
  ierr = VecViewFromOptions(lM,NULL,"-lm_view");CHKERRQ(ierr);
  ierr = VecDestroy(&lM);CHKERRQ(ierr);
  ierr = DMDestroy(&dmAux);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dm, &F);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &U);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &Udot);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &lU);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &lUdot);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &lF);CHKERRQ(ierr);
  ierr = VecSetRandom(U,r);CHKERRQ(ierr);
  ierr = VecSetRandom(Udot,r);CHKERRQ(ierr);
  ierr = VecZeroEntries(lU);CHKERRQ(ierr);
  ierr = VecZeroEntries(lUdot);CHKERRQ(ierr);
  //if (dmlocalts->boundarylocal) {ierr = (*dmlocalts->boundarylocal)(dm, time, locX, locX_t,dmlocalts->boundarylocalctx);CHKERRQ(ierr);}
  ierr = DMGlobalToLocalBegin(dm, U, INSERT_VALUES, lU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, U, INSERT_VALUES, lU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, Udot, INSERT_VALUES, lUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, Udot, INSERT_VALUES, lUdot);CHKERRQ(ierr);
  ierr = VecViewFromOptions(lU,NULL,"-lu_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(lUdot,NULL,"-ludot_view");CHKERRQ(ierr);
  ierr = VecZeroEntries(lF);CHKERRQ(ierr);
  ierr = DMPlexTSComputeIFunctionFEM(dm,t,lU,lUdot,lF,NULL);CHKERRQ(ierr);
  ierr = VecViewFromOptions(lF,NULL,"-lf_view");CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, lF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, lF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &lU);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &lUdot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &lF);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &U);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &Udot);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &F);CHKERRQ(ierr);
#endif

  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

int main(int argc, char **argv)
{
  AppCtx         ctx;
  DM             dm;
  TS             ts;
  PetscErrorCode ierr;

  ierr = PetscOptInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &ctx);CHKERRQ(ierr);
  ierr = SetupAuxDM(dm, &ctx);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSSetDMLocalPDCallbacks(ts);CHKERRQ(ierr);
  //ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  /* check DAE callbacks terms */
  {
    PetscRandom r;
    Vec         U,M = NULL,L,Udot;
    DM          dmAux;

    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&U);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&Udot);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&L);CHKERRQ(ierr);
    ierr = VecSetRandom(U,r);CHKERRQ(ierr);
    ierr = VecSetRandom(Udot,r);CHKERRQ(ierr);
    ierr = VecSetRandom(L,r);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)dm,"dmAux",(PetscObject*)&dmAux);CHKERRQ(ierr);
    if (dmAux) {
      ierr = DMGetGlobalVector(dmAux,&M);CHKERRQ(ierr);
      ierr = VecSetRandom(M,r);CHKERRQ(ierr);
    }
    ierr = TSCheckGradientDAE(ts,0.0,U,Udot,M);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&U);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&Udot);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&L);CHKERRQ(ierr);
    if (dmAux) {
      ierr = DMRestoreGlobalVector(dmAux,&M);CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  }
  //ierr = RunTest(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DestroyAppCtx(&ctx);CHKERRQ(ierr);
  ierr = PetscOptFinalize();
  return ierr;
}

/*TEST

TEST*/

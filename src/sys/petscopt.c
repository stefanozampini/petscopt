#include <petscopt/private/petscoptimpl.h>
#include <petscopt/private/tsoptimpl.h>
#include <petscopt/private/tsobjimpl.h>
#include <petscopt/private/augmentedtsimpl.h>
#include <petsctao.h>
#include <petscts.h>

PetscBool PetscOptInitializeCalled   = PETSC_FALSE;
PetscBool PetscOptPackageInitialized = PETSC_FALSE;
PetscBool PetscOptFinalizePetsc      = PETSC_FALSE;

PetscLogEvent TSOPT_Obj_Eval                = 0;
PetscLogEvent TSOPT_Opt_Eval_Grad_DAE       = 0;
PetscLogEvent TSOPT_Opt_Eval_Grad_IC        = 0;
PetscLogEvent TSOPT_Opt_Eval_Hess_DAE[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
PetscLogEvent TSOPT_Opt_Eval_Hess_IC[2][2]  = {{0,0},{0,0}};
PetscLogEvent TSOPT_Opt_SetUp               = 0;
PetscLogEvent TSOPT_FOA_Forcing             = 0;
PetscLogEvent TSOPT_FOA_Quad                = 0;
PetscLogEvent TSOPT_SOA_Forcing             = 0;
PetscLogEvent TSOPT_SOA_Quad                = 0;
PetscLogEvent TSOPT_TLM_Forcing             = 0;
PetscLogEvent TSOPT_API_Obj                 = 0;
PetscLogEvent TSOPT_API_ObjGrad             = 0;
PetscLogEvent TSOPT_API_Grad                = 0;
PetscLogEvent TSOPT_API_HSetUp              = 0;
PetscLogEvent TSOPT_API_HMult               = 0;
PetscLogEvent TSOPT_API_HMultTLM            = 0;
PetscLogEvent TSOPT_API_HMultSOA            = 0;

static PetscErrorCode PetscOptFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscOptPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode KSPCreate_AugTriangular(KSP);
PETSC_EXTERN PetscErrorCode SNESCreate_Augmented(SNES);

PetscErrorCode PetscOptInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscOptPackageInitialized) PetscFunctionReturn(0);
  if (!PetscOptInitializeCalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call PetscOptInitialize()");
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = SNESInitializePackage();CHKERRQ(ierr);
  ierr = TSInitializePackage();CHKERRQ(ierr);
  ierr = TaoInitializePackage();CHKERRQ(ierr);

  PetscOptPackageInitialized = PETSC_TRUE;
  /* Register classes */
  ierr = KSPRegister(KSPAUGTRIANGULAR,KSPCreate_AugTriangular);CHKERRQ(ierr);
  ierr = SNESRegister(SNESAUGMENTED,SNESCreate_Augmented);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("TSOptEvalGrad",     0,&TSOPT_Opt_Eval_Grad_DAE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalGradIC",   0,&TSOPT_Opt_Eval_Grad_IC);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHXX",      0,&TSOPT_Opt_Eval_Hess_DAE[0][0]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHXXdot",   0,&TSOPT_Opt_Eval_Hess_DAE[0][1]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHXM",      0,&TSOPT_Opt_Eval_Hess_DAE[0][2]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHXdotX",   0,&TSOPT_Opt_Eval_Hess_DAE[1][0]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHXdotXdot",0,&TSOPT_Opt_Eval_Hess_DAE[1][1]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHXdotM",   0,&TSOPT_Opt_Eval_Hess_DAE[1][2]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHMX",      0,&TSOPT_Opt_Eval_Hess_DAE[2][0]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHMXdot",   0,&TSOPT_Opt_Eval_Hess_DAE[2][1]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHMM",      0,&TSOPT_Opt_Eval_Hess_DAE[2][2]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHICXX",    0,&TSOPT_Opt_Eval_Hess_IC[0][0]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHICXM",    0,&TSOPT_Opt_Eval_Hess_IC[0][1]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHICMX",    0,&TSOPT_Opt_Eval_Hess_IC[1][0]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptEvalHICMM",    0,&TSOPT_Opt_Eval_Hess_IC[1][1]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptSetUp",        0,&TSOPT_Opt_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptFOAForcing",   0,&TSOPT_FOA_Forcing);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptFOAQuad",      0,&TSOPT_FOA_Quad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptSOAForcing",   0,&TSOPT_SOA_Forcing);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptSOAQuad",      0,&TSOPT_SOA_Quad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSOptTLMForcing",   0,&TSOPT_TLM_Forcing);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSObjective",       0,&TSOPT_API_Obj);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSObjGrad",         0,&TSOPT_API_ObjGrad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSGradient",        0,&TSOPT_API_Grad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSHessianSetUp",    0,&TSOPT_API_HSetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSHessianMult",     0,&TSOPT_API_HMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSHessianMultTLM",  0,&TSOPT_API_HMultTLM);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSHessianMultSOA",  0,&TSOPT_API_HMultSOA);CHKERRQ(ierr);
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  /* LCOV_EXCL_START */
  if (opt) {
    ierr = PetscStrInList("tsopt",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {
      PetscInt i,j;

      ierr = PetscLogEventDeactivate(TSOPT_Opt_Eval_Grad_DAE);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_Opt_Eval_Grad_IC);CHKERRQ(ierr);
      for (i=0;i<3;i++) {
        for (j=0;j<3;j++) {
	  ierr = PetscLogEventDeactivate(TSOPT_Opt_Eval_Hess_DAE[i][j]);CHKERRQ(ierr);
        }
      }
      for (i=0;i<2;i++) {
        for (j=0;j<2;j++) {
	  ierr = PetscLogEventDeactivate(TSOPT_Opt_Eval_Hess_IC[i][j]);CHKERRQ(ierr);
        }
      }
      ierr = PetscLogEventDeactivate(TSOPT_Opt_SetUp);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_FOA_Forcing);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_FOA_Quad);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_SOA_Forcing);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_SOA_Quad);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_TLM_Forcing);CHKERRQ(ierr);
    }
    ierr = PetscStrInList("tsoptapi",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {
      ierr = PetscLogEventDeactivate(TSOPT_API_Obj);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_API_ObjGrad);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_API_Grad);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_API_HSetUp);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_API_HMult);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_API_HMultTLM);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(TSOPT_API_HMultSOA);CHKERRQ(ierr);
    }
    ierr = PetscStrInList("tsobj",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventDeactivate(TSOPT_Obj_Eval);CHKERRQ(ierr);}
  }
  /* LCOV_EXCL_STOP */
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscOptFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOptInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscErrorCode ierr;

  if (PetscOptInitializeCalled) return 0;
  PetscOptFinalizePetsc = (PetscBool)!PetscInitializeCalled;
  ierr = PetscInitialize(argc,args,file,help); if (ierr) return ierr;
  PetscFunctionBegin;
  PetscOptInitializeCalled = PETSC_TRUE;
  ierr = PetscOptInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOptFinalize()
{
  PetscErrorCode ierr;

  if (!PetscOptInitializeCalled) {
    printf("PetscOptInitialize() must be called before PetscOptFinalize()\n");
    return(PETSC_ERR_ARG_WRONGSTATE);
  }
  if (PetscOptFinalizePetsc) {
    ierr = PetscFinalize();
    if (ierr) return ierr;
  }
  PetscFunctionReturn(0);
}

PetscBool PetscOptInitialized()
{
  return PetscOptInitializeCalled;
}

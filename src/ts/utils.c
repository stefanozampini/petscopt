#include <petscopt/tsutils.h>

PetscErrorCode TSCreateWithTS(TS ts, TS *nts)
{
  PetscErrorCode ierr;
  TSType         type;
  TSEquationType eqtype;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),nts);CHKERRQ(ierr);
  ierr = TSGetType(ts,&type);CHKERRQ(ierr);
  ierr = TSSetType(*nts,type);CHKERRQ(ierr);
  ierr = TSGetEquationType(ts,&eqtype);CHKERRQ(ierr);
  ierr = TSSetEquationType(*nts,eqtype);CHKERRQ(ierr);
  /* this is disabled since it is buggy (up to 3.13) */
#if 0
  TSProblemType  prtype;
  ierr = TSGetProblemType(ts,&prtype);CHKERRQ(ierr);
  ierr = TSSetProblemType(*nts,prtype);CHKERRQ(ierr);
#endif
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSRK,&flg);CHKERRQ(ierr);
  if (flg) {
    TSRKType rktype;

    ierr = TSRKGetType(ts,&rktype);CHKERRQ(ierr);
    ierr = TSRKSetType(*nts,rktype);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSBDF,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscInt o;

    ierr = TSBDFGetOrder(ts,&o);CHKERRQ(ierr);
    ierr = TSBDFSetOrder(*nts,o);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSTHETA,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscReal th;
    PetscBool ep;

    ierr = TSThetaGetTheta(ts,&th);CHKERRQ(ierr);
    ierr = TSThetaSetTheta(*nts,th);CHKERRQ(ierr);
    ierr = TSThetaGetEndpoint(ts,&ep);CHKERRQ(ierr);
    ierr = TSThetaSetEndpoint(*nts,ep);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSALPHA,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscReal a,b,c;

    ierr = TSAlphaGetParams(ts,&a,&b,&c);CHKERRQ(ierr);
    ierr = TSAlphaSetParams(*nts,a,b,c);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSROSW,&flg);CHKERRQ(ierr);
  if (flg) {
    TSRosWType t;

    ierr = TSRosWGetType(ts,&t);CHKERRQ(ierr);
    ierr = TSRosWSetType(*nts,t);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSARKIMEX,&flg);CHKERRQ(ierr);
  if (flg) {
    TSARKIMEXType t;

    ierr = TSARKIMEXGetType(ts,&t);CHKERRQ(ierr);
    ierr = TSARKIMEXSetType(*nts,t);CHKERRQ(ierr);
#if PETSC_VERSION_GE(3,13,0)
    ierr = TSARKIMEXGetFullyImplicit(ts,&flg);CHKERRQ(ierr);
    ierr = TSARKIMEXSetFullyImplicit(*nts,flg);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

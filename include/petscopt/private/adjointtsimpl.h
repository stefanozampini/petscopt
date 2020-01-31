#ifndef __PETSCOPT_ADJOINTTSIMPL_H
#define __PETSCOPT_ADJOINTTSIMPL_H

#include <petscopt/adjointts.h>
#include <petscopt/private/tsobjimpl.h>

typedef struct {
  Vec            design;      /* design vector (fixed) */
  PetscBool      userijac;    /* use userdefined IJacobian routine */
  TS             fwdts;       /* forward solver */
  TSObj          tsobj;       /* Objective functions linked list */
  PetscReal      t0,tf;       /* time limits, for forward time recovery */
  Vec            workinit;    /* work vector, used to initialize the adjoint variables and for Dirac's delta terms */
  Vec            quadvec;     /* vector to store the quadrature (gradient or Hessian matvec) */
  Vec            wquad;       /* work vector */
  PetscBool      dirac_delta; /* If true, means that a delta contribution needs to be added to lambda during the post step method */
  Vec            direction;   /* If present, it is a second-order adjoint */
  TS             tlmts;       /* Tangent Linear Model TS, used for Hessian matvecs */
  TS             foats;       /* First order adjoint TS, used for Hessian matvecs when solving for the second order adjoint */
  PetscBool      discrete;
  PetscErrorCode (*setup)(TS);
  PetscErrorCode (*cstep)(TS);
} AdjointCtx;

/* Check sanity of the AdjointTS */
#if !defined(PETSC_USE_DEBUG)
#define PetscCheckAdjointTS(a) do {} while (0)
#else
#define PetscCheckAdjointTS(a)                                                                                                                 \
  do {                                                                                                                                         \
    PetscErrorCode __ierr;                                                                                                                     \
    PetscContainer __c;                                                                                                                        \
    AdjointCtx    *__ac;                                                                                                                       \
    TSObj         __tsobj;                                                                                                                     \
    void          *__cc;                                                                                                                       \
    __ierr = TSGetApplicationContext((a),(void*)&__ac);CHKERRQ(__ierr);                                                                        \
    __ierr = PetscObjectQuery((PetscObject)(a),"_ts_adjctx",(PetscObject*)&__c);CHKERRQ(__ierr);                                               \
    if (!__c) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_USER,"The TS was not obtained from calling TSCreateAdjointTS()");            \
    __ierr = PetscContainerGetPointer(__c,(void**)&__cc);CHKERRQ(__ierr);                                                                      \
    if (__cc != __ac) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_USER,"You cannot change the application context for the AdjointTS"); \
    if (!__ac->fwdts) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_PLIB,"Missing forward model TS");                                    \
    __ierr = TSGetTSObj(__ac->fwdts,&__tsobj);CHKERRQ(__ierr);                                                                                 \
    if (__ac->tsobj != __tsobj) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_PLIB,"Inconsistency between TSObj objects");               \
  } while (0)
#endif

PETSC_INTERN PetscErrorCode AdjointTSSolveWithQuadrature_Private(TS);

#endif

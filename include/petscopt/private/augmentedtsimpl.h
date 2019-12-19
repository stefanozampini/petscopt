#ifndef __PETSCOPT_AUGMENTEDTSIMPL_H
#define __PETSCOPT_AUGMENTEDTSIMPL_H

#include <petscts.h>

typedef struct {
  TS             model;
  PetscInt       nqts;
  TS             *qts;
  PetscErrorCode (**updatestates)(TS,Vec,Vec);
  TSRHSJacobian  *rhsjaccoupling;
  Vec            *U;
  Vec            *Udot;
  Vec            *F;
} TSAugCtx;

/* Check sanity of the AugmentedTS */
#if !defined(PETSC_USE_DEBUG)
#define PetscCheckAugumentedTS(a) do {} while (0)
#else
#define PetscCheckAugumentedTS(a)                                                                                                                \
  do {                                                                                                                                           \
    PetscErrorCode __ierr;                                                                                                                       \
    PetscContainer __c;                                                                                                                          \
    TSAugCtx      *__ac;                                                                                                                         \
    void          *__cc;                                                                                                                         \
    __ierr = TSGetApplicationContext((a),(void*)&__ac);CHKERRQ(__ierr);                                                                          \
    __ierr = PetscObjectQuery((PetscObject)(a),"_ts_aug_ctx",(PetscObject*)&__c);CHKERRQ(__ierr);                                                \
    if (!__c) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_USER,"The TS was not obtained from calling TSCreateAugmentedTS()");            \
    __ierr = PetscContainerGetPointer(__c,(void**)&__cc);CHKERRQ(__ierr);                                                                        \
    if (__cc != __ac) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_USER,"You cannot change the application context for the AugmentedTS"); \
  } while (0)
#endif

#endif

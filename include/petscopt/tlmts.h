#ifndef __PETSCOPT_TLMTS_H
#define __PETSCOPT_TLMTS_H

#include <petscts.h>

PETSC_EXTERN PetscErrorCode TSCreateTLMTS(TS,TS*);
PETSC_EXTERN PetscErrorCode TLMTSGetRHSVec(TS,Vec*);
PETSC_EXTERN PetscErrorCode TLMTSComputeInitialConditions(TS,PetscReal,Vec);
PETSC_EXTERN PetscErrorCode TLMTSComputeForcing(TS,PetscReal,Vec,Vec,PetscBool*,Vec);
PETSC_EXTERN PetscErrorCode TLMTSSetPerturbationVec(TS,Vec);
PETSC_EXTERN PetscErrorCode TLMTSSetDesignVec(TS,Vec);
PETSC_EXTERN PetscErrorCode TLMTSGetDesignVec(TS,Vec*);
PETSC_EXTERN PetscErrorCode TLMTSGetModelTS(TS,TS*);
PETSC_EXTERN PetscErrorCode TLMTSIsDiscrete(TS,PetscBool*);
PETSC_EXTERN PetscErrorCode TLMTSSetUpStep(TS);

/* Check sanity of the TLMTS */
#if !defined(PETSC_USE_DEBUG)
#define PetscCheckTLMTS(a) do {} while (0)
#else
#define PetscCheckTLMTS(a)                                                                                                                 \
  do {                                                                                                                                     \
    PetscErrorCode __ierr;                                                                                                                 \
    PetscContainer __c;                                                                                                                    \
    void *__ac,*__cc;                                                                                                                      \
    __ierr = TSGetApplicationContext((a),(void*)&__ac);CHKERRQ(__ierr);                                                                    \
    __ierr = PetscObjectQuery((PetscObject)(a),"_ts_tlm_ctx",(PetscObject*)&__c);CHKERRQ(__ierr);                                          \
    if (!__c) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_USER,"The TS was not obtained from calling TSCreateTLMTS()");            \
    __ierr = PetscContainerGetPointer(__c,(void**)&__cc);CHKERRQ(__ierr);                                                                  \
    if (__cc != __ac) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_USER,"You cannot change the application context for the TLMTS"); \
  } while (0)
#endif

#endif

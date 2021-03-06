#ifndef __PETSCOPT_TLMTSIMPL_H
#define __PETSCOPT_TLMTSIMPL_H

#include <petscopt/tlmts.h>

typedef struct {
  TS             model;
  PetscBool      userijac;
  Vec            workrhs;
  Vec            design;
  Vec            mdelta;
  Mat            P;
  PetscBool      discrete;
  PetscErrorCode (*setup)(TS);
  PetscErrorCode (*cstep)(TS);
} TLMTS_Ctx;

#endif

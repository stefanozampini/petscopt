#if !defined(_MFEMOPT_PETSCMACROS_H)
#define _MFEMOPT_PETSCMACROS_H

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <petscsys.h>
#include <petscerror.h>

#define PCHKERRQ(obj,err) do {                                                   \
     if ((err))                                                                  \
     {                                                                           \
        PetscError(PetscObjectComm((PetscObject)(obj)),__LINE__,_MFEM_FUNC_NAME, \
                   __FILE__,(err),PETSC_ERROR_REPEAT,NULL);                      \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");                     \
     }                                                                           \
  } while(0);
#define CCHKERRQ(comm,err) do {                                \
     if ((err))                                                \
     {                                                         \
        PetscError(comm,__LINE__,_MFEM_FUNC_NAME,              \
                   __FILE__,(err),PETSC_ERROR_REPEAT,NULL);    \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");   \
     }                                                         \
  } while(0);
#endif

#endif

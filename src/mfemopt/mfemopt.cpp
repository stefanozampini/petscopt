#include <petscopt.h>
#include <mfem/general/error.hpp> 

PetscBool MFEMOptFinalizePetscOpt = PETSC_FALSE; 

void MFEMOptInitialize(int *argc,char ***argv,const char rc_file[],
                       const char help[])
{
   MFEMOptFinalizePetscOpt = (PetscBool)!PetscOptInitialized();
   PetscErrorCode ierr = PetscOptInitialize(argc,argv,rc_file,help);
   MFEM_VERIFY(!ierr,"Unable to initialize PETScOpt");
}

void MFEMOptFinalize()
{
   if (MFEMOptFinalizePetscOpt)
   {
      PetscErrorCode ierr = PetscOptFinalize();
      MFEM_VERIFY(!ierr,"Unable to finalize PETScOpt");
   }
}


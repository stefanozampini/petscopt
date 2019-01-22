#if !defined(_MFEMOPT_MFEMEXTRA_HPP)
#define _MFEMOPT_MFEMEXTRA_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemopt/reducedfunctional.hpp>
#include <mfem/linalg/petsc.hpp>

namespace mfemopt
{

class PetscNonlinearSolverOpt : public mfem::PetscNonlinearSolver
{
public:
   PetscNonlinearSolverOpt(MPI_Comm,ReducedFunctional&,
                           const std::string& = std::string(),
                           bool = true);
};

}
#endif

#endif

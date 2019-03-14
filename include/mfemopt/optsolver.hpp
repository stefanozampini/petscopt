#if !defined(_MFEMOPT_OPTSOLVER_HPP)
#define _MFEMOPT_OPTSOLVER_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemopt/reducedfunctional.hpp>
#include <mfem/linalg/vector.hpp>
#include <mfem/linalg/petsc.hpp>
#include <petsctao.h>

namespace mfemopt
{

// Abstract class for optimization solvers
class OptimizationSolver
{
protected:
   ReducedFunctional *objective;

public:
   OptimizationSolver() : objective(NULL) { } ;
   virtual void Init(ReducedFunctional& _f) { objective = &_f; };
   virtual void Solve(mfem::Vector&) = 0;
   virtual ~OptimizationSolver() { };
};

class PetscOptimizationSolver : public OptimizationSolver, public mfem::PetscSolver
{
private:
   Tao tao;
   void *private_ctx;

   void CreatePrivateContext();

public:
   PetscOptimizationSolver(MPI_Comm, const std::string &prefix = std::string());
   PetscOptimizationSolver(MPI_Comm, ReducedFunctional&, const std::string &prefix = std::string());
   operator Tao() const { return tao; }

   virtual void Init(ReducedFunctional&);
   virtual void Solve(mfem::Vector&);
   void SetMonitor(mfem::PetscSolverMonitor*);
   virtual ~PetscOptimizationSolver();
};

}
#endif

#endif

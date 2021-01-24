#if !defined(_MFEMOPT_OPTSOLVER_HPP)
#define _MFEMOPT_OPTSOLVER_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfemopt/reducedfunctional.hpp>
#include <mfem/linalg/vector.hpp>
#include <mfem/linalg/petsc.hpp>

// Forward declare Tao type
typedef struct _p_Tao *Tao;

namespace mfemopt
{

// Abstract class for optimization solvers
class OptimizationSolver
{
protected:
   ReducedFunctional *objective;

public:
   bool iterative_mode; /* MFEM */
   OptimizationSolver() : objective(NULL), iterative_mode(false) { } ;
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
   PetscOptimizationSolver(MPI_Comm,const std::string &prefix = std::string());
   PetscOptimizationSolver(MPI_Comm,ReducedFunctional&,const std::string &prefix = std::string());
   operator Tao() const { return tao; }

   virtual void Init(ReducedFunctional&);
   virtual void Solve(mfem::Vector&);
   void SetHessianType(mfem::Operator::Type);
   void SetMonitor(mfem::PetscSolverMonitor*);
   virtual ~PetscOptimizationSolver();
};

class PetscNonlinearSolverOpt : public mfem::PetscNonlinearSolver
{
public:
   PetscNonlinearSolverOpt(MPI_Comm,ReducedFunctional&,const std::string& = std::string(),bool = true);
};

}
#endif

#endif

#if !defined(_MFEMOPT_REDUCEDFUNCTIONAL_HPP)
#define _MFEMOPT_REDUCEDFUNCTIONAL_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfem/linalg/vector.hpp>
#include <mfem/linalg/handle.hpp>
#include <mfem/linalg/petsc.hpp>

namespace mfemopt
{
// Abstract class for functions of the type f : R^m -> R to be minimized
class ReducedFunctional
{
public:
   ReducedFunctional() {}

   virtual int GetParameterSize() = 0;
   virtual void GetBounds(mfem::Vector&,mfem::Vector&);
   virtual void ComputeGuess(mfem::Vector&);
   virtual void ComputeObjective(const mfem::Vector&, double*)
   { mfem::mfem_error("ReducedFunctional::Compute not overloaded!"); }
   virtual void ComputeGradient(const mfem::Vector&, mfem::Vector&)
   { mfem::mfem_error("ReducedFunctional::ComputeGradient not overloaded!"); }
   virtual void ComputeObjectiveAndGradient(const mfem::Vector& m, double *f, mfem::Vector& g)
   {
     ComputeObjective(m,f);
     ComputeGradient(m,g);
   }
   virtual mfem::Operator* GetHessian(const mfem::Vector&) { return NULL; }
   void TestFDGradient(MPI_Comm,const mfem::Vector&,double,bool=true);
   void TestFDHessian(MPI_Comm,const mfem::Vector&);
   virtual ~ReducedFunctional() {};
};

class ReducedFunctionalHessianOperatorFD : public mfem::PetscParMatrix
{
private:
   ReducedFunctional* obj;

public:
   ReducedFunctionalHessianOperatorFD(MPI_Comm,ReducedFunctional*,const mfem::Vector&);
};

}
#endif

#endif

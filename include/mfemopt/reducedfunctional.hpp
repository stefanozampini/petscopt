#if !defined(_MFEMOPT_REDUCEDFUNCTIONAL_HPP)
#define _MFEMOPT_REDUCEDFUNCTIONAL_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/linalg/vector.hpp>
#include <mfem/linalg/handle.hpp>
#include <mfem/linalg/petsc.hpp>

namespace mfemopt
{
// Abstract class for functions of the type f : R^m -> R to be minimized
class ReducedFunctional : public mfem::Operator
{
public:
   ReducedFunctional() {}

   virtual void ComputeObjective(const mfem::Vector&,double*) const
   { mfem::mfem_error("ReducedFunctional::Compute not overloaded!"); }
   virtual void ComputeGradient(const mfem::Vector&,mfem::Vector&) const
   { mfem::mfem_error("ReducedFunctional::ComputeGradient not overloaded!"); }
   virtual void ComputeObjectiveAndGradient(const mfem::Vector& m,double *f,mfem::Vector& g) const
   {
      ComputeObjective(m,f);
      ComputeGradient(m,g);
   }
   virtual mfem::Operator& GetHessian(const mfem::Vector&) const
   {
      mfem::mfem_error("ReducedFunctional::GetHessian() is not overloaded!");
      return const_cast<ReducedFunctional&>(*this);
   }

   virtual void ComputeGuess(mfem::Vector&) const;
   virtual void GetBounds(mfem::Vector&,mfem::Vector&) const;

   virtual void Init(const mfem::Vector&) {};
   virtual void Update(int,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&) {};
   virtual void PostCheck(const mfem::Vector&,mfem::Vector&,mfem::Vector&,bool &cy,bool &cw) const
   { cy = false; cw = false; }

   /* Default interface for mfem::Operator */
   virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const
   { ComputeGradient(x,y); }
   virtual mfem::Operator& GetGradient(const mfem::Vector& m) const
   { return GetHessian(m); }

   /* Testing */
   void TestFDGradient(MPI_Comm,const mfem::Vector&,double,bool=true);
   void TestFDHessian(MPI_Comm,const mfem::Vector&);
   void TestTaylor(MPI_Comm,const mfem::Vector&,bool=false);
   void TestTaylor(MPI_Comm,const mfem::Vector&,const mfem::Vector&,bool=false);

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

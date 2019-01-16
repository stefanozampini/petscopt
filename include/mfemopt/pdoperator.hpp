#if !defined(_MFEMOPT_PDOPERATOR_HPP)
#define _MFEMOPT_PDOPERATOR_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfem/linalg/operator.hpp>
#include <mfem/linalg/vector.hpp>
#include <mfem/linalg/petsc.hpp>
#include <petscts.h>

namespace mfemopt
{

class PDOperator;

class PDOperatorGradient : public mfem::Operator
{
private:
   PDOperator *op;
   mutable mfem::Vector tdst,st,m;

public:
   PDOperatorGradient(PDOperator*);
   void Update(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&);
   virtual void Mult(const mfem::Vector&,mfem::Vector&) const;
   virtual void MultTranspose(const mfem::Vector&,mfem::Vector&) const;
   virtual ~PDOperatorGradient() {}
};

class PDOperatorHessian : public mfem::Operator
{
private:
   PDOperator *op;
   mutable mfem::Vector tdst,st,m;
   mutable mfem::Vector l;
   int A,B;

public:
   PDOperatorHessian(PDOperator*,int,int);
   void Update(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&);
   virtual void Mult(const mfem::Vector&,mfem::Vector&) const;
   virtual ~PDOperatorHessian() {}
};

class PDOperatorGradientFD : public mfem::PetscParMatrix
{
private:
   PDOperator* pd;

   mfem::Vector xpIn;
   mfem::Vector xIn;
   mfem::Vector mIn;
   double t;

public:
   PDOperatorGradientFD(MPI_Comm,PDOperator*,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,double);
};

class PDOperator
{
private:
   PDOperatorGradient* GOp;
   PDOperatorHessian* HOp[3][3];

   mfem::PetscBCHandler bc; /* unused */
   mfem::PetscBCHandler hbc; /* XXX */

protected:
   friend class PDOperatorGradient;
   friend class PDOperatorHessian;

   void ApplyHomogeneousBC(mfem::Vector&);

public:
   PDOperator();

   void SetBCHandler(mfem::PetscBCHandler&);

   /* This maps parameter to state
      Given a parameter dependent F(xdot,x,t;m) residual equation, implements the action of dF/dm */
   PDOperatorGradient* GetGradientOperator();
   /* Given a parameter dependent F(xdot,x,t;m) residual equation, implements the action of
        Y = (L^T \otimes I_A)*F_AB*X -> Y = (\sum_k L_k*F^k_AB)*X
      with A,B = {xdot|x|m}, L the adjoint state and F^k the k-th residual equation. \otimes the Kronecker product
   */
   PDOperatorHessian* GetHessianOperator(int,int);

   virtual void ComputeInitialConditions(double,mfem::Vector&,const mfem::Vector&);
   virtual void SetUpFromParameters(const mfem::Vector&) = 0;
   virtual int GetStateSize() = 0;
   virtual int GetParameterSize() = 0;

   /* the action of the parameter dependent residual */
   virtual void Mult(const mfem::Vector& /* Xdot */,const mfem::Vector& /* X */,const mfem::Vector& /* M */,double,mfem::Vector&) = 0;

   virtual void ComputeGradient(const mfem::Vector& /* Xdot */,const mfem::Vector& /* X */,const mfem::Vector& /* M */,const mfem::Vector& /* Pert */,mfem::Vector&)
   { mfem::mfem_error("PDOperator::ComputeGradient not overloaded!"); }
   virtual void ComputeGradientAdjoint(const mfem::Vector& /* L */,const mfem::Vector& /* Xdot */,const mfem::Vector& /* X */,const mfem::Vector& /* M */,mfem::Vector&)
   { mfem::mfem_error("PDOperator::ComputeGradientAdjoint not overloaded!"); }
   virtual void ComputeHessian(int,int,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,
                               const mfem::Vector&,const mfem::Vector&,mfem::Vector&)
   { mfem::mfem_error("PDOperator::ComputeHessian not overloaded!"); }

   void TestFDGradient(MPI_Comm,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,double);
   virtual ~PDOperator();
};

PETSC_EXTERN PetscErrorCode mfemopt_setupts(TS,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode mfemopt_gradientdae(TS,PetscReal,Vec,Vec,Vec,Mat,void*);
PETSC_EXTERN PetscErrorCode mfemopt_hessiandae_xtm(TS,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode mfemopt_hessiandae_mxt(TS,PetscReal,Vec,Vec,Vec,Vec,Vec,Vec,void*);

}
#endif

#endif

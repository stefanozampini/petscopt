#if !defined(_MFEMOPT_MODELPROBLEMS_HPP)
#define _MFEMOPT_MODELPROBLEMS_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfemopt/pdoperator.hpp>
#include <mfemopt/pdbilininteg.hpp>
#include <mfem/fem/pfespace.hpp>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/fem/plinearform.hpp>
#include <mfem/fem/pbilinearform.hpp>
#include <mfem/linalg/petsc.hpp>

namespace mfemopt
{
/*
   The class for a linear, parameter dependent heat-like operator, i.e.
     F(xdot,x,t) = M*xdot + K*x - f(t)
   M mass matrix
   K stiffness matrix (Diffusion, CurlCurl or DivDiv depending on the space)
   f(t) forcing term (time-dependent mfem::Coefficient or mfem::VectorCoefficient for vector spaces)
*/
class ModelHeat : public mfem::TimeDependentOperator, public PDOperator
{
private:
   mfem::OperatorHandle *Mh;
   mfem::OperatorHandle *Kh;
   mfem::Operator::Type oid;

   mfem::ParFiniteElementSpace  *fes;
   int fe_range, fe_deriv;

   mfem::Coefficient*       rhs;
   mfem::VectorCoefficient* vrhs;

   mutable mfem::ParLinearForm  *rhsform;
   mutable mfem::PetscParVector *rhsvec;
   mutable mfem::Operator       *Jacobian;

   mfem::ParGridFunction *adjgf,*stgf;

   mfem::ParBilinearForm *k;
   mfem::ParBilinearForm *m;

   PDBilinearFormIntegrator *mu_pd_bilin;
   PDBilinearFormIntegrator *sigma_pd_bilin;

   mfem::PetscPreconditionerFactory *pfactory;

   void Init(mfem::ParFiniteElementSpace*,mfem::Operator::Type);
   void InitForms(mfem::Coefficient*,mfem::Coefficient*);
   void InitForms(mfem::Coefficient*,mfem::MatrixCoefficient*);
   void InitForms(PDCoefficient*,mfem::MatrixCoefficient*);
   void InitForms(PDCoefficient*,mfem::Coefficient*);
   void UpdateStiffness();
   void UpdateMass();

public:
   ModelHeat(mfem::Coefficient*,mfem::Coefficient*,mfem::ParFiniteElementSpace*,mfem::Operator::Type);
   ModelHeat(PDCoefficient*,mfem::Coefficient*,mfem::ParFiniteElementSpace*,mfem::Operator::Type);
   ModelHeat(mfem::Coefficient*,mfem::MatrixCoefficient*,mfem::ParFiniteElementSpace*,mfem::Operator::Type);
   ModelHeat(PDCoefficient*,mfem::MatrixCoefficient*,mfem::ParFiniteElementSpace*,mfem::Operator::Type);
   void SetRHS(mfem::Coefficient*);
   void SetRHS(mfem::VectorCoefficient*);

   mfem::PetscPreconditionerFactory* GetPreconditionerFactory();
   mfem::PetscPreconditionerFactory* NewPreconditionerFactory();

   /* interface for mfem::TimeDependentOperator */
   virtual void Mult(const mfem::Vector&,mfem::Vector&) const
   { mfem::mfem_error("ModelHeat::not for explicit solvers!"); }
   virtual void ImplicitMult(const mfem::Vector&,const mfem::Vector&,mfem::Vector&) const;
   virtual Operator& GetImplicitGradient(const mfem::Vector&,const mfem::Vector&,double) const;
   virtual ~ModelHeat();

   class PreconditionerFactory : public mfem::PetscPreconditionerFactory
   {
   private:
      ModelHeat& pde;
      mfem::HypreParMatrix *hA;

   public:
      PreconditionerFactory(ModelHeat& _pde, const std::string& name = std::string()): mfem::PetscPreconditionerFactory(name), pde(_pde), hA(NULL) {};
      virtual mfem::Solver* NewPreconditioner(const mfem::OperatorHandle&);
      virtual ~PreconditionerFactory() { delete hA;}
   };

   /* interface for mfemopt::PDOperator */
   virtual void SetUpFromParameters(const mfem::Vector&);
   virtual int  GetStateSize();
   virtual int  GetParameterSize();
   virtual void Mult(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&);
   virtual void ComputeGradientAdjoint(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,mfem::Vector&);
   virtual void ComputeGradient(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,mfem::Vector&);
   virtual void ComputeHessian(int,int,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,
                               const mfem::Vector&,const mfem::Vector&,mfem::Vector&);

   /* XXX */
   void GetCurrentVector(mfem::Vector&);
};

}
#endif

#endif

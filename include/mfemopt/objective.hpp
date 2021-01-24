#if !defined(_MFEMOPT_OBJECTIVE_HPP)
#define _MFEMOPT_OBJECTIVE_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfemopt/receiver.hpp>
#include <mfemopt/pdcoefficient.hpp>
#include <mfemopt/nonlininteg.hpp>
#include <mfem/general/error.hpp>
#include <mfem/linalg/vector.hpp>
#include <mfem/linalg/handle.hpp>
#include <mfem/linalg/petsc.hpp>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/fem/plinearform.hpp>
#include <mfem/fem/pbilinearform.hpp>
#include <limits>

namespace mfemopt
{
// Abstract class for objective functions of the type f : R^n x R^m x R -> R to be minimized
// f(x,m,t) : x state, m parameters, t time
class ObjectiveFunction
{
private:
   double teval;
   bool has_x,has_m;

protected:
   double scale;
   mfem::Operator *H_XX, *H_MM; /* Hessian operators, owned by the base class */

public:
   ObjectiveFunction(bool _has_x = true, bool _has_m = true, double _teval = std::numeric_limits<double>::min()) : teval(_teval), has_x(_has_x), has_m(_has_m), scale(1.0), H_XX(NULL), H_MM(NULL) {}

   virtual void SetScale(double s) { scale = s; }
   double GetEvalTime() { return teval; }

   virtual void Eval(const mfem::Vector&,const mfem::Vector&,double,double*)
   { mfem::mfem_error("ObjectiveFunction::Eval not overloaded!"); }

   inline bool HasEvalGradient_X() { return has_x; }
   virtual void EvalGradient_X(const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&)
   { mfem::mfem_error("ObjectiveFunction::EvalGradient_X not overloaded!"); }

   inline bool HasEvalGradient_M() { return has_m; }
   virtual void EvalGradient_M(const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&)
   { mfem::mfem_error("ObjectiveFunction::EvalGradient_M not overloaded!"); }

   /* Unlike the mfem::Operator::GetGradient and mfemopt::ReducedFunctional::GetHessian methods
      the Operator returned by these methods are owned by the base class and they
      should not be modified or deleted.
      Modifications can only happen through the SetUpHessian_?? methods */
   mfem::Operator* GetHessianOperator_XX() { return H_XX; }
   mfem::Operator* GetHessianOperator_MM() { return H_MM; }
   virtual void SetUpHessian_XX(const mfem::Vector&,const mfem::Vector&,double) {}
   virtual void SetUpHessian_MM(const mfem::Vector&,const mfem::Vector&,double) {}

   void TestFDGradient(MPI_Comm,const mfem::Vector&,const mfem::Vector&,double,double,bool=true);
   void TestFDHessian(MPI_Comm,const mfem::Vector&,const mfem::Vector&,double);
   virtual ~ObjectiveFunction() { delete H_XX; delete H_MM; }
};

class ObjectiveHessianOperatorFD : public mfem::PetscParMatrix
{
private:
   ObjectiveFunction* obj;

   mfem::Vector xIn;
   mfem::Vector mIn;
   double t;

   int A,B; /* 0,0 -> XX, 0,1 -> XM, 1,0 -> MX, 1,1 -> MM */

public:
   ObjectiveHessianOperatorFD(MPI_Comm,ObjectiveFunction*,const mfem::Vector&,const mfem::Vector&,double,int,int);
};

// Simple Tikhonov misfit : 1/2 \int_\Omega ||m - m_0||^2 dx
class TikhonovRegularizer : public ObjectiveFunction
{
private:
   PDCoefficient* m_pd;
   mfem::PetscParVector* m0;
   mfem::Vector x,Mx;

   void Init();

public:
   TikhonovRegularizer() : ObjectiveFunction(false,true), m_pd(NULL), m0(NULL) {}
   TikhonovRegularizer(PDCoefficient*);
   virtual void Eval(const mfem::Vector&,const mfem::Vector&,double,double*);
   virtual void EvalGradient_M(const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&);
   virtual void SetUpHessian_MM(const mfem::Vector&,const mfem::Vector&,double);
   virtual void SetScale(double);
   ~TikhonovRegularizer();
};

class TDLeastSquaresHessian;

// time dependent least squares : 1/2 sum^nreceivers_i=0 || u(x_i,t) - r_i(t) ||^2
class TDLeastSquares : public ObjectiveFunction
{
protected:
   friend class TDLeastSquaresHessian;
   mfem::ParGridFunction*              u;
   bool                                own_recv;
   mfem::Array<Receiver*>              receivers;
   mfem::Array<int>                    receivers_eid;
   mfem::Array<mfem::IntegrationPoint> receivers_ip;

   mfem::Array<mfem::VectorDeltaCoefficient*> deltacoeffs_x;
   mfem::ParLinearForm *rhsform_x;

   void InitReceivers();
   void InitDeltaCoefficients();
   void ResetDeltaCoefficients();

private:
   void Init();

public:
   TDLeastSquares();
   TDLeastSquares(const mfem::Array<Receiver*>&,mfem::ParFiniteElementSpace*,bool = false);
   void SetReceivers(const mfem::Array<Receiver*>&,bool = false);
   void SetParFiniteElementSpace(mfem::ParFiniteElementSpace*);

   virtual void Eval(const mfem::Vector&,const mfem::Vector&,double,double*);
   virtual void EvalGradient_X(const mfem::Vector&,const mfem::Vector&,double, mfem::Vector&);

   virtual ~TDLeastSquares();
};

class TDLeastSquaresHessian : public mfem::Operator
{
private:
   TDLeastSquares *ls;

public:
   TDLeastSquaresHessian() : ls(NULL) {}
   TDLeastSquaresHessian(TDLeastSquares*);
   virtual void Mult(const mfem::Vector&,mfem::Vector&) const;
   virtual ~TDLeastSquaresHessian() {}
};

// Total variation regularizer
class TVRegularizer : public ObjectiveFunction
{
private:
   PDCoefficient* m_pd;

   VTVIntegrator vtvInteg;

   mfem::Array<mfem::ParGridFunction*> wkgf;
   mfem::Array<mfem::ParGridFunction*> wkgf2;
   mfem::Array<mfem::VectorGridFunctionCoefficient*> WQ;

   mfem::Array<const mfem::FiniteElement*> els;
   mfem::Array<mfem::Vector*> mks;
   mfem::Array<mfem::Vector*> elgrads;
   mfem::Array2D<mfem::SparseMatrix*> ljacs;
   mfem::Array2D<mfem::DenseMatrix*> eljacs;

public:
   TVRegularizer(PDCoefficient*,double,bool=false,bool=true);

   double GetBeta() { return vtvInteg.GetBeta(); }
   void SetBeta(double _beta) { vtvInteg.SetBeta(_beta); }
   void Symmetrize(bool _sym = true) { vtvInteg.Symmetrize(_sym); }
   void Project(bool _prj = true) { vtvInteg.Project(_prj); }
   void UpdateDual(const mfem::Vector&);
   void UpdateDual(const mfem::Vector&,const mfem::Vector&,double,double=0.99);

   virtual void Eval(const mfem::Vector&,const mfem::Vector&,double,double*);
   virtual void EvalGradient_M(const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&);
   virtual void SetUpHessian_MM(const mfem::Vector&,const mfem::Vector&,double);

   virtual ~TVRegularizer();
};

}

/* TODO move to *.h file */
#include <petscvec.h>
PETSC_EXTERN PetscErrorCode mfemopt_eval_tdobj(Vec,Vec,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode mfemopt_eval_tdobj_x(Vec,Vec,PetscReal,Vec,void*);
PETSC_EXTERN PetscErrorCode mfemopt_eval_tdobj_xx(Vec,Vec,PetscReal,Mat,void*);

#endif

#endif

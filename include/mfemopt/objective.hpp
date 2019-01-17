#if !defined(_MFEMOPT_OBJECTIVE_HPP)
#define _MFEMOPT_OBJECTIVE_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemopt/receiver.hpp>
#include <mfemopt/pdcoefficient.hpp>
#include <mfemopt/nonlininteg.hpp>
#include <mfem/general/error.hpp>
#include <mfem/linalg/vector.hpp>
#include <mfem/linalg/handle.hpp>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/fem/plinearform.hpp>
#include <mfem/fem/pbilinearform.hpp>
#include <limits>
#include <petscvec.h>

namespace mfemopt
{
// Abstract class for objective functions of the type f : R^n x R^m x R -> R to be minimized
// f(x,m,t) : x state, m parameters, t time
class ObjectiveFunction
{
private:
   double teval;
   bool has_x,has_m;

public:
   ObjectiveFunction(bool _has_x = true, bool _has_m = true, double _teval = std::numeric_limits<double>::min()) : teval(_teval), has_x(_has_x), has_m(_has_m) {}

   double GetEvalTime() { return teval; }

   virtual void Eval(const mfem::Vector&,const mfem::Vector&,double,double*)
   { mfem::mfem_error("ObjectiveFunction::Eval not overloaded!"); }

   inline bool HasEvalGradient_X() { return has_x; }
   virtual void EvalGradient_X(const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&)
   { mfem::mfem_error("ObjectiveFunction::EvalGradient_X not overloaded!"); }

   inline bool HasEvalGradient_M() { return has_m; }
   virtual void EvalGradient_M(const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&)
   { mfem::mfem_error("ObjectiveFunction::EvalGradient_M not overloaded!"); }

   virtual void SetUpHessian_XX(const mfem::Vector&,const mfem::Vector&,double) {}
   virtual mfem::Operator* GetHessianOperator_XX() { return NULL; }
   virtual void SetUpHessian_MM(const mfem::Vector&,const mfem::Vector&,double) {}
   virtual mfem::Operator* GetHessianOperator_MM() { return NULL; }

   void TestFDGradient(MPI_Comm,const mfem::Vector&,const mfem::Vector&,double,double);
   void TestFDHessian(MPI_Comm,const mfem::Vector&,const mfem::Vector&,double);
   virtual ~ObjectiveFunction() {}
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

// Simple Tikhonov misfit : 1/2 \int_\Omega |u - u_0|^2 dx
class TikhonovRegularizer : public ObjectiveFunction
{
private:
   mfem::PetscParVector* u0;
   mfem::PetscParMatrix* M;

public:
   TikhonovRegularizer() : ObjectiveFunction(false,true), u0(NULL), M(NULL) {}
   TikhonovRegularizer(PDCoefficient*);
   virtual void Eval(const mfem::Vector&,const mfem::Vector&,double,double*);
   virtual void EvalGradient_M(const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&);
   virtual mfem::Operator* GetHessianOperator_MM() { return M; }
   ~TikhonovRegularizer();
};

class TDLeastSquaresHessian;

// time dependent least squares
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

   TDLeastSquaresHessian *H;

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
   virtual mfem::Operator* GetHessianOperator_XX();

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
   mfem::Operator *H;
   double alpha;
   double beta;
   TVIntegrator tvInteg;
   mfem::Array<mfem::ParGridFunction*> wkgf;
   mfem::Array<mfem::ParGridFunction*> wkgf2;
   mfem::Array<mfem::VectorGridFunctionCoefficient*> WQ;
   mfem::PetscParMatrix *P2D;
   mfem::Array<mfem::SparseMatrix*> ljacs;
   bool primal_dual;
   PDCoefficient* m_pd;

public:
   TVRegularizer(PDCoefficient*,double,double,bool=false);
   void SetAlpha(double _alpha) { alpha = _alpha; }
   double GetAlpha() { return alpha; }
   void Symmetrize(bool _sym = true) { tvInteg.Symmetrize(_sym); }
   void Project(bool _nrm = true) { tvInteg.Project(_nrm); }
   virtual void Eval(const mfem::Vector&,const mfem::Vector&,double,double*);
   virtual void EvalGradient_M(const mfem::Vector&,const mfem::Vector&,double,mfem::Vector&);
   virtual void SetUpHessian_MM(const mfem::Vector&,const mfem::Vector&,double);
   virtual mfem::Operator* GetHessianOperator_MM();

   void PrimalToDual(const mfem::Vector&);
   void UpdateDual(const mfem::Vector&,const mfem::Vector&,double);

   virtual ~TVRegularizer();
};

PETSC_EXTERN PetscErrorCode mfemopt_eval_tdobj(Vec,Vec,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode mfemopt_eval_tdobj_x(Vec,Vec,PetscReal,Vec,void*);
PETSC_EXTERN PetscErrorCode mfemopt_eval_tdobj_xx(Vec,Vec,PetscReal,Mat,void*);

}
#endif

#endif
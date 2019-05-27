#if !defined(_MFEMOPT_NONLININTEG_HPP)
#define _MFEMOPT_NONLININTEG_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/fem/nonlininteg.hpp>

namespace mfemopt
{

class TVIntegrator : public mfem::NonlinearFormIntegrator
{
private:
   double beta;

   double norm;

   mfem::VectorGridFunctionCoefficient *WQ;

   mfem::DenseMatrix dshape;
   mfem::Vector gradm,graddm,dualshape;

   mfem::Vector wk;
   mfem::DenseMatrix A;
   mfem::DenseMatrix tmat;

   const mfem::IntegrationRule* GetIntRule(const mfem::FiniteElement&);

protected:
   bool symmetrize;
   bool project;
   friend class TVRegularizer;

public:
   TVIntegrator(double _beta) : mfem::NonlinearFormIntegrator(), beta(_beta), norm(-1.0), WQ(NULL), symmetrize(false), project(false) {}
   void SetBeta(double _beta) { beta = _beta; }
   void SetNorm(double _norm) { norm = _norm; }
   void Symmetrize(bool _sym = true) { symmetrize = _sym; }
   void Project(bool _nrm = true) { project = _nrm; }
   void SetDualCoefficient(mfem::VectorGridFunctionCoefficient* _WQ) { WQ = _WQ; }
   virtual double GetElementEnergy(const mfem::FiniteElement&,
                                   mfem::ElementTransformation&,
                                   const mfem::Vector&);
   virtual void AssembleElementVector(const mfem::FiniteElement&,
                                      mfem::ElementTransformation&,
                                      const mfem::Vector&,mfem::Vector&);
   virtual void AssembleElementGrad(const mfem::FiniteElement&,
                                    mfem::ElementTransformation&,
                                    const mfem::Vector&,mfem::DenseMatrix&);
   virtual void AssembleElementDualUpdate(const mfem::FiniteElement&,
                                          const mfem::FiniteElement&,
                                          mfem::ElementTransformation&,
                                          const mfem::Vector&,const mfem::Vector&,
                                          mfem::Vector&,double*);
   virtual ~TVIntegrator() {}
};

}
#endif

#endif

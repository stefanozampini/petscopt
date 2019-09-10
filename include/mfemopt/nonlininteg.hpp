#if !defined(_MFEMOPT_NONLININTEG_HPP)
#define _MFEMOPT_NONLININTEG_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/fem/nonlininteg.hpp>

namespace mfemopt
{

class VTVIntegrator : public mfem::BlockNonlinearFormIntegrator
{
private:
   double beta; // TODO hyper-parameter

   const mfem::IntegrationRule *IntRule;

   mfem::Vector gradm,norms;
   mfem::Vector wk,wj;
   mfem::DenseMatrix A,tmat;

   mfem::Array<mfem::DenseMatrix*> dshapes;
   mfem::Array<mfem::Vector*> work,dwork;

   mfem::Array<mfem::VectorGridFunctionCoefficient*> WQs;

   const mfem::IntegrationRule* GetIntRule(const mfem::FiniteElement&);

protected:
   bool indep; // TODO coupling mask?
   bool symmetrize;
   bool project;
   friend class TVRegularizer;

public:
   VTVIntegrator(double,bool=true);
   void SetDualCoefficients(const mfem::Array<mfem::VectorGridFunctionCoefficient*>&);
   void SetDualCoefficients();
   void SetBeta(double _beta) { beta = _beta; }
   void Symmetrize(bool _sym = true) { symmetrize = _sym; }
   void Project(bool _prj = true) { project = _prj; }
   virtual double GetElementEnergy(const mfem::Array<const mfem::FiniteElement*>&,
                                   mfem::ElementTransformation&,
                                   const mfem::Array<const mfem::Vector*>&);
   virtual void AssembleElementVector(const mfem::Array<const mfem::FiniteElement*>&,
                                      mfem::ElementTransformation&,
                                      const mfem::Array<const mfem::Vector*>&,
                                      const mfem::Array<mfem::Vector*>&);
   virtual void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement*>&,
                                    mfem::ElementTransformation&,
                                    const mfem::Array<const mfem::Vector*>&,
                                    const mfem::Array2D<mfem::DenseMatrix*>&);
   void AssembleElementDualUpdate(const mfem::Array<const mfem::FiniteElement*>&,
                                  const mfem::Array<const mfem::FiniteElement*>&,
                                  mfem::ElementTransformation&,
                                  const mfem::Array<mfem::Vector*>&,
                                  const mfem::Array<mfem::Vector*>&,
                                  mfem::Array<mfem::Vector*>&,
                                  double*);
   virtual ~VTVIntegrator();
};

}
#endif

#endif

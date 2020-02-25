#if !defined(_MFEMOPT_PDBILININTEG_HPP)
#define _MFEMOPT_PDBILININTEG_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfemopt/pdcoefficient.hpp>
#include <mfem/fem/bilininteg.hpp>
#include <mfem/general/error.hpp>

namespace mfemopt
{

class PDBilinearFormIntegrator : public mfem::BilinearFormIntegrator
{
private:
   const mfem::IntegrationRule *oIntRule;

protected:
   PDCoefficient *pdcoeff;

public:
   PDBilinearFormIntegrator() : oIntRule(NULL), pdcoeff(NULL) {}
   PDBilinearFormIntegrator(PDCoefficient& _pdcoeff) : oIntRule(NULL), pdcoeff(&_pdcoeff) {}

   int GetParameterSize() { return pdcoeff ? pdcoeff->GetLocalSize() : 0; }
   void UpdateCoefficient(const mfem::Vector&);
   void UpdateGradient(mfem::Vector&);
   void GetCurrentVector(mfem::Vector&);

   void ComputeHessian_XM(mfem::ParGridFunction*,const mfem::Vector&,mfem::ParGridFunction*);
   void ComputeHessian_MX(mfem::ParGridFunction*,mfem::ParGridFunction*,mfem::Vector&);
   void ComputeGradient(mfem::ParGridFunction*,const mfem::Vector&,mfem::ParGridFunction*);
   void ComputeGradientAdjoint(mfem::ParGridFunction*,mfem::ParGridFunction*,mfem::Vector&);

   /* Default integration rule: specific implementations are copy-and-pasted from MFEM code */
   virtual const mfem::IntegrationRule* GetDefaultIntRule(const mfem::FiniteElement&,const mfem::FiniteElement&,mfem::ElementTransformation&,int)
   { mfem::mfem_error("PDBilinearFormIntegrator::GetDefaultIntRule not overloaded!"); return NULL; }
   const mfem::IntegrationRule* GetDefaultIntRule(const mfem::FiniteElement& el,mfem::ElementTransformation& T,int q)
   { return GetDefaultIntRule(el,el,T,q); }

   virtual void AssembleElementMatrix(const mfem::FiniteElement&,
                                      mfem::ElementTransformation&,
                                      mfem::DenseMatrix&)
   { mfem::mfem_error("PDBilinearFormIntegrator::AssembleElementMatrix not overloaded!"); }
   virtual void AssembleElementMatrix2(const mfem::FiniteElement&,
                                       const mfem::FiniteElement&,
                                       mfem::ElementTransformation&,
                                       mfem::DenseMatrix&)
   { mfem::mfem_error("PDBilinearFormIntegrator::AssembleElementMatrix2 not overloaded!"); }
   virtual void ComputeElementGradientAdjoint(const mfem::FiniteElement*,
                                              const mfem::Vector&,
                                              const mfem::FiniteElement*,
                                              const mfem::Vector&,
                                              mfem::ElementTransformation*,
                                              mfem::Vector&);
   virtual void ComputeElementGradient(const mfem::FiniteElement*,const mfem::Vector&,
                                       mfem::ElementTransformation*,const mfem::Vector&,
                                       mfem::Vector&);
   virtual void ComputeElementHessian(const mfem::FiniteElement*,const mfem::Vector&,
                                       mfem::ElementTransformation*,const mfem::Vector&,
                                       mfem::Vector&);
   virtual ~PDBilinearFormIntegrator() {}

protected:
   void ComputeElementGradient_Internal(const mfem::FiniteElement*,const mfem::Vector&,
                                        mfem::ElementTransformation*,const mfem::Vector&,
                                        mfem::Vector&,bool);
   void ComputeElementGradientAdjoint_Internal(const mfem::FiniteElement*,
                                               const mfem::Vector&,
                                               const mfem::FiniteElement*,
                                               const mfem::Vector&,
                                               mfem::ElementTransformation*,
                                               mfem::Vector&);

private:
   void ComputeGradient_Internal(mfem::ParGridFunction*,const mfem::Vector&,mfem::ParGridFunction*,bool);
   void ComputeGradientAdjoint_Internal(mfem::ParGridFunction*,mfem::ParGridFunction*,mfem::Vector&,bool);
};

class PDMassIntegrator : public mfem::MassIntegrator, public PDBilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   mfem::Vector pshape; // TODO: this should probably belong to the PDCoefficient class (which should know how to evaluate itself)
#endif

public:
   PDMassIntegrator(PDCoefficient& _pdcoeff) : PDBilinearFormIntegrator(_pdcoeff) {}
   virtual const mfem::IntegrationRule* GetDefaultIntRule(const mfem::FiniteElement&,const mfem::FiniteElement&,mfem::ElementTransformation&,int);
   using PDBilinearFormIntegrator::GetDefaultIntRule;
   virtual void AssembleElementMatrix(const mfem::FiniteElement&,
                                      mfem::ElementTransformation&,
                                      mfem::DenseMatrix&);
   virtual void AssembleElementMatrix2(const mfem::FiniteElement&,
                                       const mfem::FiniteElement&,
                                       mfem::ElementTransformation&,
                                       mfem::DenseMatrix&);
   virtual void ComputeElementGradientAdjoint(const mfem::FiniteElement*,
                                              const mfem::Vector&,
                                              const mfem::FiniteElement*,
                                              const mfem::Vector&,
                                              mfem::ElementTransformation*,
                                              mfem::Vector&);
   virtual void ComputeElementGradient(const mfem::FiniteElement*,const mfem::Vector&,
                                       mfem::ElementTransformation*,const mfem::Vector&,
                                       mfem::Vector&);
   virtual void ComputeElementHessian(const mfem::FiniteElement*,const mfem::Vector&,
                                       mfem::ElementTransformation*,const mfem::Vector&,
                                       mfem::Vector&);
};

class PDVectorFEMassIntegrator : public mfem::VectorFEMassIntegrator, public PDBilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix svshape,avshape;
   mfem::Vector pshape; // TODO: this should probably belong to the PDCoefficient class (which should know how to evaluate itself)
#endif

public:
   PDVectorFEMassIntegrator(PDCoefficient& _pdcoeff) : PDBilinearFormIntegrator(_pdcoeff) {}
   virtual const mfem::IntegrationRule* GetDefaultIntRule(const mfem::FiniteElement&,const mfem::FiniteElement&,mfem::ElementTransformation&,int);
   using PDBilinearFormIntegrator::GetDefaultIntRule;
   virtual void AssembleElementMatrix(const mfem::FiniteElement&,
                                      mfem::ElementTransformation&,
                                      mfem::DenseMatrix&);
   virtual void AssembleElementMatrix2(const mfem::FiniteElement&,
                                       const mfem::FiniteElement&,
                                       mfem::ElementTransformation&,
                                       mfem::DenseMatrix&);
   virtual void ComputeElementGradientAdjoint(const mfem::FiniteElement*,
                                              const mfem::Vector&,
                                              const mfem::FiniteElement*,
                                              const mfem::Vector&,
                                              mfem::ElementTransformation*,
                                              mfem::Vector&);
   virtual void ComputeElementGradient(const mfem::FiniteElement*,const mfem::Vector&,
                                       mfem::ElementTransformation*,const mfem::Vector&,
                                       mfem::Vector&);
   virtual void ComputeElementHessian(const mfem::FiniteElement*,const mfem::Vector&,
                                       mfem::ElementTransformation*,const mfem::Vector&,
                                       mfem::Vector&);
};

}
#endif

#endif

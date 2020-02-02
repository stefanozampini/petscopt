#if !defined(_MFEMOPT_PDBILININTEG_HPP)
#define _MFEMOPT_PDBILININTEG_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfemopt/pdcoefficient.hpp>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/fem/bilininteg.hpp>
#include <mfem/general/error.hpp>

namespace mfemopt
{

class PDBilinearFormIntegrator : public mfem::BilinearFormIntegrator
{
private:
   mfem::ParGridFunction *sworkgf;
   const mfem::IntegrationRule *oIntRule;

   void ComputeGradient_Internal(mfem::ParGridFunction*,const mfem::Vector&,const mfem::Vector&,mfem::Vector&,bool=false);
   void ComputeGradientAdjoint_Internal(mfem::ParGridFunction*,mfem::ParGridFunction*,const mfem::Vector&,mfem::Vector&,bool=false);

protected:
   PDCoefficient *pdcoeff;
   void PushPDIntRule(const mfem::FiniteElement&,mfem::ElementTransformation&,int);
   void PushPDIntRule(const mfem::FiniteElement&,const mfem::FiniteElement&,mfem::ElementTransformation&,int);
   void PopPDIntRule();

public:
   PDBilinearFormIntegrator() : sworkgf(NULL), oIntRule(NULL), pdcoeff(NULL) {}
   PDBilinearFormIntegrator(PDCoefficient& _pdcoeff) : sworkgf(NULL), oIntRule(NULL), pdcoeff(&_pdcoeff) {}

   /* XXX */
   int GetLocalSize() {if (pdcoeff) return pdcoeff->GetLocalSize(); else return 0; }
   void UpdateCoefficient(const mfem::Vector&);
   void UpdateGradient(mfem::Vector&);
   void GetCurrentVector(mfem::Vector&);
   /* XXX */
   void ComputeGradient(mfem::ParGridFunction*,const mfem::Vector&,const mfem::Vector&,mfem::Vector&);
   void ComputeGradientAdjoint(mfem::ParGridFunction*,mfem::ParGridFunction*,const mfem::Vector&,mfem::Vector&);
   void ComputeHessian_XM(mfem::ParGridFunction*,const mfem::Vector&,const mfem::Vector&,mfem::Vector&);
   void ComputeHessian_MX(mfem::ParGridFunction*,mfem::ParGridFunction*,const mfem::Vector&,mfem::Vector&);

   /* these belongs to the integrator, above to the form -> TODO: Split */
   /* Default integration rule: specific implementations are copy-and-pasted from MFEM code */
   virtual const mfem::IntegrationRule* GetDefaultIntRule(const mfem::FiniteElement&,const mfem::FiniteElement&,mfem::ElementTransformation&,int)
   { mfem::mfem_error("PDBilinearFormIntegrator::GetDefaultIntRule not overloaded!"); return NULL; }
   virtual const mfem::IntegrationRule* GetDefaultIntRule(const mfem::FiniteElement& el,mfem::ElementTransformation& T,int q)
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
                                              int,mfem::Vector&);
   virtual void ComputeElementGradient(const mfem::FiniteElement*,const mfem::Vector&,
                                       mfem::ElementTransformation*,const mfem::Vector&,int,
                                       mfem::Vector&);
   virtual void ComputeElementHessian(const mfem::FiniteElement*,const mfem::Vector&,
                                       mfem::ElementTransformation*,const mfem::Vector&,int,
                                       mfem::Vector&);
   ~PDBilinearFormIntegrator();

protected:
   virtual void ComputeElementGradient_Internal(const mfem::FiniteElement*,const mfem::Vector&,
                                                mfem::ElementTransformation*,const mfem::Vector&,int,
                                                mfem::Vector&,bool);

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
                                              int,mfem::Vector&);
private:
   virtual void ComputeElementGradient_Internal(const mfem::FiniteElement*,const mfem::Vector&,
                                                mfem::ElementTransformation*,const mfem::Vector&,int,
                                                mfem::Vector&,bool);
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
                                              int,mfem::Vector&);
private:
   virtual void ComputeElementGradient_Internal(const mfem::FiniteElement*,const mfem::Vector&,
                                                mfem::ElementTransformation*,const mfem::Vector&,int,
                                                mfem::Vector&,bool);
};

}
#endif

#endif

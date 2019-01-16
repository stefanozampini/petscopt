#if !defined(_MFEMOPT_PDBILININTEG_HPP)
#define _MFEMOPT_PDBILININTEG_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
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

   void ComputeGradient_Internal(mfem::ParGridFunction*,mfem::Vector&,mfem::Vector&,mfem::Vector&,bool=false);
   void ComputeGradientAdjoint_Internal(mfem::ParGridFunction*,mfem::ParGridFunction*,mfem::Vector&,mfem::Vector&,bool=false);

protected:
   PDCoefficient *pdcoeff;

public:
   PDBilinearFormIntegrator() : sworkgf(NULL), pdcoeff(NULL) {}
   PDBilinearFormIntegrator(PDCoefficient* _pdcoeff) : sworkgf(NULL), pdcoeff(_pdcoeff) {}

   /* XXX */
   int GetLocalSize() {if (pdcoeff) return pdcoeff->GetLocalSize(); else return 0; }
   void UpdateCoefficient(const mfem::Vector&);
   void UpdateGradient(mfem::Vector&);
   void GetCurrentVector(mfem::Vector&);
   /* XXX */
   void ComputeGradient(mfem::ParGridFunction*,mfem::Vector&,mfem::Vector&,mfem::Vector&);
   void ComputeGradientAdjoint(mfem::ParGridFunction*,mfem::ParGridFunction*,mfem::Vector&,mfem::Vector&);
   void ComputeHessian_XM(mfem::ParGridFunction*,mfem::Vector&,mfem::Vector&,mfem::Vector&);
   void ComputeHessian_MX(mfem::ParGridFunction*,mfem::ParGridFunction*,mfem::Vector&,mfem::Vector&);

   virtual void AssembleElementMatrix(const mfem::FiniteElement&,
                                      mfem::ElementTransformation&,
                                      mfem::DenseMatrix&)
   { mfem::mfem_error("PDBilinearFormIntegrator::AssembleElementMatrix not overloaded!"); }
   virtual void AssembleElementMatrix2(const mfem::FiniteElement&,
                                       const mfem::FiniteElement&,
                                       mfem::ElementTransformation&,
                                       mfem::DenseMatrix&)
   { mfem::mfem_error("PDBilinearFormIntegrator::AssembleElementMatrix2 not overloaded!"); }
   ~PDBilinearFormIntegrator();
};

class PDMassIntegrator : public mfem::MassIntegrator, public PDBilinearFormIntegrator
{
public:
   PDMassIntegrator(PDCoefficient* _pdcoeff) : PDBilinearFormIntegrator(_pdcoeff) {}
   virtual void AssembleElementMatrix(const mfem::FiniteElement&,
                                      mfem::ElementTransformation&,
                                      mfem::DenseMatrix&);
   virtual void AssembleElementMatrix2(const mfem::FiniteElement&,
                                       const mfem::FiniteElement&,
                                       mfem::ElementTransformation&,
                                       mfem::DenseMatrix&);
};

}
#endif

#endif

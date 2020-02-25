#if !defined(_MFEMOPT_PDBILINEARFORM_HPP)
#define _MFEMOPT_PDBILINEARFORM_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/linalg/petsc.hpp>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/fem/pbilinearform.hpp>

namespace mfemopt
{

class PDBilinearForm : public mfem::ParBilinearForm
{
protected:
   mfem::ParGridFunction *sgf;
   mfem::ParGridFunction *agf;
   mfem::ParGridFunction *swgf;
   mfem::ParGridFunction *awgf;

   void ComputeGradient_Internal(const mfem::Vector&,const mfem::Vector&,mfem::Vector&,bool=false);
   void ComputeGradientAdjoint_Internal(const mfem::Vector&,const mfem::Vector&,mfem::Vector&,bool=false);

public:
   PDBilinearForm(mfem::ParFiniteElementSpace *pf) : ParBilinearForm(pf), sgf(NULL), agf(NULL), swgf(NULL), awgf(NULL) {}

   virtual int GetParameterSize();
   virtual PetscInt GetParameterGlobalSize();
   virtual void UpdateParameter(const mfem::Vector&);
   virtual void ComputeGradient(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,mfem::Vector&);
   virtual void ComputeGradientAdjoint(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,mfem::Vector&);
   virtual void ComputeHessian_XM(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,mfem::Vector&);
   virtual void ComputeHessian_MX(const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,mfem::Vector&);

   virtual ~PDBilinearForm();
};

}
#endif

#endif

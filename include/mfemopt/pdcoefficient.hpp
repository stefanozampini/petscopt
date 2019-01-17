#if !defined(_MFEMOPT_PDCOEFFICIENT_HPP)
#define _MFEMOPT_PDCOEFFICIENT_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfem/linalg/petsc.hpp>
#include <mfem/fem/coefficient.hpp>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/general/error.hpp>
#include <mfem/general/socketstream.hpp>

namespace mfemopt
{

class PDCoefficient
{
private:
   int lsize,lvsize;
   bool usederiv;
   bool usefuncs;

   mfem::GridFunctionCoefficient* deriv_s_coeff;
   mfem::MatrixArrayCoefficient* deriv_m_coeff;
   mfem::GridFunctionCoefficient* s_coeff;
   mfem::MatrixArrayCoefficient* m_coeff;

   mfem::Array<PetscInt> global_cols;
   mfem::Array<bool>   pcoeffexcl;
   mfem::Array<int>    pcoeffiniti;
   mfem::Array<double> pcoeffinitv;

   mfem::Array<mfem::ParGridFunction*> pcoeffgf;
   mfem::Array<mfem::ParGridFunction*> pgradgf;
   mfem::Array<mfem::ParGridFunction*> deriv_coeffgf;
   mfem::Array<mfem::ParGridFunction*> deriv_work_coeffgf;
   mfem::PetscParMatrix* P;
   mfem::PetscParMatrix* R;

   std::vector<mfem::socketstream*> souts;


   void Reset();
   void Init(mfem::Coefficient*,mfem::VectorCoefficient*,mfem::MatrixCoefficient*,mfem::ParFiniteElementSpace*,const mfem::Array<bool>&);

protected:
   friend class PDBilinearFormIntegrator;
   void ElemDeriv(int,int,int,double=1.0);

public:
   PDCoefficient() : lsize(0), lvsize(0), usederiv(false), deriv_s_coeff(NULL), deriv_m_coeff(NULL), s_coeff(NULL), m_coeff(NULL), P(NULL) {}
   PDCoefficient(mfem::Coefficient& Q, mfem::ParFiniteElementSpace* pfes,
                 const mfem::Array<int>& excl_tag);
   PDCoefficient(mfem::Coefficient& Q, mfem::ParFiniteElementSpace* pfes,
                 const mfem::Array<bool>& excl = mfem::Array<bool>())
                 { Init(&Q,NULL,NULL,pfes,excl); }
   PDCoefficient(mfem::Coefficient& Q, mfem::ParFiniteElementSpace* pfes,
                 bool (*excl_fn)(const mfem::Vector&));
   PDCoefficient(mfem::VectorCoefficient& Q, mfem::ParFiniteElementSpace* pfes,
                 const mfem::Array<bool>& excl = mfem::Array<bool>())
                 { Init(NULL,&Q,NULL,pfes,excl); }
   PDCoefficient(mfem::MatrixCoefficient& Q, mfem::ParFiniteElementSpace* pfes,
                 const mfem::Array<bool>& excl = mfem::Array<bool>())
                 { Init(NULL,NULL,&Q,pfes,excl); }

   mfem::Coefficient * GetActiveCoefficient();
   mfem::MatrixCoefficient * GetActiveMatrixCoefficient();

   mfem::Array<mfem::ParGridFunction*>& GetCoeffs() { return pcoeffgf; }
   mfem::Array<mfem::ParGridFunction*>& GetDerivCoeffs() { return deriv_work_coeffgf; }
   mfem::Array<mfem::ParGridFunction*>& GetGradCoeffs() { return pgradgf; }
   mfem::Array<PetscInt>& GetGlobalCols() { return global_cols; }
   mfem::PetscParMatrix* GetP() { return P; }
   mfem::Array<bool>& GetExcludedElements() { return pcoeffexcl; }
   int GetLocalSize() { return lsize; }
   int GetLocalVSize() { return lvsize; }
   void GetCurrentVector(mfem::Vector&);
   void SetUseDerivCoefficients(bool=true);
   void UpdateCoefficient(const mfem::Vector&);
   void UpdateCoefficientWithGF(const mfem::Vector&,mfem::Array<mfem::ParGridFunction*>&);
   void UpdateGradient(mfem::Vector&);
   void Save(const char*);
   void Visualize(const char*);
   ~PDCoefficient();
};

}
#endif

#endif
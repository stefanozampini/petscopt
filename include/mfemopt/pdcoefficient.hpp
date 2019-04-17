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
   int lsize;
   int order;
   bool usederiv;
   bool usefuncs;

   mfem::ParFiniteElementSpace* pfes;

   mfem::Coefficient*       deriv_s_coeff;
   mfem::MatrixCoefficient* deriv_m_coeff;

   mfem::Coefficient*       s_coeff;
   mfem::MatrixCoefficient* m_coeff;

   mfem::Array<PetscInt> global_cols;
   mfem::Array<bool>   pcoeffexcl;
   mfem::Array<int>    pcoeffiniti;
   mfem::Array<double> pcoeffinitv;

   mfem::Array<mfem::ParGridFunction*> pcoeffgf;
   mfem::Array<mfem::ParGridFunction*> pgradgf;
   mfem::Array<mfem::ParGridFunction*> deriv_coeffgf;
   mfem::Array<mfem::ParGridFunction*> deriv_work_coeffgf;
   mfem::Array<mfem::Vector*> pcoeffv0;
   mfem::PetscParMatrix* P;
   mfem::PetscParMatrix* R;

   std::vector<mfem::socketstream*> souts;

   void Reset();
   void Init(mfem::Coefficient*,mfem::VectorCoefficient*,mfem::MatrixCoefficient*,mfem::ParMesh*,const mfem::FiniteElementCollection*,const mfem::Array<bool>&);

protected:
   friend class PDBilinearFormIntegrator;
   void ElemDeriv(int,int,int,double=1.0);

public:
   PDCoefficient() : lsize(0), order(-1), usederiv(false), deriv_s_coeff(NULL), deriv_m_coeff(NULL), s_coeff(NULL), m_coeff(NULL), P(NULL) {}
   PDCoefficient(mfem::Coefficient&,mfem::ParMesh*,const mfem::FiniteElementCollection*,
                 const mfem::Array<int>&);
   PDCoefficient(mfem::Coefficient&,mfem::ParMesh*,const mfem::FiniteElementCollection*,
                 const mfem::Array<bool>& = mfem::Array<bool>());
   PDCoefficient(mfem::Coefficient&,mfem::ParMesh*,const mfem::FiniteElementCollection*,
                 bool (*excl_fn)(const mfem::Vector&));
   PDCoefficient(mfem::VectorCoefficient&,mfem::ParMesh*,const mfem::FiniteElementCollection*,
                 const mfem::Array<int>&);
   PDCoefficient(mfem::VectorCoefficient&,mfem::ParMesh*,const mfem::FiniteElementCollection*,
                 const mfem::Array<bool>& = mfem::Array<bool>());
   PDCoefficient(mfem::VectorCoefficient&,mfem::ParMesh*,const mfem::FiniteElementCollection*,
                 bool (*excl_fn)(const mfem::Vector&));

   mfem::Coefficient * GetActiveCoefficient();
   mfem::MatrixCoefficient * GetActiveMatrixCoefficient();

   mfem::Array<mfem::ParGridFunction*>& GetCoeffs() { return pcoeffgf; }
   mfem::Array<mfem::ParGridFunction*>& GetDerivCoeffs() { return deriv_work_coeffgf; }
   mfem::Array<mfem::ParGridFunction*>& GetGradCoeffs() { return pgradgf; }
   mfem::Array<PetscInt>& GetGlobalCols() { return global_cols; }
   mfem::PetscParMatrix* GetP() { return P; }
   mfem::Array<bool>& GetExcludedElements() { return pcoeffexcl; }
   int GetLocalSize() { return lsize; }
   int GetOrder() { return order; }
   void GetCurrentVector(mfem::Vector&);
   void GetInitialVector(mfem::Vector&);
   void SetUseDerivCoefficients(bool=true);
   void UpdateCoefficient(const mfem::Vector&);
   void UpdateCoefficientWithGF(const mfem::Vector&,mfem::Array<mfem::ParGridFunction*>&);
   void UpdateGradient(mfem::Vector&);
   void Save(const char*);
   void SaveExcl(const char*);
   void SaveVisIt(const char*);
   void Visualize(const char* = NULL);
   ~PDCoefficient();
};

}
#endif

#endif

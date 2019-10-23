#if !defined(_MFEMOPT_PDCOEFFICIENT_HPP)
#define _MFEMOPT_PDCOEFFICIENT_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
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
   bool incl_bdr; /* XXX custom */

   mfem::ParFiniteElementSpace* pfes;

   mfem::Coefficient*       deriv_s_coeff;
   mfem::MatrixCoefficient* deriv_m_coeff;

   mfem::Coefficient*       s_coeff;
   mfem::MatrixCoefficient* m_coeff;

   mfem::Array<PetscInt> local_cols;
   mfem::Array<PetscInt> global_cols;

   mfem::Array<bool>   pcoeffexcl;
   mfem::Array<bool>   sforminteg;
   mfem::Array<int>    pcoeffiniti;
   mfem::Array<double> pcoeffinitv;
   mfem::Array<int>    piniti;
   mfem::Array<double> pinitv;
   mfem::Array<int>    pactii;
   mfem::Vector        pwork;

   mfem::Array<mfem::ParGridFunction*> pcoeffgf;
   mfem::Array<mfem::ParGridFunction*> pgradgf;
   mfem::Array<mfem::ParGridFunction*> deriv_coeffgf;
   mfem::Array<mfem::ParGridFunction*> deriv_work_coeffgf;
   mfem::Array<mfem::Vector*> pcoeffv0;
   mfem::PetscParMatrix* P;
   mfem::PetscParMatrix* R;
   mfem::PetscParMatrix* trueTransfer;

   mfem::ParGridFunction* l2gf;

   class BCHandler : public mfem::PetscBCHandler
   {
   private:
      mfem::Array<double> vals;

   public:
      BCHandler();
      BCHandler(mfem::Array<int>&,mfem::Array<double>&);
      void Update(mfem::Array<int>&,mfem::Array<double>&);
      virtual void Eval(double,mfem::Vector&);
   };
   mfem::Array<int> ess_tdof_list;
   mfem::Array<double> ess_tdof_vals;

   BCHandler bchandler;

   std::vector<mfem::socketstream*> souts;

   void Init();
   void Init(mfem::Coefficient*,mfem::VectorCoefficient*,mfem::MatrixCoefficient*,mfem::ParMesh*,const mfem::FiniteElementCollection*,const mfem::Array<bool>&);
   void Reset();
   void SetUpOperators();
   void FillExcl();
   void UpdateExcl();

protected:
   friend class PDBilinearFormIntegrator;
   void ElemDeriv(int,int,int,double=1.0);

public:
   PDCoefficient();
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
   mfem::Array<PetscInt>& GetLocalCols() { return local_cols; }
   mfem::Array<PetscInt>& GetGlobalCols() { return global_cols; }
   mfem::PetscParMatrix* GetP() { return P; }
   mfem::PetscParMatrix* GetR() { return R; }
   mfem::PetscParMatrix* GetTrueTransferOperator() { return trueTransfer; }
   mfem::Array<bool>& GetExcludedElements() { return pcoeffexcl; }
   mfem::Array<bool>& GetActiveElements() { return sforminteg; }
   int GetLocalSize() { return lsize; }
   int GetOrder() { return order; }
   void GetCurrentVector(mfem::Vector&);
   void GetInitialVector(mfem::Vector&);
   void SetUseDerivCoefficients(bool=true);
   void Distribute(const mfem::Vector&);
   void Distribute(const mfem::Vector&,mfem::Array<mfem::ParGridFunction*>&);
   void Assemble(mfem::Vector&);
   void Save(const char*);
   void SaveExcl(const char*);
   void SaveVisIt(const char*);
   void Visualize(const char* = NULL);

   mfem::PetscBCHandler* GetBCHandler() { return &bchandler; }

   void Update(bool=true);
   ~PDCoefficient();
};

}
#endif

#endif

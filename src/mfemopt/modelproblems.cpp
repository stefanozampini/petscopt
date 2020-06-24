#include <mfemopt/pdbilininteg.hpp>
#include <mfemopt/modelproblems.hpp>
#include <mfemopt/mfemextra.hpp>
#include <petscopt/tsopt.h>

namespace mfemopt
{

using namespace mfem;

ModelHeat::MFJac::MFJac(const ModelHeat& h) : heat(h), s(0.0)
{
   height = heat.Height();
   width  = heat.Width();
}

void ModelHeat::MFJac::SetShift(double shift) { s = shift; }
double ModelHeat::MFJac::GetShift() { return s; }

void ModelHeat::MFJac::Mult_Private(const Vector& x,Vector& y,bool trans) const
{
   Operator *M,*K;
   heat.Mh->Get(M);
   heat.Kh->Get(K);
   if (!s)
   {
      if (trans)
      {
         K->MultTranspose(x,y);
      }
      else
      {
         K->Mult(x,y);
      }
   }
   else
   {
      if (trans)
      {
         M->MultTranspose(x, y);
      }
      else
      {
         M->Mult(x, y);
      }
      y *= s;
      if (trans)
      {
         K->MultTranspose(x,*heat.rhsvec);
      }
      else
      {
         K->Mult(x,*heat.rhsvec);
      }
      y += *heat.rhsvec;
   }
}

void ModelHeat::MFJac::Mult(const Vector& x,Vector& y) const
{
   Mult_Private(x,y,false);
}

void ModelHeat::MFJac::MultTranspose(const Vector& x,Vector& y) const
{
   Mult_Private(x,y,true);
}

void ModelHeat::Init(ParFiniteElementSpace *_fe, Operator::Type _oid)
{
   Mh = NULL;
   Kh = NULL;
   rhsform = NULL;

   rhs = NULL;
   vrhs = NULL;

   fes = _fe;
   ParFiniteElementSpaceGetRangeAndDerivType(*fes,&fe_range,&fe_deriv);

   oid = _oid;

   adjgf = new ParGridFunction(fes);
   stgf = new ParGridFunction(fes);
   kpd = new PDBilinearForm(fes);
   mpd = new PDBilinearForm(fes);
   rhsvec = new PetscParVector(fes);

   pfactory = NULL;
}

PetscPreconditionerFactory* ModelHeat::GetPreconditionerFactory()
{
   if (!pfactory)
   {
      pfactory = (*this).NewPreconditionerFactory();
   }
   return pfactory;
}

PetscPreconditionerFactory* ModelHeat::NewPreconditionerFactory()
{
   return new PreconditionerFactory(*this,"heat model preconditioner factory");
}


void ModelHeat::InitForms(Coefficient* mu, MatrixCoefficient* sigma)
{
   MFEM_VERIFY(fes->GetVDim() == 1,"Unsupported VDIM " << fes->GetVDim());

   BilinearFormIntegrator *tm = NULL;
   BilinearFormIntegrator *ts = NULL;
   if (fe_range == FiniteElement::SCALAR)
   {
      tm = new MassIntegrator(*mu);
      ts = new DiffusionIntegrator(*sigma);
   }
   else
   {
      MFEM_VERIFY(fe_deriv != FiniteElement::DIV,"Unsupported Matrix coefficient for DivDivIntegrator ");
      tm = new VectorFEMassIntegrator(*mu);
      ts = new CurlCurlIntegrator(*sigma);
   }
   mpd->AddDomainIntegrator(tm);
   kpd->AddDomainIntegrator(ts);
}

void ModelHeat::InitForms(PDCoefficient* mu, MatrixCoefficient* sigma)
{
   MFEM_VERIFY(fes->GetVDim() == 1,"Unsupported VDIM " << fes->GetVDim());

   PDBilinearFormIntegrator *tm = NULL;
   BilinearFormIntegrator *ts = NULL;
   if (fe_range == FiniteElement::SCALAR)
   {
      tm = new PDMassIntegrator(*mu);
      ts = new DiffusionIntegrator(*sigma);
   }
   else
   {
      MFEM_VERIFY(fe_deriv != FiniteElement::DIV,"Unsupported Matrix coefficient for DivDivIntegrator");
      tm = new PDVectorFEMassIntegrator(*mu);
      ts = new CurlCurlIntegrator(*sigma);
   }
   mpd->AddDomainIntegrator(tm);
   kpd->AddDomainIntegrator(ts);
}

void ModelHeat::InitForms(Coefficient* mu, Coefficient* sigma)
{
   MFEM_VERIFY(fes->GetVDim() == 1,"Unsupported VDIM " << fes->GetVDim());

   BilinearFormIntegrator *tm = NULL;
   BilinearFormIntegrator *ts = NULL;
   if (fe_range == FiniteElement::SCALAR)
   {
      tm = new MassIntegrator(*mu);
      ts = new DiffusionIntegrator(*sigma);
   }
   else
   {
      tm = new VectorFEMassIntegrator(*mu);
      if (fe_deriv == FiniteElement::DIV)
      {
         ts = new DivDivIntegrator(*sigma);
      }
      else
      {
         ts = new CurlCurlIntegrator(*sigma);
      }
   }
   mpd->AddDomainIntegrator(tm);
   kpd->AddDomainIntegrator(ts);
}

void ModelHeat::InitForms(PDCoefficient* mu, Coefficient* sigma)
{
   MFEM_VERIFY(fes->GetVDim() == 1,"Unsupported VDIM " << fes->GetVDim());

   PDBilinearFormIntegrator *tm = NULL;
   BilinearFormIntegrator *ts = NULL;
   if (fe_range == FiniteElement::SCALAR)
   {
      tm = new PDMassIntegrator(*mu);
      ts = new DiffusionIntegrator(*sigma);
   }
   else
   {
      tm = new PDVectorFEMassIntegrator(*mu);
      if (fe_deriv == FiniteElement::DIV)
      {
         ts = new DivDivIntegrator(*sigma);
      }
      else
      {
         ts = new CurlCurlIntegrator(*sigma);
      }
   }
   mpd->AddDomainIntegrator(tm);
   kpd->AddDomainIntegrator(ts);
}

ModelHeat::ModelHeat(Coefficient* mu, MatrixCoefficient* sigma, ParFiniteElementSpace *_fe, Operator::Type _oid)
   : TimeDependentOperator(_fe->GetTrueVSize(), 0.0, TimeDependentOperator::HOMOGENEOUS), PDOperator()
{
   Init(_fe,_oid);
   InitForms(mu,sigma);

   UpdateMass();
   UpdateStiffness();
}

ModelHeat::ModelHeat(PDCoefficient* mu, MatrixCoefficient* sigma, ParFiniteElementSpace *_fe, Operator::Type _oid)
   : TimeDependentOperator(_fe->GetTrueVSize(), 0.0, TimeDependentOperator::HOMOGENEOUS), PDOperator()
{
   Init(_fe,_oid);
   InitForms(mu,sigma);

   UpdateMass();
   UpdateStiffness();
}

ModelHeat::ModelHeat(Coefficient* mu, Coefficient* sigma, ParFiniteElementSpace *_fe, Operator::Type _oid)
   : TimeDependentOperator(_fe->GetTrueVSize(), 0.0, TimeDependentOperator::HOMOGENEOUS), PDOperator()
{
   Init(_fe,_oid);
   InitForms(mu,sigma);

   UpdateMass();
   UpdateStiffness();
}

ModelHeat::ModelHeat(PDCoefficient* mu, Coefficient* sigma, ParFiniteElementSpace *_fe, Operator::Type _oid)
   : TimeDependentOperator(_fe->GetTrueVSize(), 0.0, TimeDependentOperator::HOMOGENEOUS), PDOperator()
{
   Init(_fe,_oid);
   InitForms(mu,sigma);

   UpdateMass();
   UpdateStiffness();
}

void ModelHeat::UpdateStiffness()
{
   kpd->Update();
   kpd->Assemble(0);
   kpd->Finalize(0);
   if (!Kh) Kh = new OperatorHandle(oid);
   kpd->ParallelAssemble(*Kh);
}

void ModelHeat::UpdateMass()
{
   mpd->Update();
   mpd->Assemble(0);
   mpd->Finalize(0);
   if (!Mh) Mh = new OperatorHandle(oid);
   mpd->ParallelAssemble(*Mh);
}

int ModelHeat::GetStateSize()
{
   return fes->GetTrueVSize();
}

int ModelHeat::GetParameterSize()
{
   return mpd->GetParameterSize() + kpd->GetParameterSize();
}

void ModelHeat::SetUpFromParameters(const Vector& p)
{
   MFEM_VERIFY(p.Size() >= GetParameterSize(),"Invalid Vector size " << p.Size() << ", should be (at least) " << GetParameterSize());
   /* XXX multiple coeffs */
   mpd->UpdateParameter(p);
   UpdateMass();
   UpdateStiffness();
}

void ModelHeat::Mult(const Vector& tdstate, const Vector& state, const Vector& m, double t, Vector& f)
{
   SetUpFromParameters(m);
   SetTime(t);
   ImplicitMult(state,tdstate,f);
}

/* Callback used by the adjoint solver */
void ModelHeat::ComputeGradientAdjoint(const Vector& adj, const Vector& tdstate, const Vector& state, const Vector& m, Vector& g)
{
   MFEM_VERIFY(m.Size() >= GetParameterSize(),"Invalid Vector size " << m.Size() << ", should be (at least) " << GetParameterSize());
   MFEM_VERIFY(g.Size() >= GetParameterSize(),"Invalid Vector size " << g.Size() << ", should be (at least) " << GetParameterSize());

   /* XXX multiple coeffs */
   mpd->ComputeGradientAdjoint(adj,tdstate,m,g);
}

/* Callback used by the tangent linear model solver */
void ModelHeat::ComputeGradient(const Vector& tdstate, const Vector& state, const Vector& m, const Vector &pert, Vector& o)
{
   MFEM_VERIFY(m.Size() >= GetParameterSize(),"Invalid Vector size " << m.Size() << ", should be (at least) " << GetParameterSize());
   MFEM_VERIFY(pert.Size() >= GetParameterSize(),"Invalid Vector size " << pert.Size() << ", should be (at least) " << GetParameterSize());
   MFEM_VERIFY(o.Size() == tdstate.Size(),"Invalid Vector size " << o.Size() << ", should be " << tdstate.Size());

   /* XXX multiple coeffs */
   mpd->ComputeGradient(tdstate,m,pert,o);
}

/* Callback used within the Hessian matrix-vector routines of the TS object */
void ModelHeat::ComputeHessian(int A,int B,const Vector& tdstate,const Vector& st,const Vector& m,
                          const Vector& l,const Vector& x,Vector& y)
{
   MFEM_VERIFY(m.Size() >= GetParameterSize(),"Invalid Vector size " << m.Size() << ", should be (at least) " << GetParameterSize());

   /* XXX multiple coeffs */
   if (A == 1 && B == 2) /* L^T \otimes I_N F_XtM x */
   {
      MFEM_VERIFY(y.Size() == tdstate.Size(),"Invalid Vector size " << y.Size() << ", should be " << tdstate.Size());
      MFEM_VERIFY(x.Size() >= GetParameterSize(),"Invalid Vector size " << x.Size() << ", should be (at least) " << GetParameterSize());
      mpd->ComputeHessian_XM(l,m,x,y);
   }
   else if (A == 2 && B == 1) /* L^T \otimes I_P F_MXt x */
   {
      MFEM_VERIFY(x.Size() == tdstate.Size(),"Invalid Vector size " << x.Size() << ", should be " << tdstate.Size());
      MFEM_VERIFY(y.Size() >= GetParameterSize(),"Invalid Vector size " << y.Size() << ", should be (at least) " << GetParameterSize());
      mpd->ComputeHessian_MX(l,x,m,y);
   }
   else
   {
      MFEM_VERIFY(0,"ModelHeat::ComputeHessian not implemented for " << A << ", " << B << " pair");
   }
}

void ModelHeat::SetRHS(Coefficient* _rhs)
{
   rhs = _rhs;
   vrhs = NULL;
   delete rhsform;
   rhsform = NULL;
   if (rhs)
   {
      rhsform  = new ParLinearForm(fes);
      rhsform->AddDomainIntegrator(new DomainLFIntegrator(*rhs));
   }
}

void ModelHeat::SetRHS(VectorCoefficient* _vrhs)
{
   rhs = NULL;
   vrhs = _vrhs;
   delete rhsform;
   rhsform = NULL;
   if (vrhs)
   {
      rhsform  = new ParLinearForm(fes);
      if (fe_range == FiniteElement::SCALAR)
      {
         rhsform->AddDomainIntegrator(new VectorDomainLFIntegrator(*vrhs));
      }
      else
      {
         rhsform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*vrhs));
      }
   }
}

/* Implements M*Xdot + K*X - RHS */
void ModelHeat::ImplicitMult(const Vector &x, const Vector &xdot, Vector &y) const
{
   /* Compute forcing term */
   if (rhs) rhs->SetTime(this->GetTime());
   if (vrhs) vrhs->SetTime(this->GetTime());
   if (rhsform)
   {
      rhsform->Assemble();
      rhsform->ParallelAssemble(*rhsvec);
   }
   else
   {
      *rhsvec = 0.0;
   }

   /* Get mass and stiffness operators */
   Operator *M,*K;
   Mh->Get(M);
   Kh->Get(K);

   /* Compute residual */
   M->Mult(xdot, y);
   y -= *rhsvec;
   K->Mult(x, *rhsvec);
   y += *rhsvec;
}

/* PETSc expects the Jacobian of the ODE as shift * F_xdot + F_x */
Operator& ModelHeat::GetImplicitGradient(const Vector &x, const Vector &xdot, double shift) const
{
   MFJac* Jacobian = NULL;
   std::map<double,MFJac*>::iterator it = Jacobians.find(shift);
   if (it == Jacobians.end())
   {
      Jacobian = new MFJac(*this);
      Jacobian->SetShift(shift);
      Jacobians.insert(std::pair<double,MFJac*>(shift,Jacobian));
   }
   else
   {
      Jacobian = it->second;
   }
   return *Jacobian;
}

void ModelHeat::DeleteJacobians()
{
   std::map<double,MFJac*>::iterator it;
   for (it=Jacobians.begin(); it!=Jacobians.end(); ++it)
   {
      delete it->second;
   }
}

ModelHeat::~ModelHeat()
{
   delete Mh;
   delete Kh;
   DeleteJacobians();
   delete rhsform;
   delete rhsvec;
   delete mpd;
   delete kpd;
   delete adjgf;
   delete stgf;
   delete pfactory;
}

#include <mfemopt/private/mfemoptpetscmacros.h>
#if defined(PETSC_HAVE_HYPRE)
#include "petscmathypre.h"
#endif

Solver* ModelHeat::PreconditionerFactory::NewPreconditioner(const OperatorHandle& oh)
{
  Solver *solver = NULL;
  PetscErrorCode ierr;

  if (oh.Type() == Operator::PETSC_MATSHELL)
  {
     PetscParMatrix *oJ;
     oh.Get(oJ);

     Operator* ctx;
     ierr = MatShellGetContext(*oJ,&ctx); PCHKERRQ(*oJ,ierr);
     ModelHeat::MFJac *J = dynamic_cast<ModelHeat::MFJac *>(ctx);
     MFEM_VERIFY(J,"Not a ModelHeat::MFJac operator");
     double s = J->GetShift();
     Operator *M,*K;
     pde.Mh->Get(M);
     pde.Kh->Get(K);
     HypreParMatrix *hK = dynamic_cast<HypreParMatrix *>(K);
     HypreParMatrix *hM = dynamic_cast<HypreParMatrix *>(M);
     PetscParMatrix *pK = dynamic_cast<PetscParMatrix *>(K);
     PetscParMatrix *pM = dynamic_cast<PetscParMatrix *>(M);
     // this matrix will be owned by the solver
     HypreParMatrix *hA = NULL;
     if (hK && hM) // When using HypreParMatrix, take advantage of the solvers
     {
        if (s != 0.0) hA = Add(s,*hM,1.0,*hK);
        else hA = new HypreParMatrix(*hM);
        if (pde.bc)
        {
           HypreParVector X(*hA,false);
           HypreParVector B(*hA,true);
           X = 0.0;
           hA->EliminateRowsCols(pde.bc->GetTDofs(),X,B);
        }
        if (pde.fe_deriv == FiniteElement::DIV)
        {
           HypreADS *t = new HypreADS(*hA,pde.fes);
           t->SetPrintLevel(0);
           solver = new SymmetricSolver(t,true,hA,true);
        }
        else if (pde.fe_deriv == FiniteElement::CURL)
        {
           HypreAMS *t = new HypreAMS(*hA,pde.fes);
           t->SetPrintLevel(0);
           solver = new SymmetricSolver(t,true,hA,true);
        }
        else
        {
           HypreBoomerAMG *t = new HypreBoomerAMG(*hA);
           t->SetPrintLevel(0);
           solver = new SymmetricSolver(t,true,hA,true);
        }
     }
     else if (pK && pM) // When using PetscParMatrix, either use BDDC (PETSC_MATIS), or test command line preconditioners (all the other types)
     {
        PetscBool ismatis;
        Mat       pJ;

        if (s != 0.0)
        {
           ierr = MatDuplicate(*pK,MAT_COPY_VALUES,&pJ); PCHKERRQ(*pK,ierr);
           ierr = MatAXPY(pJ,s,*pM,SAME_NONZERO_PATTERN); PCHKERRQ(*pK,ierr);
        }
        else
        {
           ierr = MatDuplicate(*pM,MAT_COPY_VALUES,&pJ); PCHKERRQ(*pK,ierr);
        }
        if (s >= 0.0) { ierr = MatSetOption(pJ,MAT_SPD,PETSC_TRUE); PCHKERRQ(*pK,ierr); }
        PetscParMatrix ppJ(pJ,false); // Do not take reference, since the Mat will be owned by the preconditioner
        if (pde.bc)
        {
           PetscParVector dummy(PetscObjectComm((PetscObject)pJ),0);
           ppJ.EliminateRowsCols(pde.bc->GetTDofs(),dummy,dummy);
        }
        ierr = PetscObjectTypeCompare((PetscObject)pJ,MATIS,&ismatis); PCHKERRQ(*pK,ierr);
        if (ismatis)
        {
           PetscBDDCSolverParams opts;
           opts.SetSpace(pde.fes);
           if (pde.bc)
           {
              Array<int>& ess = pde.bc->GetTDofs();
              opts.SetEssBdrDofs(&ess);
           }
           solver = new PetscBDDCSolver(ppJ,opts,"heat_");
        }
        else
        {
           solver = new PetscPreconditioner(ppJ,"heat_");
        }
     }
     else
     {
        std::ostringstream errstr;
        errstr << "Invalid combination: K ";
        errstr << (hK ? "HypreParMatrix" : (pK ? "PetscParMatrix" : "not recognized"));
        errstr << ", M ";
        errstr << (hM ? "HypreParMatrix" : (pM ? "PetscParMatrix" : "not recognized"));
        mfem_error(errstr.str().c_str());
     }
  }
  else
  {
     std::ostringstream errstr;
     errstr << "Unhandled operator type ";
     errstr << oh.Type();
     errstr << ". Run using PetscODESolver::SetJacobianType(Operator::ANY_TYPE);";
     errstr << " and using your favourite OperatorType in the ModelHeat constructor.";
     mfem_error(errstr.str().c_str());
  }
  return solver;
}

}

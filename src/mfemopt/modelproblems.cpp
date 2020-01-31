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

void ModelHeat::MFJac::Mult_Private(const mfem::Vector& x,mfem::Vector& y,bool trans) const
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
};

void ModelHeat::MFJac::Mult(const mfem::Vector& x,mfem::Vector& y) const
{
   Mult_Private(x,y,false);
}

void ModelHeat::MFJac::MultTranspose(const mfem::Vector& x,mfem::Vector& y) const
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

   mu_pd_bilin = NULL;
   sigma_pd_bilin = NULL;

   fes = _fe;
   ParFiniteElementSpaceGetRangeAndDeriv(*fes,&fe_range,&fe_deriv);

   oid = _oid;

   adjgf = new ParGridFunction(fes);
   stgf = new ParGridFunction(fes);
   k = new ParBilinearForm(fes);
   m = new ParBilinearForm(fes);
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
   m->AddDomainIntegrator(tm);
   k->AddDomainIntegrator(ts);
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
   m->AddDomainIntegrator(tm);
   k->AddDomainIntegrator(ts);

   /* just flag we have a parameter dependent bilinear form */
   mu_pd_bilin = tm;
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
   m->AddDomainIntegrator(tm);
   k->AddDomainIntegrator(ts);
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
   m->AddDomainIntegrator(tm);
   k->AddDomainIntegrator(ts);

   /* just flag we have a parameter dependent bilinear form */
   mu_pd_bilin = tm;
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
   k->Update();
   k->Assemble(0);
   k->Finalize(0);
   if (!Kh) Kh = new OperatorHandle(oid);
   k->ParallelAssemble(*Kh);
}

void ModelHeat::UpdateMass()
{
   m->Update();
   m->Assemble(0);
   m->Finalize(0);
   if (!Mh) Mh = new OperatorHandle(oid);
   m->ParallelAssemble(*Mh);
}

void ModelHeat::GetCurrentVector(Vector &m)
{
   MFEM_VERIFY(m.Size() >= GetParameterSize(),"Invalid Vector size " << m.Size() << ", should be (at least) " << GetParameterSize());
   double *data = m.GetData();
   if (mu_pd_bilin)
   {
      int ls = mu_pd_bilin->GetLocalSize();
      Vector pm(data,ls);
      mu_pd_bilin->GetCurrentVector(pm);
      data += ls;
      pm.SetData(NULL); /* XXX clang static analysis */
   }
   if (sigma_pd_bilin)
   {
      int ls = sigma_pd_bilin->GetLocalSize();
      Vector pm(data,ls);
      sigma_pd_bilin->GetCurrentVector(pm);
      pm.SetData(NULL); /* XXX clang static analysis */
   }
}

int ModelHeat::GetStateSize()
{
   return fes->GetTrueVSize();
}

int ModelHeat::GetParameterSize()
{
   int ls = 0;
   if (mu_pd_bilin) ls += mu_pd_bilin->GetLocalSize();
   if (sigma_pd_bilin) ls += sigma_pd_bilin->GetLocalSize();
   return ls;
}

void ModelHeat::SetUpFromParameters(const Vector& p)
{
   MFEM_VERIFY(p.Size() >= GetParameterSize(),"Invalid Vector size " << p.Size() << ", should be (at least) " << GetParameterSize());
   double *data = p.GetData();
   if (mu_pd_bilin)
   {
      int ls = mu_pd_bilin->GetLocalSize();
      Vector pp(data,ls);
      mu_pd_bilin->UpdateCoefficient(pp);
      data += ls;
      pp.SetData(NULL); /* XXX clang static analysis */
   }
   if (sigma_pd_bilin)
   {
      int ls = sigma_pd_bilin->GetLocalSize();
      Vector pp(data,ls);
      sigma_pd_bilin->UpdateCoefficient(pp);
      pp.SetData(NULL); /* XXX clang static analysis */
   }
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

   double *mdata = m.GetData();
   double *gdata = g.GetData();

   adjgf->Distribute(adj);
   stgf->Distribute(tdstate);
   if (mu_pd_bilin)
   {
      int ls = mu_pd_bilin->GetLocalSize();
      Vector pm(mdata,ls);
      Vector pg(gdata,ls);
      mu_pd_bilin->ComputeGradientAdjoint(adjgf,stgf,pm,pg);
   }
   /* XXX multiple coeffs */
}

/* Callback used by the tangent linear model solver */
void ModelHeat::ComputeGradient(const Vector& tdstate, const Vector& state, const Vector& m, const Vector &pert, Vector& o)
{
   MFEM_VERIFY(m.Size() >= GetParameterSize(),"Invalid Vector size " << m.Size() << ", should be (at least) " << GetParameterSize());
   MFEM_VERIFY(pert.Size() >= GetParameterSize(),"Invalid Vector size " << pert.Size() << ", should be (at least) " << GetParameterSize());
   MFEM_VERIFY(o.Size() == tdstate.Size(),"Invalid Vector size " << o.Size() << ", should be " << tdstate.Size());

   double *mdata = m.GetData();
   double *pdata = pert.GetData();
   double *odata = o.GetData();

   stgf->Distribute(tdstate);
   if (mu_pd_bilin)
   {
      int ls = mu_pd_bilin->GetLocalSize();
      Vector pm(mdata,ls);
      Vector ppert(pdata,ls);
      Vector po(odata,tdstate.Size());
      mu_pd_bilin->ComputeGradient(stgf,pm,ppert,po);
   }
   /* XXX multiple coeffs */
}

/* Callback used within the Hessian matrix-vector routines of the TS object */
void ModelHeat::ComputeHessian(int A,int B,const Vector& tdstate,const Vector& st,const Vector& m,
                          const Vector& l,const Vector& x,Vector& y)
{
   MFEM_VERIFY(m.Size() >= GetParameterSize(),"Invalid Vector size " << m.Size() << ", should be (at least) " << GetParameterSize());

   double *mdata = m.GetData();
   double *ydata = y.GetData();
   double *xdata = x.GetData();

   /* XXX multiple coeffs */
   if (A == 1 && B == 2) /* L^T \otimes I_N F_XtM x */
   {
      MFEM_VERIFY(y.Size() == tdstate.Size(),"Invalid Vector size " << y.Size() << ", should be " << tdstate.Size());
      MFEM_VERIFY(x.Size() >= GetParameterSize(),"Invalid Vector size " << x.Size() << ", should be (at least) " << GetParameterSize());
      adjgf->Distribute(l);
      if (mu_pd_bilin)
      {
         int ls = mu_pd_bilin->GetLocalSize();
         Vector pm(mdata,ls);
         Vector ppert(xdata,ls);
         Vector py(ydata,tdstate.Size());
         mu_pd_bilin->ComputeHessian_XM(adjgf,pm,ppert,py);
      }
   }
   else if (A == 2 && B == 1) /* L^T \otimes I_P F_MXt x */
   {
      MFEM_VERIFY(x.Size() == tdstate.Size(),"Invalid Vector size " << x.Size() << ", should be " << tdstate.Size());
      MFEM_VERIFY(y.Size() >= GetParameterSize(),"Invalid Vector size " << y.Size() << ", should be (at least) " << GetParameterSize());
      adjgf->Distribute(l);
      stgf->Distribute(x);
      if (mu_pd_bilin)
      {
         int ls = mu_pd_bilin->GetLocalSize();
         Vector py(ydata,ls);
         Vector pm(mdata,ls);
         mu_pd_bilin->ComputeHessian_MX(adjgf,stgf,pm,py);
      }
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
   delete m;
   delete k;
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
        hA = Add(s,*hM,1.0,*hK);
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

        ierr = MatDuplicate(*pK,MAT_COPY_VALUES,&pJ); PCHKERRQ(*pK,ierr);
        ierr = MatAXPY(pJ,s,*pM,SAME_NONZERO_PATTERN); PCHKERRQ(*pK,ierr);
        if (s > 0.0) { ierr = MatSetOption(pJ,MAT_SPD,PETSC_TRUE); PCHKERRQ(*pK,ierr); }
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
           solver = new PetscBDDCSolver(ppJ,opts);
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

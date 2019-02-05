#include <mfemopt/modelproblems.hpp>
#include <mfemopt/mfemextra.hpp>
#include <petscopt/tsopt.h>

namespace mfemopt
{

using namespace mfem;

void ModelHeat::Init(ParFiniteElementSpace *_fe, Operator::Type _oid)
{
   Mh = NULL;
   Kh = NULL;
   rhsform = NULL;
   Jacobian = NULL;

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
      pfactory = new PreconditionerFactory(*this,"heat model preconditioner factory");
   }
   return pfactory;
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
      //mu_pd_bilin->UpdateCoefficient(m); /* WHY THIS WAS HERE? */
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
      //mu_pd_bilin->UpdateCoefficient(m); /* WHY THIS WAS HERE? */
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
         //mu_pd_bilin->UpdateCoefficient(m); /* WHY THIS WAS HERE? */
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
         //mu_pd_bilin->UpdateCoefficient(m); /* WHY THIS WAS HERE? */
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
   delete rhsform;
   rhsform  = new ParLinearForm(fes);
   rhsform->AddDomainIntegrator(new DomainLFIntegrator(*rhs));
}

void ModelHeat::SetRHS(VectorCoefficient* _vrhs)
{
   vrhs = _vrhs;
   delete rhsform;
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
   switch (Mh->Type())
   {
      case Operator::Hypre_ParCSR:
      {
         delete Jacobian;
         Jacobian = Add(shift,*(Mh->As<HypreParMatrix>()),1.0,*(Kh->As<HypreParMatrix>()));
         break;
      }
      case Operator::PETSC_MATHYPRE:
      case Operator::PETSC_MATAIJ:
      case Operator::PETSC_MATIS:
      {
         if (!Jacobian) Jacobian = new PetscParMatrix();
         PetscParMatrix *pJacobian = dynamic_cast<PetscParMatrix *>(Jacobian);
         *pJacobian  = *(Mh->As<PetscParMatrix>());
         *pJacobian *= shift;
         *pJacobian += *(Kh->As<PetscParMatrix>());
         break;
      }
      default:
      {
         MFEM_ABORT("To be implemented");
         break;
      }
   }
   return *Jacobian;
}

ModelHeat::~ModelHeat()
{
   delete Mh;
   delete Kh;
   delete Jacobian;
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

  if (oh.Type() == Operator::PETSC_MATIS)
  {
     PetscParMatrix *pA;
     oh.Get(pA);
     MatSetOption(*pA,MAT_SPD,PETSC_TRUE);
     PetscBDDCSolverParams opts;
     opts.SetSpace(pde.fes);
     if (pde.bc)
     {
        Array<int>& ess = pde.bc->GetTDofs();
        opts.SetEssBdrDofs(&ess);
     }
     solver = new PetscBDDCSolver(*pA,opts);
  }
#if 0
|| defined(PETSC_HAVE_HYPRE)
  else if (oh.Type() == Operator::PETSC_MATHYPRE)
  {
     PetscParMatrix *pA;
     oh.Get(pA);
     MatSetOption(*pA,MAT_SPD,PETSC_TRUE);

     hypre_ParCSRMatrix *parcsr;
     ierr = MatHYPREGetParCSR(*pA,&parcsr); PCHKERRQ(*pA,ierr);
     delete hA;
     hA = new HypreParMatrix(parcsr,false);
     if (pde.fe_range == FiniteElement::SCALAR)
     {
        HypreBoomerAMG *t = new HypreBoomerAMG(*hA);
        t->SetPrintLevel(0);
        solver = t;
     }
     else
     {
        if (pde.fe_deriv == FiniteElement::DIV)
        {
           HypreADS *t = new HypreADS(*hA,pde.fes);
           t->SetPrintLevel(0);
           solver = t;
        }
        else
        {
           HypreAMS *t = new HypreAMS(*hA,pde.fes);
           t->SetPrintLevel(0);
           solver = t;
        }
     }
  }
#endif
  else if (oh.Type() == Operator::Hypre_ParCSR)
  {
     HypreParMatrix *ohA;
     oh.Get(ohA);

     if (pde.fe_range == FiniteElement::SCALAR)
     {
        HypreBoomerAMG *t = new HypreBoomerAMG(*ohA);
        t->SetPrintLevel(0);
        solver = t;
     }
     else
     {
        if (pde.fe_deriv == FiniteElement::DIV)
        {
           HypreADS *t = new HypreADS(*ohA,pde.fes);
           t->SetPrintLevel(0);
           solver = t;
        }
        else
        {
           HypreAMS *t = new HypreAMS(*ohA,pde.fes);
           t->SetPrintLevel(0);
           solver = t;
        }
     }
  }
  else if (oh.Type() == Operator::PETSC_MATAIJ)
  {
     PetscParMatrix *pA;
     oh.Get(pA);
     ierr = MatSetOption(*pA,MAT_SPD,PETSC_TRUE); PCHKERRQ(*pA,ierr);

     if (pde.fe_range == FiniteElement::SCALAR)
     {
        solver = new PetscPreconditioner(*pA);
     }
     else
     {
#if defined(PETSC_HAVE_HYPRE)
        // convert and attach the converted Mat to the original one so that it will get destroyed
        Mat tA;
        ierr = MatConvert(*pA,MATHYPRE,MAT_INITIAL_MATRIX,&tA); PCHKERRQ(*pA,ierr);
        ierr = PetscObjectCompose(*pA,"__mfemopt_temporary_convmat",(PetscObject)tA); PCHKERRQ(*pA,ierr);
        ierr = PetscObjectDereference((PetscObject)tA); PCHKERRQ(*pA,ierr);

        delete hA;
        hypre_ParCSRMatrix *parcsr;
        ierr = MatHYPREGetParCSR(tA,&parcsr); PCHKERRQ(*pA,ierr);
        hA = new HypreParMatrix(parcsr,false);

        if (pde.fe_deriv == FiniteElement::DIV)
        {
           HypreADS *t = new HypreADS(*hA,pde.fes);
           t->SetPrintLevel(0);
           solver = t;
        }
        else
        {
           HypreAMS *t = new HypreAMS(*hA,pde.fes);
           t->SetPrintLevel(0);
           solver = t;
        }
#else
        solver = new PetscPreconditioner(*pA);
#endif
    }
  }
  else
  {
     mfem_error("Unhandled operator type");
  }
  return solver;
}

}

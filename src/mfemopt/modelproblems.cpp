#include <mfemopt/modelproblems.hpp>

namespace mfemopt
{

using namespace mfem;

void ModelHeat::Init(ParFiniteElementSpace *_fe, Operator::Type _oid)
{
   Mh = NULL;
   Kh = NULL;
   rhsform = NULL;
   Jacobian = NULL;
   mu_bilin = NULL;

   fes = _fe;
   oid = _oid;

   adjgf = new ParGridFunction(fes);
   stgf = new ParGridFunction(fes);
   k = new ParBilinearForm(fes);
   m = new ParBilinearForm(fes);
   rhsvec = new PetscParVector(fes);
}

void ModelHeat::InitForms(Coefficient* mu, MatrixCoefficient* sigma)
{
   m->AddDomainIntegrator(new MassIntegrator(*mu));
   k->AddDomainIntegrator(new DiffusionIntegrator(*sigma));
}

void ModelHeat::InitForms(PDCoefficient* mu, MatrixCoefficient* sigma)
{
   mu_bilin = new PDMassIntegrator(mu); /* TODO: fix inconsistency */
   m->AddDomainIntegrator(mu_bilin);
   k->AddDomainIntegrator(new DiffusionIntegrator(*sigma));
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
   /* XXX multiple coeffs */
   if (mu_bilin) mu_bilin->GetCurrentVector(m);
}

int ModelHeat::GetStateSize()
{
   return fes->GetTrueVSize();
}

int ModelHeat::GetParameterSize()
{
   int ls = 0;
   /* XXX multiple coeffs */
   if (mu_bilin) ls += mu_bilin->GetLocalSize();
   return ls;
}

void ModelHeat::SetUpFromParameters(const Vector& p)
{
   /* XXX multiple coeffs */
   if (mu_bilin) mu_bilin->UpdateCoefficient(p);
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
   /* XXX multiple coeffs */
   adjgf->Distribute(adj);
   stgf->Distribute(tdstate);
   if (mu_bilin)
   {
      mu_bilin->UpdateCoefficient(m); /* TODO: I should take note of when the coefficient gets updated */
      Vector pm(m.GetData(),mu_bilin->GetLocalSize()); /* TODO FIX CONSTS */
      Vector pg(g.GetData(),mu_bilin->GetLocalSize());
      mu_bilin->ComputeGradientAdjoint(adjgf,stgf,pm,pg);
   }
}

/* Callback used by the tangent linear model solver */
void ModelHeat::ComputeGradient(const Vector& tdstate, const Vector& state, const Vector& m, const Vector &pert, Vector& o)
{
   /* XXX multiple coeffs */
   stgf->Distribute(tdstate);
   if (mu_bilin)
   {
      mu_bilin->UpdateCoefficient(m); /* TODO: I should take note of when the coefficient gets updated */
      Vector ppert(pert.GetData(),mu_bilin->GetLocalSize());
      Vector po(o.GetData(),tdstate.Size());
      Vector pm(m.GetData(),mu_bilin->GetLocalSize()); /* TODO FIX CONSTS */
      mu_bilin->ComputeGradient(stgf,pm,ppert,po);
   }
}

/* Used by */
void ModelHeat::ComputeHessian(int A,int B,const Vector& tdst,const Vector& st,const Vector& m,
                          const Vector& l,const Vector& x,Vector& y)
{
   /* XXX multiple coeffs */
   if (A == 1 && B == 2) /* L^T \otimes I_N F_XtM x */
   {
      adjgf->Distribute(l);
      if (mu_bilin)
      {
         mu_bilin->UpdateCoefficient(m); /* TODO: I should take note of when the coefficient gets updated */
         Vector ppert(x.GetData(),mu_bilin->GetLocalSize());
         Vector py(y.GetData(),tdst.Size());
         Vector pm(m.GetData(),mu_bilin->GetLocalSize()); /* TODO FIX CONSTS */
         mu_bilin->ComputeHessian_XM(adjgf,pm,ppert,py);
      }
   }
   else if (A == 2 && B == 1) /* L^T \otimes I_P F_MXt x */
   {
      adjgf->Distribute(l);
      stgf->Distribute(x);
      if (mu_bilin)
      {
         mu_bilin->UpdateCoefficient(m); /* TODO: I should take note of when the coefficient gets updated */
         Vector py(y.GetData(),mu_bilin->GetLocalSize());
         Vector pm(m.GetData(),mu_bilin->GetLocalSize()); /* TODO FIX CONSTS */
         mu_bilin->ComputeHessian_MX(adjgf,stgf,pm,py);
      }
   }
   else
   {
      y = 0.0;
   }
}

void ModelHeat::SetRHS(Coefficient* _rhs)
{
   delete rhsform;
   rhs.SetSize(0);
   rhs.Append(_rhs);
   rhsform  = new ParLinearForm(fes);
   for (int i = 0; i < rhs.Size(); i++) rhsform->AddDomainIntegrator(new DomainLFIntegrator(*rhs[i]));
}

/* Implements M*Xdot + K*X - RHS */
void ModelHeat::ImplicitMult(const Vector &x, const Vector &xdot, Vector &y) const
{
   /* Compute forcing term */
   for (int i = 0; i < rhs.Size(); i++) rhs[i]->SetTime(this->GetTime());
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
}

}

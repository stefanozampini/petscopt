#include <mfemopt/objective.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <mfem/mesh/pmesh.hpp>
#include <mfem/linalg/petsc.hpp>
#include <petscmat.h>
#include <petsc/private/petscimpl.h>
#include <cmath>
#include <limits>

static PetscErrorCode ObjComputeHessianMFFD_Private(void*,Vec,Vec);

typedef struct
{
   mfemopt::ObjectiveFunction *obj;
   mfem::Vector *xIn;
   mfem::Vector *mIn;
   double t;
   int which;
} __obj_fdhessian_ctx;

namespace mfemopt
{

using namespace mfem;

ObjectiveHessianOperatorFD::ObjectiveHessianOperatorFD(MPI_Comm _comm, ObjectiveFunction *_obj, const Vector& _xIn, const Vector& _mIn, double _t,  int _A, int _B) : PetscParMatrix()
{
   MFEM_VERIFY(_A >=0 && _A < 2 && _B >=0 && _B < 2,"A and B should be in [0,1]");

   obj = _obj;
   xIn = _xIn;
   mIn = _mIn;
   t = _t;
   A = _A;
   B = _B;

   height = (A == 0) ? xIn.Size() : mIn.Size();
   width = (B == 0) ? xIn.Size() : mIn.Size();

   PetscErrorCode ierr;

   __obj_fdhessian_ctx *mffd;
   ierr = PetscNew(&mffd);CCHKERRQ(_comm,ierr);
   mffd->obj = obj;
   mffd->xIn = &xIn;
   mffd->mIn = &mIn;
   mffd->t = _t;
   mffd->which = B;

   Mat H;
   ierr = MatCreate(_comm,&H);CCHKERRQ(_comm,ierr);
   ierr = MatSetSizes(H,height,width,PETSC_DECIDE,PETSC_DECIDE);CCHKERRQ(_comm,ierr);
   ierr = MatSetType(H,MATMFFD);CCHKERRQ(_comm,ierr);
   ierr = MatSetUp(H);CCHKERRQ(_comm,ierr);
   if (A == 0)
   {
      PetscParVector X(_comm,xIn,true);
      ierr = MatMFFDSetBase(H,X,NULL);CCHKERRQ(_comm,ierr);
   }
   else
   {
      PetscParVector X(_comm,mIn,true);
      ierr = MatMFFDSetBase(H,X,NULL);CCHKERRQ(_comm,ierr);
   }
   ierr = MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))ObjComputeHessianMFFD_Private,mffd);CCHKERRQ(_comm,ierr);
   ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CCHKERRQ(_comm,ierr);
   ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CCHKERRQ(_comm,ierr);

   PetscContainer c;
   ierr = PetscContainerCreate(_comm,&c);CCHKERRQ(_comm,ierr);
   ierr = PetscContainerSetPointer(c,mffd);CCHKERRQ(_comm,ierr);
   ierr = PetscContainerSetUserDestroy(c,PetscContainerUserDestroyDefault);CCHKERRQ(_comm,ierr);
   ierr = PetscObjectCompose((PetscObject)H,"__mffd_ctx",(PetscObject)c);CCHKERRQ(_comm,ierr);
   ierr = PetscContainerDestroy(&c);CCHKERRQ(_comm,ierr);

   SetMat(H);
   ierr = MatDestroy(&H);CCHKERRQ(_comm,ierr);
}

void ObjectiveFunction::TestFDGradient(MPI_Comm comm, const Vector& xIn, const Vector& mIn, double t, double delta, bool progress)
{
   PetscErrorCode ierr;
   PetscBool verbose = PETSC_FALSE;
   ierr = PetscOptionsGetBool(NULL,NULL,"-fd_gradient_verbose",&verbose,NULL); CCHKERRQ(comm,ierr);

   PetscReal h = delta;

   PetscParVector px(comm,xIn,true);
   PetscParVector pm(comm,mIn,true);

   PetscMPIInt rank;
   ierr = MPI_Comm_rank(comm,&rank);CCHKERRQ(comm,ierr);

   ierr = PetscPrintf(comm,"ObjectiveFunction::TestFDGradient, delta: %g\n",(double)h);CCHKERRQ(comm,ierr);
   if (has_x)
   {
      ierr = PetscPrintf(comm,"-> hang tight while computing state gradient");CCHKERRQ(comm,ierr);
      PetscParVector g(comm,xIn);
      (*this).EvalGradient_X(xIn,mIn,t,g);

      PetscParVector fdg(comm,xIn);
      fdg = 0.0;
      PetscParVector x(px);
      for (PetscInt i = 0; i < g.GlobalSize(); i++)
      {
         double f1=0.0,f2=0.0;
         Array<PetscInt> idx(1);
         Array<PetscScalar> vals(1);

         if (progress)
         {
            ierr = PetscPrintf(comm,"\r-> hang tight while computing state gradient : %f%%",(i*100.0)/g.GlobalSize());CCHKERRQ(comm,ierr);
         }

         x = px;

         idx[0] = i;

         vals[0] = !rank ? -h : 0.0;
         x.AddValues(idx,vals);
         (*this).Eval(x,mIn,t,&f1);
         vals[0] = !rank ? 2*h : 0.0;
         x.AddValues(idx,vals);
         (*this).Eval(x,mIn,t,&f2);

         vals[0] = (f2 - f1)/(2.0*h);
         fdg.SetValues(idx,vals);
      }
      if (progress)
      {
         ierr = PetscPrintf(comm,"\r-> hang tight while computing state gradient : 100.000000%%\n");CCHKERRQ(comm,ierr);
      }
      else
      {
         ierr = PetscPrintf(comm,"\n");CCHKERRQ(comm,ierr);
      }
      PetscParVector dg(g);
      dg = g;
      dg -= fdg;
      double gn = std::sqrt(InnerProduct(comm,g,g));
      double dgn = std::sqrt(InnerProduct(comm,dg,dg));
      double fdgn = std::sqrt(InnerProduct(comm,fdg,fdg));
      double cosine = InnerProduct(comm,g,fdg)/(gn*fdgn);
      ierr = PetscPrintf(comm,"||g|| = %g, ||g_fd|| = %g, ||g - g_fd|| = %g, ||g-g_fd||/||g_fd|| = %g, cos() = %g\n",gn,fdgn,dgn,dgn/fdgn,cosine);CCHKERRQ(comm,ierr);

      if (verbose)
      {
         ierr = PetscPrintf(comm,"FINITE DIFFERENCE\n");CCHKERRQ(comm,ierr);
         fdg.Print();
         ierr = PetscPrintf(comm,"COMPUTED\n");CCHKERRQ(comm,ierr);
         g.Print();
         ierr = PetscPrintf(comm,"DIFFERENCE\n");CCHKERRQ(comm,ierr);
         dg.Print();
      }
   }
   if (has_m)
   {
      ierr = PetscPrintf(comm,"-> hang tight while computing design gradient");CCHKERRQ(comm,ierr);

      PetscParVector g(comm,mIn);
      (*this).EvalGradient_M(xIn,mIn,t,g);

      PetscParVector fdg(comm,mIn);
      PetscParVector m(pm);
      for (PetscInt i = 0; i < g.GlobalSize(); i++)
      {
         if (progress)
         {
            ierr = PetscPrintf(comm,"\r-> hang tight while computing design gradient : %f%%",(i*100.0)/g.GlobalSize());CCHKERRQ(comm,ierr);
         }
         double f1=0.0,f2=0.0;
         Array<PetscInt> idx(1);
         Array<PetscScalar> vals(1);
         idx[0] = i;

         m = pm;
         vals[0] = !rank ? -h : 0.0;
         m.AddValues(idx,vals);
         (*this).Eval(xIn,m,t,&f1);
         vals[0] = !rank ? 2*h : 0.0;
         m.AddValues(idx,vals);
         (*this).Eval(xIn,m,t,&f2);

         vals[0] = (f2 - f1)/(2.0*h);
         fdg.SetValues(idx,vals);
      }
      if (progress)
      {
         ierr = PetscPrintf(comm,"\r-> hang tight while computing state gradient : 100.000000%%\n");CCHKERRQ(comm,ierr);
      }
      else
      {
         ierr = PetscPrintf(comm,"\n");CCHKERRQ(comm,ierr);
      }

      PetscParVector dg(g);
      dg = g;
      dg -= fdg;
      double gn = std::sqrt(InnerProduct(comm,g,g));
      double dgn = std::sqrt(InnerProduct(comm,dg,dg));
      double fdgn = std::sqrt(InnerProduct(comm,fdg,fdg));
      double cosine = InnerProduct(comm,g,fdg)/(gn*fdgn);
      ierr = PetscPrintf(comm,"||g|| = %g, ||g_fd|| = %g, ||g - g_fd|| = %g, ||g-g_fd||/||g_fd|| = %g, cos() = %g\n",gn,fdgn,dgn,dgn/fdgn,cosine);CCHKERRQ(comm,ierr);

      if (verbose)
      {
         ierr = PetscPrintf(comm,"FINITE DIFFERENCE\n");CCHKERRQ(comm,ierr);
         fdg.Print();
         ierr = PetscPrintf(comm,"COMPUTED\n");CCHKERRQ(comm,ierr);
         g.Print();
         ierr = PetscPrintf(comm,"DIFFERENCE\n");CCHKERRQ(comm,ierr);
         dg.Print();
      }
   }
}

void ObjectiveFunction::TestFDHessian(MPI_Comm comm, const Vector& xIn, const Vector& mIn, double t)
{
   PetscErrorCode ierr;
   PetscBool verbose = PETSC_FALSE;
   ierr = PetscOptionsGetBool(NULL,NULL,"-fd_hessian_verbose",&verbose,NULL); CCHKERRQ(comm,ierr);

   Operator *H;

   (*this).SetUpHessian_XX(xIn,mIn,t);
   H = GetHessianOperator_XX();
   if (H)
   {
      ierr = PetscPrintf(comm,"ObjectiveFunction::TestFDHessian state x state\n");CCHKERRQ(comm,ierr);
      ObjectiveHessianOperatorFD fdH(comm,this,xIn,mIn,t,0,0);
      PetscParMatrix *pfdH = new PetscParMatrix(comm,&fdH,Operator::PETSC_MATAIJ);
      PetscParMatrix *pH = new PetscParMatrix(comm,H,Operator::PETSC_MATAIJ);
      if (verbose)
      {
         pfdH->Print();
         pH->Print();
      }
      PetscReal nrm,nrmd,nrminf;
      PetscParMatrix *diff = new PetscParMatrix();
      *diff = *pH;
      *diff -= *pfdH;
      ierr = MatNorm(*pH,NORM_INFINITY,&nrm);CCHKERRQ(comm,ierr);
      ierr = MatNorm(*pfdH,NORM_INFINITY,&nrmd);CCHKERRQ(comm,ierr);
      ierr = MatNorm(*diff,NORM_INFINITY,&nrminf);CCHKERRQ(comm,ierr);
      ierr = PetscPrintf(comm,"||H||_inf = %g, ||H_fd||_inf = %g, ||H - H_fd||_inf = %g, ||H-H_fd||_inf/||H_fd||_inf = %g\n",nrm,nrmd,nrminf,nrminf/nrmd);CCHKERRQ(comm,ierr);
      delete diff;
      delete pH;
      delete pfdH;
   }

   (*this).SetUpHessian_MM(xIn,mIn,t);
   H = GetHessianOperator_MM();
   if (H)
   {
      ierr = PetscPrintf(comm,"ObjectiveFunction::TestFDHessian design x design\n");CCHKERRQ(comm,ierr);
      ObjectiveHessianOperatorFD fdH(comm,this,xIn,mIn,t,1,1);
      PetscParMatrix *pfdH = new PetscParMatrix(comm,&fdH,Operator::PETSC_MATAIJ);
      PetscParMatrix *pH = new PetscParMatrix(comm,H,Operator::PETSC_MATAIJ);
      if (verbose)
      {
         pfdH->Print();
         pH->Print();
      }
      PetscReal nrm,nrmd,nrminf;
      PetscParMatrix *diff = new PetscParMatrix();
      *diff = *pH;
      *diff -= *pfdH;
      if (verbose)
      {
         diff->Print();
      }
      ierr = MatNorm(*pH,NORM_INFINITY,&nrm);CCHKERRQ(comm,ierr);
      ierr = MatNorm(*pfdH,NORM_INFINITY,&nrmd);CCHKERRQ(comm,ierr);
      ierr = MatNorm(*diff,NORM_INFINITY,&nrminf);CCHKERRQ(comm,ierr);
      ierr = PetscPrintf(comm,"||H||_inf = %g, ||H_fd||_inf = %g, ||H - H_fd||_inf = %g, ||H-H_fd||_inf/||H_fd||_inf = %g\n",nrm,nrmd,nrminf,nrminf/nrmd);CCHKERRQ(comm,ierr);
      delete diff;
      delete pH;
      delete pfdH;
   }
}

// TikhonovRegularizer
TikhonovRegularizer::TikhonovRegularizer(PDCoefficient *_u0) : ObjectiveFunction(false,true)
{
   Array<ParGridFunction*> &pgf = _u0->GetCoeffs();
   if (!pgf.Size()) return;
   MFEM_VERIFY(pgf.Size() == 1,"Not yet coded");

   BilinearForm *m = new BilinearForm(pgf[0]->ParFESpace());
   m->AddDomainIntegrator(new MassIntegrator());
   m->Assemble(0);
   m->Finalize(0);

   PetscParMatrix *tM = new PetscParMatrix(pgf[0]->ParFESpace()->GetParMesh()->GetComm(),
                                           pgf[0]->ParFESpace()->GlobalVSize(),
                                           (PetscInt*)pgf[0]->ParFESpace()->GetDofOffsets(),
                                           &(m->SpMat()),
                                           Operator::PETSC_MATAIJ);
   delete m;

   PetscParMatrix *P = _u0->GetP();

   H_MM = RAP(tM,P);
   delete tM;

   u0 = new PetscParVector(*P);
   _u0->GetCurrentVector(*u0);
}

void TikhonovRegularizer::Eval(const mfem::Vector& u,const mfem::Vector& m,double t,double* f)
{
   Vector x(m),Mx; /* XXX work array */
   x -= *u0;
   Mx.SetSize(H_MM->Height());
   H_MM->Mult(x,Mx);
   *f = 0.5*InnerProduct(u0->GetComm(),x,Mx);
}

void TikhonovRegularizer::EvalGradient_M(const mfem::Vector& u,const mfem::Vector& m,double t,mfem::Vector &g)
{
   Vector x(m);
   x -= *u0;
   H_MM->Mult(x,g);
}

TikhonovRegularizer::~TikhonovRegularizer()
{
   delete u0;
}

// least squares sampling : 1/2 || E - D ||^2
static double F_ls(const Vector& E, const Vector& D)
{
   Vector T(E);
   T -= D;
   return 0.5*(T*T);
}

// derivative least squares sampling : E = (E - D)
static void dFdu_ls(Vector& E, const Vector& D)
{
   E -= D;
}

void TDLeastSquares::Init()
{
   u = NULL;
   rhsform_x = NULL;
   own_recv = false;
}

TDLeastSquares::TDLeastSquares() : ObjectiveFunction(true,false)
{
   Init();
}


TDLeastSquares::TDLeastSquares(const Array<Receiver*>& _receivers, ParFiniteElementSpace* _fe, bool _own_recv) : ObjectiveFunction(true,false)
{
   Init();
   SetReceivers(_receivers,_own_recv);
   SetParFiniteElementSpace(_fe);
   H_XX = new TDLeastSquaresHessian(this);
}

void TDLeastSquares::SetReceivers(const Array<Receiver*>& _receivers, bool _own_recv)
{
   if (own_recv)
   {
      for (int i=0; i<receivers.Size(); i++)
      {
         delete receivers[i];
      }
   }
   own_recv = _own_recv;
   receivers.SetSize(_receivers.Size());
   receivers.Assign(_receivers);
   receivers_eid.SetSize(0);
}

void TDLeastSquares::SetParFiniteElementSpace(ParFiniteElementSpace* _fe)
{
   delete u;
   u = new ParGridFunction(_fe);
   ResetDeltaCoefficients();
   receivers_eid.SetSize(0);
   InitReceivers();
   InitDeltaCoefficients();
}

void TDLeastSquares::InitReceivers()
{
   MFEM_VERIFY(u,"TDLeastSquares::InitReceivers() ParGridFunction not present");
   if (receivers.Size() == receivers_eid.Size()) return;

   ParMesh *mesh = u->ParFESpace()->GetParMesh();

   int nc = receivers.Size();
   int sdim = mesh->SpaceDimension();
   receivers_eid.SetSize(nc);
   receivers_ip.SetSize(nc);
   receivers_eid = -1;

   DenseMatrix centers(sdim,nc);
   for (int i = 0; i < nc; i++)
   {
      const Vector& center = receivers[i]->Center();
      MFEM_VERIFY(center.Size() == sdim,
                  "Point dim " << center.Size() <<
                  " does not match space dim " << sdim)
      centers.SetCol(i,center);
   }
   mesh->FindPoints(centers,receivers_eid,receivers_ip,false);
}

void TDLeastSquares::Eval(const Vector& state, const Vector& m, double time, double *f)
{
   InitReceivers();

   u->Distribute(state);
   double lval = 0.0;
   for (int i=0; i<receivers.Size(); i++)
   {
      int eid = receivers_eid[i];
      if (eid < 0) continue;
      const IntegrationPoint& ip = receivers_ip[i];

      Vector E,D;
      u->GetVectorValue(eid,ip,E);
      D.SetSize(E.Size());
      receivers[i]->GetIData(time,D);
      lval += F_ls(E,D);
   }
   ParFiniteElementSpace *fes = u->ParFESpace();
   lval *= scale;
   MPI_Allreduce(&lval,f,1,MPI_DOUBLE_PRECISION,MPI_SUM,fes->GetParMesh()->GetComm());
}

void TDLeastSquares::InitDeltaCoefficients()
{
   InitReceivers();
   if (deltacoeffs_x.Size() == receivers_eid.Size()) return;
   ResetDeltaCoefficients();

   ParFiniteElementSpace *fes = u->ParFESpace();
   rhsform_x  = new ParLinearForm(fes);
   for (int i=0; i<receivers.Size(); i++)
   {
      if (!fes->GetParMesh()->GetNE()) { deltacoeffs_x.Append(NULL); continue; } //XXX empty meshes
      int eid = receivers_eid[i];
      const FiniteElement *FElem = fes->GetFE(eid < 0 ? 0 : eid);
      int vdim = (FElem->GetRangeType() == FiniteElement::SCALAR) ? fes->GetVDim() : fes->GetParMesh()->SpaceDimension();
      Vector V(vdim);
      V = 0.0;
      VectorDeltaCoefficient *vd = new VectorDeltaCoefficient(V);
      DeltaCoefficient &d = vd->GetDeltaCoefficient();
      d.SetDeltaCenter(receivers[i]->Center());
      if (FElem->GetRangeType() == FiniteElement::SCALAR)
      {
         rhsform_x->AddDomainIntegrator(new VectorDomainLFIntegrator(*vd));
      }
      else
      {
         rhsform_x->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*vd));
      }
      deltacoeffs_x.Append(vd);
   }
}

void TDLeastSquares::EvalGradient_X(const Vector& state, const Vector& m, double time, mfem::Vector& g)
{
   InitDeltaCoefficients();

   u->Distribute(state);
   for (int i=0; i<receivers.Size(); i++)
   {
      int eid = receivers_eid[i];
      if (eid < 0) continue;
      const IntegrationPoint& ip = receivers_ip[i];

      Vector E,D;
      u->GetVectorValue(eid,ip,E);
      D.SetSize(E.Size());
      receivers[i]->GetIData(time,D);
      dFdu_ls(E,D); // E = E - D
      deltacoeffs_x[i]->SetDirection(E);
      deltacoeffs_x[i]->SetScale(scale);
   }
   rhsform_x->Assemble();
   rhsform_x->ParallelAssemble(g);
}

void TDLeastSquares::ResetDeltaCoefficients()
{
   for (int i=0; i<deltacoeffs_x.Size(); i++)
   {
      delete deltacoeffs_x[i];
   }
   deltacoeffs_x.SetSize(0);
   delete rhsform_x;
   rhsform_x = NULL;
}

TDLeastSquares::~TDLeastSquares()
{
   delete u;
   if (own_recv)
   {
      for (int i=0; i<receivers.Size(); i++)
      {
         delete receivers[i];
      }
   }
   ResetDeltaCoefficients();
}

TDLeastSquaresHessian::TDLeastSquaresHessian(TDLeastSquares *ls)
{
   ParFiniteElementSpace *fes = ls->u->ParFESpace();
   height = width = fes->GetTrueVSize();
   (*this).ls = ls;
}

void TDLeastSquaresHessian::Mult(const Vector& x, Vector& y) const
{
   ls->InitDeltaCoefficients();

   ls->u->Distribute(x);
   for (int i=0; i<ls->receivers.Size(); i++)
   {
      int eid = ls->receivers_eid[i];
      if (eid < 0) continue;
      const IntegrationPoint& ip = ls->receivers_ip[i];

      Vector E;
      ls->u->GetVectorValue(eid,ip,E);
      ls->deltacoeffs_x[i]->SetDirection(E);
      ls->deltacoeffs_x[i]->SetScale(ls->scale);
   }
   ls->rhsform_x->Assemble();
   ls->rhsform_x->ParallelAssemble(y);
}

TVRegularizer::TVRegularizer(PDCoefficient* _m_pd, double _alpha, double _beta, bool _primal_dual) : ObjectiveFunction(false,true), beta(_beta), tvInteg(beta), wkgf(), P2D(NULL), ljacs(), primal_dual(_primal_dual), m_pd(_m_pd)
{
   SetScale(_alpha);
   if (!m_pd) return;
   Array<ParGridFunction*> &pgf = m_pd->GetDerivCoeffs();
   if (!pgf.Size()) return;
   const FiniteElement *el = pgf[0]->ParFESpace()->GetFE(0);
   const int dim = el->GetDim();
   const int ord = el->GetOrder();
   L2_FECollection *l2 = new L2_FECollection(ord, dim); /* XXX the correct space should have the same order I think */
   ParFiniteElementSpace *gfes = new ParFiniteElementSpace(pgf[0]->ParFESpace()->GetParMesh(), l2, dim);
   for (int i = 0; i < pgf.Size(); i++)
   {
      ParGridFunction *gf = new ParGridFunction(gfes);
      *gf = 0.0;
      if (!i) gf->MakeOwner(l2);
      wkgf.Append(gf);
      WQ.Append(new VectorGridFunctionCoefficient(gf));
      wkgf2.Append(new ParGridFunction(gfes));
   }

   //ParFiniteElementSpace *mfes = pgf[0]->ParFESpace();
   //ParMesh *mesh = mfes->GetParMesh();
   //Vector m(m_pd->GetLocalSize());
   //m = 1.0;
   //m_pd->SetUseDerivCoefficients(true);
   //m_pd->UpdateCoefficient(m);
   //integ_exclude.SetSize(mesh->GetNE());
   //for (int i = 0; i < mesh->GetNE(); i++)
   //{
   //   Array<int> dofs;
   //   mfes->GetElementVDofs(i,dofs);
   //   Vector vals;
   //   pgf[0]->GetSubVector(dofs,vals);
   //   bool lex = false;
   //   for (int v = 0; v < vals.Size(); v++) if (vals[v] == 0.0) lex = true;
   //   integ_exclude[i] = lex;
   //}
   //Array<bool>& cexcl = m_pd->GetActiveElements();
   //integ_exclude.SetSize(mesh->GetNE());
   //integ_exclude.Assign(cexcl);
   ParDiscreteLinearOperator* intOp = new ParDiscreteLinearOperator(pgf[0]->ParFESpace(),gfes);
   intOp->AddDomainInterpolator(new GradientInterpolator);
   intOp->Assemble();
   intOp->Finalize();
   HypreParMatrix *H = intOp->ParallelAssemble();
   delete intOp;

   PetscParMatrix *pH = new PetscParMatrix(H,Operator::PETSC_MATAIJ);
   delete H;

   Array<PetscInt> rows(pH->GetNumRows());
   PetscInt rst = pH->GetRowStart();
   for (int i = 0; i < rows.Size(); i++)
   {
      rows[i] = i + rst;
   }
   P2D = new PetscParMatrix(*pH,rows,m_pd->GetGlobalCols());
   delete pH;
}

void TVRegularizer::UpdateDual(const Vector& m)
{
   if (!m_pd) return;
   if (!wkgf.Size()) return;
   int n = P2D->Width();
   double *data = m.GetData();
   for (int i=0; i<wkgf.Size(); i++)
   {
      Vector pmi(data,n);
      P2D->Mult(pmi,*wkgf[i]);
      data += n;
      pmi.SetData(NULL); /* XXX clang static analysis */
   }
}

void TVRegularizer::UpdateDual(const Vector& m, const Vector& dm, double lambda)
{
   Vector mk,dmk,wk,dwk;
   DenseMatrix dM;
   LUFactors dMinv;

   Array<int> vdofs,dvdofs,ipiv;

   if (!primal_dual || !m_pd) return;
   Array<ParGridFunction*> &pgf  = m_pd->GetDerivCoeffs();
   Array<ParGridFunction*> &pgf2 = m_pd->GetGradCoeffs();
   if (!pgf.Size()) return;
   ParFiniteElementSpace *fes = pgf[0]->ParFESpace();
   ParMesh *mesh = fes->GetParMesh();
   for (int i = 0; i < pgf.Size(); i++)
   {
      ParFiniteElementSpace *pfes = pgf[i]->ParFESpace();
      MFEM_VERIFY(mesh == pfes->GetParMesh(),"Different meshes not supported");
      ParFiniteElementSpace *pfes2 = pgf2[i]->ParFESpace();
      MFEM_VERIFY(mesh == pfes2->GetParMesh(),"Different meshes not supported");
   }
   m_pd->UpdateCoefficientWithGF(m,pgf);
   m_pd->UpdateCoefficientWithGF(dm,pgf2);

   Array<bool>& integ_exclude = m_pd->GetExcludedElements();
   double llopt = std::numeric_limits<double>::max(), lopt;
   VectorMassIntegrator dminteg;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      if (integ_exclude[e]) continue;
      /* For multiple: SetDenominator */
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         /* Compute element mass matrix for dual dofs */
         GridFunction *dgf = WQ[pg]->GetGridFunction();
         FiniteElementSpace *dfes = dgf->FESpace();
         const FiniteElement *del = dfes->GetFE(e);
         ElementTransformation *dT = dfes->GetElementTransformation(e);
         dminteg.AssembleElementMatrix(*del,*dT,dM);

         /* factor element mass matrix */
         int ddof = dM.Width();
         ipiv.SetSize(ddof);
         dMinv.data = dM.GetData();
         dMinv.ipiv = ipiv.GetData();
         dMinv.Factor(ddof);

         /* compute rhs for dual update */
         ParFiniteElementSpace *fes = pgf[pg]->ParFESpace();
         fes->GetElementVDofs(e, vdofs);
         pgf[pg]->GetSubVector(vdofs, mk);
         pgf2[pg]->GetSubVector(vdofs, dmk);
         dmk *= -1.0; /* XXX PETSc gets back -du */
         const FiniteElement *el = fes->GetFE(e);
         ElementTransformation *T = fes->GetElementTransformation(e);

         tvInteg.SetDualCoefficient(WQ[pg]);
         tvInteg.AssembleElementDualUpdate(*el,*del,*T,
                                           mk,dmk,dwk,&lopt);
         llopt = std::min(lopt,llopt);

         /* solve for the update */
         dMinv.Solve(ddof,1,dwk.GetData());

         /* store update */
         GridFunction *dwgf = wkgf2[pg];
         dfes->GetElementVDofs(e, dvdofs);
         dwgf->SetSubVector(dvdofs, dwk);
      }
   }
   MPI_Allreduce(&llopt,&lopt,1,MPI_DOUBLE_PRECISION,MPI_MIN,mesh->GetComm());

   /* XXX TODO add customization for these? */
   bool uselambda = (tvInteg.symmetrize && tvInteg.project);
   //uselambda = true;
   //std::cout << "LOPT " << lopt << std::endl;
   lopt = uselambda ? lambda : 0.99*lopt;
   /* update dual dofs */
   for (int pg = 0; pg < pgf.Size(); pg++)
   {
      GridFunction *dgf = WQ[pg]->GetGridFunction();
      GridFunction *dwgf = wkgf2[pg];
      dgf->Add(lopt,*dwgf);
   }
}

TVRegularizer::~TVRegularizer()
{
   for (int i = 0; i < wkgf.Size(); i++)
   {
      delete wkgf[i];
      delete wkgf2[i];
      delete WQ[i];
   }
   for (int i = 0; i < ljacs.Size(); i++)
   {
      delete ljacs[i];
   }
   delete P2D;
}

void TVRegularizer::Eval(const Vector& state, const Vector& m, double time, double *f)
{
   Vector     mk;
   Array<int> vdofs;

   *f = 0.0;
   if (!m_pd) return;
   Array<ParGridFunction*> pgf;
   pgf = m_pd->GetDerivCoeffs();
   if (!pgf.Size()) return;

   ParFiniteElementSpace *fes = pgf[0]->ParFESpace();
   ParMesh *mesh = fes->GetParMesh();
   for (int i = 0; i < pgf.Size(); i++)
   {
      ParFiniteElementSpace *pfes = pgf[i]->ParFESpace();
      MFEM_VERIFY(mesh == pfes->GetParMesh(),"Different meshes not supported");
   }

   m_pd->UpdateCoefficientWithGF(m,pgf);

   double lf = 0.0;
   Array<bool>& integ_exclude = m_pd->GetExcludedElements();
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      if (integ_exclude[e]) continue;
      /* For multiple: SetDenominator */
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         ParFiniteElementSpace *fes = pgf[pg]->ParFESpace();
         fes->GetElementVDofs(e, vdofs);
         pgf[pg]->GetSubVector(vdofs, mk);
         const FiniteElement *el = fes->GetFE(e);
         ElementTransformation *T = fes->GetElementTransformation(e);
         lf += scale * tvInteg.GetElementEnergy(*el,*T,mk);
      }
   }
   MPI_Allreduce(&lf,f,1,MPI_DOUBLE_PRECISION,MPI_SUM,mesh->GetComm());
}

void TVRegularizer::EvalGradient_M(const Vector& state, const Vector& m, double time, mfem::Vector& g)
{
   Vector     mk,elgrad;
   Array<int> vdofs;

   g = 0.0;
   if (!m_pd) return;
   Array<ParGridFunction*> &pgf = m_pd->GetDerivCoeffs();
   if (!pgf.Size()) return;

   ParFiniteElementSpace *fes = pgf[0]->ParFESpace();
   ParMesh *mesh = fes->GetParMesh();
   for (int i = 0; i < pgf.Size(); i++)
   {
      ParFiniteElementSpace *pfes = pgf[i]->ParFESpace();
      MFEM_VERIFY(mesh == pfes->GetParMesh(),"Different meshes not supported");
   }

   m_pd->UpdateCoefficientWithGF(m,pgf);

   Array<ParGridFunction*> &pgradgf = m_pd->GetGradCoeffs();
   for (int pg = 0; pg < pgf.Size(); pg++) *pgradgf[pg] = 0.0;

   Array<bool>& integ_exclude = m_pd->GetExcludedElements();
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      if (integ_exclude[e]) continue;
      /* For multiple: SetDenominator */
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         ParFiniteElementSpace *fes = pgf[pg]->ParFESpace();
         fes->GetElementVDofs(e, vdofs);
         pgf[pg]->GetSubVector(vdofs, mk);
         const FiniteElement *el = fes->GetFE(e);
         ElementTransformation *T = fes->GetElementTransformation(e);
         tvInteg.AssembleElementVector(*el,*T,mk,elgrad);
         elgrad *= scale;
         pgradgf[pg]->AddElementVector(vdofs,elgrad);
      }
   }
   m_pd->UpdateGradient(g);
}

void TVRegularizer::SetUpHessian_MM(const Vector& x,const Vector& m,double t)
{
   if (!m_pd) return;
   Array<ParGridFunction*> &pgf = m_pd->GetDerivCoeffs();
   if (!pgf.Size()) return;

   ParFiniteElementSpace *fes = pgf[0]->ParFESpace();
   ParMesh *mesh = fes->GetParMesh();
   for (int i = 0; i < pgf.Size(); i++)
   {
      ParFiniteElementSpace *pfes = pgf[i]->ParFESpace();
      MFEM_VERIFY(mesh == pfes->GetParMesh(),"Different meshes not supported");
   }

   m_pd->UpdateCoefficientWithGF(m,pgf);

   if (!ljacs.Size())
   {
      for (int i = 0; i < pgf.Size(); i++)
      {
         ljacs.Append(new SparseMatrix(pgf[i]->ParFESpace()->GetVSize()));
      }
   }
   else
   {
      for (int i = 0; i < pgf.Size(); i++)
      {
         *ljacs[i] = 0.0;
      }
   }

   int skz = 0;
   Array<bool>& integ_exclude = m_pd->GetExcludedElements();
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      if (integ_exclude[e]) continue;
      /* For multiple: SetDenominator */
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         DenseMatrix elmat;
         Vector      mk;
         Array<int>  vdofs;

         ParFiniteElementSpace *fes = pgf[pg]->ParFESpace();
         fes->GetElementVDofs(e, vdofs);
         pgf[pg]->GetSubVector(vdofs, mk);
         const FiniteElement *el = fes->GetFE(e);
         ElementTransformation *T = fes->GetElementTransformation(e);
         if (primal_dual) tvInteg.SetDualCoefficient(WQ[pg]);
         else tvInteg.SetDualCoefficient(NULL);
         tvInteg.AssembleElementGrad(*el,*T,mk,elmat);
         elmat *= scale;
         ljacs[pg]->AddSubMatrix(vdofs,vdofs,elmat,skz);
      }
   }
   for (int i = 0; i < pgf.Size(); i++)
   {
      ljacs[i]->Finalize(skz);
   }

   PetscParMatrix *uH = new PetscParMatrix(mesh->GetComm(),fes->GlobalVSize(),
                                           (PetscInt*)fes->GetDofOffsets(),ljacs[0], /* XXX */
                                           Operator::PETSC_MATAIJ);

   PetscParMatrix *H = RAP(uH,m_pd->GetP());
   if (!H_MM) H_MM = H;
   else
   {
      PetscParMatrix *pH_MM = dynamic_cast<mfem::PetscParMatrix *>(H_MM);
      MFEM_VERIFY(pH_MM,"Unsupported operator type");

      Mat B;
      B = H->ReleaseMat(false);

      PetscErrorCode ierr;
      ierr = MatHeaderReplace(*pH_MM,&B); CCHKERRQ(mesh->GetComm(),ierr);
      delete H;
   }
   delete uH;
}

PetscErrorCode mfemopt_eval_tdobj(Vec U,Vec M,PetscReal t,PetscReal* f,void* ctx)
{
   mfemopt::ObjectiveFunction *obj = (mfemopt::ObjectiveFunction*)ctx;
   double lf;

   PetscFunctionBeginUser;
   mfem::PetscParVector u(U,true);
   mfem::PetscParVector m(M,true);
   obj->Eval(u,m,t,&lf);
   *f = lf;
   PetscFunctionReturn(0);
}

PetscErrorCode mfemopt_eval_tdobj_x(Vec U,Vec M,PetscReal t,Vec G,void* ctx)
{
   PetscErrorCode ierr;
   mfemopt::ObjectiveFunction *obj = (mfemopt::ObjectiveFunction*)ctx;

   PetscFunctionBeginUser;
   mfem::PetscParVector u(U,true);
   mfem::PetscParVector m(M,true);
   mfem::PetscParVector g(G,true);
   obj->EvalGradient_X(u,m,t,g);
   ierr = PetscObjectStateIncrease((PetscObject)G);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

PetscErrorCode mfemopt_eval_tdobj_xx(Vec U,Vec M,PetscReal t,Mat A,void* ctx)
{
   PetscErrorCode ierr;
   mfemopt::ObjectiveFunction *obj = (mfemopt::ObjectiveFunction*)ctx;

   PetscFunctionBeginUser;
   mfem::PetscParVector u(U,true);
   mfem::PetscParVector m(M,true);
   obj->SetUpHessian_XX(u,m,t);
   if (A) { ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr); }
   PetscFunctionReturn(0);
}

}

PetscErrorCode ObjComputeHessianMFFD_Private(void *ctx, Vec x, Vec y)
{
   __obj_fdhessian_ctx *octx = (__obj_fdhessian_ctx*)ctx;
   mfemopt::ObjectiveFunction *obj = octx->obj;
   mfem::Vector *xIn = octx->xIn;
   mfem::Vector *mIn = octx->mIn;
   double t = octx->t;
   int which = octx->which;

   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   mfem::PetscParVector X(x,true);
   mfem::PetscParVector Y(y,true);
   if (which == 0)
   {
      if (obj->HasEvalGradient_X())
      {
         obj->EvalGradient_X(X,*mIn,t,Y);
      }
      else
      {
         Y = 0.0;
      }
   }
   else
   {
      if (obj->HasEvalGradient_M())
      {
         obj->EvalGradient_M(*xIn,X,t,Y);
      }
      else
      {
         Y = 0.0;
      }
   }
   ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

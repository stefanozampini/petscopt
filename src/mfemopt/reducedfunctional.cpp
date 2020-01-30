#include <mfemopt/reducedfunctional.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <mfem/linalg/petsc.hpp>
#include <petscopt/petscmat_cl.h>
#include <limits>

static PetscErrorCode ComputeHessianMFFD_Private(void*,Vec,Vec);

typedef struct
{
   mfemopt::ReducedFunctional *obj;
} __fdhessian_ctx;

namespace mfemopt
{

using namespace mfem;

void ReducedFunctional::ComputeGuess(mfem::Vector& m) const
{
   Vector l,u;

   (*this).GetBounds(l,u);
   m.SetSize(Height());
   MFEM_VERIFY(m.Size() <= l.Size(),"Wrong sizes: m" << m.Size() << ", l " << l.Size());
   MFEM_VERIFY(m.Size() <= u.Size(),"Wrong sizes: m" << m.Size() << ", u " << u.Size());
   for (int i = 0; i < m.Size(); i++)
   {
      m[i] = m[i] < l[i] ? l[i] : m[i];
      m[i] = m[i] > u[i] ? u[i] : m[i];
   }
}

void ReducedFunctional::GetBounds(mfem::Vector& l, mfem::Vector& u) const
{
   l.SetSize(Height());
   u.SetSize(Height());
   l = std::numeric_limits<double>::min();
   u = std::numeric_limits<double>::max();
}

void ReducedFunctional::TestFDGradient(MPI_Comm comm, const mfem::Vector& mIn, double delta, bool progress)
{
   PetscErrorCode ierr;

   PetscBool verbose = PETSC_FALSE;
   ierr = PetscOptionsGetBool(NULL,NULL,"-fd_gradient_verbose",&verbose,NULL); CCHKERRQ(comm,ierr);

   PetscReal h = delta;

   PetscParVector pm(comm,mIn,true);

   PetscMPIInt rank;
   ierr = MPI_Comm_rank(comm,&rank);CCHKERRQ(comm,ierr);

   PetscParVector g(comm,mIn);
   (*this).ComputeGradient(mIn,g);

   PetscParVector m(pm);
   PetscParVector fdg(comm,mIn);
   fdg = 0.0;
   ierr = PetscPrintf(comm,"ReducedFunctional::TestFDGradient, delta: %g\n",(double)h);CCHKERRQ(comm,ierr);
   ierr = PetscPrintf(comm,"-> hang tight while computing state gradient");CCHKERRQ(comm,ierr);

   for (PetscInt i = 0; i < g.GlobalSize(); i++)
   {
      double f1=0.0,f2=0.0;
      Array<PetscInt> idx(1);
      Array<PetscScalar> vals(1);

      if (progress)
      {
         ierr = PetscPrintf(comm,"\r-> hang tight while computing state gradient : %f%%",(i*100.0)/g.GlobalSize());CCHKERRQ(comm,ierr);
      }
      m = pm;

      idx[0] = i;

      vals[0] = !rank ? -h : 0.0;
      m.AddValues(idx,vals);
      (*this).ComputeObjective(m,&f1);
      vals[0] = !rank ? 2*h : 0.0;
      m.AddValues(idx,vals);
      (*this).ComputeObjective(m,&f2);

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

void ReducedFunctional::TestFDHessian(MPI_Comm comm, const Vector& mIn)
{
   PetscErrorCode ierr;
   PetscBool verbose = PETSC_FALSE;
   ierr = PetscOptionsGetBool(NULL,NULL,"-fd_hessian_verbose",&verbose,NULL); CCHKERRQ(comm,ierr);

   Operator &H = GetHessian(mIn);
   ierr = PetscPrintf(comm,"ReducedFunctional::TestFDHessian\n");CCHKERRQ(comm,ierr);
   ReducedFunctionalHessianOperatorFD fdH(comm,this,mIn);
   PetscParMatrix *pfdH = new PetscParMatrix(comm,&fdH,Operator::PETSC_MATAIJ);
   PetscParMatrix *pH = new PetscParMatrix(comm,&H,Operator::PETSC_MATAIJ);
   if (verbose)
   {
      ierr = PetscPrintf(comm,"FINITE DIFFERENCE\n");CCHKERRQ(comm,ierr);
      pfdH->Print();
      ierr = PetscPrintf(comm,"COMPUTED\n");CCHKERRQ(comm,ierr);
      pH->Print();
   }
   PetscReal nrm,nrmd,nrminf;
   PetscParMatrix *diff = new PetscParMatrix();
   *diff = *pH;
   *diff -= *pfdH;
   if (verbose)
   {
      ierr = PetscPrintf(comm,"DIFFERENCE\n");CCHKERRQ(comm,ierr);
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

void ReducedFunctional::TestTaylor(MPI_Comm comm, const Vector& mIn, bool testhess)
{
   /* Try to make tests reproducible on a given machine */
   PetscErrorCode ierr;
   PetscInt s = 22011982;
   ierr = PetscOptionsGetInt(NULL,NULL,"-taylor_seed",&s,NULL); CCHKERRQ(comm,ierr);

   Vector dmIn(mIn.Size());
   dmIn.Randomize(s);

   (*this).TestTaylor(comm,mIn,dmIn,testhess);
}

void ReducedFunctional::TestTaylor(MPI_Comm comm, const Vector& mIn, const Vector& dmIn, bool testhess)
{
   PetscErrorCode ierr;

   PetscReal h;
   PetscInt n;
   ierr = PetscOptionsGetReal(NULL,NULL,"-taylor_h",(h=1./8.,&h),NULL); CCHKERRQ(comm,ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-taylor_steps",(n=4,&n),NULL); CCHKERRQ(comm,ierr);

   double *tG = new double[n];
   double *tH = new double[n];

   double obj;
   Vector g(mIn.Size());
   (*this).ComputeObjectiveAndGradient(mIn,&obj,g);

   double v1,v2 = 0.0;
   v1 = InnerProduct(comm,g,dmIn);

   if (testhess)
   {
      Vector m2(mIn.Size());
      Operator &H = GetHessian(mIn);
      H.Mult(dmIn,m2);
      v2 = InnerProduct(comm,m2,dmIn);
   }

   Vector m(mIn.Size());
   for (PetscInt i = 0; i < n; i++, h /= 2.)
   {
      m = dmIn;
      m *= h;
      m += mIn;

      double hobj;
      (*this).ComputeObjective(m,&hobj);

      tG[i] = std::abs(hobj-obj-h*v1);

      tH[i] = 0.0;
      if (testhess)
      {
         tH[i] = std::abs(hobj-obj-h*v1-h*h*v2/2.);
      }
   }

   ierr = PetscPrintf(comm,"-------------------------- ReducedFunctional::TestTaylor ---------------------------\n"); CCHKERRQ(comm,ierr);
   ierr = PetscPrintf(comm,"\t\tGradient"); CCHKERRQ(comm,ierr);
   if (testhess) {
     ierr = PetscPrintf(comm,"\t\t\t\tHessian"); CCHKERRQ(comm,ierr);
   } else {
     ierr = PetscPrintf(comm,"\t\t\tHessian not tested"); CCHKERRQ(comm,ierr);
   }
   ierr = PetscPrintf(comm,"\n--------------------------------------------------------------------------\n"); CCHKERRQ(comm,ierr);
   for (PetscInt i = 0; i < n; i++) {
     PetscReal rate;

     rate = (i > 0 && tG[i] != tG[i-1]) ? -PetscLogReal(tG[i]/tG[i-1])/PetscLogReal(2.0) : 0.0;
     ierr = PetscPrintf(comm,"%-#8g\t%-#8g\t%D",tG[i],rate,(PetscInt)PetscMin(PetscRoundReal(rate),2)); CCHKERRQ(comm,ierr);
     rate = (i > 0 && tH[i] != tH[i-1]) ? -PetscLogReal(tH[i]/tH[i-1])/PetscLogReal(2.0) : 0.0;
     ierr = PetscPrintf(comm,"\t%-#8g\t%-#8g\t%D",tH[i],rate,(PetscInt)PetscMin(PetscRoundReal(rate),3)); CCHKERRQ(comm,ierr);
     ierr = PetscPrintf(comm,"\n"); CCHKERRQ(comm,ierr);
   }
   delete [] tG;
   delete [] tH;
}

ReducedFunctionalHessianOperatorFD::ReducedFunctionalHessianOperatorFD(MPI_Comm _comm, ReducedFunctional *_obj, const Vector& _mIn) : PetscParMatrix()
{
   obj = _obj;
   height = width = _mIn.Size();

   PetscErrorCode ierr;

   __fdhessian_ctx *mffd;
   ierr = PetscNew(&mffd);CCHKERRQ(_comm,ierr);
   mffd->obj = obj;

   Mat H;
   ierr = MatCreate(_comm,&H);CCHKERRQ(_comm,ierr);
   ierr = MatSetSizes(H,height,width,PETSC_DECIDE,PETSC_DECIDE);CCHKERRQ(_comm,ierr);
   ierr = MatSetType(H,MATMFFD);CCHKERRQ(_comm,ierr);
   ierr = MatSetUp(H);CCHKERRQ(_comm,ierr);

   PetscParVector X(_comm,_mIn,true);
   ierr = MatMFFDSetBase(H,X,NULL);CCHKERRQ(_comm,ierr);
   ierr = MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))ComputeHessianMFFD_Private,mffd);CCHKERRQ(_comm,ierr);
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

}

#include <petsc/private/petscimpl.h>

PetscErrorCode ComputeHessianMFFD_Private(void *ctx, Vec x, Vec y)
{
   __fdhessian_ctx *octx = (__fdhessian_ctx*)ctx;
   mfemopt::ReducedFunctional *obj = octx->obj;

   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   mfem::PetscParVector X(x,true);
   mfem::PetscParVector Y(y,true);
   obj->ComputeGradient(X,Y);
   ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

#include <mfemopt/pdoperator.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <mfemopt/private/utils.hpp>
#include <mfem/linalg/petsc.hpp>
#include <petsc/private/petscimpl.h>
#include <petscopt/petscmat_cl.h>

static PetscErrorCode PDOperatorGradientMFFD_Private(void*,Vec,Vec);

namespace mfemopt
{

using namespace mfem;

typedef struct
{
   PDOperator *pd;
   Vector *xpIn;
   Vector *xIn;
   double t;
} __pdop_fdgradient_ctx;


PDOperatorGradientFD::PDOperatorGradientFD(MPI_Comm _comm, PDOperator *_pd, const Vector& _xpIn, const Vector& _xIn, const Vector& _mIn, double _t) : PetscParMatrix()
{
   pd = _pd;
   xpIn = _xpIn;
   xIn = _xIn;
   t = _t;

   height = xIn.Size();
   width = _mIn.Size();

   PetscErrorCode ierr;

   __pdop_fdgradient_ctx *mffd;
   ierr = PetscNew(&mffd);CCHKERRQ(_comm,ierr);
   mffd->pd = pd;
   mffd->xpIn = &xpIn;
   mffd->xIn = &xIn;
   mffd->t = _t;

   Mat H;
   ierr = MatCreate(_comm,&H);CCHKERRQ(_comm,ierr);
   ierr = MatSetSizes(H,height,width,PETSC_DECIDE,PETSC_DECIDE);CCHKERRQ(_comm,ierr);
   ierr = MatSetType(H,MATMFFD);CCHKERRQ(_comm,ierr);
   ierr = MatSetUp(H);CCHKERRQ(_comm,ierr);
   PetscParVector X(_comm,_mIn,true);
   ierr = MatMFFDSetBase(H,X,NULL);CCHKERRQ(_comm,ierr);
   ierr = MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))PDOperatorGradientMFFD_Private,mffd);CCHKERRQ(_comm,ierr);
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

PDOperator::PDOperator()
{
   GOp = NULL;
   bc  = NULL;
   HOp[0][0] = NULL;
   HOp[0][1] = NULL;
   HOp[0][2] = NULL;
   HOp[1][0] = NULL;
   HOp[1][1] = NULL;
   HOp[1][2] = NULL;
   HOp[2][0] = NULL;
   HOp[2][1] = NULL;
   HOp[2][2] = NULL;
}

PDOperator::~PDOperator()
{
   delete GOp;
   delete HOp[0][0];
   delete HOp[0][1];
   delete HOp[0][2];
   delete HOp[1][0];
   delete HOp[1][1];
   delete HOp[1][2];
   delete HOp[2][0];
   delete HOp[2][1];
   delete HOp[2][2];
}

void PDOperator::SetBCHandler(PetscBCHandler* _bc)
{
   /* Model bchandler */
   bc = _bc;

   /* bchandler for homogenous bc application */
   Array<int> empty(0);
   hbc.SetTDofs(bc ? bc->GetTDofs() : empty);
}

void PDOperator::ApplyHomogeneousBC(Vector& x)
{
   hbc.ApplyBC(x);
}

void PDOperator::ComputeInitialConditions(double t0,Vector& x0,const Vector& m)
{
   x0.SetSize((*this).GetStateSize());
   x0 = 0.0;
}

PDOperatorGradient* PDOperator::GetGradientOperator()
{
   return GOp ? GOp : GOp = new PDOperatorGradient(this);
}

PDOperatorHessian* PDOperator::GetHessianOperator(int A, int B)
{
   return HOp[A][B] ? HOp[A][B] : HOp[A][B] = new PDOperatorHessian(this,A,B);
}

void PDOperator::TestFDGradient(MPI_Comm comm, const Vector& xpIn, const Vector& xIn, const Vector& mIn, double t)
{
   PetscErrorCode ierr;
   PetscBool verbose = PETSC_FALSE;
   ierr = PetscOptionsGetBool(NULL,NULL,"-fd_gradient_verbose",&verbose,NULL); CCHKERRQ(comm,ierr);

   PetscMPIInt size;
   ierr = MPI_Comm_size(comm,&size);CCHKERRQ(comm,ierr);

   PetscParVector xdot(comm,xpIn,true);
   PetscParVector x(comm,xIn,true);
   PetscParVector m(comm,mIn,true);

   ierr = PetscPrintf(comm,"PDOperator::TestFDGradient wrt design\n");CCHKERRQ(comm,ierr);

   Mat GExpl,GT,GTExpl,check;
   PetscReal errfd, normfd,normG,normGT,err;

   PDOperatorGradientFD fdG(comm,this,xpIn,xIn,mIn,t);
   PetscParMatrix *pfdG = new PetscParMatrix(comm,&fdG,Operator::PETSC_MATAIJ);
   pfdG->EliminateRows(hbc.GetTDofs()); /* XXX */
   {
      Mat T = *pfdG;
      ierr = PetscObjectSetName((PetscObject)T,"G_fd");CCHKERRQ(comm,ierr);
   }
   ierr = MatViewFromOptions(*pfdG,NULL,"-test_pdoperatorgradient_Gfd_view");CCHKERRQ(comm,ierr);
   ierr = MatNorm(*pfdG,NORM_INFINITY,&normfd);CCHKERRQ(comm,ierr);

   PDOperatorGradient *G = (*this).GetGradientOperator();
   PetscParMatrix *pG = new PetscParMatrix(comm,G,Operator::PETSC_MATSHELL);
   G->Update(xdot,x,m);

   ierr = MatComputeOperator(*pG,MATAIJ,&GExpl);CCHKERRQ(comm,ierr);
   ierr = MatConvert(GExpl,size > 1 ? MATMPIAIJ : MATSEQAIJ,MAT_INPLACE_MATRIX,&GExpl);CCHKERRQ(comm,ierr);
   ierr = MatNorm(GExpl,NORM_INFINITY,&normG);CCHKERRQ(comm,ierr);
   ierr = PetscObjectSetName((PetscObject)GExpl,"G");CCHKERRQ(comm,ierr);
   ierr = MatViewFromOptions(GExpl,NULL,"-test_pdoperatorgradient_G_view");CCHKERRQ(comm,ierr);

   ierr = MatCreateTranspose(*pG,&GT);CCHKERRQ(comm,ierr);
   ierr = MatComputeOperator(GT,MATAIJ,&GTExpl);CCHKERRQ(comm,ierr);
   ierr = MatConvert(GTExpl,size > 1 ? MATMPIAIJ : MATSEQAIJ,MAT_INPLACE_MATRIX,&GTExpl);CCHKERRQ(comm,ierr);
   ierr = MatNorm(GTExpl,NORM_INFINITY,&normGT);CCHKERRQ(comm,ierr);

   ierr = MatTranspose(GTExpl,MAT_INITIAL_MATRIX,&check);CCHKERRQ(comm,ierr);
   { /* XXX */
      PetscParMatrix TT(check,true);
      TT.EliminateRows(hbc.GetTDofs());
   }
   ierr = MatDestroy(&GTExpl);CCHKERRQ(comm,ierr);
   ierr = MatTranspose(check,MAT_INITIAL_MATRIX,&GTExpl);CCHKERRQ(comm,ierr);
   ierr = PetscObjectSetName((PetscObject)GTExpl,"GT");CCHKERRQ(comm,ierr);
   ierr = MatViewFromOptions(GTExpl,NULL,"-test_pdoperatorgradient_GT_view");CCHKERRQ(comm,ierr);

   ierr = PetscObjectSetName((PetscObject)check,"(G^T)^T");CCHKERRQ(comm,ierr);
   ierr = MatViewFromOptions(check,NULL,"-test_pdoperatorgradient_GTT_view");CCHKERRQ(comm,ierr);
   ierr = MatAXPY(check,-1.0,GExpl,DIFFERENT_NONZERO_PATTERN);CCHKERRQ(comm,ierr);
   ierr = MatScale(check,1./normG);CCHKERRQ(comm,ierr);
   ierr = PetscObjectSetName((PetscObject)check,"||G - (G^T)^T||/||G||");CCHKERRQ(comm,ierr);
   ierr = MatViewFromOptions(check,NULL,"-test_pdoperatorgradient_check_view");CCHKERRQ(comm,ierr);
   ierr = MatNorm(check,NORM_INFINITY,&err);CCHKERRQ(comm,ierr);

   ierr = MatAXPY(GExpl,-1.0,*pfdG,DIFFERENT_NONZERO_PATTERN);CCHKERRQ(comm,ierr);
   ierr = MatNorm(GExpl,NORM_INFINITY,&errfd);CCHKERRQ(comm,ierr);
   ierr = PetscPrintf(comm,"||G|| = %g, ||G^T|| = %g, ||G - (G^T)^T||/||G|| = %g, ||G_fd|| = %g, ||G_fd - G|| = %g\n",(double)normG,(double)normGT,(double)err,(double)normfd,(double)errfd);CCHKERRQ(comm,ierr);
   ierr = MatViewFromOptions(GExpl,NULL,"-test_pdoperatorgradient_diff_view");CCHKERRQ(comm,ierr);
   ierr = MatDestroy(&check);CCHKERRQ(comm,ierr);
   ierr = MatDestroy(&GExpl);CCHKERRQ(comm,ierr);
   ierr = MatDestroy(&GTExpl);CCHKERRQ(comm,ierr);
   ierr = MatDestroy(&GT);CCHKERRQ(comm,ierr);
   delete pG;
   delete pfdG;
}

PDOperatorGradient::PDOperatorGradient(PDOperator* _op)
{
   height = _op->GetStateSize();
   width = _op->GetParameterSize();
   op = _op;
}

void PDOperatorGradient::Mult(const Vector& im,Vector& o) const
{
   /* im as input, M = F_m(tdst,st,m) -> output  o = M * im */
   op->ComputeGradient(tdst,st,m,im,o);
   op->ApplyHomogeneousBC(o);
}

void PDOperatorGradient::MultTranspose(const Vector& l,Vector& o) const
{
   /* l as input, M = F_m(tdst,st,m) -> output  o = l^t * M */
   op->ComputeGradientAdjoint(l,tdst,st,m,o);
}

void PDOperatorGradient::Update(const Vector& itdst,const Vector& ist, const Vector &im)
{
   tdst.SetDataAndSize(itdst.GetData(),itdst.Size());
   st.SetDataAndSize(ist.GetData(),ist.Size());
   m.SetDataAndSize(im.GetData(),im.Size());
}

PDOperatorHessian::PDOperatorHessian(PDOperator* _op, int _A, int _B)
{
   height = _A < 2 ? _op->GetStateSize() : _op->GetParameterSize();
   width = _A < 2 ? _op->GetParameterSize() : _op->GetStateSize();
   op = _op;
   A = _A;
   B = _B;
}

void PDOperatorHessian::Mult(const Vector& x,Vector& y) const
{
   op->ComputeHessian(A,B,tdst,st,m,l,x,y);
   if (A < 2)
   {
      op->ApplyHomogeneousBC(y);
   }
}

void PDOperatorHessian::Update(const Vector& itdst,const Vector& ist, const Vector &im, const Vector &il)
{
   tdst.SetDataAndSize(itdst.GetData(),itdst.Size());
   st.SetDataAndSize(ist.GetData(),ist.Size());
   m.SetDataAndSize(im.GetData(),im.Size());
   l.SetDataAndSize(il.GetData(),il.Size());
}

}

PetscErrorCode mfemopt_setupts(TS ts,Vec X0,Vec M,void* ctx)
{
   PetscErrorCode ierr;
   PetscReal      t0;

   PetscFunctionBeginUser;
   mfemopt::PDOperator *pdop = mfemopt::mi_void_safe_cast<mfemopt::PDOperator,mfem::TimeDependentOperator>(ctx);
   if (!pdop) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing PDOperator");
   mfem::PetscParVector x0(X0,true);
   mfem::PetscParVector m(M,true);
   ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
   pdop->SetUpFromParameters(m);
   pdop->ComputeInitialConditions(t0,x0,m);
   ierr = PetscObjectStateIncrease((PetscObject)X0);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

PetscErrorCode mfemopt_gradientdae(TS ts,PetscReal t,Vec X,Vec Xdot,Vec M,Mat A,void* ctx)
{
   PetscErrorCode ierr;
   mfem::Operator *op;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,&op);CHKERRQ(ierr);
   mfemopt::PDOperatorGradient *gop = mfemopt::mi_void_safe_cast<mfemopt::PDOperatorGradient,mfem::Operator>(op);
   if (!gop) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"Missing PDOperatorGradient");
   mfem::PetscParVector x(X,true);
   mfem::PetscParVector xdot(Xdot,true);
   mfem::PetscParVector m(M,true);
   gop->Update(xdot,x,m);
   PetscFunctionReturn(0);
}

/* f_xtm  : Y = (Ldot^T \otimes I_N)*F_UdotM*R */
PetscErrorCode mfemopt_hessiandae_xtm(TS ts,PetscReal t,Vec u,Vec u_t,Vec M,Vec Ldot,Vec R,Vec Y,void *ctx)
{
   PetscFunctionBeginUser;
   mfemopt::PDOperator *pdop = mfemopt::mi_void_safe_cast<mfemopt::PDOperator,mfem::TimeDependentOperator>(ctx);
   if (!pdop) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing PDOperator");
   mfemopt::PDOperatorHessian *hop = pdop->GetHessianOperator(1,2);
   if (!hop) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing PDOperatorHessian");
   mfem::PetscParVector x(u,true);
   mfem::PetscParVector xdot(u_t,true);
   mfem::PetscParVector m(M,true);
   mfem::PetscParVector ldot(Ldot,true);
   hop->Update(xdot,x,m,ldot);
   mfem::PetscParVector r(R,true);
   mfem::PetscParVector y(Y,true);
   hop->Mult(r,y);
   PetscFunctionReturn(0);
}

/* f_mxt  : Y = (L^T \otimes I_P)*F_MUdot*Xdot */
PetscErrorCode mfemopt_hessiandae_mxt(TS ts,PetscReal t,Vec u,Vec u_t,Vec M,Vec Ldot,Vec R,Vec Y,void *ctx)
{
   PetscFunctionBeginUser;
   mfemopt::PDOperator *pdop = mfemopt::mi_void_safe_cast<mfemopt::PDOperator,mfem::TimeDependentOperator>(ctx);
   if (!pdop) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing PDOperator");
   mfemopt::PDOperatorHessian *hop = pdop->GetHessianOperator(2,1);
   if (!hop) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing PDOperatorHessian");
   mfem::PetscParVector x(u,true);
   mfem::PetscParVector xdot(u_t,true);
   mfem::PetscParVector m(M,true);
   mfem::PetscParVector ldot(Ldot,true);
   hop->Update(xdot,x,m,ldot);
   mfem::PetscParVector r(R,true);
   mfem::PetscParVector y(Y,true);
   hop->Mult(r,y);
   PetscFunctionReturn(0);
}

PetscErrorCode PDOperatorGradientMFFD_Private(void *ctx, Vec x, Vec y)
{
   mfemopt::__pdop_fdgradient_ctx *octx = (mfemopt::__pdop_fdgradient_ctx*)ctx;
   mfemopt::PDOperator *pd = octx->pd;
   mfem::Vector *xpIn = octx->xpIn;
   mfem::Vector *xIn = octx->xIn;
   double t = octx->t;

   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   mfem::PetscParVector X(x,true);
   mfem::PetscParVector Y(y,true);
   pd->Mult(*xpIn,*xIn,X,t,Y);
   ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

#include <mfemopt/nonlininteg.hpp>
#include <cmath>
#include <limits>

namespace mfemopt
{

using namespace mfem;

const IntegrationRule* TVIntegrator::GetIntRule(const FiniteElement& el)
{
   if (IntRule) return IntRule;
   const int dim = el.GetDim();
   int order;
   if (el.Space() == FunctionSpace::Pk)
   {
      order = 2*el.GetOrder(); /* fine in 2D for order 1,2 , check in 3D */
   }
   else
   {
      order = 2*el.GetOrder() + dim - 1;
   }

   if (el.Space() == FunctionSpace::rQk)
   {
      IntRule = &RefinedIntRules.Get(el.GetGeomType(), order);
   }
   else
   {
      IntRule = &IntRules.Get(el.GetGeomType(), order);
   }
   return IntRule;
}

double TVIntegrator::GetElementEnergy(const FiniteElement& el,
                                      ElementTransformation& T,
                                      const Vector& mk)
{
   const int dim = el.GetDim();
   const int dof = el.GetDof();

   gradm.SetSize(dim);
   dshape.SetSize(dof, dim);
   const IntegrationRule *ir = GetIntRule(el);
   double en = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcPhysDShape(T,dshape);
      dshape.MultTranspose(mk,gradm);
      en += T.Weight()* ip.weight *(norm < 0.0 ? std::sqrt(InnerProduct(gradm,gradm) + beta) : norm);
   }
   return en;
}

void TVIntegrator::AssembleElementVector(const FiniteElement& el,
                                         ElementTransformation& T,
                                         const Vector& mk,Vector& elvect)
{
   const int dim = el.GetDim();
   const int dof = el.GetDof();

   gradm.SetSize(dim);
   dshape.SetSize(dof, dim);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = GetIntRule(el);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcPhysDShape(T,dshape);
      dshape.MultTranspose(mk,gradm);
      double s = T.Weight()*ip.weight/(norm < 0.0 ? std::sqrt(InnerProduct(gradm,gradm) + beta) : norm);
      dshape.AddMult_a(s,gradm,elvect);
   }
}

//static double nwmax = -1;

void TVIntegrator::AssembleElementGrad(const FiniteElement& el,
                                       ElementTransformation& T,
                                       const Vector& mk,DenseMatrix& elmat)
{
   const int dim = el.GetDim();
   const int dof = el.GetDof();

   tmat.SetSize(dim, dof);
   gradm.SetSize(dim);
   dshape.SetSize(dof, dim);
   elmat.SetSize(dof, dof);
   elmat = 0.0;

   const IntegrationRule *ir = GetIntRule(el);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcPhysDShape(T,dshape);
      dshape.MultTranspose(mk,gradm);

      double s = 1.0/(norm < 0.0 ? std::sqrt(InnerProduct(gradm,gradm) + beta) : norm);
      if (WQ) /* primal-dual approach */
      {
         double nw;
         WQ->Eval(wk, T, ip);
         nw = std::sqrt(InnerProduct(wk,wk));
         if (project)
         {
            wk /= std::max(1.0,nw);
         }
         //if (nw > 1.0 && nw > nwmax) { nwmax = nw; std::cout << "WARNING NWMAX " << nwmax << std::endl; }
         //wk *= s; /* XXX REMOVE (used to check correctness of the code)*/
      }
      else
      {
         wk = gradm;
         wk *= s;
      }


      A.Diag(s,dim);
      if (symmetrize)
      {
         AddMult_a_VWt(-s*s/2.0,wk,gradm,A);
         AddMult_a_VWt(-s*s/2.0,gradm,wk,A);
      }
      else
      {
         AddMult_a_VWt(-s*s,wk,gradm,A);
      }

#if 0
      if (symmetrize) {
        double ll[3] = {0., 0., 0.},vv[9];

        A.CalcEigenvalues(ll,vv);
        if (ll[0] < 0. || ll[1] < 0. || ll[2] < 0.)
        {
           std::cout << "EIGS: " << ll[0] << "," << ll[1] << ", " << ll[2] << std::endl;
           wk.Print();
        }
      }
#endif
      double w = ip.weight*T.Weight();
      A *= w;

      MultABt(A,dshape,tmat);
      AddMult(dshape,tmat,elmat);
   }
}

static inline double __mfemopt_sign(double x) {
#if 0
   /* a dummy (portable) implementation of sign */
   static const char testpz[16] = {'\0'};
   static const double testpn = +NAN;
   return x != x ? ( memcmp(&x,&testpn,sizeof(double)) ? -1.0 : +1.0) : ( x == -x ? ( memcmp(&x,testpz,sizeof(double)) ? -1.0 : 1.0) : (x < 0.0 ? -1.0 : +1.0) );
#else
   return std::signbit(x) ? -1.0 : 1.0;
#endif
}

static inline double larger_quadratic_roots(double a, double b, double c)
{
   //double temp = -0.5 * (b + std::copysign(1.0, b) * sqrt(b*b - 4*a*c));
   double temp = -0.5 * (b + __mfemopt_sign(b) * std::sqrt(b*b - 4*a*c));
   double x1 = temp / a;
   double x2 = c / temp;
   return x1 < x2 ? x2 : x1;
}

void TVIntegrator::AssembleElementDualUpdate(const FiniteElement& el,
                                             const FiniteElement& del,
                                             ElementTransformation& T,
                                             const Vector& mk,const Vector& dmk,
                                             Vector& dwk, double *lopt)
{
   MFEM_VERIFY(WQ,"missing coefficient for dual variables");

   const int dim = el.GetDim();
   const int dof = el.GetDof();
   const int ddof = del.GetDof();

   /* local optimal dual steplength */
   Vector lalpha(dim),lbeta(dim);
   double llopt;

   gradm.SetSize(dim);
   graddm.SetSize(dim);
   dshape.SetSize(dof, dim);
   dualshape.SetSize(ddof);
   dwk.SetSize(ddof*dim);
   dwk = 0.0;

   *lopt = std::numeric_limits<double>::max();
   const IntegrationRule *ir = GetIntRule(el);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcPhysDShape(T,dshape);
      del.CalcPhysShape(T,dualshape);

      dshape.MultTranspose(mk,gradm);
      dshape.MultTranspose(dmk,graddm);
      double s = 1./(norm < 0.0 ? std::sqrt(InnerProduct(gradm,gradm) + beta) : norm);

      WQ->Eval(wk, T, ip);
      double nw = 1.0;
      if (project) /* XXX Here to check againts Georg's */
      {
         nw = std::max(1.0,std::sqrt(InnerProduct(wk,wk)));
         wk /= nw;
      }

      A.Diag(s,dim);
      if (symmetrize && project)  /* XXX Here to check againts Georg's */
      {
         AddMult_a_VWt(-s*s/2.0,wk,gradm,A);
         AddMult_a_VWt(-s*s/2.0,gradm,wk,A);
      }
      else
      {
         AddMult_a_VWt(-s*s,wk,gradm,A);
      }
      gradm *= s;
      A.AddMult(graddm,gradm);
      double w = ip.weight*T.Weight();
      for (int d = 0; d < dim; d++)
      {
         Vector dwkd(dwk.GetData() + d*ddof,ddof);
         dwkd.Add(w*(gradm(d) - wk(d)*nw),dualshape);
         lalpha(d) = gradm(d) - wk(d)*nw;
         lbeta(d) = wk(d);
      }

      /* determine optimal step length for dual variables */
      double eA = InnerProduct(lalpha,lalpha);
      double eB = 2.*InnerProduct(lalpha,lbeta);
      double eC = InnerProduct(lbeta,lbeta)-1.0;

      llopt = larger_quadratic_roots(eA,eB,eC);
      *lopt = std::min(llopt,*lopt);
   }
}

}

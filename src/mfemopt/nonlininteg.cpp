#include <mfemopt/nonlininteg.hpp>
#include <cmath>
#include <limits>

namespace mfemopt
{

using namespace mfem;

VTVIntegrator::VTVIntegrator(double _beta, bool _ind)
{
   beta = _beta;
   indep = _ind;
   IntRule = NULL;
   symmetrize = false;
   project = false;
}

void VTVIntegrator::SetDualCoefficients(const Array<VectorGridFunctionCoefficient*>& _WQs)
{
   WQs.SetSize(_WQs.Size());
   WQs.Assign(_WQs);
}

void VTVIntegrator::SetDualCoefficients()
{
   WQs.SetSize(0);
}

VTVIntegrator::~VTVIntegrator()
{
   for (int j = 0; j < work.Size(); j++) delete work[j];
   for (int j = 0; j < dwork.Size(); j++) delete dwork[j];
   for (int j = 0; j < dshapes.Size(); j++) delete dshapes[j];
}

const IntegrationRule* VTVIntegrator::GetIntRule(const FiniteElement& el)
{
   if (IntRule) return IntRule;
   int order = 2*el.GetOrder();

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

double VTVIntegrator::GetElementEnergy(const Array<const FiniteElement*>& els,
                                       ElementTransformation& T,
                                       const Array<const Vector*>& mks)
{
   double en = 0.0;
   if (!els.Size()) return en;

   const int dim = els[0]->GetDim();
   const int nb = els.Size();
   norms.SetSize(nb);
   gradm.SetSize(dim);

   if (dshapes.Size() < nb)
   {
      for (int i = dshapes.Size(); i < nb; i++)
      {
         dshapes.Append(new DenseMatrix());
      }
   }

   for (int j = 0; j < nb; j++)
   {
      const int dof = els[j]->GetDof();
      dshapes[j]->SetSize(dof, dim);
   }

   const IntegrationRule *ir = GetIntRule(*(els[0]));
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      const double w = T.Weight() * ip.weight;

      for (int j = 0; j < nb; j++)
      {
         els[j]->CalcPhysDShape(T,*(dshapes[j]));
         dshapes[j]->MultTranspose(*(mks[j]),gradm);
         norms[j] = InnerProduct(gradm,gradm);
      }
      if (indep)
      {
         for (int j = 0; j < nb; j++)
         {
            en += w * std::sqrt(norms[j] + beta);
         }
      }
      else
      {
         double c = 0.0;
         for (int j = 0; j < nb; j++)
         {
            c += norms[j];
         }
         en += w * std::sqrt(c + beta);
      }
   }
   return en;
}

void VTVIntegrator::AssembleElementVector(const Array<const FiniteElement*>& els,
                                          ElementTransformation& T,
                                          const Array<const Vector*>& mks,
                                          const Array<Vector*>& elvects)
{
   if (!els.Size()) return;

   const int dim = els[0]->GetDim();
   const int nb = els.Size();
   norms.SetSize(nb);
   gradm.SetSize(dim);

   if (work.Size() < nb)
   {
      for (int i = work.Size(); i < nb; i++)
      {
         work.Append(new Vector());
      }
   }
   if (dshapes.Size() < nb)
   {
      for (int i = dshapes.Size(); i < nb; i++)
      {
         dshapes.Append(new DenseMatrix());
      }
   }
   for (int j = 0; j < nb; j++)
   {
      const int dof = els[j]->GetDof();
      elvects[j]->SetSize(dof);
      *elvects[j] = 0.0;
      work[j]->SetSize(dof);
      dshapes[j]->SetSize(dof, dim);
   }

   const IntegrationRule *ir = GetIntRule(*(els[0]));
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      double c = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      const double w = T.Weight() * ip.weight;

      for (int j = 0; j < nb; j++)
      {
         els[j]->CalcPhysDShape(T,*(dshapes[j]));
         dshapes[j]->MultTranspose(*(mks[j]),gradm);
         norms[j] = InnerProduct(gradm,gradm);
         if (indep)
         {
            const double s = w/std::sqrt(norms[j] + beta);
            dshapes[j]->AddMult_a(s,gradm,*(elvects[j]));
         }
         else
         {
            const double s = w;
            dshapes[j]->Mult(gradm,*(work[j]));
            *(work[j]) *= s;
            c += norms[j];
         }
      }
      if (!indep)
      {
         for (int j = 0; j < nb; j++)
         {
            *(work[j]) /= std::sqrt(c + beta);
            *(elvects[j]) += *(work[j]);
         }
      }
   }
}

void VTVIntegrator::AssembleElementGrad(const Array<const FiniteElement*>& els,
                                        ElementTransformation& T,
                                        const Array<const Vector*>& mks,
                                        const Array2D<DenseMatrix*>& elmats)
{
   if (!els.Size()) return;

   const int dim = els[0]->GetDim();
   wk.SetSize(dim);

   const int nb = els.Size();
   norms.SetSize(nb);

   if (dshapes.Size() < nb)
   {
      for (int i = dshapes.Size(); i < nb; i++)
      {
         dshapes.Append(new DenseMatrix());
      }
   }
   if (work.Size() < nb)
   {
      for (int i = work.Size(); i < nb; i++)
      {
         work.Append(new Vector());
      }
   }
   for (int j = 0; j < nb; j++)
   {
      const int jdof = els[j]->GetDof();
      for (int k = 0; k < nb; k++)
      {
         const int kdof = els[k]->GetDof();
         elmats(j,k)->SetSize(jdof, kdof);
         *(elmats(j,k)) = 0.0;
      }
      dshapes[j]->SetSize(jdof, dim);
      work[j]->SetSize(dim);
   }

   const bool pd = WQs.Size() ? true : false;
   const IntegrationRule *ir = GetIntRule(*(els[0]));
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      double c = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      const double w = ip.weight * T.Weight();

      for (int j = 0; j < nb; j++)
      {
         els[j]->CalcPhysDShape(T,*(dshapes[j]));
         dshapes[j]->MultTranspose(*(mks[j]),*(work[j]));
         norms[j] = InnerProduct(*(work[j]),*(work[j]));
         c += norms[j];
      }

      if (indep)
      {
         for (int j = 0; j < nb; j++)
         {
            const double s = 1.0/std::sqrt(norms[j] + beta);
            const int dof = els[j]->GetDof();

            if (pd) /* primal-dual approach */
            {
               WQs[j]->Eval(wj, T, ip);
               if (project)
               {
                  double nw = std::sqrt(InnerProduct(wj,wj));
                  wj /= std::max(1.0,nw);
               }
#if 0
               /* (used to check correctness of the code, if duals updated via P2D(m) )*/
               wj *= s;
#endif
            }
            else
            {
               wj = *(work[j]);
               wj *= s;
            }

            A.Diag(s,dim);
            if (symmetrize)
            {
               AddMult_a_VWt(-s*s/2.0,wj,*(work[j]),A);
               AddMult_a_VWt(-s*s/2.0,*(work[j]),wj,A);
            }
            else
            {
               AddMult_a_VWt(-s*s,wj,*(work[j]),A);
            }

            A *= w;

            tmat.SetSize(dim,dof);
            MultABt(A,*(dshapes[j]),tmat);
            AddMult(*(dshapes[j]),tmat,*(elmats(j,j)));
         }
      }
      else
      {
         const double s = 1.0/std::sqrt(c + beta);

         for (int j = 0; j < nb; j++)
         {
            if (pd)
            {
               WQs[j]->Eval(wj, T, ip);
               if (project)
               {
                  double nw = std::sqrt(InnerProduct(wj,wj));
                  wj /= std::max(1.0,nw);
               }
#if 0
               /* (used to check correctness of the code, if duals updated via P2D(m) )*/
               wj *= s;
#endif
            }
            else
            {
               wj = *(work[j]);
               wj *= s;
            }

            for (int k = 0; k < nb; k++)
            {
               const int dof = els[k]->GetDof();
               A.Diag(j == k ? s : 0.0,dim);
               if (symmetrize)
               {
                  if (pd)
                  {
                     WQs[k]->Eval(wk, T, ip);
                     if (project)
                     {
                        double nw = std::sqrt(InnerProduct(wk,wk));
                        wk /= std::max(1.0,nw);
                     }
#if 0
                     /* (used to check correctness of the code, if duals updated via P2D(m) )*/
                     wk *= s;
#endif
                  }
                  else
                  {
                     wk = *(work[k]);
                     wk *= s;
                  }
                  AddMult_a_VWt(-s*s/2.0,wj,*(work[k]),A);
                  AddMult_a_VWt(-s*s/2.0,*(work[j]),wk,A);
               }
               else
               {
                  AddMult_a_VWt(-s*s,wj,*(work[k]),A);
               }

               A *= w;

               tmat.SetSize(dim,dof);
               MultABt(A,*(dshapes[k]),tmat);
               AddMult(*(dshapes[j]),tmat,*(elmats(j,k)));
            }
         }
      }
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

void VTVIntegrator::AssembleElementDualUpdate(const Array<const FiniteElement*>& els,
                                              const Array<const FiniteElement*>& dels,
                                              ElementTransformation& T,
                                              const Array<Vector*>& mks,
                                              const Array<Vector*>& dmks,
                                              Array<Vector*>& dwks,
                                              double *lopt)
{
   MFEM_VERIFY(WQs.Size(),"missing coefficient for dual variables");

   if (!els.Size()) return;

   const int dim = els[0]->GetDim();
   wk.SetSize(dim);
   wj.SetSize(dim);

   const int nb = els.Size();
   norms.SetSize(nb);
   gradm.SetSize(dim);

   /* local optimal dual steplength */
   Vector lalpha(dim),lbeta(dim);
   double llopt;

   Vector dwkd,dualshape;

   if (dshapes.Size() < nb)
   {
      for (int i = dshapes.Size(); i < nb; i++)
      {
         dshapes.Append(new DenseMatrix());
      }
   }
   if (work.Size() < nb)
   {
      for (int i = work.Size(); i < nb; i++)
      {
         work.Append(new Vector());
      }
   }
   if (dwork.Size() < nb)
   {
      for (int i = dwork.Size(); i < nb; i++)
      {
         dwork.Append(new Vector());
      }
   }
   for (int i = 0; i < nb; i++)
   {
      const int dof = els[i]->GetDof();
      const int ddof = dels[i]->GetDof();
      work[i]->SetSize(dim);
      dwork[i]->SetSize(dim);
      dshapes[i]->SetSize(dof,dim);
      dwks[i]->SetSize(ddof*dim);
      *(dwks[i]) = 0.0;
   }

   if (lopt) *lopt = std::numeric_limits<double>::max();

   const IntegrationRule *ir = GetIntRule(*els[0]);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      double c = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);

      const double w = ip.weight * T.Weight();

      for (int j = 0; j < nb; j++)
      {
         els[j]->CalcPhysDShape(T,*(dshapes[j]));
         dshapes[j]->MultTranspose(*(mks[j]),*(work[j]));
         dshapes[j]->MultTranspose(*(dmks[j]),*(dwork[j]));
         norms[j] = InnerProduct(*(work[j]),*(work[j]));
         c += norms[j];
      }

      if (indep)
      {
         for (int j = 0; j < nb; j++)
         {
            const double s = 1.0/std::sqrt(norms[j] + beta);
            const int ddof = dels[j]->GetDof();
            double nw = 1.0;

            gradm = *(work[j]);

            WQs[j]->Eval(wj, T, ip);
            if (project)
            {
               nw = std::max(1.0,std::sqrt(InnerProduct(wj,wj)));
               wj /= nw;
            }
            A.Diag(s,dim);

            if (symmetrize)
            {
               AddMult_a_VWt(-s*s/2.0,wj,gradm,A);
               AddMult_a_VWt(-s*s/2.0,gradm,wj,A);
            }
            else
            {
               AddMult_a_VWt(-s*s,wj,gradm,A);
            }
            gradm *= s;

            A.AddMult(*(dwork[j]),gradm);

            /* assemble element contribution to rhs update */
            dualshape.SetSize(ddof);
            dels[j]->CalcPhysShape(T,dualshape);
            for (int d = 0; d < dim; d++)
            {
               dwkd.SetDataAndSize(dwks[j]->GetData() + d*ddof,ddof);
               dwkd.Add(w*(gradm(d) - wj(d)*nw),dualshape);
               lalpha(d) = gradm(d) - wj(d)*nw;
               lbeta(d) = wj(d);
            }

            /* determine optimal step length for dual variables */
            if (lopt)
            {
               const double eA = InnerProduct(lalpha,lalpha);
               const double eB = 2.*InnerProduct(lalpha,lbeta);
               const double eC = InnerProduct(lbeta,lbeta)-1.0;

               llopt = larger_quadratic_roots(eA,eB,eC);
               *lopt = std::min(llopt,*lopt);
            }
         }
      }
      else
      {
         const double s = 1.0/std::sqrt(c + beta);

         for (int j = 0; j < nb; j++)
         {
            const int ddof = dels[j]->GetDof();
            double nw = 1.0;

            WQs[j]->Eval(wj, T, ip);
            if (project)
            {
               nw = std::max(1.0,std::sqrt(InnerProduct(wj,wj)));
               wj /= nw;
            }

            gradm = *(work[j]);
            gradm *= s;

            for (int k = 0; k < nb; k++)
            {
               A.Diag(j == k ? s : 0.0,dim);
               if (symmetrize)
               {
                  WQs[k]->Eval(wk, T, ip);
                  if (project)
                  {
                     double nw2 = std::sqrt(InnerProduct(wk,wk));
                     wk /= std::max(1.0,nw2);
                  }
                  AddMult_a_VWt(-s*s/2.0,wj,*(work[k]),A);
                  AddMult_a_VWt(-s*s/2.0,*(work[j]),wk,A);
               }
               else
               {
                  AddMult_a_VWt(-s*s,wj,*(work[k]),A);
               }

               A.AddMult(*(dwork[k]),gradm);
            }

            /* assemble element contribution to rhs update */
            dualshape.SetSize(ddof);
            dels[j]->CalcPhysShape(T,dualshape);
            for (int d = 0; d < dim; d++)
            {
               dwkd.SetDataAndSize(dwks[j]->GetData() + d*ddof,ddof);
               dwkd.Add(w*(gradm(d) - wj(d)*nw),dualshape);
               lalpha(d) = gradm(d) - wj(d)*nw;
               lbeta(d) = wj(d);
            }

            /* determine optimal step length for dual variables */
            if (lopt)
            {
               const double eA = InnerProduct(lalpha,lalpha);
               const double eB = 2.*InnerProduct(lalpha,lbeta);
               const double eC = InnerProduct(lbeta,lbeta)-1.0;

               llopt = larger_quadratic_roots(eA,eB,eC);
               *lopt = std::min(llopt,*lopt);
            }
         }
      }
   }
}

}

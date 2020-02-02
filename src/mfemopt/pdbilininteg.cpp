#include <mfemopt/pdbilininteg.hpp>
#include <mfemopt/pdcoefficient.hpp>
#include <mfem/general/socketstream.hpp>

namespace mfemopt
{
using namespace mfem;

PDBilinearFormIntegrator::~PDBilinearFormIntegrator()
{
   delete sworkgf;
}

void PDBilinearFormIntegrator::UpdateCoefficient(const Vector& m)
{
   if (!pdcoeff) return;
   pdcoeff->SetUseDerivCoefficients(false);
   pdcoeff->Distribute(m);
}

void PDBilinearFormIntegrator::GetCurrentVector(Vector& m)
{
   if (!pdcoeff) return;
   pdcoeff->SetUseDerivCoefficients(false);
   pdcoeff->GetCurrentVector(m);
}

void PDBilinearFormIntegrator::ComputeHessian_XM(ParGridFunction *agf, const Vector& m, const Vector& pertIn, Vector& out)
{
   ComputeGradient_Internal(agf,m,pertIn,out,true);
}

void PDBilinearFormIntegrator::ComputeHessian_MX(ParGridFunction *agf, ParGridFunction *sgf, const Vector& m, Vector& out)
{
   ComputeGradientAdjoint_Internal(agf,sgf,m,out,true);
}

void PDBilinearFormIntegrator::ComputeGradient(ParGridFunction *sgf, const Vector& m, const Vector& pertIn, Vector& out)
{
   ComputeGradient_Internal(sgf,m,pertIn,out);
}

void PDBilinearFormIntegrator::ComputeGradientAdjoint(ParGridFunction *agf, ParGridFunction *sgf, const Vector& m, Vector& out)
{
   ComputeGradientAdjoint_Internal(agf,sgf,m,out);
}

void PDBilinearFormIntegrator::ComputeGradient_Internal(ParGridFunction *sgf, const Vector& m, const Vector& pertIn ,Vector& out, bool hessian)
{
   Vector      ovals,svals,pvals;
   Array<int>  vdofs,pdofs;
   DenseMatrix elmat;

   /* get deriv_work_coeff */
   Array<ParGridFunction*>& pgf = pdcoeff->GetDerivCoeffs();
   if (!pgf.Size()) return;

   ParFiniteElementSpace *sfes = sgf->ParFESpace();
   ParMesh *pmesh = sfes->GetParMesh();
   for (int i = 0; i < pgf.Size(); i++)
   {
      ParFiniteElementSpace *pfes = pgf[i]->ParFESpace();
      MFEM_VERIFY(pmesh == pfes->GetParMesh(),"Different meshes not supported");
   }

   /* updates deriv_work_coeff */
   pdcoeff->SetUseDerivCoefficients();
   pdcoeff->Distribute(pertIn);
   pdcoeff->SetUseDerivCoefficients(false);

   /* XXX different adjoint space */
   if (!sworkgf) sworkgf = new ParGridFunction(sfes);
   *sworkgf = 0.0;

   Array<bool>& elactive = pdcoeff->GetActiveElements();
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      if (!elactive[e]) continue;

      sfes->GetElementVDofs(e, vdofs);
      sgf->GetSubVector(vdofs, svals);

      /* This could be done more efficiently if we require specialized routines
         for the element assembly */
      const FiniteElement *sel = sfes->GetFE(e);
      ElementTransformation *eltrans = sfes->GetElementTransformation(e);
      ParFiniteElementSpace *pfes = pdcoeff->pfes;
      pfes->GetElementVDofs(e, pdofs);
      pvals.SetSize(pdofs.Size()*pgf.Size());
      double *ptr = pvals.GetData();
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         Vector vals(ptr,pdofs.Size());
         pgf[pg]->GetSubVector(pdofs, vals);
         ptr += pdofs.Size();
         vals.SetData(NULL); /* XXX clang static analysis */
      }
      if (hessian)
      {
         ComputeElementHessian(sel,svals,eltrans,pvals,e,ovals);
      }
      else
      {
         ComputeElementGradient(sel,svals,eltrans,pvals,e,ovals);
      }
      ptr = ovals.GetData();
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         Vector vals(ptr,vdofs.Size());
         sworkgf->AddElementVector(vdofs,vals);
         ptr += vdofs.Size();
         vals.SetData(NULL); /* XXX clang static analysis */
      }
   }
   sworkgf->ParallelAssemble(out);
}

void PDBilinearFormIntegrator::ComputeGradientAdjoint_Internal(ParGridFunction *agf, ParGridFunction *sgf, const Vector& m, Vector& g, bool hessian)
{
   Array<int>  vdofs,pdofs;
   Vector      avals,svals,ovals;

   Array<ParGridFunction*>& pgf = pdcoeff->GetGradCoeffs();
   if (!pgf.Size()) return;

   ParFiniteElementSpace *sfes = sgf->ParFESpace();
   ParFiniteElementSpace *afes = agf->ParFESpace();
   ParMesh *pmesh = sfes->GetParMesh();
   MFEM_VERIFY(sfes == afes,"Different adjoint space not supported");
   MFEM_VERIFY(pmesh == afes->GetParMesh(),"Different meshes not supported");
   for (int i = 0; i < pgf.Size(); i++)
   {
      ParFiniteElementSpace *pfes = pgf[i]->ParFESpace();
      MFEM_VERIFY(pmesh == pfes->GetParMesh(),"Different meshes not supported");
   }
   for (int i = 0; i < pgf.Size(); i++)
   {
      *(pgf[i]) = 0.0;
   }

   Array<bool>& elactive = pdcoeff->GetActiveElements();
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      if (!elactive[e]) continue;
      sfes->GetElementVDofs(e, vdofs);
      sgf->GetSubVector(vdofs, svals);
      agf->GetSubVector(vdofs, avals);

      const FiniteElement *sel = sfes->GetFE(e);
      const FiniteElement *ael = sfes->GetFE(e);
      ElementTransformation *eltrans = sfes->GetElementTransformation(e);
      ComputeElementGradientAdjoint(sel,svals,ael,avals,eltrans,e,ovals);
      ParFiniteElementSpace *pfes = pdcoeff->pfes;
      pfes->GetElementVDofs(e, pdofs);
      double *ptr = ovals.GetData();
      const int n = pdofs.Size();
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         Vector vals(ptr,n);
         pgf[pg]->AddElementVector(pdofs,vals);
         ptr += n;
         vals.SetData(NULL); /* XXX clang static analysis */
      }
   }
   pdcoeff->Assemble(g);
}

void PDBilinearFormIntegrator::PushPDIntRule(const FiniteElement &el,
                                             ElementTransformation &Trans,
                                             int qorder)
{
   MFEM_VERIFY(!oIntRule,"You forgot to pop the PDIntRule")
   if (!IntRule)
   {
      IntRule = GetDefaultIntRule(el,Trans,0);
   }
   oIntRule = IntRule;
   IntRule = GetDefaultIntRule(el,Trans,qorder);
}

void PDBilinearFormIntegrator::PushPDIntRule(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
                                             ElementTransformation &Trans,
                                             int qorder)
{
   MFEM_VERIFY(!oIntRule,"You forgot to pop the PDIntRule")
   if (!IntRule)
   {
      IntRule = GetDefaultIntRule(trial_fe,test_fe,Trans,0);
   }
   oIntRule = IntRule;
   IntRule = GetDefaultIntRule(trial_fe,test_fe,Trans,qorder);
}

void PDBilinearFormIntegrator::PopPDIntRule()
{
   MFEM_VERIFY(oIntRule,"You forgot to push the PDIntRule")
   IntRule = oIntRule;
   oIntRule = NULL;
}

const IntegrationRule* PDMassIntegrator::GetDefaultIntRule(const FiniteElement &trial_fe,
                                                           const FiniteElement &test_fe,
                                                           ElementTransformation &Trans,
                                                           int qo)
{
   int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() + qo;

   const IntegrationRule *ir = NULL;
   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      ir = &RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   else if (test_fe.Space() == FunctionSpace::rQk)
   {
      ir = &RefinedIntRules.Get(test_fe.GetGeomType(), order);
   }
   else
   {
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }
   return ir;
}

void PDMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                             ElementTransformation &Trans,
                                             DenseMatrix &elmat)
{
   Coefficient *oQ = MassIntegrator::Q;
   MassIntegrator::Q = pdcoeff->GetActiveCoefficient();

   PushPDIntRule(el,Trans,pdcoeff->GetOrder());
   MassIntegrator::AssembleElementMatrix(el,Trans,elmat);
   PopPDIntRule();

   MassIntegrator::Q = oQ;
}

void PDMassIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              ElementTransformation &Trans,
                                              DenseMatrix &elmat)
{
   Coefficient *oQ = MassIntegrator::Q;
   MassIntegrator::Q = pdcoeff->GetActiveCoefficient();

   PushPDIntRule(trial_fe,test_fe,Trans,pdcoeff->GetOrder());
   MassIntegrator::AssembleElementMatrix2(trial_fe,test_fe,Trans,elmat);
   PopPDIntRule();

   MassIntegrator::Q = oQ;
}

void PDMassIntegrator::ComputeElementGradientAdjoint(const FiniteElement *sel, const Vector& svals,
                                                     const FiniteElement *ael, const Vector& avals,
                                                     ElementTransformation *eltrans, int e,
                                                     Vector& ovals)
{
   MFEM_VERIFY(pdcoeff,"Missing PDCoefficient");
   const int ngf = pdcoeff->GetComponents();
   MFEM_VERIFY(ngf == 1,"Invalid PDCoefficient components " << ngf);

   const int nsd = sel->GetDof();
   const int nad = ael->GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   shape.SetSize(nsd);
   te_shape.SetSize(nad);

   ParFiniteElementSpace *pfes = pdcoeff->GetPFES();
#ifdef MFEM_THREAD_SAFE
   Vector pshape;
#endif
   const FiniteElement *pel = pfes->GetFE(e);
   const int npd = pel->GetDof();
   pshape.SetSize(npd);

   ovals.SetSize(npd);
   ovals = 0.0;

   const IntegrationRule *ir = GetDefaultIntRule(*sel,*ael,*eltrans,pdcoeff->GetOrder());

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      eltrans->SetIntPoint(&ip);
      sel->CalcShape(ip, shape);
      if (ael != sel) ael->CalcShape(ip, te_shape);
      pel->CalcShape(ip, pshape);

      const double w = eltrans->Weight() * ip.weight;
      const double L = (ael != sel) ? avals*te_shape : avals*shape;
      const double R = shape*svals;
      for (int k = 0; k < npd; k++)
      {
         ovals(k) += L*w*pshape(k)*R;
      }
   }
}

void PDMassIntegrator::ComputeElementGradient_Internal(const FiniteElement *sel, const Vector& svals,
                                                       ElementTransformation *eltrans, const Vector& pvals, int e,
                                                       Vector& ovals, bool hessian)
{
   MFEM_VERIFY(pdcoeff,"Missing PDCoefficient");
   const int ngf = pdcoeff->GetComponents();
   MFEM_VERIFY(ngf == 1,"Invalid PDCoefficient components " << ngf);

   const int nsd = sel->GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector shape;
#endif
   shape.SetSize(nsd);

#ifdef MFEM_THREAD_SAFE
   Vector pshape;
#endif
   ParFiniteElementSpace *pfes = pdcoeff->GetPFES();
   const FiniteElement *pel = pfes->GetFE(e);
   const int npd = pel->GetDof();
   pshape.SetSize(npd);

   ovals.SetSize(nsd);
   ovals = 0.0;

   const IntegrationRule *ir = GetDefaultIntRule(*sel,*sel,*eltrans,pdcoeff->GetOrder());

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      eltrans->SetIntPoint(&ip);
      sel->CalcShape(ip, shape);
      pel->CalcShape(ip, pshape);

      const double w = eltrans->Weight() * ip.weight;
      const double R = shape*svals;
      for (int k = 0; k < npd; k++)
      {
         const double ww = w*pshape(k)*pvals(k)*R;
         ovals.Add(ww,shape);
      }
   }
}

const IntegrationRule* PDVectorFEMassIntegrator::GetDefaultIntRule(const FiniteElement &trial_fe,
                                                                   const FiniteElement &test_fe,
                                                                   ElementTransformation &Trans,
                                                                   int qo)
{
   int order = Trans.OrderW() + trial_fe.GetOrder() + test_fe.GetOrder() + qo;
   return &IntRules.Get(trial_fe.GetGeomType(), order);
}

void PDVectorFEMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                     ElementTransformation &Trans,
                                                     DenseMatrix &elmat)
{
   Coefficient       *oQ = VectorFEMassIntegrator::Q;
   VectorCoefficient *VQ = VectorFEMassIntegrator::VQ;
   MatrixCoefficient *MQ = VectorFEMassIntegrator::MQ;
   VectorFEMassIntegrator::Q = pdcoeff->GetActiveCoefficient();
   VectorFEMassIntegrator::VQ = NULL;
   VectorFEMassIntegrator::MQ = pdcoeff->GetActiveMatrixCoefficient();

   PushPDIntRule(el,Trans,pdcoeff->GetOrder());
   VectorFEMassIntegrator::AssembleElementMatrix(el,Trans,elmat);
   PopPDIntRule();

   VectorFEMassIntegrator::Q  = oQ;
   VectorFEMassIntegrator::VQ = VQ;
   VectorFEMassIntegrator::MQ = MQ;
}

void PDVectorFEMassIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                                      const FiniteElement &test_fe,
                                                      ElementTransformation &Trans,
                                                      DenseMatrix &elmat)
{
   Coefficient       *oQ = VectorFEMassIntegrator::Q;
   VectorCoefficient *VQ = VectorFEMassIntegrator::VQ;
   MatrixCoefficient *MQ = VectorFEMassIntegrator::MQ;
   VectorFEMassIntegrator::Q = pdcoeff->GetActiveCoefficient();
   VectorFEMassIntegrator::VQ = NULL;
   VectorFEMassIntegrator::MQ = pdcoeff->GetActiveMatrixCoefficient();

   PushPDIntRule(trial_fe,test_fe,Trans,pdcoeff->GetOrder());
   VectorFEMassIntegrator::AssembleElementMatrix2(trial_fe,test_fe,Trans,elmat);
   PopPDIntRule();

   VectorFEMassIntegrator::Q  = oQ;
   VectorFEMassIntegrator::VQ = VQ;
   VectorFEMassIntegrator::MQ = MQ;
}

void PDVectorFEMassIntegrator::ComputeElementGradientAdjoint(const FiniteElement *sel, const Vector& svals,
                                                             const FiniteElement *ael, const Vector& avals,
                                                             ElementTransformation *eltrans, int e,
                                                             Vector& ovals)
{
   MFEM_VERIFY(pdcoeff,"Missing PDCoefficient");
   const int ngf = pdcoeff->GetComponents();
   if (ngf != 1) // XXX not implemented
   {
      PDBilinearFormIntegrator::ComputeElementGradientAdjoint(sel,svals,ael,avals,eltrans,e,ovals);
      return;
   }
   if (sel->GetRangeType() != FiniteElement::VECTOR || ael->GetRangeType() != FiniteElement::VECTOR)
      mfem_error("PDVectorFEMassIntegrator::ComputeElementGradientAdjoint(...)\n"
                 "   is not implemented for non vector state and adjoint bases.");

   const int dim = sel->GetDim();
   const int nsd = sel->GetDof();
   const int nad = ael->GetDof();
#ifdef MFEM_THREAD_SAFE
   DenseMatrix svshape(nsd,dim);
   DenseMatrix avshape(nad,dim);
#else
   svshape.SetSize(nsd,dim);
   avshape.SetSize(nad,dim);
#endif
   Vector R(dim),L(dim);

   ParFiniteElementSpace *pfes = pdcoeff->GetPFES();
#ifdef MFEM_THREAD_SAFE
   Vector pshape;
#endif
   const FiniteElement *pel = pfes->GetFE(e);
   const int npd = pel->GetDof();
   pshape.SetSize(npd);

   ovals.SetSize(npd);
   ovals = 0.0;

   const IntegrationRule *ir = GetDefaultIntRule(*sel,*ael,*eltrans,pdcoeff->GetOrder());

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      eltrans->SetIntPoint(&ip);

      sel->CalcVShape(*eltrans, svshape);
      svshape.MultTranspose(svals,R);
      if (sel != ael)
      {
         ael->CalcVShape(*eltrans, avshape);
         avshape.MultTranspose(avals,L);
      }
      else
      {
         svshape.MultTranspose(avals,L);
      }
      const double w = eltrans->Weight() * ip.weight;

      // assuming scalar coefficients
      const double w2 = L*R;
      pel->CalcShape(ip, pshape);
      for (int k = 0; k < npd; k++)
      {
         ovals(k) += w*pshape(k)*w2;
      }
   }
}

void PDVectorFEMassIntegrator::ComputeElementGradient_Internal(const FiniteElement *sel, const Vector& svals,
                                                               ElementTransformation *eltrans, const Vector& pvals, int e,
                                                               Vector& ovals, bool hessian)
{
   MFEM_VERIFY(pdcoeff,"Missing PDCoefficient");
   const int ngf = pdcoeff->GetComponents();
   if (ngf != 1) // XXX not implemented
   {
      PDBilinearFormIntegrator::ComputeElementGradient_Internal(sel,svals,eltrans,pvals,e,ovals,hessian);
      return;
   }
   if (sel->GetRangeType() != FiniteElement::VECTOR)
      mfem_error("PDVectorFEMassIntegrator::ComputeElementGradient_Internal(...)\n"
                 "   is not implemented for non vector bases.");

   const int dim = sel->GetDim();
   const int nsd = sel->GetDof();
#ifdef MFEM_THREAD_SAFE
   DenseMatrix svshape(nsd,dim);
#else
   svshape.SetSize(nsd,dim);
#endif
   Vector R(dim),L(nsd);

   ParFiniteElementSpace *pfes = pdcoeff->GetPFES();
#ifdef MFEM_THREAD_SAFE
   Vector pshape;
#endif
   const FiniteElement *pel = pfes->GetFE(e);
   const int npd = pel->GetDof();
   pshape.SetSize(npd);

   ovals.SetSize(nsd);
   ovals = 0.0;

   const IntegrationRule *ir = GetDefaultIntRule(*sel,*sel,*eltrans,pdcoeff->GetOrder());

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      eltrans->SetIntPoint(&ip);

      sel->CalcVShape(*eltrans, svshape);
      svshape.MultTranspose(svals,R);
      const double w = eltrans->Weight() * ip.weight;

      // assuming scalar coefficients
      pel->CalcShape(ip, pshape);
      svshape.Mult(R,L);
      for (int k = 0; k < npd; k++)
      {
         const double ww = w*pshape(k)*pvals(k);
         ovals.Add(ww,L);
      }
   }
}



void PDBilinearFormIntegrator::ComputeElementGradientAdjoint(const FiniteElement *sel, const Vector& svals,
                                                             const FiniteElement *ael, const Vector& avals,
                                                             ElementTransformation *eltrans, int e,
                                                             Vector& ovals)
{
   pdcoeff->SetUseDerivCoefficients();
   ParFiniteElementSpace *pfes = pdcoeff->pfes;

   Array<int> pdofs;
   pfes->GetElementVDofs(e, pdofs);

   const int ngf = pdcoeff->GetComponents();
   ovals.SetSize(pdofs.Size()*ngf);

   DenseMatrix elmat;
   for (int g = 0; g < ngf; g++)
   {
      for (int k = 0; k < pdofs.Size(); k++)
      {
         pdcoeff->ElemDeriv(g,e,k,1.0);

         if (sel != ael)
         {
            AssembleElementMatrix2(*ael,*sel,*eltrans,elmat);
         }
         else
         {
            AssembleElementMatrix(*sel,*eltrans,elmat);
         }
         ovals[k + g*pdofs.Size()] = elmat.InnerProduct(avals,svals);
      }
   }
   pdcoeff->SetUseDerivCoefficients(false);
}

void PDBilinearFormIntegrator::ComputeElementGradient(const FiniteElement *sel, const Vector& svals,
                                                      ElementTransformation *eltrans, const Vector& pvals, int e,
                                                      Vector& ovals)
{
   ComputeElementGradient_Internal(sel,svals,eltrans,pvals,e,ovals,false);
}

void PDBilinearFormIntegrator::ComputeElementHessian(const FiniteElement *sel, const Vector& svals,
                                                     ElementTransformation *eltrans, const Vector& pvals, int e,
                                                     Vector& ovals)
{
   ComputeElementGradient_Internal(sel,svals,eltrans,pvals,e,ovals,true);
}

void PDBilinearFormIntegrator::ComputeElementGradient_Internal(const FiniteElement *sel, const Vector& svals,
                                                               ElementTransformation *eltrans, const Vector& pvals, int e,
                                                               Vector& ovals, bool hessian)
{
   ParFiniteElementSpace *pfes = pdcoeff->pfes;

   Array<int> pdofs;
   pfes->GetElementVDofs(e, pdofs);

   const int ngf = pdcoeff->GetComponents();
   const int nvd = svals.Size();
   const int npd = pdofs.Size();
   ovals.SetSize(nvd*ngf);
   ovals = 0.0;

   DenseMatrix elmat;
   double *ptr = ovals.GetData();
   pdcoeff->SetUseDerivCoefficients();
   for (int g = 0; g < ngf; g++)
   {
      Vector vals(ptr,nvd);

      for (int k = 0; k < npd; k++)
      {
         /* update deriv_coeff */
         pdcoeff->ElemDeriv(g,e,k,pvals[k + g*npd]);

         AssembleElementMatrix(*sel,*eltrans,elmat);
         if (!hessian) /* this is used in F_m */
         {
            elmat.AddMult(svals,vals);
         }
         else /* this is used in (L^T \otimes I_N)F_xm R */
         {
            elmat.AddMultTranspose(svals,vals);
         }
      }
      ptr += nvd;
      vals.SetData(NULL); /* XXX clang static analysis */
   }
   pdcoeff->SetUseDerivCoefficients(false);
}

}

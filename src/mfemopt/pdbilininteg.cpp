#include <mfemopt/pdbilininteg.hpp>
#include <mfemopt/pdcoefficient.hpp>
#include <mfem/general/socketstream.hpp>

namespace mfemopt
{
using namespace mfem;

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

void PDBilinearFormIntegrator::ComputeHessian_XM(ParGridFunction *agf, const Vector& pertIn, ParGridFunction *ogf)
{
   ComputeGradient_Internal(agf,pertIn,ogf,true);
}

void PDBilinearFormIntegrator::ComputeHessian_MX(ParGridFunction *agf, ParGridFunction *sgf, Vector& out)
{
   ComputeGradientAdjoint_Internal(agf,sgf,out,true);
}

void PDBilinearFormIntegrator::ComputeGradient(ParGridFunction *sgf, const Vector& pertIn, ParGridFunction *ogf)
{
   ComputeGradient_Internal(sgf,pertIn,ogf,false);
}

void PDBilinearFormIntegrator::ComputeGradientAdjoint(ParGridFunction *agf, ParGridFunction *sgf, Vector& out)
{
   ComputeGradientAdjoint_Internal(agf,sgf,out,false);
}

void PDBilinearFormIntegrator::ComputeGradient_Internal(ParGridFunction *sgf, const Vector& pertIn, ParGridFunction *ogf, bool hessian)
{
   Vector      ovals,svals,pvals;
   Array<int>  vdofs,pdofs;
   DenseMatrix elmat;

   *ogf = 0.0;

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

   Array<bool>& elactive = pdcoeff->GetActiveElements();
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      if (!elactive[e]) continue;

      sfes->GetElementVDofs(e, vdofs);
      sgf->GetSubVector(vdofs, svals);

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
         ComputeElementHessian(sel,svals,eltrans,pvals,ovals);
      }
      else
      {
         ComputeElementGradient(sel,svals,eltrans,pvals,ovals);
      }
      ptr = ovals.GetData();
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         Vector vals(ptr,vdofs.Size());
         ogf->AddElementVector(vdofs,vals);
         ptr += vdofs.Size();
         vals.SetData(NULL); /* XXX clang static analysis */
      }
   }
}

void PDBilinearFormIntegrator::ComputeGradientAdjoint_Internal(ParGridFunction *agf, ParGridFunction *sgf, Vector& g, bool hessian)
{
   Array<int>  svdofs,avdofs,pdofs;
   Vector      avals,svals,ovals;

   Array<ParGridFunction*>& pgf = pdcoeff->GetGradCoeffs();
   if (!pgf.Size()) return;

   ParFiniteElementSpace *sfes = sgf->ParFESpace();
   ParFiniteElementSpace *afes = agf->ParFESpace();
   ParMesh *pmesh = sfes->GetParMesh();
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
      sfes->GetElementVDofs(e, svdofs);
      sgf->GetSubVector(svdofs, svals);
      afes->GetElementVDofs(e, avdofs);
      agf->GetSubVector(avdofs, avals);

      const FiniteElement *sel = sfes->GetFE(e);
      const FiniteElement *ael = afes->GetFE(e);
      ElementTransformation *eltrans = sfes->GetElementTransformation(e);
      ComputeElementGradientAdjoint(sel,svals,ael,avals,eltrans,ovals);
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

const IntegrationRule* PDMassIntegrator::GetDefaultIntRule(const FiniteElement &trial_fe,
                                                           const FiniteElement &test_fe,
                                                           ElementTransformation &Trans,
                                                           int qo)
{
   //int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() + qo;
   int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();

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

   const IntegrationRule *oir = MassIntegrator::IntRule;
   MassIntegrator::IntRule = GetDefaultIntRule(el,Trans,pdcoeff->GetOrder());

   MassIntegrator::AssembleElementMatrix(el,Trans,elmat);

   MassIntegrator::IntRule = oir;
   MassIntegrator::Q = oQ;
}

void PDMassIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              ElementTransformation &Trans,
                                              DenseMatrix &elmat)
{
   Coefficient *oQ = MassIntegrator::Q;
   MassIntegrator::Q = pdcoeff->GetActiveCoefficient();

   const IntegrationRule *oir = MassIntegrator::IntRule;
   MassIntegrator::IntRule = GetDefaultIntRule(trial_fe,test_fe,Trans,pdcoeff->GetOrder());

   MassIntegrator::AssembleElementMatrix2(trial_fe,test_fe,Trans,elmat);

   MassIntegrator::IntRule = oir;
   MassIntegrator::Q = oQ;
}

void PDMassIntegrator::ComputeElementGradientAdjoint(const FiniteElement *sel, const Vector& svals,
                                                     const FiniteElement *ael, const Vector& avals,
                                                     ElementTransformation *eltrans,
                                                     Vector& ovals)
{
   MFEM_VERIFY(pdcoeff,"Missing PDCoefficient");
   MFEM_VERIFY(pdcoeff->Scalar(),"Need a scalar coefficient");

   const int nsd = sel->GetDof();
   const int nad = ael->GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   shape.SetSize(nsd);
   te_shape.SetSize(nad);

#ifdef MFEM_THREAD_SAFE
   Vector pshape;
#endif

   const IntegrationRule *ir = GetDefaultIntRule(*sel,*ael,*eltrans,pdcoeff->GetOrder());

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      eltrans->SetIntPoint(&ip);
      sel->CalcShape(ip, shape);
      if (ael != sel) ael->CalcShape(ip, te_shape);

      const double w = eltrans->Weight() * ip.weight;
      const double L = (ael != sel) ? avals*te_shape : avals*shape;
      const double R = shape*svals;
      const double f = w*L*R;
      if (!i)
      {
         pdcoeff->EvalDerivShape(*eltrans,ovals);
         ovals *= f;
      }
      else
      {
         pdcoeff->EvalDerivShape(*eltrans,pshape);
         ovals.Add(f,pshape);
      }
   }
}

void PDMassIntegrator::ComputeElementHessian(const FiniteElement *sel, const Vector& svals,
                                             ElementTransformation *eltrans, const Vector& pvals,
                                             Vector& ovals)
{
   ComputeElementGradient(sel,svals,eltrans,pvals,ovals);
}

void PDMassIntegrator::ComputeElementGradient(const FiniteElement *sel, const Vector& svals,
                                              ElementTransformation *eltrans, const Vector& pvals,
                                              Vector& ovals)
{
   MFEM_VERIFY(pdcoeff,"Missing PDCoefficient");
   MFEM_VERIFY(pdcoeff->Scalar(),"Need a scalar coefficient");

   const int nsd = sel->GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector shape;
#endif
   shape.SetSize(nsd);

   ovals.SetSize(nsd);
   ovals = 0.0;

   const IntegrationRule *ir = GetDefaultIntRule(*sel,*sel,*eltrans,pdcoeff->GetOrder());

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      eltrans->SetIntPoint(&ip);
      sel->CalcShape(ip, shape);
      double Q;
      pdcoeff->EvalDerivCoefficient(*eltrans,pvals,&Q);
      const double w = eltrans->Weight() * ip.weight;
      const double R = shape*svals;
      ovals.Add(Q*w*R,shape);
   }
}

const IntegrationRule* PDVectorFEMassIntegrator::GetDefaultIntRule(const FiniteElement &trial_fe,
                                                                   const FiniteElement &test_fe,
                                                                   ElementTransformation &Trans,
                                                                   int qo)
{
   //int order = Trans.OrderW() + trial_fe.GetOrder() + test_fe.GetOrder() + qo;
   int order = Trans.OrderW() + trial_fe.GetOrder() + test_fe.GetOrder();
   return &IntRules.Get(trial_fe.GetGeomType(), order);
}

void PDVectorFEMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                     ElementTransformation &Trans,
                                                     DenseMatrix &elmat)
{
   Coefficient       *oQ = VectorFEMassIntegrator::Q;
   VectorCoefficient *VQ = VectorFEMassIntegrator::DQ;
   MatrixCoefficient *MQ = VectorFEMassIntegrator::MQ;
   VectorFEMassIntegrator::Q = pdcoeff->GetActiveCoefficient();
   VectorFEMassIntegrator::DQ = NULL;
   VectorFEMassIntegrator::MQ = pdcoeff->GetActiveMatrixCoefficient();

   const IntegrationRule *oir = VectorFEMassIntegrator::IntRule;
   VectorFEMassIntegrator::IntRule = GetDefaultIntRule(el,Trans,pdcoeff->GetOrder());

   VectorFEMassIntegrator::AssembleElementMatrix(el,Trans,elmat);

   VectorFEMassIntegrator::IntRule = oir;

   VectorFEMassIntegrator::Q  = oQ;
   VectorFEMassIntegrator::DQ = VQ;
   VectorFEMassIntegrator::MQ = MQ;
}

void PDVectorFEMassIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                                      const FiniteElement &test_fe,
                                                      ElementTransformation &Trans,
                                                      DenseMatrix &elmat)
{
   Coefficient       *oQ = VectorFEMassIntegrator::Q;
   VectorCoefficient *VQ = VectorFEMassIntegrator::DQ;
   MatrixCoefficient *MQ = VectorFEMassIntegrator::MQ;
   VectorFEMassIntegrator::Q = pdcoeff->GetActiveCoefficient();
   VectorFEMassIntegrator::DQ = NULL;
   VectorFEMassIntegrator::MQ = pdcoeff->GetActiveMatrixCoefficient();

   const IntegrationRule *oir = VectorFEMassIntegrator::IntRule;
   VectorFEMassIntegrator::IntRule = GetDefaultIntRule(trial_fe,test_fe,Trans,pdcoeff->GetOrder());

   VectorFEMassIntegrator::AssembleElementMatrix2(trial_fe,test_fe,Trans,elmat);

   VectorFEMassIntegrator::IntRule = oir;

   VectorFEMassIntegrator::Q  = oQ;
   VectorFEMassIntegrator::DQ = VQ;
   VectorFEMassIntegrator::MQ = MQ;
}

void PDVectorFEMassIntegrator::ComputeElementGradientAdjoint(const FiniteElement *sel, const Vector& svals,
                                                             const FiniteElement *ael, const Vector& avals,
                                                             ElementTransformation *eltrans,
                                                             Vector& ovals)
{
   if (sel->GetRangeType() != FiniteElement::VECTOR || ael->GetRangeType() != FiniteElement::VECTOR)
      mfem_error("PDVectorFEMassIntegrator::ComputeElementGradientAdjoint(...)\n"
                 "   is not implemented for non vector state and adjoint bases.");
   MFEM_VERIFY(pdcoeff,"Missing PDCoefficient");

   const int dim = eltrans->GetSpaceDim();
   const int nsd = sel->GetDof();
   const int nad = ael->GetDof();
#ifdef MFEM_THREAD_SAFE
   DenseMatrix svshape(nsd,dim);
   DenseMatrix avshape(nad,dim);
#else
   svshape.SetSize(nsd,dim);
   avshape.SetSize(nad,dim);
#endif
   Vector R(dim),L(dim),Rt(dim);

   DenseTensor K;

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

      if (pdcoeff->Scalar())
      {
         const double w2 = w*(L*R);
         if (!i)
         {
            pdcoeff->EvalDerivShape(*eltrans,ovals);
            ovals *= w2;
         }
         else
         {
            pdcoeff->EvalDerivShape(*eltrans,pshape);
            ovals.Add(w2,pshape);
         }
      }
      else
      {
         pdcoeff->EvalDerivShape(*eltrans,K);
         const int npd = K.SizeK();
         if (!i)
         {
            ovals.SetSize(npd);
            ovals = 0.0;
         }
         for (int k = 0; k < npd; k++)
         {
            K(k).Mult(R,Rt);
            const double w2 = w*(L*Rt);

            ovals(k) += w2;
         }
      }
   }
}

void PDVectorFEMassIntegrator::ComputeElementGradient(const FiniteElement *sel, const Vector& svals,
                                                      ElementTransformation *eltrans, const Vector& pvals,
                                                      Vector& ovals)
{
   if (sel->GetRangeType() != FiniteElement::VECTOR)
      mfem_error("PDVectorFEMassIntegrator::ComputeElementGradient(...)\n"
                 "   is not implemented for non vector bases.");

   MFEM_VERIFY(pdcoeff,"Missing PDCoefficient");
   const int ngf = pdcoeff->GetComponents();
   const int dim = eltrans->GetSpaceDim();
   const int nsd = sel->GetDof();
#ifdef MFEM_THREAD_SAFE
   DenseMatrix svshape(nsd,dim);
#else
   svshape.SetSize(nsd,dim);
#endif
   Vector R(dim),L(nsd),Rt(dim);
   DenseTensor K(dim,dim,ngf);

   ovals.SetSize(nsd*ngf);
   ovals = 0.0;

   const IntegrationRule *ir = GetDefaultIntRule(*sel,*sel,*eltrans,pdcoeff->GetOrder());

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      eltrans->SetIntPoint(&ip);

      sel->CalcVShape(*eltrans, svshape);
      svshape.MultTranspose(svals,R);
      const double w = eltrans->Weight() * ip.weight;

      if (pdcoeff->Scalar())
      {
         double Q;
         pdcoeff->EvalDerivCoefficient(*eltrans,pvals,&Q);
         svshape.Mult(R,L);
         ovals.Add(Q*w,L);
      }
      else
      {
         double *ptr = ovals.GetData();
         pdcoeff->EvalDerivCoefficient(*eltrans,pvals,K);
         for (int k = 0; k < ngf; k++)
         {
            K(k).Mult(R,Rt);

            Vector oovals(ptr,nsd);
            svshape.Mult(Rt,L);
            oovals.Add(w,L);
            ptr += nsd;
            oovals.SetData(NULL);
         }
      }
   }
}

void PDVectorFEMassIntegrator::ComputeElementHessian(const FiniteElement *sel, const Vector& svals,
                                                     ElementTransformation *eltrans, const Vector& pvals,
                                                     Vector& ovals)
{
   ComputeElementGradient(sel,svals,eltrans,pvals,ovals);
}

/* default implementations */
void PDBilinearFormIntegrator::ComputeElementGradient(const FiniteElement *sel, const Vector& svals,
                                                      ElementTransformation *eltrans, const Vector& pvals,
                                                      Vector& ovals)
{
   ComputeElementGradient_Internal(sel,svals,eltrans,pvals,ovals,false);
}

void PDBilinearFormIntegrator::ComputeElementHessian(const FiniteElement *sel, const Vector& svals,
                                                     ElementTransformation *eltrans, const Vector& pvals,
                                                     Vector& ovals)
{
   ComputeElementGradient_Internal(sel,svals,eltrans,pvals,ovals,true);
}

void PDBilinearFormIntegrator::ComputeElementGradientAdjoint(const FiniteElement *sel, const Vector& svals,
                                                             const FiniteElement *ael, const Vector& avals,
                                                             ElementTransformation *eltrans,
                                                             Vector& ovals)
{
   ComputeElementGradientAdjoint_Internal(sel,svals,ael,avals,eltrans,ovals);
}

/* reference implementations (should work with all bilinear integrators) */
void PDBilinearFormIntegrator::ComputeElementGradientAdjoint_Internal(const FiniteElement *sel, const Vector& svals,
                                                                      const FiniteElement *ael, const Vector& avals,
                                                                      ElementTransformation *eltrans,
                                                                      Vector& ovals)
{
   pdcoeff->SetUseDerivCoefficients();
   ParFiniteElementSpace *pfes = pdcoeff->pfes;

   int e = eltrans->ElementNo;
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


void PDBilinearFormIntegrator::ComputeElementGradient_Internal(const FiniteElement *sel, const Vector& svals,
                                                               ElementTransformation *eltrans, const Vector& pvals,
                                                               Vector& ovals, bool hessian)
{
   ParFiniteElementSpace *pfes = pdcoeff->pfes;

   int e = eltrans->ElementNo;
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

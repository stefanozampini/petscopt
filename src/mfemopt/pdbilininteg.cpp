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

   /* updates deriv_work_coeff: this also flags the needed coefficient for the element assembly */
   pdcoeff->SetUseDerivCoefficients();
   pdcoeff->Distribute(pertIn);

   /* XXX different adjoint space */
   if (!sworkgf) sworkgf = new ParGridFunction(sfes);
   *sworkgf = 0.0;

   Array<bool>& elactive = pdcoeff->GetActiveElements();
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      if (!elactive[e]) continue;

      sfes->GetElementVDofs(e, vdofs);
      sgf->GetSubVector(vdofs, svals);
      ovals.SetSize(vdofs.Size());

      /* This could be done more efficiently if we require specialized routines
         for the element assembly */
      const FiniteElement *sel = sfes->GetFE(e);
      ElementTransformation *eltrans = sfes->GetElementTransformation(e);
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         ParFiniteElementSpace *pfes = pgf[pg]->ParFESpace();

         ovals = 0.0;

         pfes->GetElementVDofs(e, pdofs);
         pgf[pg]->GetSubVector(pdofs, pvals);

         for (int k = 0; k < pdofs.Size(); k++)
         {
            /* update deriv_coeff */
            pdcoeff->ElemDeriv(pg,e,k,pvals[k]);

            /* TODO Nonlinear forms (use AssembleElementVector, with slot in residual evaluation for parameter) */
            AssembleElementMatrix(*sel,*eltrans,elmat);
            if (!hessian) /* this is used in F_m */
            {
               elmat.AddMult(svals,ovals);
            }
            else /* this is used in (L \otimes I_N)F_xm R */
            {
               elmat.AddMultTranspose(svals,ovals);
            }
         }
         sworkgf->AddElementVector(vdofs,ovals);
      }
   }
   sworkgf->ParallelAssemble(out);
   pdcoeff->SetUseDerivCoefficients(false);
}

void PDBilinearFormIntegrator::ComputeGradientAdjoint_Internal(ParGridFunction *agf, ParGridFunction *sgf, const Vector& m, Vector& g, bool hessian)
{
   Array<int>  vdofs,pdofs;
   Vector      avals,svals,ovals;
   DenseMatrix elmat;

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

   /* needed by element assembly */
   pdcoeff->SetUseDerivCoefficients();

   Array<bool>& elactive = pdcoeff->GetActiveElements();
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      if (!elactive[e]) continue;
      sfes->GetElementVDofs(e, vdofs);
      sgf->GetSubVector(vdofs, svals);
      agf->GetSubVector(vdofs, avals);

      /* This could be done more efficiently if we require specialized routines
         for the element assembly */
      const FiniteElement *sel = sfes->GetFE(e);
      const FiniteElement *ael = sfes->GetFE(e);
      ElementTransformation *eltrans = sfes->GetElementTransformation(e);
      for (int pg = 0; pg < pgf.Size(); pg++)
      {
         ParFiniteElementSpace *pfes = pgf[pg]->ParFESpace();

         pfes->GetElementVDofs(e, pdofs);
         ovals.SetSize(pdofs.Size());
         for (int k = 0; k < pdofs.Size(); k++)
         {
            pdcoeff->ElemDeriv(pg,e,k,1.0);

            /* TODO Nonlinear forms */
            AssembleElementMatrix2(*ael,*sel,*eltrans,elmat);
            ovals[k] = elmat.InnerProduct(avals,svals);
         }
         pgf[pg]->AddElementVector(pdofs,ovals);
      }
   }
   /* disable usage of deriv_work_coeffs */
   pdcoeff->SetUseDerivCoefficients(false);
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

}

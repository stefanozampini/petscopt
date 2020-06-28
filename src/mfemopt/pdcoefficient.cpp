#include <mfemopt/pdcoefficient.hpp>
#include <mfemopt/mfemextra.hpp>
#include <mfem/fem/datacollection.hpp>
#include <mfem/linalg/petsc.hpp>
#include <mfem/fem/plinearform.hpp>
#include <mfem/fem/pbilinearform.hpp>
#include <petscmat.h>
#include <fstream>

namespace mfemopt
{
using namespace mfem;

void PDCoefficient::Reset()
{
   scalar = false;
   usederiv = false;
   delete l2gf;
   pcoeffiniti.SetSize(0);
   pcoeffinitv.SetSize(0);
   pcoeffexcl.SetSize(0);
   sforminteg.SetSize(0);
   piniti.SetSize(0);
   pinitv.SetSize(0);
   pactii.SetSize(0);
   pwork.SetSize(0);
   delete P;
   delete R;
   delete trueTransfer;
   delete s_coeff;
   delete m_coeff;
   delete deriv_s_coeff;
   delete deriv_m_coeff;
   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      delete pcoeffgf[i];
   }
   for (int i=0; i<pgradgf.Size(); i++)
   {
      delete pgradgf[i];
   }
   for (int i=0; i<deriv_coeffgf.Size(); i++)
   {
      delete deriv_coeffgf[i];
   }
   for (int i=0; i<deriv_work_coeffgf.Size(); i++)
   {
      delete deriv_work_coeffgf[i];
   }
   for (int i=0; i<pcoeffv0.Size(); i++)
   {
      delete pcoeffv0[i];
   }
   pcoeffgf.SetSize(0);
   pgradgf.SetSize(0);
   deriv_coeffgf.SetSize(0);
   deriv_work_coeffgf.SetSize(0);
   pcoeffv0.SetSize(0);
   local_cols.SetSize(0);
   global_cols.SetSize(0);
   delete pfes;
   lsize = 0;
   order = -1;
   ngf = 0;
   for (unsigned int i=0; i<souts.size(); i++)
   {
      delete souts[i];
   }
   souts.resize(0);
   delete cg;
   delete M;
   delete bM;
   delete mass;
}

void PDCoefficient::Init()
{
   lsize = 0;
   order = -1;
   ngf = -1;
   scalar = false;
   usederiv = false;
   incl_bdr = true;
   deriv_s_coeff = NULL;
   deriv_m_coeff = NULL;
   s_coeff = NULL;
   m_coeff = NULL;
   P = NULL;
   R = NULL;
   pfes = NULL;
   l2gf = NULL;
   trueTransfer = NULL;
   cg = NULL;
   M = NULL;
   bM = NULL;
   mass = NULL;
   mid = Operator::PETSC_MATAIJ;
}

PDCoefficient::PDCoefficient()
{
   Init();
}

PDCoefficient::PDCoefficient(Coefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
               const Array<bool>& excl, bool incl_bdr)
{
   Init(&Q,NULL,NULL,mesh,fec,excl,incl_bdr);
}

PDCoefficient::PDCoefficient(Coefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
                             bool (*excl_fn)(const Vector&), bool incl_bdr)
{
   Array<bool> excl;
   MeshGetElementsTagged(mesh,excl_fn,excl);
   Init(&Q,NULL,NULL,mesh,fec,excl,incl_bdr);
}

PDCoefficient::PDCoefficient(Coefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
               const Array<int>& excl_tag, bool incl_bdr)
{
   Array<bool> excl;
   MeshGetElementsTagged(mesh,excl_tag,excl);
   Init(&Q,NULL,NULL,mesh,fec,excl,incl_bdr);
}

PDCoefficient::PDCoefficient(VectorCoefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
               const Array<bool>& excl, bool incl_bdr)
{
   Init(NULL,&Q,NULL,mesh,fec,excl,incl_bdr);
}

PDCoefficient::PDCoefficient(VectorCoefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
                             bool (*excl_fn)(const Vector&), bool incl_bdr)
{
   Array<bool> excl;
   MeshGetElementsTagged(mesh,excl_fn,excl);
   Init(NULL,&Q,NULL,mesh,fec,excl,incl_bdr);
}

PDCoefficient::PDCoefficient(VectorCoefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
               const Array<int>& excl_tag,bool incl_bdr)
{
   Array<bool> excl;
   MeshGetElementsTagged(mesh,excl_tag,excl);
   Init(NULL,&Q,NULL,mesh,fec,excl,incl_bdr);
}

void PDCoefficient::Project(const Vector& x, Vector& y)
{
   /* populate with proper values at the boundary of optmization region */
   SetUseDerivCoefficients(false);
   Distribute(x,deriv_work_coeffgf);
   ProjectCoefficientInternal(NULL,NULL,NULL,deriv_work_coeffgf);
   GetCurrentVector(y,deriv_work_coeffgf);
}

void PDCoefficient::ProjectCoefficient(Coefficient& Q)
{
   ProjectCoefficientInternal(&Q,NULL,NULL,pcoeffgf);
}

void PDCoefficient::ProjectCoefficient(VectorCoefficient& Q)
{
   ProjectCoefficientInternal(NULL,&Q,NULL,pcoeffgf);
}

void PDCoefficient::ProjectCoefficientInternal(Coefficient* Q, VectorCoefficient* VQ, MatrixCoefficient *MQ, Array<ParGridFunction*> gf)
{
   if (MQ) { mfem_error("Not yet implemented"); }
   MFEM_VERIFY(ngf > 0,"NGF ERROR");
   MFEM_VERIFY(ngf == gf.Size(),"NGF ERROR");
   MFEM_VERIFY(pfes,"Missing pfes");

   int map_type;
   ParFiniteElementSpaceGetRangeAndDerivMapType(*pfes,&map_type,NULL);

   bool inout = false;
   BilinearFormIntegrator *mass_integ = NULL;
   Array<LinearFormIntegrator*> b_integ;
   Array<Coefficient*> cQ;
   if (map_type == FiniteElement::VALUE ||
       map_type == FiniteElement::INTEGRAL)
   {
      if (VQ)
      {
         MFEM_VERIFY(ngf == VQ->GetVDim(),"NGF ERROR");
         for (int i = 0; i < ngf; i++)
         {
            *gf[i] = 0.0;
            cQ.Append(new ComponentCoefficient(*VQ,i));
            b_integ.Append(new DomainLFIntegrator(*cQ[i]));
            gf[i]->ProjectDiscCoefficient(*cQ[i],GridFunction::ARITHMETIC);
         }
      }
      else if (Q)
      {
         *gf[0] = 0.0;
         gf[0]->ProjectDiscCoefficient(*Q,GridFunction::ARITHMETIC);
         b_integ.Append(new DomainLFIntegrator(*Q));
      }
      else /* use gf as coefficient (input/output) */
      {
         inout = true;
      }
      mass_integ = new MassIntegrator;
   }
   else if (map_type == FiniteElement::H_DIV ||
            map_type == FiniteElement::H_CURL)
   {
      if (Q)
      {
         MFEM_ABORT("Not supported scalar + vectorFE");
      }
      else if (VQ)
      {
         MFEM_VERIFY(ngf == 1,"NGF ERROR");
         b_integ.Append(new VectorFEDomainLFIntegrator(*VQ));
         *gf[0] = 0.0;
         gf[0]->ProjectDiscCoefficient(*VQ,GridFunction::ARITHMETIC);
      }
      else /* use gf as coefficient (input/output) */
      {
         MFEM_VERIFY(ngf == 1,"NGF ERROR");
         inout = true;
      }
      mass_integ = new VectorFEMassIntegrator;
   }
   else
   {
      MFEM_ABORT("unknown type of FE space");
   }

   Vector B(pfes->GetTrueVSize());
   Vector X(pfes->GetTrueVSize());
   if (!cg) { cg = new PetscPCGSolver(pfes->GetParMesh()->GetComm(),"coeff_l2_"); }
   if (!mass)
   {
      MFEM_VERIFY(!M,"This should not happen");
      OperatorHandle oM(Operator::PETSC_MATAIJ);
      mass = new ParBilinearForm(pfes);
      mass->AddDomainIntegrator(mass_integ);
      mass->Assemble(0);
      mass->Finalize(0);
      mass->ParallelAssemble(oM);
      oM.SetOperatorOwner(false);
      oM.Get(M);
      cg->SetOperator(*M);
   }
   else
   {
      MFEM_VERIFY(M,"This should not happen");
      cg->SetOperator(*M);
      delete mass_integ;
   }
   cg->iterative_mode = !inout;
   for (int i = 0; i < ngf; i++)
   {
      if (!inout)
      {
         ParLinearForm b(pfes);
         b.AddDomainIntegrator(b_integ[i]);
         b.Assemble();
         b.ParallelAssemble(B);
         gf[i]->ParallelProject(X);
      }
      else
      {
         gf[i]->ParallelProject(B);
      }
      cg->Mult(B, X);
      gf[i]->Distribute(X);
   }
   cg->iterative_mode = false;

   /* cleanup */
   for (int i = 0; i < cQ.Size(); i++) { delete cQ[i]; }
}

void PDCoefficient::Init(Coefficient *Q, VectorCoefficient *VQ, MatrixCoefficient *MQ, ParMesh *mesh, const FiniteElementCollection *fec, const Array<bool>& excl, bool include_bdr)
{
   Init();
   incl_bdr = include_bdr;
   const FiniteElement *fe = NULL;
   if (mesh->GetNE())
   {
      fe = fec->FiniteElementForGeometry(mesh->GetElementBaseGeometry(0));
   }
   /* Reduce on the comm */
   bool fescalar;
   {
      int loc[2] = {-1,-1};
      int glob[2];

      if (fe)
      {
         loc[0] = fe->GetOrder();
         loc[1] = fe->GetRangeType();
      }
      MPI_Allreduce(loc,glob,2,MPI_INT,MPI_MAX,mesh->GetComm());
      if (loc[0] != -1)
      {
         MFEM_VERIFY(loc[0] == glob[0],"Different order not supported " << loc[0] << " != " << glob[0]); /* XXX Not exhaustive check */
         MFEM_VERIFY(loc[1] == glob[1],"Different range not supported " << loc[1] << " != " << glob[1]); /* XXX Not exhaustive check */
      }
      order = glob[0];
      fescalar = (glob[1] == FiniteElement::SCALAR);
   }

   /* store values of projected initial coefficients
      and create actual coefficients to be used */
   ngf = 0;
   if (MQ) { mfem_error("Not yet implemented"); }
   else if (VQ)
   {
      /* XXX MASK */
      if (fescalar)
      {
         ngf = VQ->GetVDim();
      }
      else
      {
         MFEM_VERIFY(mesh->SpaceDimension() == VQ->GetVDim(),"Cannot represent a vector coefficient with VDIM " << VQ->GetVDim() << " using with space dimension " << mesh->SpaceDimension());
         ngf = 1;
      }
   }
   else if (Q)
   {
      MFEM_VERIFY(fescalar,"Cannot represent a scalar coefficient with a vector finite element space");
      ngf = 1;
   }
   else mfem_error("Unhandled case");

   pfes = new ParFiniteElementSpace(mesh,fec,1,Ordering::byVDIM);
   { /* only relevant for refinement operations */
      pfes->SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
   }

   /* store values of projected initial coefficients
      and create actual coefficients to be used */
   scalar = false;
   if (MQ) { mfem_error("Not yet implemented"); }
   else if (VQ)
   {
      if (fescalar)
      {
         for (int i = 0; i < ngf; i++)
         {
            pcoeffgf.Append(new ParGridFunction(pfes));
            deriv_coeffgf.Append(new ParGridFunction(pfes));
            deriv_work_coeffgf.Append(new ParGridFunction(pfes));
            pgradgf.Append(new ParGridFunction(pfes));
         }
         /* XXX MASK */
         MatrixArrayCoefficient *tmp_m_coeff = new MatrixArrayCoefficient(VQ->GetVDim());
         MatrixArrayCoefficient *tmp_deriv_m_coeff = new MatrixArrayCoefficient(VQ->GetVDim());
         for (int i = 0; i < ngf; i++)
         {
            tmp_m_coeff->Set(i,i,new GridFunctionCoefficient(pcoeffgf[i]));
            tmp_deriv_m_coeff->Set(i,i,new GridFunctionCoefficient(deriv_coeffgf[i]));
         }
         m_coeff = tmp_m_coeff;
         deriv_m_coeff = tmp_deriv_m_coeff;
      }
      else
      {
         /* XXX MASK ERROR */
         ngf = 1;
         pcoeffgf.Append(new ParGridFunction(pfes));
         deriv_coeffgf.Append(new ParGridFunction(pfes));
         deriv_work_coeffgf.Append(new ParGridFunction(pfes));
         pgradgf.Append(new ParGridFunction(pfes));

         m_coeff = new DiagonalMatrixCoefficient(new VectorGridFunctionCoefficient(pcoeffgf[0]),true);
         deriv_m_coeff = new DiagonalMatrixCoefficient(new VectorGridFunctionCoefficient(deriv_coeffgf[0]),true);
      }
   }
   else if (Q)
   {
      pcoeffgf.Append(new ParGridFunction(pfes));
      deriv_coeffgf.Append(new ParGridFunction(pfes));
      deriv_work_coeffgf.Append(new ParGridFunction(pfes));
      pgradgf.Append(new ParGridFunction(pfes));

      s_coeff = new GridFunctionCoefficient(pcoeffgf[0]);
      deriv_s_coeff = new GridFunctionCoefficient(deriv_coeffgf[0]);
      scalar = true;
   }

   /* perform real L2 projection */
   ProjectCoefficientInternal(Q,VQ,MQ,pcoeffgf);

   /* Store conforming initial values and distribute, because
      the code always uses P to compute the vdofs values
      When using a H1 space, in the case of element-wise jumps below

      0 - 1 - 2
      | v | w |
      3 - 4 - 5
      |   y   |
      6 ----- 7

      vdof #4 will have a value from project (v+w+y) and
      a value from P*true_dofs_vals ((y+v) + (y+w))/2
   */
   for (int i = 0; i < ngf; i++)
   {
      int tvsize = pcoeffgf[i]->ParFESpace()->GetTrueVSize();

      pcoeffv0.Append(new Vector(tvsize));
      pcoeffgf[i]->ParallelProject(*pcoeffv0[i]);
      pcoeffgf[i]->Distribute(*pcoeffv0[i]);
      *pgradgf[i] = 0.0;
      *deriv_coeffgf[i] = 0.0;
      *deriv_work_coeffgf[i] = 0.0;
   }

   /* process excluded elements */
   pcoeffexcl.SetSize(mesh->GetNE());
   pcoeffexcl = false;
   for (int i = 0; i < std::min(pcoeffexcl.Size(),excl.Size()); i++) pcoeffexcl[i] = excl[i];

   PetscBool lhas = PETSC_FALSE,has;
   for (int i = 0; i < pcoeffexcl.Size(); i++) if (pcoeffexcl[i]) { lhas = PETSC_TRUE; break; }
   MPI_Allreduce(&lhas,&has,1,MPIU_BOOL,MPI_LOR,mesh->GetComm());

   if (has)
   {
      /* L2 Space for cell markers to be used in Update() */
      FiniteElementCollection *l2fec = new L2_FECollection(0, mesh->Dimension());
      ParFiniteElementSpace   *l2fes = new ParFiniteElementSpace(mesh,l2fec);
      l2gf = new ParGridFunction(l2fes);
      l2gf->MakeOwner(l2fec);
   }

   SetUpOperators();

   FillExcl();
}

/*
   XXX This assumes pcoeffv0 and pcoeffgf are initialized with proper initial values
       when excluded regions are present
*/
void PDCoefficient::SetUpOperators()
{
   MFEM_VERIFY(pfes,"Missing ParFiniteElementSpace()!");
   delete bM;
   delete P;
   delete R;
   delete trueTransfer;
   trueTransfer = NULL;

   /* mask for elements that are active in the integration of forms for state variables */
   sforminteg.SetSize(pcoeffexcl.Size());

   ess_tdof_list.SetSize(0);
   ess_tdof_vals.SetSize(0);
   if (!l2gf) /* no excluded regions */
   {
      sforminteg = true;

      P = new PetscParMatrix(pfes->Dof_TrueDof_Matrix(),Operator::PETSC_MATAIJ);
      R = new PetscParMatrix(pfes->GetParMesh()->GetComm(),pfes->GetRestrictionMatrix(),Operator::PETSC_MATAIJ);
      PetscInt cst = P->GetColStart();
      local_cols.SetSize(P->Width());
      global_cols.SetSize(P->Width());
      for (int i = 0; i < P->Width(); i++)
      {
         local_cols[i] = i;
         global_cols[i] = i + cst;
      }
      pcoeffiniti.SetSize(0);
      pcoeffinitv.SetSize(0);
      piniti.SetSize(0);
      pinitv.SetSize(0);
      pactii.SetSize(P->Width());
      for (int i = 0; i < pactii.Size(); i++) pactii[i] = i;
   }
   else
   {
      ParMesh *mesh = pfes->GetParMesh();
      Array<int> vdofs_mark(pfes->Dof_TrueDof_Matrix()->Height());
      Array<int> tdofs_mark(pfes->Dof_TrueDof_Matrix()->Width());
      vdofs_mark = 0;
      tdofs_mark = 0;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         if (incl_bdr ? !pcoeffexcl[i] : pcoeffexcl[i])
         {
            Array<int> dofs;
            pfes->GetElementVDofs(i,dofs);
            for (int j = 0; j < dofs.Size(); j++) vdofs_mark[dofs[j]] = 1;
         }
      }

      /* reduce the vdofs marker with neighbors */
      pfes->Synchronize(vdofs_mark);

      /* make it conforming */
      pfes->GetRestrictionMatrix()->BooleanMult(vdofs_mark,tdofs_mark);
      pfes->Dof_TrueDof_Matrix()->BooleanMult(1,tdofs_mark,1,vdofs_mark);
      if (incl_bdr)
      {
         for (int i = 0; i < vdofs_mark.Size(); i++) vdofs_mark[i] = !vdofs_mark[i];
         for (int i = 0; i < tdofs_mark.Size(); i++) tdofs_mark[i] = !tdofs_mark[i];
      }

      /* elements used for state space form integration */
      for (int i = 0; i < mesh->GetNE(); i++) sforminteg[i] = !pcoeffexcl[i];
      if (incl_bdr)
      {
         /* when we include the boundary dofs, we need to integrate one layer more
            in the state space */
         Array<int> dofs;

         for (int i = 0; i < mesh->GetNE(); i++)
         {
            if (pcoeffexcl[i])
            {
               pfes->GetElementVDofs(i,dofs);
               for (int j = 0; j < dofs.Size(); j++)
               {
                  if (vdofs_mark[dofs[j]]) { sforminteg[i] = true; break; }
               }
            }
         }
      }

      /* store vdofs indices from excluded regions */
      pcoeffiniti.Reserve(vdofs_mark.Size());
      pcoeffiniti.SetSize(0);
      for (int i = 0; i < vdofs_mark.Size(); i++)
      {
         if (vdofs_mark[i])
         {
            pcoeffiniti.Append(i);
         }
      }

      /* compute dof_truedof and restriction on active dofs */
      PetscParMatrix *PT = new PetscParMatrix(pfes->Dof_TrueDof_Matrix(),Operator::PETSC_MATAIJ);

      local_cols.SetSize(0);
      local_cols.Reserve(PT->Width());

      global_cols.SetSize(0);
      global_cols.Reserve(PT->Width());

      piniti.SetSize(0);
      pactii.SetSize(0);
      piniti.Reserve(PT->Width());
      pactii.Reserve(PT->Width());

      PetscInt cst = PT->GetColStart();
      for (int i = 0; i < tdofs_mark.Size(); i++)
      {
         if (!tdofs_mark[i])
         {
            global_cols.Append(i + cst);
            local_cols.Append(i);
            pactii.Append(i);
         }
         else
         {
            piniti.Append(i);
         }
      }

      /* list of essential dofs (single component) */
      Array<int> esslist;
      esslist.SetSize(0);
      esslist.Reserve(pactii.Size());
      if (incl_bdr)
      {
         Array<int> tdofs_markb(pfes->Dof_TrueDof_Matrix()->Width());
         Array<int> vdofs_markb(pfes->Dof_TrueDof_Matrix()->Height());
         vdofs_markb = 0;
         tdofs_markb = 0;
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            if (pcoeffexcl[i])
            {
               Array<int> dofs;
               pfes->GetElementVDofs(i,dofs);
               for (int j = 0; j < dofs.Size(); j++) vdofs_markb[dofs[j]] = 1;
            }
         }

         /* reduce the vdofs marker */
         pfes->Synchronize(vdofs_markb);

         /* make it conforming */
         pfes->GetRestrictionMatrix()->BooleanMult(vdofs_markb,tdofs_markb);
         pfes->Dof_TrueDof_Matrix()->BooleanMult(1,tdofs_markb,1,vdofs_markb);
         Array<int> ess_vdofs;

         for (int i = 0; i < vdofs_markb.Size(); i++)
         {
            if (vdofs_markb[i] && !vdofs_mark[i]) continue;
            vdofs_markb[i] = 0;
         }
         pfes->GetRestrictionMatrix()->BooleanMult(vdofs_markb,tdofs_markb);
         for (int i = 0; i < pactii.Size(); i++)
         {
            if (tdofs_markb[pactii[i]])
            {
               esslist.Append(i);
            }
         }
      }

      /* extract submatrices */
      Array<PetscInt> rows(PT->GetNumRows());
      PetscInt rst = PT->GetRowStart();
      for (int i = 0; i < rows.Size(); i++)
      {
         rows[i] = i + rst;
      }
      P = new PetscParMatrix(*PT,rows,global_cols);
      delete PT;
      PT = new PetscParMatrix(pfes->GetParMesh()->GetComm(),pfes->GetRestrictionMatrix(),Operator::PETSC_MATAIJ);
      R = new PetscParMatrix(*PT,global_cols,rows);
      delete PT;

      /* store true- and v- dofs for excluded regions */
      pcoeffinitv.SetSize(pcoeffiniti.Size()*pcoeffgf.Size());
      pinitv.SetSize(piniti.Size()*pcoeffgf.Size());
      ess_tdof_list.SetSize(esslist.Size()*pcoeffgf.Size());
      ess_tdof_vals.SetSize(esslist.Size()*pcoeffgf.Size());
      for (int g = 0; g < pcoeffgf.Size(); g++)
      {
         /* vdofs */
         Vector& vgf = *(pcoeffgf[g]);
         const int stgf = g*pcoeffiniti.Size();
         for (int i = 0; i < pcoeffiniti.Size(); i++)
         {
            pcoeffinitv[i+stgf] = vgf[pcoeffiniti[i]];
         }

         /* tdofs */
         Vector& v0 = *(pcoeffv0[g]);
         const int st0 = g*piniti.Size();
         for (int i = 0; i < piniti.Size(); i++)
         {
            pinitv[i+st0] = v0[piniti[i]];
         }
         const int ste = g*esslist.Size();
         const int stp = g*pactii.Size();
         for (int i = 0; i < esslist.Size(); i++)
         {
            ess_tdof_list[i+ste] = esslist[i] + stp;
            ess_tdof_vals[i+ste] = v0[pactii[esslist[i]]];
         }
      }
      /* mass matrix */
      PetscParMatrix *nM = new PetscParMatrix(*M,global_cols,global_cols);
      delete M;
      M = nM;
      cg->SetOperator(*M);
   }

   /* block operator for mass matrix */
   if (ngf > 1)
   {
      Array<int> boff(ngf+1);
      boff[0] = 0;
      for (int i = 0; i < ngf; i++) boff[i+1] = P->Width();
      boff.PartialSum();
      BlockOperator bOp(boff);
      for (int i = 0; i < ngf; i++) bOp.SetBlock(i,i,M);
      bOp.owns_blocks = 0;
      bM = new PetscParMatrix(pfes->GetParMesh()->GetComm(),&bOp,mid);
   }

   /* Update local size */
   lsize = ngf*P->Width();

   /* Update work vector */
   pwork.SetSize(pfes->GetTrueVSize());

   /* Update BCHandler */
   bchandler.Update(ess_tdof_list,ess_tdof_vals);
}

void PDCoefficient::SaveExcl(const char* filename)
{
   FillExcl();
   if (!l2gf) return;

   ParMesh *pmesh = l2gf->ParFESpace()->GetParMesh();
   PetscMPIInt rank;
   MPI_Comm comm = pmesh->GetComm();
   MPI_Comm_rank(comm,&rank);

   std::ostringstream fname;
   fname << filename  << "." << std::setfill('0') << std::setw(6) << rank;

   std::ofstream oofs(fname.str().c_str());
   oofs.precision(8);

   pmesh->Print(oofs);
   l2gf->Save(oofs);
}

void PDCoefficient::Save(const char* filename)
{
   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      ParMesh *pmesh = pcoeffgf[i]->ParFESpace()->GetParMesh();
      PetscMPIInt rank;
      MPI_Comm comm = pmesh->GetComm();
      MPI_Comm_rank(comm,&rank);

      std::ostringstream fname;
      fname << filename  << "-" << i << "." << std::setfill('0') << std::setw(6) << rank;

      std::ofstream oofs(fname.str().c_str());
      oofs.precision(8);

      pmesh->Print(oofs);
      pcoeffgf[i]->Save(oofs);
   }
}

void PDCoefficient::SaveVisIt(const char* filename)
{
   if (!pcoeffgf.Size()) return;

   ParMesh *pmesh = pcoeffgf[0]->ParFESpace()->GetParMesh();
   DataCollection *dc = NULL;
   dc = new VisItDataCollection(filename, pmesh);
   dc->SetPrecision(8);
   dc->SetFormat(DataCollection::SERIAL_FORMAT);
   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      std::ostringstream fname;
      fname << "coeff" << "-" << i;
      dc->RegisterField(fname.str(), pcoeffgf[i]);
   }
   dc->SetCycle(0);
   dc->SetTime(0.0);
   dc->Save();
   delete dc;
}

void PDCoefficient::SaveParaView(const char* filename)
{
   if (!pcoeffgf.Size()) return;

   ParMesh *pmesh = pcoeffgf[0]->ParFESpace()->GetParMesh();
   DataCollection *dc = NULL;
   dc = new ParaViewDataCollection(filename, pmesh);
   dc->SetPrecision(8);
   dc->SetFormat(DataCollection::SERIAL_FORMAT);
   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      std::ostringstream fname;
      fname << "coeff" << "-" << i;
      dc->RegisterField(fname.str(), pcoeffgf[i]);
   }
   dc->SetCycle(0);
   dc->SetTime(0.0);
   dc->Save();
   delete dc;
}

void PDCoefficient::Visualize(const char* keys)
{
   std::string fkeys = keys ? keys : "c";
   bool disabled = false, first = false;
   if (!souts.size())
   {
      souts.resize(pcoeffgf.Size());
      for (int i=0; i<pcoeffgf.Size(); i++)
      {
         ParMesh *pmesh = pcoeffgf[i]->ParFESpace()->GetParMesh();
         PetscMPIInt rank;
         MPI_Comm comm = pmesh->GetComm();
         MPI_Comm_rank(comm,&rank);

         char vishost[] = "localhost";
         int  visport   = 19916;
         souts[i] = new socketstream;
         souts[i]->precision(8);
         souts[i]->open(vishost, visport);
         if (!souts[i] || !souts[i]->is_open())
         {
            if (!rank)
            {
               std::cout << "Unable to connect to GLVis server at "
                    << vishost << ':' << visport << std::endl;
               std::cout << "GLVis visualization disabled." << std::endl;
            }
            disabled = true;
         }
      }
      first = true;
   }
   if (disabled)
   {
      for (unsigned int i=0; i<souts.size(); i++)
      {
         delete souts[i];
      }
      souts.resize(0);
      return;
   }

   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      ParMesh *pmesh = pcoeffgf[i]->ParFESpace()->GetParMesh();
      PetscMPIInt rank,size;
      MPI_Comm comm = pmesh->GetComm();
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&size);

      if (!souts[i] || !souts[i]->is_open()) continue;
      socketstream& sock = *(souts[i]);
      sock << "parallel " << size << " " << rank << "\n";
      sock << "solution\n" << *pmesh << *pcoeffgf[i];
      if (first)
      {
         sock << "window_size 800 800\n";
         if (pcoeffgf.Size() > 1)
         {
            sock << "window_title 'Target-" << i << "'\n";
         }
         else
         {
            sock << "window_title 'Target'\n";
         }
         sock << "keys " << fkeys << "\n";
      }
      sock << "pause\n";
      sock << std::flush;
   }
}

void PDCoefficient::SetUseDerivCoefficients(bool use)
{
   usederiv = use;
}

void PDCoefficient::ElemDeriv(int pg, int el, int d, double val)
{
   MFEM_VERIFY(pg < deriv_coeffgf.Size(),"Invalid index " << pg << ", max " << deriv_coeffgf.Size()-1);
   for (int k = 0; k < deriv_coeffgf.Size(); k++)
   {
      ParFiniteElementSpace *pfes = deriv_coeffgf[k]->ParFESpace();

      Array<int> pdofs;
      pfes->GetElementVDofs(el, pdofs);
      deriv_coeffgf[k]->SetSubVector(pdofs, 0.0);
      if (k == pg)
      {
         /* XXX */
         int dd = pdofs[d] < 0 ? -1-pdofs[d] : pdofs[d];
         double vv = pdofs[d] < 0 ? -val : val;
         (*deriv_coeffgf[k])[dd] = vv;
      }
   }
}

void PDCoefficient::EvalDerivCoefficient(ElementTransformation& T, const Vector& vals, double *f)
{
   MFEM_VERIFY(ngf == 1,"Invalid ngf " << ngf << " (expected 1)");
   ParFiniteElementSpace *pfes = deriv_coeffgf[0]->ParFESpace();
   Array<int> dofs;
   pfes->GetElementVDofs(T.ElementNo, dofs);
   deriv_coeffgf[0]->SetSubVector(dofs, vals);
   *f = deriv_s_coeff->Eval(T,T.GetIntPoint());
}

void PDCoefficient::EvalDerivCoefficient(ElementTransformation& T, const Vector& vals, DenseTensor& K)
{
   MFEM_VERIFY(!scalar,"Only for non-scalar coefficients");
   /* TODO CHECK K */
   Vector zeros;
   if (deriv_m_coeff)
   {
      K = 0.0;
      double *ptr = vals.GetData();
      for (int i = 0; i < ngf; i++)
      {
         ParFiniteElementSpace *pfes = deriv_coeffgf[i]->ParFESpace();
         Array<int> dofs;
         pfes->GetElementVDofs(T.ElementNo,dofs);
         Vector pvals(ptr,dofs.Size());
         if (!i)
         {
            zeros.SetSize(dofs.Size());
            zeros = 0.0;
         }
         for (int k = 0; k < ngf; k++)
         {
            if (k == i) continue;
            deriv_coeffgf[k]->SetSubVector(dofs,zeros);
         }
         deriv_coeffgf[i]->SetSubVector(dofs,pvals);
         deriv_m_coeff->Eval(K(i),T,T.GetIntPoint());
         ptr += dofs.Size();
         pvals.SetData(NULL);
      }
   }
   else
   {
      double f;
      EvalDerivCoefficient(T,vals,&f);
      K = 0.0;
      for (int d = 0; d < K.SizeI(); d++) K(0)(d,d) = f;
   }
}

void PDCoefficient::EvalDerivShape(ElementTransformation& T, Vector& pshape)
{
   MFEM_VERIFY(scalar,"Only for scalar coefficients");
   ParFiniteElementSpace *pfes = deriv_coeffgf[0]->ParFESpace();

   const IntegrationPoint &ip = T.GetIntPoint();
   const int e = T.ElementNo;
   const FiniteElement *pel = pfes->GetFE(e);
   const int npd = pel->GetDof();
   pshape.SetSize(npd);
   pel->CalcShape(ip, pshape);
}

void PDCoefficient::EvalDerivShape(ElementTransformation& T, DenseTensor& K)
{
   /* TODO full matrix case */
   MFEM_VERIFY(!scalar,"Only for non-scalar coefficients");
   MFEM_VERIFY(deriv_m_coeff,"Missing deriv_m_coeff");
   const int vdim = deriv_m_coeff->GetWidth();
   MFEM_VERIFY(ngf == 1 || ngf == vdim,"Invalid ngf " << ngf << " != 1 and != " << vdim);
   const int e = T.ElementNo;
   if (ngf == 1) // vector space
   {
      ParFiniteElementSpace *pfes = deriv_coeffgf[0]->ParFESpace();
      const FiniteElement *pel = pfes->GetFE(e);
      const int npd = pel->GetDof();
      K.SetSize(vdim,vdim,npd);
      K = 0.0;
      DenseMatrix pshape(npd,vdim);
      pel->CalcVShape(T, pshape);
      for (int k = 0; k < npd; k++)
      {
         DenseMatrix& Kk = K(k);
         for (int d = 0; d < vdim; d++)
         {
            Kk(d,d) = pshape(k,d);
         }
      }
   }
   else if (ngf == vdim) // multi-scalar
   {
      ParFiniteElementSpace *pfes = deriv_coeffgf[0]->ParFESpace();
      const IntegrationPoint &ip = T.GetIntPoint();
      const FiniteElement *pel = pfes->GetFE(e);
      const int npd = pel->GetDof();
      K.SetSize(vdim,vdim,npd*ngf);
      K = 0.0;
      Vector pshape(npd);
      pel->CalcShape(ip, pshape);
      for (int d = 0; d < npd; d++)
      {
         const double v = pshape(d);
         for (int k = 0; k < ngf; k++)
         {
            K(k,k,k*npd+d) = v;
         }
      }
   }
   else
   {
      mfem_error("Not implemented");
   }

}

Coefficient* PDCoefficient::GetActiveCoefficient()
{
   if (usederiv)
   {
      return deriv_s_coeff;
   }
   else
   {
      return s_coeff;
   }
}

MatrixCoefficient* PDCoefficient::GetActiveMatrixCoefficient()
{
   if (usederiv)
   {
      return deriv_m_coeff;
   }
   else
   {
      return m_coeff;
   }
}

/* XXX conform with MFEM terminology? ->Project */
void PDCoefficient::GetCurrentVector(Vector& m)
{
   GetCurrentVector(m,pcoeffgf);
}

void PDCoefficient::GetCurrentVector(Vector& m, Array<ParGridFunction*>& agf)
{
   MFEM_VERIFY(agf.Size() == pcoeffgf.Size(),"Invalid array size " << agf.Size() << "!. Should be " << pcoeffgf.Size());
   MFEM_VERIFY(m.Size() == lsize,"Invalid Vector size " << m.Size() << "!. Should be " << lsize);
   double *data = m.GetData();
   int n = R->Height();
   for (int i=0; i<agf.Size(); i++)
   {
      Vector pmi(data,n);
      R->Mult(*agf[i],pmi);
      data += n;
      pmi.SetData(NULL); /* XXX clang static analysis */
   }
}

void PDCoefficient::GetInitialVector(Vector& m)
{
   MFEM_VERIFY(m.Size() == lsize,"Invalid Vector size " << m.Size() << "!. Should be " << lsize);
   double *data = m.GetData();
   int n = P->Width();
   for (int i=0; i<pcoeffv0.Size(); i++)
   {
      Vector pmi(data,n);
      for (int j = 0; j < pactii.Size(); j++) pmi[j] = (*pcoeffv0[i])[pactii[j]];
      data += n;
      pmi.SetData(NULL); /* XXX clang static analysis */
   }
}

void PDCoefficient::Distribute(const Vector& m)
{
   if (usederiv) Distribute(m,deriv_work_coeffgf);
   else          Distribute(m,pcoeffgf);
}

void PDCoefficient::Distribute(const Vector& m, Array<ParGridFunction*>& agf)
{
   MFEM_VERIFY(agf.Size() == pcoeffgf.Size(),"Invalid array size " << agf.Size() << "!. Should be " << pcoeffgf.Size());
   MFEM_VERIFY(m.Size() == lsize,"Invalid Vector size " << m.Size() << "!. Should be " << lsize);
   for (int i = 0; i < agf.Size(); i++)
   {
      pwork = 0.0;
      const int st1 = i*piniti.Size();
      for (int j = 0; j < piniti.Size(); j++) pwork[piniti[j]] = usederiv ? 0.0 : pinitv[st1 + j];
      const int st2 = i*pactii.Size();
      for (int j = 0; j < pactii.Size(); j++) pwork[pactii[j]] = m[st2 + j];
      agf[i]->Distribute(pwork);
   }
}

void PDCoefficient::Assemble(Vector& g)
{
   MFEM_VERIFY(g.Size() == lsize,"Invalid Vector size " << g.Size() << "!. Should be " << lsize);
   double *data = g.GetData();
   int n = P->Width();
   for (int i=0; i<pgradgf.Size(); i++)
   {
      Vector pgi(data,n);
      ParGridFunction *gf = pgradgf[i];

      /* values from excluded regions need to be zeroed */
      for (int j = 0; j < pcoeffiniti.Size(); j++)
      {
         (*gf)[pcoeffiniti[j]] = 0.0;
      }

      /*
         equivalent to:

         gf->ParallelAssemble(pwork);
         for (j = 0; j < pactii.Size(); j++) g[st + j] = pwork[pactii[j]];
      */
      P->MultTranspose(*gf,pgi);
      data += n;
      pgi.SetData(NULL); /* XXX clang static analysis */
   }
}

void PDCoefficient::FillExcl()
{
   if (!l2gf) return;

   ParFiniteElementSpace *l2fes = l2gf->ParFESpace();
   ParMesh *mesh = l2fes->GetParMesh();

   Array<int> dofs;
   Vector     vals;

   *l2gf = 1.0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      if (!pcoeffexcl[e]) continue;
      l2fes->GetElementVDofs(e,dofs);
      vals.SetSize(dofs.Size());
      vals = 0.0;
      l2gf->SetSubVector(dofs, vals);
   }
}

void PDCoefficient::UpdateExcl()
{
   ParMesh *mesh = pfes->GetParMesh();
   pcoeffexcl.SetSize(mesh->GetNE());
   pcoeffexcl = false;
   if (!l2gf) return;

   ParFiniteElementSpace *l2fes = l2gf->ParFESpace();
   long fseq = l2fes->GetSequence();
   l2fes->Update(); /* XXX since l2gf owns fes, it should also call fes->Update() during l2gf->Update() but it does not! */
   if (fseq == l2fes->GetSequence()) return;

   l2gf->Update();

   pcoeffexcl = true;
   Array<int> dofs;
   Vector     vals;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      l2fes->GetElementVDofs(e,dofs);
      l2gf->GetSubVector(dofs, vals);
      for (int j = 0; j < vals.Size(); j++)
      {
         if (vals[j] != 0.0)
         {
            pcoeffexcl[e] = false;
            break;
         }
      }
   }
}

void PDCoefficient::Update(bool want_true)
{
   long fseq = pfes->GetSequence();
   pfes->Update(true);
   if (fseq == pfes->GetSequence()) return;

   PetscParMatrix *oP = NULL;
   if (want_true) /* save previous dof_truedof */
   {
      oP = new PetscParMatrix(P,Operator::ANY_TYPE);
   }

   /* Update excluded regions */
   UpdateExcl();

   /* GetCurrent and UpdateCoeff ? */
   for (int i = 0; i < pcoeffgf.Size(); i++)
   {
      const int st = i*pcoeffgf[i]->Size();
      for (int j = 0; j < pcoeffiniti.Size(); j++)
      {
         (*pcoeffgf[i])[pcoeffiniti[j]] = pcoeffinitv[j+st];
      }
      pcoeffgf[i]->Update();
   }

   /* XXX Store Initial coefficient and resample? */
   for (int i = 0; i < pcoeffgf.Size(); i++)
   {
      int tvsize = pcoeffgf[i]->ParFESpace()->GetTrueVSize();

      pcoeffv0[i]->SetSize(tvsize);
      pcoeffgf[i]->ParallelProject(*pcoeffv0[i]);
      pcoeffgf[i]->Distribute(*pcoeffv0[i]);
   }

   for (int i = 0; i < pgradgf.Size(); i++)
   {
      pgradgf[i]->Update();
      *pgradgf[i] = 0.0;
   }
   for (int i = 0; i < deriv_coeffgf.Size(); i++)
   {
      deriv_coeffgf[i]->Update();
      *deriv_coeffgf[i] = 0.0;
   }
   for (int i = 0; i < deriv_work_coeffgf.Size(); i++)
   {
      deriv_work_coeffgf[i]->Update();
      *deriv_work_coeffgf[i] = 0.0;
   }

   if (mass)
   {
      OperatorHandle oM(mid);
      mass->Update();
      mass->Assemble(0);
      mass->Finalize(0);
      mass->ParallelAssemble(oM);
      oM.SetOperatorOwner(false);
      delete M;
      oM.Get(M);
      cg->SetOperator(*M);
   }

   /* SetUp operators and conforming dofs. For excl regions, also the fixed values
      Deletes any previous P,R and trueTransfer set */
   SetUpOperators();

   /* transfer operator for true data as triple matrix product (R * gftransfer * P) */
   if (want_true)
   {
      PetscParMatrix pT(pfes->GetParMesh()->GetComm(),pfes->GetUpdateOperator(),Operator::PETSC_MATAIJ);
      trueTransfer = mfem::TripleMatrixProduct(R,&pT,oP);
   }
   delete oP;

   /* No need to update s_coeff, m_coeff, deriv_s_coeff, deriv_m_coeff */
}

PDCoefficient::~PDCoefficient()
{
   Reset();
}

PDCoefficient::BCHandler::BCHandler()
{
   SetType(CONSTANT);
}

PDCoefficient::BCHandler::BCHandler(Array<int>& el, Array<double>& v)
{
   SetType(CONSTANT);
   Update(el,v);
}

void PDCoefficient::BCHandler::Update(Array<int>& el, Array<double>& v)
{
   SetTDofs(el);
   vals.SetSize(v.Size());
   for (int i = 0; i < v.Size(); i++) vals[i] = v[i];
}

void PDCoefficient::BCHandler::Eval(double t,Vector& g)
{
   Array<int>& td = GetTDofs();
   for (int i = 0; i < td.Size(); i++) g[td[i]] = vals[i];
}

}

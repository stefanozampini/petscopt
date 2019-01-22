#include <mfemopt/pdcoefficient.hpp>
#include <petscmat.h>
#include <fstream>

namespace mfemopt
{
using namespace mfem;

void PDCoefficient::Reset()
{
   usederiv = false;
   pcoeffiniti.SetSize(0);
   pcoeffinitv.SetSize(0);
   pcoeffexcl.SetSize(0);
   delete P;
   delete R;
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
   pcoeffgf.SetSize(0);
   pgradgf.SetSize(0);
   deriv_coeffgf.SetSize(0);
   deriv_work_coeffgf.SetSize(0);
   global_cols.SetSize(0);
   lvsize = 0;
   lsize = 0;
   for (unsigned int i=0; i<souts.size(); i++)
   {
      delete souts[i];
   }
   souts.resize(0);
}

PDCoefficient::PDCoefficient(Coefficient& Q, ParFiniteElementSpace* pfes,
                             bool (*excl_fn)(const Vector&))
{
   ParMesh *mesh = pfes->GetParMesh();
   Vector pt(mesh->SpaceDimension());
   Array<bool> excl(mesh->GetNE());
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      mesh->GetElementTransformation(e)->Transform(
         Geometries.GetCenter(mesh->GetElementBaseGeometry(e)), pt);
      excl[e] = (*excl_fn)(pt);
   }
   Init(&Q,NULL,NULL,pfes,excl);
}

PDCoefficient::PDCoefficient(Coefficient& Q, ParFiniteElementSpace* pfes,
               const Array<int>& excl_tag)
{
   ParMesh *mesh = pfes->GetParMesh();
   Array<bool> excl(mesh->GetNE());
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      bool lexcl = false;
      int eatt = mesh->GetAttribute(i);
      for (int r = 0; r < excl_tag.Size(); r++)
      {
         if (excl_tag[r] == eatt)
         {
            lexcl = true;
            break;
         }
      }
      excl[i] = lexcl;
   }
   Init(&Q,NULL,NULL,pfes,excl);
}

void PDCoefficient::Init(Coefficient *Q, VectorCoefficient *VQ, MatrixCoefficient *MQ, ParFiniteElementSpace* pfes,const Array<bool>& excl)
{
   lsize = 0;
   lvsize = 0;
   usederiv = false;
   deriv_s_coeff = NULL;
   deriv_m_coeff = NULL;
   s_coeff = NULL;
   m_coeff = NULL;
   P = NULL;
   R = NULL;

   /* coefficients */
   int ngf = 1;
   if (MQ) { mfem_error("Not yet implemented"); }
   else if (VQ) { mfem_error("Not yet implemented"); }
   else if (Q) { ngf = 1; };

   for (int i = 0; i < ngf; i++)
   {
      pcoeffgf.Append(new ParGridFunction(pfes));
      deriv_coeffgf.Append(new ParGridFunction(pfes));
      deriv_work_coeffgf.Append(new ParGridFunction(pfes));
      pgradgf.Append(new ParGridFunction(pfes));
      lvsize += pfes->GetVSize();
   }

   /* store values of projected initial coefficients
      and create actual coefficients to be used */
   if (MQ) { mfem_error("Not yet implemented"); }
   else if (VQ) { mfem_error("Not yet implemented"); }
   else if (Q)
   {
      *pcoeffgf[0] = 0.0;
      // XXX BUG: GROUPCOMM not created!
      pcoeffgf[0]->ProjectDiscCoefficient(*Q,GridFunction::ARITHMETIC);
      //pcoeffgf[0]->ProjectCoefficient(*Q);
      s_coeff = new GridFunctionCoefficient(pcoeffgf[0]);
      deriv_s_coeff = new GridFunctionCoefficient(deriv_coeffgf[0]);
   }

   pcoeffexcl.SetSize(pfes->GetParMesh()->GetNE());
   pcoeffexcl = false;
   for (int i = 0; i < std::min(pcoeffexcl.Size(),excl.Size()); i++) pcoeffexcl[i] = excl[i];

   PetscBool lhas = PETSC_FALSE,has;
   for (int i = 0; i < pcoeffexcl.Size(); i++) if (pcoeffexcl[i]) { lhas = PETSC_TRUE; break; }
   MPI_Allreduce(&lhas,&has,1,MPIU_BOOL,MPI_LOR,pfes->GetParMesh()->GetComm());
   if (has)
   {
      ParMesh *pmesh = pfes->GetParMesh();
      ParGridFunction *gf = pgradgf[0];
      *gf = 0.0;

      PetscParMatrix *PT = new PetscParMatrix(pfes->Dof_TrueDof_Matrix(),Operator::PETSC_MATAIJ);
      Array<PetscInt> rows(PT->GetNumRows());
      PetscInt rst = PT->GetRowStart();
      for (int i = 0; i < rows.Size(); i++)
      {
         rows[i] = i + rst;
      }

      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         if (!pcoeffexcl[i])
         //if (pcoeffexcl[i])
         {
            Array<int> dofs;
            pfes->GetElementVDofs(i,dofs);
            Vector vals(dofs.Size());
            vals = 1.0;
            gf->SetSubVector(dofs,vals);
         }
      }

      /* store dofs for excluded regions */
      Array<int> initi(gf->Size());
      {
         int cum = 0;
         Vector lwork(gf->GetData(),gf->Size());
         for (int i = 0; i < lwork.Size(); i++)
         {
            if (std::abs(lwork(i)) < 1.e-12)
            //if (std::abs(lwork(i)) > 0.0)
            {
               initi[cum++] = i;
            }
         }
         initi.SetSize(cum);
      }

      /* we don't exclude those elements that are
         neighbours of the domain of interest */
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         Array<int> dofs;
         pfes->GetElementVDofs(i,dofs);
         Vector vals;
         gf->GetSubVector(dofs,vals);
         pcoeffexcl[i] = vals.Normlinf() > 1.e-12 ? false : true;
      }

      /* restrict on active dofs (XXX is this robust?) */
      Array<PetscInt> local_cols;
      global_cols.Reserve(PT->Width());
      local_cols.Reserve(PT->Width());
      PetscInt cst = PT->GetColStart();
      HypreParVector *work = gf->ParallelAssemble();
      for (int i = 0; i < work->Size(); i++)
      {
         if (std::abs((*work)(i)) > 1.e-12)
         //if (std::abs((*work)(i)) < 1.e-12)
         {
            global_cols.Append(i + cst);
            local_cols.Append(i);
         }
      }
      P = new PetscParMatrix(*PT,rows,global_cols);
      delete PT;

      for (int i = 0; i < rows.Size(); i++)
      {
         rows[i] = i;
      }
      PT = new PetscParMatrix(pfes->GetRestrictionMatrix());
      R = new PetscParMatrix(*PT,local_cols,rows);
      delete PT;
      delete work;

      initi.Copy(pcoeffiniti);
      pcoeffinitv.SetSize(initi.Size()*ngf);
      for (int g = 0; g < ngf; g++)
      {
         const int st = g*gf->Size();
         ParGridFunction *vgf = pcoeffgf[g];
         for (int i = 0; i < initi.Size(); i++)
         {
            pcoeffinitv[i+st] = (*vgf)[initi[i]];
         }
      }
   }
   else
   {
      P = new PetscParMatrix(pfes->Dof_TrueDof_Matrix(),Operator::PETSC_MATAIJ);
      R = new PetscParMatrix(PETSC_COMM_SELF,pfes->GetRestrictionMatrix());
      PetscInt cst = P->GetColStart();
      global_cols.SetSize(P->Width());
      for (int i = 0; i < P->Width(); i++)
      {
         global_cols[i] = i + cst;
      }
   }
   lsize = ngf*P->Width();
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
         sock << "window_title 'Target'\n";
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

void PDCoefficient::GetCurrentVector(Vector& m)
{
   int off = 0;
   m.SetSize(lsize);
   for (int i=0; i<pgradgf.Size(); i++)
   {
      int n = P->Width();
      Vector pmi(m.GetData()+off,n);
      ParGridFunction *gf = pcoeffgf[i];
      MFEM_VERIFY(!usederiv,"This should not happen");
      if (usederiv)
      {
         gf = deriv_work_coeffgf[i];
      }
      else
      {
         gf = pcoeffgf[i];
      }
      R->Mult(*gf,pmi);
      off += n;
   }
}

void PDCoefficient::UpdateCoefficient(const Vector& m)
{
   MFEM_VERIFY(m.Size() == lsize,"Invalid Vector size " << m.Size() << "!. Should be " << lsize);
   int off = 0;
   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      int n = P->Width();
      Vector pmi;
      pmi.SetDataAndSize(m.GetData()+off,n);
      ParGridFunction *gf;
      if (usederiv)
      {
         gf = deriv_work_coeffgf[i];
      }
      else
      {
         gf = pcoeffgf[i];
      }
      P->Mult(pmi,*gf);

      /* restore values from excluded regions */
      const int st = i*gf->Size();
      for (int j = 0; j < pcoeffiniti.Size(); j++)
      {
         if (usederiv) (*gf)[pcoeffiniti[j]] = 0;
         else (*gf)[pcoeffiniti[j]] = pcoeffinitv[j+st];
      }
      off += n;
   }
}

void PDCoefficient::UpdateCoefficientWithGF(const Vector& m, Array<ParGridFunction*>& agf)
{
   MFEM_VERIFY(agf.Size() == pcoeffgf.Size(),"Invalid array size " << agf.Size() << "!. Should be " << pcoeffgf.Size());
   MFEM_VERIFY(m.Size() == lsize,"Invalid Vector size " << m.Size() << "!. Should be " << lsize);
   int off = 0;
   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      int n = P->Width();
      Vector pmi;
      pmi.SetDataAndSize(m.GetData()+off,n);

      ParGridFunction *gf = agf[i];
      P->Mult(pmi,*gf);

      /* restore values from excluded regions */
      const int st = i*gf->Size();
      for (int j = 0; j < pcoeffiniti.Size(); j++)
      {
         if (usederiv) (*gf)[pcoeffiniti[j]] = 0;
         else (*gf)[pcoeffiniti[j]] = pcoeffinitv[j+st];
      }
      off += n;
   }
}

void PDCoefficient::UpdateGradient(Vector& g)
{
   int off = 0;
   g.SetSize(lsize);
   for (int i=0; i<pgradgf.Size(); i++)
   {
      int n = P->Width();
      Vector pgi(g.GetData()+off,n);
      ParGridFunction *gf = pgradgf[i];
      P->MultTranspose(*gf,pgi);
      off += n;
   }
}

PDCoefficient::~PDCoefficient()
{
   Reset();
}

}

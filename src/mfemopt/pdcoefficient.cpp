#include <mfemopt/pdcoefficient.hpp>
#include <mfemopt/mfemextra.hpp>
#include <mfem/fem/datacollection.hpp>
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
   for (int i=0; i<pcoeffv0.Size(); i++)
   {
      delete pcoeffv0[i];
   }
   pcoeffgf.SetSize(0);
   pgradgf.SetSize(0);
   deriv_coeffgf.SetSize(0);
   deriv_work_coeffgf.SetSize(0);
   pcoeffv0.SetSize(0);
   global_cols.SetSize(0);
   delete pfes;
   lsize = 0;
   order = -1;
   for (unsigned int i=0; i<souts.size(); i++)
   {
      delete souts[i];
   }
   souts.resize(0);
}


PDCoefficient::PDCoefficient(Coefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
               const Array<bool>& excl)
{
   Init(&Q,NULL,NULL,mesh,fec,excl);
}

PDCoefficient::PDCoefficient(Coefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
                             bool (*excl_fn)(const Vector&))
{
   Array<bool> excl;
   MeshGetElementsTagged(mesh,excl_fn,excl);
   Init(&Q,NULL,NULL,mesh,fec,excl);
}

PDCoefficient::PDCoefficient(Coefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
               const Array<int>& excl_tag)
{
   Array<bool> excl;
   MeshGetElementsTagged(mesh,excl_tag,excl);
   Init(&Q,NULL,NULL,mesh,fec,excl);
}

PDCoefficient::PDCoefficient(VectorCoefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
               const Array<bool>& excl)
{
   Init(NULL,&Q,NULL,mesh,fec,excl);
}

PDCoefficient::PDCoefficient(VectorCoefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
                             bool (*excl_fn)(const Vector&))
{
   Array<bool> excl;
   MeshGetElementsTagged(mesh,excl_fn,excl);
   Init(NULL,&Q,NULL,mesh,fec,excl);
}

PDCoefficient::PDCoefficient(VectorCoefficient& Q, ParMesh *mesh, const FiniteElementCollection *fec,
               const Array<int>& excl_tag)
{
   Array<bool> excl;
   MeshGetElementsTagged(mesh,excl_tag,excl);
   Init(NULL,&Q,NULL,mesh,fec,excl);
}

void PDCoefficient::Init(Coefficient *Q, VectorCoefficient *VQ, MatrixCoefficient *MQ, ParMesh *mesh, const FiniteElementCollection *fec, const Array<bool>& excl)
{
   lsize = 0;
   usederiv = false;
   deriv_s_coeff = NULL;
   deriv_m_coeff = NULL;
   s_coeff = NULL;
   m_coeff = NULL;
   P = NULL;
   R = NULL;

   const FiniteElement *fe = NULL;
   if (mesh->GetNE())
   {
      fe = fec->FiniteElementForGeometry(mesh->GetElementBaseGeometry(0));
   }
   /* Reduce on the comm */
   bool scalar;
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
      scalar = (glob[1] == FiniteElement::SCALAR);
   }

   /* store values of projected initial coefficients
      and create actual coefficients to be used */
   int ngf = 0;
   if (MQ) { mfem_error("Not yet implemented"); }
   else if (VQ)
   {
      /* XXX MASK */
      if (scalar)
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
      MFEM_VERIFY(scalar,"Cannot represent a scalar coefficient with a vector finite element space");
      ngf = 1;
   }
   else mfem_error("Unhandled case");

   pfes = new ParFiniteElementSpace(mesh,fec,1,Ordering::byVDIM);

   /* store values of projected initial coefficients
      and create actual coefficients to be used */
   if (MQ) { mfem_error("Not yet implemented"); }
   else if (VQ)
   {
      if (scalar)
      {
         for (int i = 0; i < ngf; i++)
         {
            pcoeffgf.Append(new ParGridFunction(pfes));
            deriv_coeffgf.Append(new ParGridFunction(pfes));
            deriv_work_coeffgf.Append(new ParGridFunction(pfes));
            pgradgf.Append(new ParGridFunction(pfes));

            ComponentCoefficient Q(*VQ,i);
            *pcoeffgf[i] = 0.0;
            pcoeffgf[i]->ProjectDiscCoefficient(Q,GridFunction::ARITHMETIC);
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
         *pcoeffgf[0] = 0.0;
         pcoeffgf[0]->ProjectDiscCoefficient(*VQ,GridFunction::ARITHMETIC);

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

      *pcoeffgf[0] = 0.0;
      pcoeffgf[0]->ProjectDiscCoefficient(*Q,GridFunction::ARITHMETIC);
      s_coeff = new GridFunctionCoefficient(pcoeffgf[0]);
      deriv_s_coeff = new GridFunctionCoefficient(deriv_coeffgf[0]);
   }

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
      /* excluded elements mark their dofs as fixed */
      Array<int> vdofs_mark(pfes->Dof_TrueDof_Matrix()->Height());
      Array<int> tdofs_mark(pfes->Dof_TrueDof_Matrix()->Width());
      vdofs_mark = 0;
      tdofs_mark = 0;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         if (pcoeffexcl[i])
         {
            Array<int> dofs;
            pfes->GetElementVDofs(i,dofs);
            for (int j = 0; j < dofs.Size(); j++) vdofs_mark[dofs[j]] = 1;
         }
      }

      /* reduce the vdofs marker */
      pfes->Synchronize(vdofs_mark);

      /* make it conforming */
      pfes->GetRestrictionMatrix()->BooleanMult(vdofs_mark,tdofs_mark);
      pfes->Dof_TrueDof_Matrix()->BooleanMult(1,tdofs_mark,1,vdofs_mark);

      /* store vdofs indices and initial values from excluded regions */
      pcoeffiniti.Reserve(vdofs_mark.Size());
      for (int i = 0; i < vdofs_mark.Size(); i++)
      {
         if (vdofs_mark[i])
         {
            pcoeffiniti.Append(i);
         }
      }
      pcoeffinitv.SetSize(pcoeffiniti.Size()*ngf);
      for (int g = 0; g < ngf; g++)
      {
         ParGridFunction *vgf = pcoeffgf[g];
         const int st = g*vgf->Size();
         for (int i = 0; i < pcoeffiniti.Size(); i++)
         {
            pcoeffinitv[i+st] = (*vgf)[pcoeffiniti[i]];
         }
      }

      /* dof_truedof on active dofs */
      PetscParMatrix *PT = new PetscParMatrix(pfes->Dof_TrueDof_Matrix(),Operator::PETSC_MATAIJ);
      Array<PetscInt> rows(PT->GetNumRows());
      PetscInt rst = PT->GetRowStart();
      for (int i = 0; i < rows.Size(); i++)
      {
         rows[i] = i + rst;
      }

      Array<PetscInt> local_cols;
      global_cols.Reserve(PT->Width());
      local_cols.Reserve(PT->Width());
      PetscInt cst = PT->GetColStart();

      for (int i = 0; i < tdofs_mark.Size(); i++)
      {
         if (!tdofs_mark[i])
         {
            global_cols.Append(i + cst);
            local_cols.Append(i);
         }
      }
      P = new PetscParMatrix(*PT,rows,global_cols);
      delete PT;

      /* Restriction on active dofs */
      for (int i = 0; i < rows.Size(); i++)
      {
         rows[i] = i;
      }
      PT = new PetscParMatrix(pfes->GetRestrictionMatrix());
      R = new PetscParMatrix(*PT,local_cols,rows);
      delete PT;
   }
   else /* no elements to be excluded, convert to PETSc operators for later usage */
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
      // XXX SHOULD I DO THIS HERE ?
#if 0
      /* restore values from excluded regions */
      ParGridFunction *gf = pcoeffgf[i];
      const int st = i*gf->Size();
      for (int j = 0; j < pcoeffiniti.Size(); j++)
      {
         (*gf)[pcoeffiniti[j]] = pcoeffinitv[j+st];
      }
#endif
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
   MFEM_VERIFY(!usederiv,"This should not happen");
   MFEM_VERIFY(m.Size() == lsize,"Invalid Vector size " << m.Size() << "!. Should be " << lsize);
   double *data = m.GetData();
   int n = P->Width();
   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      Vector pmi(data,n);
      R->Mult(*pcoeffgf[i],pmi);
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
      R->Mult(*pcoeffv0[i],pmi);
      data += n;
      pmi.SetData(NULL); /* XXX clang static analysis */
   }
}

void PDCoefficient::UpdateCoefficient(const Vector& m)
{
   if (usederiv) UpdateCoefficientWithGF(m,deriv_work_coeffgf);
   else          UpdateCoefficientWithGF(m,pcoeffgf);
}

void PDCoefficient::UpdateCoefficientWithGF(const Vector& m, Array<ParGridFunction*>& agf)
{
   MFEM_VERIFY(agf.Size() == pcoeffgf.Size(),"Invalid array size " << agf.Size() << "!. Should be " << pcoeffgf.Size());
   MFEM_VERIFY(m.Size() == lsize,"Invalid Vector size " << m.Size() << "!. Should be " << lsize);
   double *data = m.GetData();
   int n = P->Width();
   for (int i=0; i<pcoeffgf.Size(); i++)
   {
      Vector pmi(data,n);

      ParGridFunction *gf = agf[i];
      P->Mult(pmi,*gf);
      /* restore values from excluded regions */
      const int st = i*gf->Size();
      for (int j = 0; j < pcoeffiniti.Size(); j++)
      {
         (*gf)[pcoeffiniti[j]] = usederiv ? 0.0 : pcoeffinitv[j+st];
      }
      data += n;
      pmi.SetData(NULL); /* XXX clang static analysis */
   }
}

void PDCoefficient::UpdateGradient(Vector& g)
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

      P->MultTranspose(*gf,pgi);
      data += n;
      pgi.SetData(NULL); /* XXX clang static analysis */
   }
}

PDCoefficient::~PDCoefficient()
{
   Reset();
}

}

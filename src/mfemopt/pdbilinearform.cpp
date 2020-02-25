#include <mfemopt/pdbilinearform.hpp>
#include <mfemopt/pdbilininteg.hpp>
#include <petscsys.h>

namespace mfemopt
{
using namespace mfem;

PDBilinearForm::~PDBilinearForm()
{
   delete sgf;
   delete agf;
   delete swgf;
   delete awgf;
}

int PDBilinearForm::GetParameterSize()
{
   int ls = 0;
   for (int i = 0; i < dbfi.Size(); i++)
   {
      PDBilinearFormIntegrator *pdb = dynamic_cast<PDBilinearFormIntegrator *> (dbfi[i]);
      if (pdb) ls += pdb->GetParameterSize();
   }
   return ls;
}

PetscInt PDBilinearForm::GetParameterGlobalSize()
{
   PetscInt l = GetParameterSize(), g;
   MPI_Comm comm = ParFESpace()->GetParMesh()->GetComm();
   MPI_Allreduce(&l,&g,1,MPIU_INT,MPI_SUM,comm);
   return g;
}

void PDBilinearForm::UpdateParameter(const Vector& m)
{
   double *data = m.GetData();
   for (int i = 0; i < dbfi.Size(); i++)
   {
      PDBilinearFormIntegrator *pdb = dynamic_cast<PDBilinearFormIntegrator *> (dbfi[i]);
      if (pdb)
      {
         int ls = pdb->GetParameterSize();
         Vector mm(data,ls);
         pdb->UpdateCoefficient(mm);
         data += ls;
         mm.SetData(NULL);
      }
   }
}

/* these are independent from the value of the parameter m */
void PDBilinearForm::ComputeHessian_XM(const Vector& a, const Vector& m, const Vector& pertIn, Vector& out)
{
   ComputeGradient_Internal(a,pertIn,out,true);
}

void PDBilinearForm::ComputeHessian_MX(const Vector& a, const Vector& s, const Vector& m, Vector& out)
{
   ComputeGradientAdjoint_Internal(a,s,out,true);
}

void PDBilinearForm::ComputeGradient(const Vector& s, const Vector& m, const Vector& pertIn, Vector& out)
{
   ComputeGradient_Internal(s,pertIn,out);
}

void PDBilinearForm::ComputeGradientAdjoint(const Vector& a, const Vector& s, const Vector& m, Vector& out)
{
   ComputeGradientAdjoint_Internal(a,s,out);
}

void PDBilinearForm::ComputeGradient_Internal(const Vector& in, const Vector& pertIn, Vector& out, bool hessian)
{
   out = 0.0;
   if (!GetParameterGlobalSize()) return;

   ParGridFunction *ingf,*wgf;
   if (hessian)
   {
      /* XXX different adjoint space */
      if (!agf) agf = new ParGridFunction(ParFESpace());
      if (!awgf) awgf = new ParGridFunction(ParFESpace());
      ingf = agf;
      wgf = awgf;
   }
   else
   {
      if (!sgf) sgf = new ParGridFunction(ParFESpace());
      if (!swgf) swgf = new ParGridFunction(ParFESpace());
      ingf = sgf;
      wgf = swgf;
   }
   ingf->Distribute(in);

   double *data = pertIn.GetData();
   for (int i = 0; i < dbfi.Size(); i++)
   {
      PDBilinearFormIntegrator *pdb = dynamic_cast<PDBilinearFormIntegrator *> (dbfi[i]);
      if (pdb)
      {
         Vector tout(out);
         int ls = pdb->GetParameterSize();
         Vector pp(data,ls);
         if (hessian)
         {
            pdb->ComputeHessian_XM(ingf,pp,wgf);
         }
         else
         {
            pdb->ComputeGradient(ingf,pp,wgf);
         }
         tout = 0.0;
         wgf->ParallelAssemble(tout);
         out += tout;
         data += ls;
         pp.SetData(NULL);
      }
   }
}

void PDBilinearForm::ComputeGradientAdjoint_Internal(const Vector& ain, const Vector& sin, Vector& g, bool hessian)
{
   g = 0.0;
   if (!GetParameterGlobalSize()) return;

   /* XXX different adjoint space */
   if (!agf) agf = new ParGridFunction(ParFESpace());
   if (!awgf) awgf = new ParGridFunction(ParFESpace());
   if (!sgf) sgf = new ParGridFunction(ParFESpace());
   if (!swgf) swgf = new ParGridFunction(ParFESpace());
   agf->Distribute(ain);
   sgf->Distribute(sin);

   double *data = g.GetData();
   for (int i = 0; i < dbfi.Size(); i++)
   {
      PDBilinearFormIntegrator *pdb = dynamic_cast<PDBilinearFormIntegrator *> (dbfi[i]);
      if (pdb)
      {
         int ls = pdb->GetParameterSize();
         Vector gg(data,ls);
         if (hessian)
         {
            pdb->ComputeHessian_MX(agf,sgf,gg);
         }
         else
         {
            pdb->ComputeGradientAdjoint(agf,sgf,gg);
         }
         data += ls;
         gg.SetData(NULL);
      }
   }
}

}

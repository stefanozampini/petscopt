#include <mfemopt/mfemextra.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>

static void __mfemopt_snes_obj(mfem::Operator*,const mfem::Vector&,double*);
static void __mfemopt_snes_postcheck(mfem::Operator*,const mfem::Vector&,mfem::Vector&,mfem::Vector&,bool&,bool&);

namespace mfemopt
{
using namespace mfem;

ReplicatedParMesh::ReplicatedParMesh(MPI_Comm comm, Mesh &mesh, int nrep, bool contig)
{
   PetscErrorCode ierr;
   PetscSubcomm   subcomm;
   PetscMPIInt    size;

   ierr = MPI_Comm_size(comm,&size); CCHKERRQ(comm,ierr);
   MFEM_VERIFY(!(size%nrep),"Size of comm must be a multiple of the number of replicas");
   MFEM_VERIFY(mesh.Conforming(),"Not supported");

   ierr = PetscSubcommCreate(comm, &subcomm); CCHKERRQ(comm,ierr);
   ierr = PetscSubcommSetNumber(subcomm, (PetscInt)nrep); CCHKERRQ(comm,ierr);
   ierr = PetscSubcommSetType(subcomm, contig ? PETSC_SUBCOMM_CONTIGUOUS : PETSC_SUBCOMM_INTERLACED); CCHKERRQ(comm,ierr);

   ierr = PetscCommDuplicate(comm,&parent_comm,NULL); CCHKERRQ(comm,ierr);
   ierr = PetscCommDuplicate(subcomm->child,&child_comm,NULL); CCHKERRQ(comm,ierr);
   color = subcomm->color;

   PetscMPIInt child_size, parent_size;
   ierr = MPI_Comm_size(child_comm, &child_size); CCHKERRQ(child_comm,ierr);
   ierr = MPI_Comm_size(parent_comm, &parent_size); CCHKERRQ(child_comm,ierr);

   int *child_part = mesh.GeneratePartitioning(child_size, 1);

   child_mesh = new ParMesh(child_comm, mesh, child_part, 1);
   // there's no simple way to create a ParMesh and change the comm
   parent_mesh = new ParMesh(parent_comm, mesh, child_part, 1);

   delete [] child_part;

   ierr = PetscSubcommDestroy(&subcomm); CCHKERRQ(comm,ierr);
}

ReplicatedParMesh::~ReplicatedParMesh()
{
   PetscErrorCode ierr;
   ierr = PetscCommDestroy(&parent_comm); CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = PetscCommDestroy(&child_comm); CCHKERRQ(PETSC_COMM_SELF,ierr);
   delete child_mesh;
   delete parent_mesh;
}

ParMesh* ParMeshTest(MPI_Comm comm, Mesh &mesh)
{
   int size;
   MPI_Comm_size(comm,&size);
   int nel = mesh.GetNE();
   int *test_partitioning = new int[nel];
   for (int i = 0; i < nel; i++) test_partitioning[i] = (int)((1.0*i*size)/(1.0*nel));

   ParMesh *pmesh = new ParMesh(comm,mesh,test_partitioning);
   delete[] test_partitioning;
   return pmesh;
}

void MeshGetElementsTagged(Mesh *mesh, bool (*tag_fn)(const Vector&),Array<bool>& tag)
{
   tag.SetSize(mesh->GetNE());
   Vector pt(mesh->SpaceDimension());
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      mesh->GetElementTransformation(e)->Transform(
         Geometries.GetCenter(mesh->GetElementBaseGeometry(e)), pt);
      tag[e] = (*tag_fn)(pt);
   }
}

void MeshGetElementsTagged(Mesh *mesh,const Array<int>& which_tag,Array<bool>& tag)
{
   tag.SetSize(mesh->GetNE());
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      bool ltag = false;
      int eatt = mesh->GetAttribute(i);
      for (int r = 0; r < which_tag.Size(); r++)
      {
         if (which_tag[r] == eatt)
         {
            ltag = true;
            break;
         }
      }
      tag[i] = ltag;
   }
}

void FiniteElementSpaceGetRangeAndDeriv(FiniteElementSpace& fes, int* r, int* d)
{
   const FiniteElement* fe = fes.GetFE(0);
   if (r) *r = fe ? fe->GetRangeType() : -1;
   if (d) *d = fe ? fe->GetDerivType() : -1;
}

void ParFiniteElementSpaceGetRangeAndDeriv(ParFiniteElementSpace& fes, int* r, int* d)
{
   int lr[2] = {-1,-1}, gr[2];
   FiniteElementSpaceGetRangeAndDeriv(fes,lr,lr+1);
   /* reduce for empty meshes */
   MPI_Allreduce(&lr,&gr,2,MPI_INT,MPI_MAX,fes.GetParMesh()->GetComm());
   if (r) *r = gr[0];
   if (d) *d = gr[1];
}

double FunctionOfCoefficient::Eval(ElementTransformation &T,
                              const IntegrationPoint &ip)
{
   g->SetTime(GetTime());
   return f ? f(g->Eval(T,ip)) : g->Eval(T,ip);
}

ComponentCoefficient::ComponentCoefficient(VectorCoefficient& _VQ, int _c)
{
   MFEM_VERIFY(_c > -1,"Invalid component " << _c );
   MFEM_VERIFY(_c < _VQ.GetVDim(),"Component " << _c << " too large. Max " << _VQ.GetVDim());
   time = 0.0;
   VQ = &_VQ;
   c  = _c;
   w.SetSize(VQ->GetVDim());
}

double ComponentCoefficient::Eval(ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   VQ->Eval(w,T,ip);
   return w[c];
}

DiagonalMatrixCoefficient::DiagonalMatrixCoefficient(VectorCoefficient* _VQ, bool _own) : MatrixCoefficient(0) /* XXX the class have the explicit constructor */
{
   VQ = _VQ;
   own = _own;
   int vd = VQ ? VQ->GetVDim() : 0;
   w.SetSize(vd);
   height = width = vd;
   time = 0.0;
}

void DiagonalMatrixCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   VQ->Eval(w,T,ip);
   K.Diag(w.GetData(),w.Size());
}

DiagonalMatrixCoefficient::~DiagonalMatrixCoefficient()
{
  if (own) delete VQ;
}

PetscNonlinearSolverOpt::PetscNonlinearSolverOpt(MPI_Comm comm, ReducedFunctional &rf,
                         const std::string &prefix, bool obj) : PetscNonlinearSolver(comm,rf,prefix)
{
   if (obj)
   {
      SetObjective(__mfemopt_snes_obj);
   }
   SetPostCheck(__mfemopt_snes_postcheck);
}

}

void __mfemopt_snes_obj(mfem::Operator *op, const mfem::Vector& u, double *f)
{
   mfemopt::ReducedFunctional *rf = dynamic_cast<mfemopt::ReducedFunctional*>(op);
   MFEM_VERIFY(rf,"Not a mfemopt::ReducedFunctional operator");
   rf->ComputeObjective(u,f);
}

void __mfemopt_snes_postcheck(mfem::Operator *op, const mfem::Vector& X, mfem::Vector& Y, mfem::Vector &W, bool& cy, bool& cw)
{
   mfemopt::ReducedFunctional *rf = dynamic_cast<mfemopt::ReducedFunctional*>(op);
   MFEM_VERIFY(rf,"Not a mfemopt::ReducedFunctional operator");
   cy = false;
   cw = false;
   rf->PostCheck(X,Y,W,cy,cw);
}

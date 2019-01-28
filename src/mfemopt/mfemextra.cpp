#include <mfemopt/mfemextra.hpp>

static void __mfemopt_snes_obj(mfem::Operator*,const mfem::Vector&,double*);
static void __mfemopt_snes_postcheck(mfem::Operator*,const mfem::Vector&,mfem::Vector&,mfem::Vector&,bool&,bool&);

namespace mfemopt
{
using namespace mfem;

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
   return w(c);
}

DiagonalMatrixCoefficient::DiagonalMatrixCoefficient(VectorCoefficient* _VQ, bool _own) : MatrixCoefficient(0) /* XXX the class have the explicit constructor */
{
   MFEM_VERIFY(_VQ,"Missing VectorCoefficient");
   MFEM_VERIFY(_VQ->GetVDim() > 0,"Invalid dim " << _VQ->GetVDim());
   VQ = _VQ;
   own = _own;
   w.SetSize(VQ->GetVDim());
   height = width = VQ->GetVDim();
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

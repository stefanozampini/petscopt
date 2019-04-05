#include <fstream>
#include <mfemopt/mfemextra.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <petscsf.h>

static void __mfemopt_snes_obj(mfem::Operator*,const mfem::Vector&,double*);
static void __mfemopt_snes_postcheck(mfem::Operator*,const mfem::Vector&,mfem::Vector&,mfem::Vector&,bool&,bool&);
static void __mfemopt_snes_update(mfem::Operator*,int,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&,const mfem::Vector&);

namespace mfemopt
{
using namespace mfem;

ReplicatedParMesh::ReplicatedParMesh(MPI_Comm comm, Mesh &mesh, int nrep, bool contig)
{
   PetscErrorCode ierr;
   PetscSubcomm   subcomm;
   MPI_Comm       rcomm;
   PetscMPIInt    size,crank;

   MFEM_VERIFY(nrep > 0,"Number of replicas should be positive");
   ierr = MPI_Comm_size(comm,&size); CCHKERRQ(comm,ierr);
   MFEM_VERIFY(!(size%nrep),"Size of comm must be a multiple of the number of replicas");

   ierr = PetscSubcommCreate(comm, &subcomm); CCHKERRQ(comm,ierr);
   ierr = PetscSubcommSetNumber(subcomm, (PetscInt)nrep); CCHKERRQ(comm,ierr);
   ierr = PetscSubcommSetType(subcomm, contig ? PETSC_SUBCOMM_CONTIGUOUS : PETSC_SUBCOMM_INTERLACED); CCHKERRQ(comm,ierr);

   color = subcomm->color;

   /* original comm */
   ierr = PetscCommDuplicate(comm,&parent_comm,NULL); CCHKERRQ(comm,ierr);
   /* comm for replicated mesh */
   ierr = PetscCommDuplicate(subcomm->child,&child_comm,NULL); CCHKERRQ(subcomm->child,ierr);
   /* reduction comm */
   ierr = MPI_Comm_rank(child_comm,&crank);CCHKERRQ(child_comm,ierr);
   ierr = MPI_Comm_split(parent_comm,crank,color,&rcomm);CCHKERRQ(comm,ierr);
   ierr = PetscCommDuplicate(rcomm,&red_comm,NULL); CCHKERRQ(rcomm,ierr);
   ierr = MPI_Comm_free(&rcomm);CCHKERRQ(PETSC_COMM_SELF,ierr);

   PetscMPIInt child_size, parent_size;
   ierr = MPI_Comm_size(child_comm, &child_size); CCHKERRQ(child_comm,ierr);
   ierr = MPI_Comm_size(parent_comm, &parent_size); CCHKERRQ(parent_comm,ierr);

   int *child_part = mesh.GeneratePartitioning(child_size, 1);
   child_mesh = new ParMesh(child_comm, mesh, child_part, 1);
   if (!contig) for (int i = 0; i < mesh.GetNE(); i++) child_part[i] *= nrep;
   parent_mesh = new ParMesh(parent_comm, mesh, child_part, 1);

   delete [] child_part;
   ierr = PetscSubcommDestroy(&subcomm); CCHKERRQ(comm,ierr);
}

ReplicatedParMesh::~ReplicatedParMesh()
{
   PetscErrorCode ierr;
   ierr = PetscCommDestroy(&parent_comm); CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = PetscCommDestroy(&child_comm); CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = PetscCommDestroy(&red_comm); CCHKERRQ(PETSC_COMM_SELF,ierr);
   delete child_mesh;
   delete parent_mesh;
}

ReplicatedParFiniteElementSpace::ReplicatedParFiniteElementSpace(ReplicatedParMesh *pm, const FiniteElementCollection *f,
                                                                 int dim, int ordering)
{
   cfes = new ParFiniteElementSpace(pm->GetChild(),f,dim,ordering);
   pfes = new ParFiniteElementSpace(pm->GetParent(),f,dim,ordering);

   /* SF for distribute/scatter */
   PetscErrorCode ierr;
   PetscInt nroots,nleaves,lsize;
   PetscSFNode *iremote;
   MPI_Comm red_comm;

   /* True dofs */
   lsize = (PetscInt)(cfes->GetTrueVSize());
   nroots = pm->IsMaster() ? lsize : 0;
   nleaves = lsize;
   ierr = PetscMalloc1(nleaves,&iremote); CCHKERRQ(PETSC_COMM_SELF,ierr);
   for (PetscInt i = 0; i < nleaves; i++)
   {
      iremote[i].rank  = 0;
      iremote[i].index = i;
   }
   red_comm = pm->GetRedComm();
   ierr = PetscSFCreate(red_comm,&red_T_sf); CCHKERRQ(red_comm,ierr);
   ierr = PetscSFSetGraph(red_T_sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER); PCHKERRQ(red_T_sf,ierr);
   ierr = PetscSFSetUp(red_T_sf); PCHKERRQ(red_T_sf,ierr);

   /* Vdofs */
   lsize = (PetscInt)(cfes->GetVSize());
   nroots = pm->IsMaster() ? lsize : 0;
   nleaves = lsize;
   ierr = PetscMalloc1(nleaves,&iremote); CCHKERRQ(PETSC_COMM_SELF,ierr);
   for (PetscInt i = 0; i < nleaves; i++)
   {
      iremote[i].rank  = 0;
      iremote[i].index = i;
   }
   red_comm = pm->GetRedComm();
   ierr = PetscSFCreate(red_comm,&red_V_sf); CCHKERRQ(red_comm,ierr);
   ierr = PetscSFSetGraph(red_V_sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER); PCHKERRQ(red_V_sf,ierr);
   ierr = PetscSFSetUp(red_V_sf); PCHKERRQ(red_V_sf,ierr);
}

void ReplicatedParFiniteElementSpace::Broadcast(const Vector& x, Vector &y)
{
   PetscErrorCode ierr;

   PetscInt nleaves,nroots;
   ierr = PetscSFGetGraph(red_V_sf,&nroots,&nleaves,NULL,NULL); PCHKERRQ(red_V_sf,ierr);
   MFEM_VERIFY(x.Size() >= nroots,"Invalid size for x: " << x.Size() << " < " << nroots);
   MFEM_VERIFY(y.Size() >= nleaves,"Invalid size for y: " << y.Size() << " < " << nleaves);

   double *xd,*yd;
   xd = x.GetData();
   yd = y.GetData();
   ierr = PetscSFBcastBegin(red_V_sf,MPI_DOUBLE_PRECISION,xd,yd); PCHKERRQ(red_V_sf,ierr);
   ierr = PetscSFBcastEnd(red_V_sf,MPI_DOUBLE_PRECISION,xd,yd); PCHKERRQ(red_V_sf,ierr);
}

void ReplicatedParFiniteElementSpace::Reduce(const Vector& x, Vector &y, MPI_Op op)
{
   PetscErrorCode ierr;

   PetscInt nleaves,nroots;
   ierr = PetscSFGetGraph(red_V_sf,&nroots,&nleaves,NULL,NULL); PCHKERRQ(red_V_sf,ierr);
   MFEM_VERIFY(x.Size() >= nleaves,"Invalid size for x: " << x.Size() << " < " << nleaves);
   MFEM_VERIFY(y.Size() >= nroots,"Invalid size for y: " << y.Size() << " < " << nroots);

   double *xd,*yd;
   xd = x.GetData();
   yd = y.GetData();
   ierr = PetscSFReduceBegin(red_V_sf,MPI_DOUBLE_PRECISION,xd,yd,op); PCHKERRQ(red_V_sf,ierr);
   ierr = PetscSFReduceEnd(red_V_sf,MPI_DOUBLE_PRECISION,xd,yd,op); PCHKERRQ(red_V_sf,ierr);
}

void ReplicatedParFiniteElementSpace::TBroadcast(const Vector& x, Vector &y)
{
   PetscErrorCode ierr;

   PetscInt nleaves,nroots;
   ierr = PetscSFGetGraph(red_T_sf,&nroots,&nleaves,NULL,NULL); PCHKERRQ(red_T_sf,ierr);
   MFEM_VERIFY(x.Size() >= nroots,"Invalid size for x: " << x.Size() << " < " << nroots);
   MFEM_VERIFY(y.Size() >= nleaves,"Invalid size for y: " << y.Size() << " < " << nleaves);

   double *xd,*yd;
   xd = x.GetData();
   yd = y.GetData();
   ierr = PetscSFBcastBegin(red_T_sf,MPI_DOUBLE_PRECISION,xd,yd); PCHKERRQ(red_T_sf,ierr);
   ierr = PetscSFBcastEnd(red_T_sf,MPI_DOUBLE_PRECISION,xd,yd); PCHKERRQ(red_T_sf,ierr);
}

void ReplicatedParFiniteElementSpace::TReduce(const Vector& x, Vector &y, MPI_Op op)
{
   PetscErrorCode ierr;

   PetscInt nleaves,nroots;
   ierr = PetscSFGetGraph(red_T_sf,&nroots,&nleaves,NULL,NULL); PCHKERRQ(red_T_sf,ierr);
   MFEM_VERIFY(x.Size() >= nleaves,"Invalid size for x: " << x.Size() << " < " << nleaves);
   MFEM_VERIFY(y.Size() >= nroots,"Invalid size for y: " << y.Size() << " < " << nroots);

   double *xd,*yd;
   xd = x.GetData();
   yd = y.GetData();
   ierr = PetscSFReduceBegin(red_T_sf,MPI_DOUBLE_PRECISION,xd,yd,op); PCHKERRQ(red_T_sf,ierr);
   ierr = PetscSFReduceEnd(red_T_sf,MPI_DOUBLE_PRECISION,xd,yd,op); PCHKERRQ(red_T_sf,ierr);
}

ReplicatedParFiniteElementSpace::~ReplicatedParFiniteElementSpace()
{
   PetscErrorCode ierr;
   ierr = PetscSFDestroy(&red_V_sf); CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = PetscSFDestroy(&red_T_sf); CCHKERRQ(PETSC_COMM_SELF,ierr);
   delete cfes;
   delete pfes;
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

void ParMeshPrint(ParMesh& mesh, const char* filename)
{
   PetscMPIInt rank;
   MPI_Comm_rank(mesh.GetComm(),&rank);

   std::ostringstream fname;
   fname << filename  << "." << std::setfill('0') << std::setw(6) << rank;

   std::ofstream oofs(fname.str().c_str());
   oofs.precision(8);

   mesh.Print(oofs);
}

void MeshGetElementsTagged(Mesh *mesh, bool (*tag_fn)(const Vector&), Array<bool>& tag, bool center)
{
   tag.SetSize(mesh->GetNE());
   tag = true;
   Vector pt(mesh->SpaceDimension());
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      if (center)
      {
         mesh->GetElementTransformation(e)->Transform(
            Geometries.GetCenter(mesh->GetElementBaseGeometry(e)), pt);
         tag[e] = (*tag_fn)(pt);
      }
      else
      {
         Array<int> vv;
         mesh->GetElementVertices(e,vv);
         for (int v = 0; v < vv.Size(); v++)
         {
            mesh->GetNode(vv[v], pt.GetData());
            tag[e] = tag[e] && (*tag_fn)(pt);
         }
      }
   }
}

void MeshGetElementsTagged(Mesh *mesh, const Array<int>& which_tag, Array<bool>& tag)
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
   MPI_Allreduce(lr,gr,2,MPI_INT,MPI_MAX,fes.GetParMesh()->GetComm());
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
   SetUpdate(__mfemopt_snes_update);
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

void __mfemopt_snes_update(mfem::Operator *op, int it, const mfem::Vector& X, const mfem::Vector& Y, const mfem::Vector &W, const mfem::Vector &Z)
{
   mfemopt::ReducedFunctional *rf = dynamic_cast<mfemopt::ReducedFunctional*>(op);
   MFEM_VERIFY(rf,"Not a mfemopt::ReducedFunctional operator");
   rf->Update(it,X,Y,W,Z);
}

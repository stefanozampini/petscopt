#include <fstream>
#include <mfemopt/mfemextra.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <mfem/linalg/petsc.hpp>
#include <mfem/fem/plinearform.hpp>
#include <mfem/fem/pbilinearform.hpp>
#include <petscsf.h>

namespace mfemopt
{
using namespace mfem;

ReplicatedParMesh::ReplicatedParMesh(MPI_Comm comm, Mesh &mesh, int nrep, bool contig, int **part) : drep(comm,nrep,contig)
{
   PetscErrorCode ierr;
   PetscMPIInt child_size, parent_size;
   ierr = MPI_Comm_size(drep.GetChildComm(), &child_size); CCHKERRQ(drep.GetChildComm(),ierr);
   ierr = MPI_Comm_size(drep.GetParentComm(), &parent_size); CCHKERRQ(drep.GetParentComm(),ierr);

   // TODO : Get rid of stupid parent mesh
   int *child_part = mesh.GeneratePartitioning(child_size, 1);
   child_mesh = new ParMesh(drep.GetChildComm(), mesh, child_part, 1);
   if (!contig) for (int i = 0; i < mesh.GetNE(); i++) child_part[i] *= nrep;
   parent_mesh = new ParMesh(drep.GetParentComm(), mesh, child_part, 1);

   if (part) *part = child_part;
   else delete [] child_part;
}

ReplicatedParMesh::~ReplicatedParMesh()
{
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

void FiniteElementSpaceGetRangeAndDerivType(FiniteElementSpace& fes, int* r, int* d)
{
   const FiniteElement* fe = fes.GetFE(0);
   if (r) *r = fe ? fe->GetRangeType() : INT_MIN;
   if (d) *d = fe ? fe->GetDerivType() : INT_MIN;
}

void FiniteElementSpaceGetRangeAndDerivMapType(FiniteElementSpace& fes, int* r, int* d)
{
   const FiniteElement* fe = fes.GetFE(0);
   if (r) *r = fe ? fe->GetMapType() : INT_MIN;
   if (d) *d = fe ? fe->GetDerivMapType() : INT_MIN;
}

void ParFiniteElementSpaceGetRangeAndDerivType(ParFiniteElementSpace& fes, int* r, int* d)
{
   int lr[2], gr[2];
   FiniteElementSpaceGetRangeAndDerivType(fes,lr,lr+1);
   /* reduce for empty meshes */
   MPI_Allreduce(lr,gr,2,MPI_INT,MPI_MAX,fes.GetParMesh()->GetComm());
   if (r) *r = gr[0];
   if (d) *d = gr[1];
}

void ParFiniteElementSpaceGetRangeAndDerivMapType(ParFiniteElementSpace& fes, int* r, int* d)
{
   int lr[2], gr[2];
   FiniteElementSpaceGetRangeAndDerivMapType(fes,lr,lr+1);
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

#if 0
void ProjectCoefficient_internal(Coefficient* Q, VectorCoefficient* VQ, ParGridFunction& proj)
{
  ParFiniteElementSpace *pfes = proj.ParFESpace();
  int map_type;
  ParFiniteElementSpaceGetRangeAndDerivMapType(*pfes,&map_type,NULL);

  BilinearFormIntegrator *mass_integ = NULL;
  LinearFormIntegrator *b_integ = NULL;
  if (map_type == FiniteElement::VALUE ||
      map_type == FiniteElement::INTEGRAL)
  {
     if (VQ)
     {
        mass_integ = new VectorMassIntegrator;
        b_integ = new VectorDomainLFIntegrator(*VQ);
     }
     else
     {
        mass_integ = new MassIntegrator;
        b_integ = new DomainLFIntegrator(*Q);
     }
  }
  else if (map_type == FiniteElement::H_DIV ||
           map_type == FiniteElement::H_CURL)
  {
     if (Q)
     {
        MFEM_ABORT("Not supported scalar + vectorFE");
     }
     else
     {
       mass_integ = new VectorFEMassIntegrator;
       b_integ = new VectorFEDomainLFIntegrator(*VQ);
     }
  }
  else
  {
     MFEM_ABORT("unknown type of FE space");
  }
  if (Q)
  {
     proj.ProjectDiscCoefficient(*Q,GridFunction::ARITHMETIC);
  }
  else
  {
     proj.ProjectDiscCoefficient(*VQ,GridFunction::ARITHMETIC);
  }
  Vector B(pfes->GetTrueVSize());
  Vector X(pfes->GetTrueVSize());
  ParLinearForm b(pfes);
  b.AddDomainIntegrator(b_integ);
  b.Assemble();
  b.ParallelAssemble(B);

  OperatorHandle A(Operator::PETSC_MATAIJ);
  ParBilinearForm a(pfes);
  a.AddDomainIntegrator(mass_integ);
  a.Assemble(0);
  a.Finalize(0);
  a.ParallelAssemble(A);
  proj.ParallelProject(X);
  PetscPCGSolver cg(pfes->GetParMesh()->GetComm(),"coeff_l2_");
  cg.SetOperator(*A.Ptr());
  cg.iterative_mode = true;
  cg.Mult(B, X);
  proj.Distribute(X);
}

void ProjectCoefficient(Coefficient& Q, ParGridFunction& proj)
{
   ProjectCoefficient_internal(&Q,NULL,proj);
}

void ProjectCoefficient(VectorCoefficient& VQ, ParGridFunction& proj)
{
   ProjectCoefficient_internal(NULL,&VQ,proj);
}
#endif

SymmetricSolver::SymmetricSolver(Solver *solver, bool owner, Operator* op, bool opowner)
{
  height = solver->Height();
  width = solver->Width();
  isolver = solver;
  ownsolver = owner;
  if (op) MFEM_VERIFY(height == op->Height() && width == op->Width(),"Invalid operator sizes: height " << height << " " << op->Height() << ", width " << width << " " << op->Width());
  iop = op;
  ownop = opowner;
}

void SymmetricSolver::SetOperator(const Operator &op)
{
  mfem_error("SymmetricSolver::SetOperator(const Operator &op) is not intended to be used");
}

void SymmetricSolver::Mult(const Vector &b, Vector& x) const
{
  isolver->Mult(b,x);
}

void SymmetricSolver::MultTranspose(const Vector &b, Vector& x) const
{
  isolver->Mult(b,x);
}

SymmetricSolver::~SymmetricSolver()
{
  if (ownop) delete iop;
  if (ownsolver) delete isolver;
}

}

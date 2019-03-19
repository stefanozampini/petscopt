static const char help[] = "Tests Replicated{ParMesh|ParFiniteElementSpace}.";

#include <mfemopt.hpp>

typedef enum {FEC_L2, FEC_H1, FEC_HCURL, FEC_HDIV} FECType;
static const char *FECTypes[] = {"L2","H1","HCURL","HDIV","FecType","FEC_",0};

using namespace mfem;
using namespace mfemopt;

int dim;
double kappa = 1.0;

double Es_fun(const Vector &x)
{
   double v = 1.0;
   for (int i = 0; i < x.Size(); i++) v *= sin(kappa * x(i));
   return v;
}

void Ev_fun(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

int main(int argc, char *argv[])
{
   MFEMInitializePetsc(&argc,&argv,NULL,help);

   char      meshfile[PETSC_MAX_PATH_LEN] = "../../../../share/petscopt/meshes/inline_quad.mesh";
   PetscInt  srl = 0, nrep = 1;
   PetscBool contig = PETSC_TRUE, nc_mesh = PETSC_FALSE;
   FECType   fec_type = FEC_H1;
   bool      scal = true;
   double    norm;

   /* Process options */
   {
      PetscErrorCode ierr;

      ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for the tests",NULL);CHKERRQ(ierr);
      ierr = PetscOptionsString("-meshfile","Mesh filename",NULL,meshfile,meshfile,sizeof(meshfile),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-srl","Number of sequential refinements",NULL,srl,&srl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-nrep","Number of replicas",NULL,nrep,&nrep,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-contig","Contiguous or interleaved splitting",NULL,contig,&contig,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-nc_mesh","NC mesh",NULL,nc_mesh,&nc_mesh,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnum("-fec_type","FEC to be tested","",FECTypes,(PetscEnum)fec_type,(PetscEnum*)&fec_type,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
   }

   Mesh *mesh = new Mesh(meshfile, 1, 0);
   for (int lev = 0; lev < srl; lev++)
   {
      mesh->UniformRefinement();
   }
   if (nc_mesh) mesh->EnsureNCMesh();
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   ReplicatedParMesh *pmesh = new ReplicatedParMesh(PETSC_COMM_WORLD,*mesh,nrep,contig);
   delete mesh;

#if 0
   ParMeshPrint(*(pmesh->GetParent()),"parent");

   std::stringstream cfname;
   cfname << "child_" << pmesh->GetColor();
   ParMeshPrint(*(pmesh->GetChild()),cfname.str().c_str());
#endif

   FiniteElementCollection *fec = NULL;
   switch (fec_type)
   {
      case FEC_L2:
         scal = true;
         fec = new L2_FECollection(2,dim);
         break;
      case FEC_H1:
         scal = true;
         fec = new H1_FECollection(2,dim);
         break;
      case FEC_HCURL:
         scal = false;
         fec = new ND_FECollection(2,dim);
         break;
      case FEC_HDIV:
         scal = false;
         fec = new RT_FECollection(2,dim);
         break;
      default:
         MFEM_ABORT("Unhandled FEC Type");
         break;
   }

   ReplicatedParFiniteElementSpace *pfes = new ReplicatedParFiniteElementSpace(pmesh,fec);

   ParFiniteElementSpace *ppfes = pfes->GetParent();
   ParFiniteElementSpace *cpfes = pfes->GetChild();

   ParGridFunction *x_ex = new ParGridFunction(ppfes);
   PetscParVector *px_ex = new PetscParVector(ppfes);
   if (scal)
   {
      FunctionCoefficient E(Es_fun);
      x_ex->ProjectCoefficient(E);
      x_ex->ParallelProject(*px_ex);
   }
   else
   {
      VectorFunctionCoefficient E(sdim, Ev_fun);
      x_ex->ProjectCoefficient(E);
      x_ex->ParallelProject(*px_ex);
   }

   ParGridFunction *x = new ParGridFunction(ppfes);
   ParGridFunction *y = new ParGridFunction(cpfes);
   PetscParVector *px = new PetscParVector(ppfes);
   PetscParVector *py = new PetscParVector(cpfes);

   *y = PETSC_MAX_REAL;
   *py = PETSC_MAX_REAL;

   /* Broadcast/Reduce GridFunction */
   *x = 0.0;
   pfes->Broadcast(*x_ex,*y);
   pfes->Reduce(*y,*x);
   *x *= -1.0/nrep;
   *x += *x_ex;
   norm = ParNormlp(*x,2,PETSC_COMM_WORLD);
   MFEM_VERIFY(norm < PETSC_SMALL,"Error vdofs " << norm);

   /* Broadcast/Reduce distributed vector */
   *px = 0.0;
   pfes->TBroadcast(*px_ex,*py);
   pfes->TReduce(*py,*px);
   *px *= -1.0/nrep;
   *px += *px_ex;
   norm = ParNormlp(*px,2,PETSC_COMM_WORLD);
   MFEM_VERIFY(norm < PETSC_SMALL,"Error true dofs " << norm);

   /* Use replicator class */
   DataReplicator *drep = new DataReplicator(PETSC_COMM_WORLD,nrep,contig);

   *y = PETSC_MAX_REAL;
   *py = PETSC_MAX_REAL;

   double f_ex = 1.6792302;
   double f1, f2 = PETSC_MAX_REAL;

   *x = 0.0;
   drep->Broadcast("gridf",*x_ex,*y);
   drep->Reduce("gridf",*y,*x);
   *x *= -1.0/nrep;
   *x += *x_ex;
   norm = ParNormlp(*x,2,PETSC_COMM_WORLD);
   MFEM_VERIFY(norm < PETSC_SMALL,"Error drep " << norm);

   *px = 0.0;
   drep->Broadcast("distv",*px_ex,*py);
   drep->Reduce("distv",*py,*px);
   *px *= -1.0/nrep;
   *px += *px_ex;
   norm = ParNormlp(*px,2,PETSC_COMM_WORLD);
   MFEM_VERIFY(norm < PETSC_SMALL,"Error drep " << norm);

   f1 = 0.0;
   drep->Broadcast(f_ex,&f2);
   drep->Reduce(f2,&f1);
   f1 = drep->IsMaster() ? f1 - f_ex*nrep : f1;
   f1 = std::abs(f1);
   MFEM_VERIFY(f1 < PETSC_SMALL,"Error drep " << f1);

   int m = 3,n = 5;
   DenseMatrix work(m,n), lwork;
   for (int j = 0; j < n; j++)
     for (int i = 0; i < m; i++)
       work(i,j) = 100.*j + i;
   drep->Split(work,lwork);
   norm = 0.0;
   f1 = lwork.FNorm2();
   drep->Reduce(f1,&norm);
   norm = drep->IsMaster() ? norm - work.FNorm2() : 0.0;
   norm = std::abs(norm);
   MFEM_VERIFY(norm < PETSC_SMALL,"Error split " << norm);

   delete drep;

   delete x_ex;
   delete x;
   delete y;
   delete px_ex;
   delete px;
   delete py;
   delete pfes;
   delete fec;
   delete pmesh;
   MFEMFinalizePetsc();
   return 0;
}

/*TEST

   build:
     requires: mfemopt

   test:
     suffix: rep1
     output_file: output/ex4.out
     nsize: {{1 2}}
     args: -meshfile ${petscopt_dir}/share/petscopt/meshes/inline_quad.mesh -contig {{0 1}} -fec_type {{L2 H1 HCURL HDIV}}

   test:
     suffix: repn
     output_file: output/ex4.out
     nsize: 3
     args: -meshfile ${petscopt_dir}/share/petscopt/meshes/inline_quad.mesh -contig {{0 1}} -nrep 3 -fec_type {{L2 H1 HCURL HDIV}}
TEST*/

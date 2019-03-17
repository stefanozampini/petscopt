static const char help[] = "Tests Replicated{ParMesh|ParFiniteElementSpace}.";

//include <petscopt.h>
#include <mfemopt.hpp>

//#include <mfem.hpp>

//#include <iostream>
//#include <sstream>

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
   PetscBool contig = PETSC_TRUE;
   FECType   fec_type = FEC_H1;
   bool      scal = true;

   /* Process options */
   {
      PetscErrorCode ierr;

      ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for the tests",NULL);CHKERRQ(ierr);
      ierr = PetscOptionsString("-meshfile","Mesh filename",NULL,meshfile,meshfile,sizeof(meshfile),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-srl","Number of sequential refinements",NULL,srl,&srl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-nrep","Number of replicas",NULL,nrep,&nrep,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-contig","Contiguous or interleaved splitting",NULL,contig,&contig,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnum("-fec_type","FEC to be tested","",FECTypes,(PetscEnum)fec_type,(PetscEnum*)&fec_type,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
   }

   Mesh *mesh = new Mesh(meshfile, 1, 0);
   for (int lev = 0; lev < srl; lev++)
   {
      mesh->UniformRefinement();
   }
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
      x_ex->ParallelAverage(*px_ex);
   }
   else
   {
      VectorFunctionCoefficient E(sdim, Ev_fun);
      x_ex->ProjectCoefficient(E);
      x_ex->ParallelAverage(*px_ex);
   }

   ParGridFunction *x = new ParGridFunction(ppfes);
   ParGridFunction *y = new ParGridFunction(cpfes);
   PetscParVector *px = new PetscParVector(ppfes);
   PetscParVector *py = new PetscParVector(cpfes);

   *x = *x_ex;
   *x *= (-1.0*nrep);
   *y = PETSC_MAX_REAL;

   *px = *px_ex;
   *px *= (-1.0*nrep);
   *py = PETSC_MAX_REAL;

   /* Broadcast/Reduce GridFunction */
   pfes->Broadcast(*x_ex,*y);
   pfes->Reduce(*y,*x);
   double norm = ParNormlp(*x,2,PETSC_COMM_WORLD);
   MFEM_VERIFY(norm < PETSC_SMALL,"Error vdofs " << norm);

   /* Broadcast/Reduce distributed vector */
   pfes->TBroadcast(*px_ex,*py);
   pfes->TReduce(*py,*px);
   double pnorm = ParNormlp(*px,2,PETSC_COMM_WORLD);
   MFEM_VERIFY(pnorm < PETSC_SMALL,"Error true dofs " << pnorm);

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

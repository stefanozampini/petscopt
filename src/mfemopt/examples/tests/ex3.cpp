static const char help[] = "Tests PDOperator assembly.";

#include <petscopt.h>
#include <mfemopt.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>

#include <mfem.hpp>

#include <iostream>
#include <sstream>

using namespace mfem;
using namespace mfemopt;

typedef enum {FEC_L2, FEC_H1, FEC_HCURL, FEC_HDIV} FECType;
static const char *FECTypes[] = {"L2","H1","HCURL","HDIV","FecType","FEC_",0};
// TODO
//typedef enum {SIGMA_NONE, SIGMA_SCALAR, SIGMA_DIAG, SIGMA_FULL} SIGMAType;
//static const char *SIGMATypes[] = {"NONE","SCALAR","DIAG","FULL","SigmaType","SIGMA_",0};

static int refine_fn(const Vector &x)
{
   for (int d = 0; d < x.Size(); d++) if (x(d) < 0.25 || x(d) > 0.75) return 0;
   return 1;
}

void NCRefinement(Mesh *mesh, int (*fn)(const Vector&))
{
   Vector pt(mesh->SpaceDimension());
   Array<int> el_to_refine;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      mesh->GetElementTransformation(e)->Transform(
         Geometries.GetCenter(mesh->GetElementBaseGeometry(e)), pt);
      int refineme = (*fn)(pt);
      if (refineme) el_to_refine.Append(e);
   }
   mesh->GeneralRefinement(el_to_refine,1);
}

static bool excl_fn(const Vector &x)
{
   for (int d = 0; d < x.Size(); d++) if (x(d) < 0.2 || x(d) > 0.8) return true;
   return false;
}

static double mu_fn(const Vector &x)
{
   double v = 1.0;
   for (int i = 0; i < x.Size(); i++) v += x(i)*x(i);
   return v;
}

static void mu_vec_fn(const Vector& x,Vector& V)
{
   V = 0.0;
   for (int i = 0; i < std::min(V.Size(),x.Size()); i++)
   {
      V(i) = x(i)*x(i);
   }
}

static void mu_mat_fn(const Vector& x,DenseMatrix& K)
{
   K = 0.0;
   for (int i = 0; i < std::min(K.Height(),x.Size()); i++)
   {
      for (int j = 0; j < std::min(K.Width(),x.Size()); j++)
      {
         K(i,j) = x(i)*x(j);
      }
   }
}

class TestOperator
{
private:
   ParBilinearForm pbform;
   ParMixedBilinearForm pbformmix;
   OperatorHandle  Mh;
   OperatorHandle  Mhmix;
public:
   TestOperator(ParFiniteElementSpace*,BilinearFormIntegrator*,Operator::Type = Operator::PETSC_MATAIJ);
   Operator* GetOperator();
   Operator* GetOperatorMixed();
   ~TestOperator();
};

TestOperator::TestOperator(ParFiniteElementSpace *s_fes, BilinearFormIntegrator *pd_bilin, Operator::Type oid) : pbform(s_fes), pbformmix(s_fes,s_fes), Mh(oid), Mhmix(oid)
{
   pbform.AddDomainIntegrator(pd_bilin);
   pbformmix.AddDomainIntegrator(pd_bilin);
}

Operator* TestOperator::GetOperator()
{
   pbform.Update();
   pbform.Assemble(0);
   pbform.Finalize(0);
   pbform.ParallelAssemble(Mh);

   Operator *Mop;
   Mh.Get(Mop);
   return Mop;
}

Operator* TestOperator::GetOperatorMixed()
{
   pbformmix.Update();
   pbformmix.Assemble(0);
   pbformmix.Finalize(0);
   pbformmix.ParallelAssemble(Mhmix);

   Operator *Mop;
   Mhmix.Get(Mop);
   return Mop;
}

TestOperator::~TestOperator()
{
   /* MFEM gets ownership of the BilinearFormIntegrators */
   {
      Array<BilinearFormIntegrator*> *dbfi = pbform.GetDBFI();
      for (int i = 0; i < dbfi->Size(); i++) (*dbfi)[i] = NULL;
   }
   {
      Array<BilinearFormIntegrator*> *dbfi = pbformmix.GetDBFI();
      for (int i = 0; i < dbfi->Size(); i++) (*dbfi)[i] = NULL;
   }
}

static void PetscParMatrixInftyNorm(PetscParMatrix& op_M,PetscReal *norm)
{
   PetscErrorCode ierr;
   ierr = MatNorm(op_M,NORM_INFINITY,norm); PCHKERRQ(op_M,ierr);
}

static void RunTest(ParFiniteElementSpace *fes, BilinearFormIntegrator *bilin, PDBilinearFormIntegrator *bilin_pd, const std::string& name)
{

   TestOperator    op(fes,bilin);
   TestOperator op_pd(fes,bilin_pd);

   PetscParMatrix      op_M(PETSC_COMM_WORLD,   op.GetOperator(),Operator::PETSC_MATAIJ);
   PetscParMatrix   op_M_pd(PETSC_COMM_WORLD,op_pd.GetOperator(),Operator::PETSC_MATAIJ);
   PetscParMatrix    op_M_m(PETSC_COMM_WORLD,   op.GetOperatorMixed(),Operator::PETSC_MATAIJ);
   PetscParMatrix op_M_pd_m(PETSC_COMM_WORLD,op_pd.GetOperatorMixed(),Operator::PETSC_MATAIJ);

   PetscReal norm;
   Mat T;
   std::string oname;

   T = op_M;
   oname = name + "-op_M";
   PetscObjectSetName((PetscObject)T,oname.c_str());
   MatViewFromOptions(T,NULL,"-test_view");

   T = op_M_pd;
   oname = name + "-op_M_pd";
   PetscObjectSetName((PetscObject)T,oname.c_str());
   MatViewFromOptions(T,NULL,"-test_view");

   T = op_M_m;
   oname = name + "-op_M_m";
   PetscObjectSetName((PetscObject)T,oname.c_str());
   MatViewFromOptions(T,NULL,"-test_view");

   T = op_M_pd_m;
   oname = name + "-op_M_pd_m";
   PetscObjectSetName((PetscObject)T,oname.c_str());
   MatViewFromOptions(T,NULL,"-test_view");

   op_M_pd -= op_M;
   PetscParMatrixInftyNorm(op_M_pd,&norm);
   PetscPrintf(PETSC_COMM_WORLD,"[%s] ||   op_M_pd - op_M||: %g\n",name.c_str(),(double)norm);

   T = op_M_pd;
   oname = name + "-diff-op_M_pd";
   PetscObjectSetName((PetscObject)T,oname.c_str());
   MatViewFromOptions(T,NULL,"-test_diff_view");

   op_M_m -= op_M;
   PetscParMatrixInftyNorm(op_M_m,&norm);
   PetscPrintf(PETSC_COMM_WORLD,"[%s] ||    op_M_m - op_M||: %g\n",name.c_str(),(double)norm);

   T = op_M_m;
   oname = name + "-diff-op_M_m";
   PetscObjectSetName((PetscObject)T,oname.c_str());
   MatViewFromOptions(T,NULL,"-test_diff_view");

   op_M_pd_m -= op_M;
   PetscParMatrixInftyNorm(op_M_pd_m,&norm);
   PetscPrintf(PETSC_COMM_WORLD,"[%s] || op_M_pd_m - op_M||: %g\n",name.c_str(),(double)norm);

   T = op_M_pd_m;
   oname = name + "-diff-op_M_pd_m";
   PetscObjectSetName((PetscObject)T,oname.c_str());
   MatViewFromOptions(T,NULL,"-test_diff_view");
}

int main(int argc, char *argv[])
{
   MFEMInitializePetsc(&argc,&argv,NULL,help);

   char meshfile[PETSC_MAX_PATH_LEN] = "../../../../share/petscopt/meshes/inline_quad.mesh";
   PetscInt srl = 0, prl = 0, ncrl = 0;

   FECType   s_fec_type = FEC_H1;
   PetscInt  s_ord = 1;

   FECType   mu_fec_type = FEC_H1;
   PetscInt  mu_ord = 1;
   PetscInt  n_mu_excl = 1024;
   PetscInt  mu_excl[1024];
   PetscBool mu_excl_fn = PETSC_FALSE;

   PetscBool glvis = PETSC_FALSE, test_part = PETSC_FALSE, test_progress = PETSC_TRUE;

   /* Process options */
   {
      PetscBool      flg;
      PetscErrorCode ierr;

      ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for the tests",NULL);CHKERRQ(ierr);
      /* Mesh */
      ierr = PetscOptionsString("-meshfile","Mesh filename",NULL,meshfile,meshfile,sizeof(meshfile),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-srl","Number of sequential refinements",NULL,srl,&srl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-prl","Number of parallel refinements",NULL,prl,&prl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-ncrl","Number of non-conforming refinements (refines element with center in [0.25,0.25]^d)",NULL,ncrl,&ncrl,NULL);CHKERRQ(ierr);

      /* State space */
      ierr = PetscOptionsInt("-state_ord","Polynomial order approximation for the state",NULL,s_ord,&s_ord,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnum("-state_fec_type","FEC for the state","",FECTypes,(PetscEnum)s_fec_type,(PetscEnum*)&s_fec_type,NULL);CHKERRQ(ierr);

      /* Parameter space */
      ierr = PetscOptionsInt("-mu_ord","Polynomial order approximation for mu",NULL,mu_ord,&mu_ord,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnum("-mu_fec_type","FEC for mu","",FECTypes,(PetscEnum)mu_fec_type,(PetscEnum*)&mu_fec_type,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-mu_exclude_fn","Excludes elements outside [0.5,0.5]^d for mu optimization",NULL,mu_excl_fn,&mu_excl_fn,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray("-mu_exclude","Elements' tag to exclude for mu optimization",NULL,mu_excl,&n_mu_excl,&flg);CHKERRQ(ierr);
      if (!flg) n_mu_excl = 0;

      ierr = PetscOptionsBool("-glvis","Activate GLVis monitoring",NULL,glvis,&glvis,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-test_partitioning","Test with a fixed element partition",NULL,test_part,&test_part,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-test_progress","Report progress when testing",NULL,test_progress,&test_progress,NULL);CHKERRQ(ierr);

      ierr = PetscOptionsEnd();CHKERRQ(ierr);
   }

   Array<int> mu_excl_a((int)n_mu_excl);
   for (int i = 0; i < n_mu_excl; i++) mu_excl_a[i] = (int)mu_excl[i];

   /* Create mesh and finite element space for the independent variable */
   ParMesh *pmesh = NULL;
   {
      Mesh *mesh = new Mesh(meshfile, 1, 1);
      for (int lev = 0; lev < srl; lev++)
      {
         mesh->UniformRefinement();
      }
      if (ncrl) mesh->EnsureNCMesh();
      for (int lev = 0; lev < ncrl; lev++)
      {
         NCRefinement(mesh,refine_fn);
      }

      if (test_part)
      {
         pmesh = ParMeshTest(PETSC_COMM_WORLD, *mesh);
      }
      else
      {
         pmesh = new ParMesh(PETSC_COMM_WORLD, *mesh);
      }
      delete mesh;
      for (int lev = 0; lev < prl; lev++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   /* Optimization space */
   FiniteElementCollection *s_fec = NULL;
   bool s_scal = false;
   switch (s_fec_type)
   {
      case FEC_L2:
         s_scal = true;
         s_fec = new L2_FECollection(s_ord,pmesh->Dimension());
         break;
      case FEC_H1:
         s_scal = true;
         s_fec = new H1_FECollection(s_ord,pmesh->Dimension());
         break;
      case FEC_HCURL:
         s_scal = false;
         s_fec = new ND_FECollection(s_ord,pmesh->Dimension());
         break;
      case FEC_HDIV:
         s_scal = false;
         s_fec = new RT_FECollection(s_ord,pmesh->Dimension());
         break;
      default:
         MFEM_ABORT("Unhandled FEC Type");
         break;
   }

   MatrixFunctionCoefficient *mu_mat = new MatrixFunctionCoefficient(pmesh->SpaceDimension(),mu_mat_fn);
   VectorFunctionCoefficient *mu_vec = new VectorFunctionCoefficient(pmesh->SpaceDimension(),mu_vec_fn);
   FunctionCoefficient *mu = new FunctionCoefficient(mu_fn);

   /* Optimization space */
   bool mu_scal = false;
   FiniteElementCollection *mu_fec = NULL;
   switch (mu_fec_type)
   {
      case FEC_L2:
         mu_scal = true;
         mu_fec = new L2_FECollection(mu_ord,pmesh->Dimension());
         break;
      case FEC_H1:
         mu_scal = true;
         mu_fec = new H1_FECollection(mu_ord,pmesh->Dimension());
         break;
      case FEC_HCURL:
         mu_scal = false;
         mu_fec = new ND_FECollection(mu_ord,pmesh->Dimension());
         break;
      case FEC_HDIV:
         mu_scal = false;
         mu_fec = new RT_FECollection(mu_ord,pmesh->Dimension());
         break;
      default:
         MFEM_ABORT("Unhandled FEC Type");
         break;
   }

   if (!PetscGlobalRank)
   {
      std::cout << "Using param fec type " << FECTypes[mu_fec_type] << ", order " << mu_ord << std::endl;
      std::cout << "Using state fec type " << FECTypes[ s_fec_type] << ", order " <<  s_ord << std::endl;
   }

   /* The parameter dependent coefficient for mu */
   PDCoefficient *mu_pd = NULL;
   BilinearFormIntegrator *mu_bilin = NULL;
   PDBilinearFormIntegrator *mu_bilin_pd = NULL;

   std::string testname;

   /* scalar space for param */
   if (mu_scal)
   {
      if (mu_excl_fn) mu_pd = new PDCoefficient(*mu,pmesh,mu_fec,excl_fn);
      else            mu_pd = new PDCoefficient(*mu,pmesh,mu_fec,mu_excl_a);
      if (glvis) mu_pd->Visualize();

      ParFiniteElementSpace* fes = new ParFiniteElementSpace(pmesh,s_fec);

      if (s_scal) /* test scalar space integrators */
      {
         testname    = "SP_SS";
         mu_bilin    = new MassIntegrator(*mu);
         mu_bilin_pd = new PDMassIntegrator(*mu_pd);
      }
      else /* test vector space integrators */
      {
         testname    = "SP_VS";
         mu_bilin    = new VectorFEMassIntegrator(*mu);
         mu_bilin_pd = new PDVectorFEMassIntegrator(*mu_pd);
      }

      RunTest(fes,mu_bilin,mu_bilin_pd,testname);

      delete mu_pd;
      delete mu_bilin;
      delete mu_bilin_pd;

      /* test vector coefficient with vector integrators */
      if (!s_scal)
      {
         if (mu_excl_fn) mu_pd = new PDCoefficient(*mu_vec,pmesh,mu_fec,excl_fn);
         else            mu_pd = new PDCoefficient(*mu_vec,pmesh,mu_fec,mu_excl_a);
         if (glvis) mu_pd->Visualize();

         mu_bilin    = new VectorFEMassIntegrator(*mu_vec);
         mu_bilin_pd = new PDVectorFEMassIntegrator(*mu_pd);

         testname += "_VI";
         RunTest(fes,mu_bilin,mu_bilin_pd,testname);

         delete mu_pd;
         delete mu_bilin;
         delete mu_bilin_pd;
      }
      delete fes;
   }
   else if (!s_scal) /* vector space for param */ /* TODO: remove !s_scal when PDVectorMassIntegrator is implemented */
   {
      if (mu_excl_fn) mu_pd = new PDCoefficient(*mu_vec,pmesh,mu_fec,excl_fn);
      else            mu_pd = new PDCoefficient(*mu_vec,pmesh,mu_fec,mu_excl_a);
      if (glvis) mu_pd->Visualize();

      ParFiniteElementSpace* fes = NULL;
#if 0
      if (s_scal)
      {
         testname    = "VP_SS";
         fes         = new ParFiniteElementSpace(pmesh,s_fec,pmesh->Dimension());
         mu_bilin    = new VectorMassIntegrator(*mu);
         mu_bilin_pd = new PDVectorMassIntegrator(*mu_pd);
      }
      else
#endif
      {
         testname    = "VP_VS";
         fes         = new ParFiniteElementSpace(pmesh,s_fec);
         mu_bilin    = new VectorFEMassIntegrator(*mu_vec);
         mu_bilin_pd = new PDVectorFEMassIntegrator(*mu_pd);
      }

      RunTest(fes,mu_bilin,mu_bilin_pd,testname);

      delete mu_pd;
      delete mu_bilin;
      delete mu_bilin_pd;
      delete fes;
   }

   delete mu;
   delete mu_vec;
   delete mu_mat;
   delete mu_fec;
   delete s_fec;
   delete pmesh;

   MFEMFinalizePetsc();
   return 0;
}

/*TEST

   build:
     requires: mfemopt

   test:
     suffix: conforming
     nsize: 2
     args: -glvis 0 -meshfile ${petscopt_dir}/share/petscopt/meshes/inline_quad.mesh -state_fec_type {{H1 HCURL HDIV L2}separate output} -state_ord 2 -mu_fec_type {{H1 HCURL HDIV L2}separate output} -mu_ord 2

   test:
     suffix: nonconforming
     nsize: 1
     args: -glvis 0 -meshfile ${petscopt_dir}/share/petscopt/meshes/inline_quad.mesh -state_fec_type {{H1 HCURL HDIV L2}separate output} -state_ord 2 -mu_fec_type {{H1 HCURL HDIV L2}separate output} -mu_ord 2 -ncrl 1

TEST*/

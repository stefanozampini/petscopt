static const char help[] = "A simple total-variation based, primal-dual image regularization.";
// Example runs:
// - mpiexec -n 4  ./ex1 -image ${PETSCOPT_DIR}/share/petscopt/data/img_medium.bmp -monitor -primaldual -noise 0.1 -tv_alpha 0.001 -tv_beta 0.001 -ksp_type cg -pc_type gamg -primaldual 1 -symmetrize 1

#include <mfemopt.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <mfem.hpp>
#include <limits>

using namespace mfem;
using namespace mfemopt;

/* an Image is represented through the MFEM Coefficient class */
class Image : public Coefficient
{
private:
   /* double precision pixel representation */
   double *data;
   int    nex,ney;
   double hx,hy;

   /* FEM representation */
   FiniteElementCollection *fec;
   ParFiniteElementSpace   *pfes;
   ParMesh                 *pmesh;

   /* read implementations */
   void ReadBMP(const char*,bool=true);
   void ReadTXT(const char*);


public:
   Image() : data(NULL), nex(0), ney(0), fec(NULL), pfes(NULL), pmesh(NULL) {}
   Image(MPI_Comm,const char*,int=1,bool=true,bool=false);

   virtual double Eval(ElementTransformation&,
                       const IntegrationPoint&);

   PDCoefficient* CreatePDCoefficient();
   ParFiniteElementSpace* ParFESpace() { return pfes; }

   void AddNoise(double);
   void Normalize();
   void Save(const char*);
   void Visualize(const char*);

   virtual ~Image() { delete[] data; delete fec; delete pfes; delete pmesh; }
};

Image::Image(MPI_Comm comm, const char* filename, int ord, bool quad, bool test_part)
{
   nex = ney = 0;
   std::string fname(filename);
   size_t lastdot = fname.find_last_of(".");
   std::string fext = fname.substr(lastdot);
   if (fext == ".bmp")
   {
      ReadBMP(filename,true);
   }
   else if (fext == ".txt")
   {
      ReadTXT(filename);
   }
   else
   {
      std::cout << "Unkwnown extension (" << fext << ") in " << fname << std::endl;
      mfem_error("Unsupported filename");
   }

   double Lx = double(ney)/double(nex);
   double Ly = 1.0;

   hx = Lx/(nex-1.0);
   hy = Ly/(ney-1.0);

   Mesh *mesh = new Mesh(nex,ney,quad ? Element::QUADRILATERAL : Element::TRIANGLE,1,Lx,Ly);

   /* For testing purposes, we speficy a partitioning */
   if (test_part)
   {
      pmesh = ParMeshTest(comm,*mesh);
   }
   else
   {
      pmesh = new ParMesh(comm,*mesh);
   }
   delete mesh;

   fec = new H1_FECollection(std::max(ord,1), 2);
   pfes = new ParFiniteElementSpace(pmesh, fec, 1, Ordering::byVDIM);

}

void Image::Normalize()
{
   double im = std::numeric_limits<double>::max();
   double iM = std::numeric_limits<double>::min();
   for (int i = 0; i < ney*nex; i++)
   {
      im = data[i] < im ? data[i] : im;
      iM = data[i] > iM ? data[i] : iM;
   }
   for (int i = 0; i < ney*nex; i++) data[i] = (data[i] - im)/(iM - im);
}

void Image::AddNoise(double s)
{
   GaussianNoise noise;
   Vector vnoise;
   noise.random(vnoise,nex*ney);
   vnoise *= s;
   for (int i = 0; i < ney*nex; i++) data[i] += vnoise[i];
}

/* the evaluation routine */
double Image::Eval(ElementTransformation &T,
                   const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);
   int ix = std::floor(x[0]/hx);
   int iy = std::floor(x[1]/hy);
   MFEM_VERIFY(0 <= ix && ix < nex,"Wrong ix " << ix);
   MFEM_VERIFY(0 <= iy && iy < ney,"Wrong iy " << iy);
   return data[iy + ix*ney];
}

void Image::ReadTXT(const char* filename)
{
   std::ifstream input(filename);
   MFEM_VERIFY(!input.fail(),"Missing file " << filename);

   input >> nex;
   input >> ney;

   data = new double[nex*ney];
   for (int i = 0; i < nex; i++)
   {
     for (int j = 0; j < ney; j++)
     {
        input >> data[j + i*ney];
     }
   }
}

void Image::ReadBMP(const char* filename, bool grayscale)
{
   FILE* f = fopen(filename, "rb");
   MFEM_VERIFY(f,"Unable to open file");

   // Read 54-bytes header
   unsigned char info[54];
   size_t bread = fread(info, sizeof(unsigned char), 54, f);
   MFEM_VERIFY(54 == bread,"Error in fread: 54 expected items, read instead: " << bread);
   int nx = *(int*)&info[18];
   int ny = *(int*)&info[22];
   short bits = *(short*)&info[28];
   MFEM_VERIFY(bits/8 && !(bits%8),"Unsupported number of bits per pixel " << bits);
   int bb = bits/8;

   int vdim = grayscale ? 1 : bits/8;
   nex = nx;
   ney = std::abs(ny);

   data = new double[nex*ney*vdim];
   for (int i = 0; i < nex*ney*vdim; i++) data[i] = 0.0;

   int rowsize = std::ceil((bits*nex)/32)*4;
   unsigned char* bmpdata = new unsigned char[rowsize];

   for (int i = 0; i < ney; i++)
   {
      size_t err = fread(bmpdata,sizeof(unsigned char),rowsize,f);
      MFEM_VERIFY(err == (size_t)rowsize,"Unexpected EOF " << i << ": read " << err << ", expected " << rowsize);
      for (int j = 0; j < nex; j++)
      {
          int ind = j + i*nex;
          if (!grayscale)
          {
             for (int b = 0; b < bb; b++)
             {
                data[b*nex*ney + ind] = (double)bmpdata[bb*j+bb-b-1];
             }
          }
          else
          {
             for (int b = 0; b < std::max(bb,3); b++)
             {
                data[ind] += (double)bmpdata[bb*j+b];
             }
          }
      }
   }
   fclose(f);

   delete[] bmpdata;
}

void Image::Save(const char* filename)
{
   MPI_Comm comm = pmesh->GetComm();

   PetscErrorCode ierr;
   PetscMPIInt rank;
   ierr = MPI_Comm_rank(comm,&rank); CCHKERRQ(comm,ierr);

   std::ostringstream fname;
   fname << filename << "." << std::setfill('0') << std::setw(6) << rank;

   std::ofstream ofs(fname.str().c_str());
   ofs.precision(8);

   pmesh->Print(ofs);
   ParGridFunction gf(pfes);
   gf.ProjectDiscCoefficient(*this,GridFunction::ARITHMETIC);
   gf.Save(ofs);
}

void Image::Visualize(const char* name)
{
   MPI_Comm comm = pmesh->GetComm();
   std::string sname(name);

   PetscErrorCode ierr;
   PetscMPIInt rank,size;
   ierr = MPI_Comm_rank(comm,&rank); CCHKERRQ(comm,ierr);
   ierr = MPI_Comm_size(comm,&size); CCHKERRQ(comm,ierr);

   ParGridFunction gf(pfes);
   gf.ProjectDiscCoefficient(*this,GridFunction::ARITHMETIC);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sock(vishost, visport);
   sock << "parallel " << size << " " << rank << "\n";
   sock.precision(8);
   sock << "solution\n" << *pmesh << gf << std::flush;
   sock << "window_size 800 800\n";
   sock << "window_title '" << sname << "'\n";
   sock << "keys RJlc\n" << std::flush;
}

PDCoefficient* Image::CreatePDCoefficient()
{
   return new PDCoefficient(*this,(*this).pfes);
}

/* the objective functional as a sum of Tikhonov and TV terms */
class ImageFunctional : public ReducedFunctional
{
private:
   mutable PetscParMatrix H;
   mutable TikhonovRegularizer *tk;
   mutable TVRegularizer *tv;

public:
   ImageFunctional(int,TikhonovRegularizer*,TVRegularizer*);
   virtual void ComputeObjective(const Vector&,double*) const;
   virtual void ComputeGradient(const Vector&,Vector&) const;
   virtual Operator& GetHessian(const Vector&) const;
   virtual void PostCheck(const Vector&,Vector&,Vector&,bool&,bool&) const;
};

ImageFunctional::ImageFunctional(int _lsize, TikhonovRegularizer *_tk, TVRegularizer *_tv) : H(), tk(_tk), tv(_tv)
{
   height = width = _lsize;
}

void ImageFunctional::ComputeObjective(const Vector& u, double *f) const
{
   Vector dummy;
   double f1,f2;
   tk->Eval(dummy,u,0.,&f1);
   tv->Eval(dummy,u,0.,&f2);
   *f = f1 + f2;
}

void ImageFunctional::ComputeGradient(const Vector& u, Vector& g) const
{
   Vector dummy,g1;
   g1.SetSize(g.Size());
   tk->EvalGradient_M(dummy,u,0.,g);
   tv->EvalGradient_M(dummy,u,0.,g1);
   g += g1;
}

Operator& ImageFunctional::GetHessian(const Vector& u) const
{
   Vector dummy;
   tk->SetUpHessian_MM(dummy,u,0.);
   tv->SetUpHessian_MM(dummy,u,0.);
   Operator* Htk = tk->GetHessianOperator_MM();
   Operator* Htv = tv->GetHessianOperator_MM();
   PetscParMatrix *pHtk = dynamic_cast<mfem::PetscParMatrix *>(Htk);
   PetscParMatrix *pHtv = dynamic_cast<mfem::PetscParMatrix *>(Htv);
   MFEM_VERIFY(pHtk,"Unsupported operator type");
   MFEM_VERIFY(pHtv,"Unsupported operator type");

   /* Pointers to operators returned by the ObjectiveFunction class should not be
      deleted or modified */
   H = *pHtk;

   /* These matrices have the same pattern, the MFEM overloaded += operator uses DIFFERENT_NONZERO_PATTERN */
   {
      PetscErrorCode ierr;
      ierr = MatAXPY(H,1.0,*pHtv,SAME_NONZERO_PATTERN); PCHKERRQ(pHtv,ierr);
   }
   return H;
}

/* This method is called after a successful line search
   We use it to update the dual variable for the TV regularizer */
void ImageFunctional::PostCheck(const Vector& X, Vector& Y, Vector &W, bool& cy, bool& cw) const
{
   /* we don't change the step (Y) or the updated solution (W = X - lambda*Y) */
   cy = false;
   cw = false;
   double lambda = X.Size() ? (X[0] - W[0])/Y[0] : 0.0;
   tv->UpdateDual(X,Y,lambda);
}

/* the main routine */
int main(int argc, char* argv[])
{
   MFEMInitializePetsc(&argc,&argv,NULL,help);

   /* process options */
   PetscErrorCode ierr;
   char imgfile[PETSC_MAX_PATH_LEN] = "../../../../share/petscopt/data/logo_noise.txt";
   PetscBool save = PETSC_FALSE, viz = PETSC_TRUE, mon = PETSC_TRUE;
   PetscBool primal_dual = PETSC_FALSE, symmetrize = PETSC_FALSE, project = PETSC_FALSE, quad = PETSC_FALSE;
   PetscReal noise = 0.0, tv_alpha = 0.0007, tv_beta = 0.1;
   PetscInt  ord = 1;
   PetscBool test = PETSC_FALSE, test_progress = PETSC_FALSE, test_part = PETSC_FALSE;

   ierr = PetscOptionsGetBool(NULL,NULL,"-save",&save,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-glvis",&viz,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&mon,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-primaldual",&primal_dual,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-symmetrize",&symmetrize,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-project",&project,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-quad",&quad,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-order",&ord,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-noise",&noise,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-tv_alpha",&tv_alpha,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-tv_beta",&tv_beta,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetString(NULL,NULL,"-image",imgfile,sizeof(imgfile),NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-test",&test,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-test_progress",&test_progress,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-test_partitioning",&test_part,NULL);CHKERRQ(ierr);

   /* indent to have stacked objects destroyed before PetscFinalize() is called */
   {
      Image img(PETSC_COMM_WORLD,imgfile,ord,quad,test_part);
      img.Normalize();
      if (viz)
      {
         img.Visualize("Image to be reconstructed");
      }
      img.AddNoise(noise);
      if (viz && noise != 0.0)
      {
         img.Visualize("Image to be reconstructed (noisy)");
      }

      /* The optimization variable */
      PDCoefficient* imgpd = img.CreatePDCoefficient();

      /* Regularizers */
      TikhonovRegularizer tk(imgpd);
      TVRegularizer tv(imgpd,tv_alpha,tv_beta,primal_dual);
      tv.Symmetrize(symmetrize);
      tv.Project(project);

      /* The full objective */
      ImageFunctional objective(imgpd->GetLocalSize(),&tk,&tv);

      Vector dummy;
      Vector u(imgpd->GetLocalSize());

      /* Testing */
      if (test)
      {
         double f1,f2,f;
         u = 1.0;

         ierr = PetscPrintf(PETSC_COMM_WORLD,"---------------------------------------\n");CHKERRQ(ierr);
         ierr = PetscPrintf(PETSC_COMM_WORLD,"TV tests\n");CHKERRQ(ierr);
         tv.Eval(dummy,u,0.,&f1);
         tv.TestFDGradient(PETSC_COMM_WORLD,dummy,u,0.0,1.e-6,test_progress);
         tv.TestFDHessian(PETSC_COMM_WORLD,dummy,u,0.0);
         ierr = PetscPrintf(PETSC_COMM_WORLD,"---------------------------------------\n");CHKERRQ(ierr);
         ierr = PetscPrintf(PETSC_COMM_WORLD,"Tikhonov tests\n");CHKERRQ(ierr);
         tk.Eval(dummy,u,0.,&f2);
         tk.TestFDGradient(PETSC_COMM_WORLD,dummy,u,0.0,1.e-6,test_progress);
         tk.TestFDHessian(PETSC_COMM_WORLD,dummy,u,0.0);
         ierr = PetscPrintf(PETSC_COMM_WORLD,"---------------------------------------\n");CHKERRQ(ierr);
         ierr = PetscPrintf(PETSC_COMM_WORLD,"Image tests\n");CHKERRQ(ierr);
         objective.ComputeObjective(u,&f);
         objective.TestFDGradient(PETSC_COMM_WORLD,u,1.e-6,test_progress);
         objective.TestFDHessian(PETSC_COMM_WORLD,u);
         ierr = PetscPrintf(PETSC_COMM_WORLD,"---------------------------------------\n");CHKERRQ(ierr);
         MFEM_VERIFY(std::abs(f1+f2-f) < PETSC_SMALL,"Error eval " << std::abs(f1+f2-f));
      }

      PetscNonlinearSolverOpt newton(PETSC_COMM_WORLD,objective);

      NewtonMonitor mymonitor;
      if (mon) newton.SetMonitor(&mymonitor);

      /* solve via Newton */
      newton.Mult(dummy,u);
      if (viz || save)
      {
         imgpd->UpdateCoefficient(u);
         if (save) imgpd->Save("reconstructed_image");
         if (viz) imgpd->Visualize("RJlc");
      }
      delete imgpd;
   }
   MFEMFinalizePetsc();
   return 0;
}

/*TEST

  build:
    requires: mfemopt

  test:
    filter: sed -e "s/-nan/nan/g"
    nsize: {{1 2}}
    suffix: test
    args: -glvis 0 -test_partitioning -test -test_progress 0 -image ${petscopt_dir}/share/petscopt/data/img_small.bmp -monitor 0 -snes_converged_reason -quad 0 -order 2

  test:
    filter: sed -e "s/-nan/nan/g"
    nsize: {{1 2}}
    suffix: test_pd
    args: -glvis 0 -test_partitioning -test -test_progress 0 -image ${petscopt_dir}/share/petscopt/data/img_small.bmp -monitor 0 -snes_converged_reason -quad 0 -order 2 -primaldual

  test:
    nsize: {{1 2}}
    suffix: tv
    args: -glvis 0 -test_partitioning -image ${petscopt_dir}/share/petscopt/data/logo_noise.txt -quad 1 -order 1 -snes_converged_reason -snes_rtol 1.e-10 -snes_atol 1.e-10 -ksp_rtol 1.e-10 -ksp_atol 1.e-10 -primaldual 0 -symmetrize 0 -monitor 0 -snes_converged_reason

  test:
    nsize: {{1 2}}
    suffix: tv_pd
    args: -glvis 0 -test_partitioning -image ${petscopt_dir}/share/petscopt/data/logo_noise.txt -quad 1 -order 1 -snes_converged_reason -snes_rtol 1.e-10 -snes_atol 1.e-10 -ksp_rtol 1.e-10 -ksp_atol 1.e-10 -ksp_type cg -pc_type gamg -primaldual 1 -symmetrize 1 -monitor 0 -snes_converged_reason

  test:
    nsize: {{1 2}}
    suffix: tv_pd_project
    args: -glvis 0 -test_partitioning -image ${petscopt_dir}/share/petscopt/data/logo_noise.txt -quad 1 -order 1 -snes_converged_reason -snes_rtol 1.e-10 -snes_atol 1.e-10 -ksp_rtol 1.e-10 -ksp_atol 1.e-10 -ksp_type cg -pc_type gamg -primaldual 1 -symmetrize 1 -project -monitor 0 -snes_converged_reason

TEST*/

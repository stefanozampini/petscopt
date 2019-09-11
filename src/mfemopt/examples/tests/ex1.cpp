static const char help[] = "A simple total-variation based, primal-dual image regularization.";
// Example runs:
// - mpiexec -n 4  ./ex1 -image ${PETSCOPT_DIR}/share/petscopt/data/img_medium.bmp -monitor -primaldual -noise 0.1 -tv_alpha 0.001 -tv_beta 0.001 -ksp_type cg -pc_type gamg -primaldual 1 -symmetrize 1

#include <mfemopt.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <mfem.hpp>
#include <limits>

using namespace mfem;
using namespace mfemopt;

/* an Image represented through the MFEM Coefficient class */
class Image;
class ImageCoefficient : public Coefficient
{
private:
   Image *img;
public:
   ImageCoefficient(Image *_img) : img(_img) { time = 0.0; }
   virtual double Eval(ElementTransformation&,
                       const IntegrationPoint&);
   ~ImageCoefficient() {};
};

class ImageVectorCoefficient : public VectorCoefficient
{
private:
   Image *img;
public:
   ImageVectorCoefficient(Image *_img);
   virtual void Eval(Vector&,ElementTransformation&,
                     const IntegrationPoint&);
   ~ImageVectorCoefficient() {};
};

/* an Image is represented through the MFEM Coefficient class */
class Image
{
private:
   Coefficient *imgcoeff;
   VectorCoefficient *imgcoeffv;

protected:
   friend class ImageCoefficient;
   friend class ImageVectorCoefficient;

   /* double precision pixel representation */
   double *data;
   int    nex,ney;
   double hx,hy;
   int    vdim;

   /* FEM representation */
   FiniteElementCollection *fec;
   ParFiniteElementSpace   *pfes,*sfes;
   ParMesh                 *pmesh;

   /* read implementations */
   void ReadBMP(const char*,bool=true);
   void ReadTXT(const char*,bool=true);

public:
   Image() : imgcoeff(NULL), imgcoeffv(NULL), data(NULL), nex(0), ney(0), fec(NULL), pfes(NULL), sfes(NULL), pmesh(NULL) {}
   Image(MPI_Comm,const char*,int=1,bool=true,bool=true,bool=false);

   PDCoefficient* CreatePDCoefficient();

   void AddNoise(double);
   void Normalize();
   void Save(const char*);
   void Visualize(const char*);

   ~Image() { delete imgcoeff; delete imgcoeffv; delete[] data; delete fec; delete pfes; delete sfes; delete pmesh; }
};

Image::Image(MPI_Comm comm, const char* filename, int ord, bool quad, bool vector, bool test_part)
{
   nex = ney = 0;
   std::string fname(filename);
   size_t lastdot = fname.find_last_of(".");
   std::string fext = fname.substr(lastdot);
   if (fext == ".bmp")
   {
      ReadBMP(filename,!vector);
   }
   else if (fext == ".txt")
   {
      ReadTXT(filename,!vector);
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
   pfes = new ParFiniteElementSpace(pmesh, fec, vdim, Ordering::byVDIM);
   sfes = new ParFiniteElementSpace(pmesh, fec, 1, Ordering::byVDIM);

   imgcoeffv = new ImageVectorCoefficient(this);
   imgcoeff = new ImageCoefficient(this);
}

void Image::Normalize()
{
   std::vector<double> im(vdim);
   im.assign(vdim,std::numeric_limits<double>::max());
   std::vector<double> iM(vdim);
   iM.assign(vdim,std::numeric_limits<double>::min());
   for (int i = 0; i < ney*nex; i++)
   {
      for (int v = 0; v < vdim; v++)
      {
         im[v] = data[vdim*i+v] < im[v] ? data[vdim*i+v] : im[v];
         iM[v] = data[vdim*i+v] > iM[v] ? data[vdim*i+v] : iM[v];
      }
   }
   for (int v = 0; v < vdim; v++)
   {
      if (iM[v] > im[v])
      {
         for (int i = 0; i < ney*nex; i++)
         {
            data[vdim*i+v] = (data[vdim*i+v] - im[v])/(iM[v] - im[v]);
         }
      }
      else
      {
         for (int i = 0; i < ney*nex; i++)
         {
            data[vdim*i+v] = 0.0;
         }
      }
   }
}

void Image::AddNoise(double s)
{
   GaussianNoise noise;
   Vector vnoise;
   noise.Randomize(vnoise,vdim*nex*ney);
   vnoise *= s;
   for (int i = 0; i < vdim*ney*nex; i++) data[i] += vnoise[i];
}

void Image::ReadTXT(const char* filename, bool single)
{
   std::ifstream input(filename);
   MFEM_VERIFY(!input.fail(),"Missing file " << filename);

   int nc;
   input >> nex;
   input >> ney;
   input >> nc;
   vdim = single ? 1 : nc;

   data = new double[nex*ney*vdim];
   for (int i = 0; i < nex*ney*vdim; i++) data[i] = 0.0;

   for (int c = 0; c < nc; c++)
   {
      const int v = single ? 0 : c;
      for (int i = 0; i < nex; i++)
      {
         for (int j = 0; j < ney; j++)
         {
            double d;
            input >> d;

            data[vdim*(j + i*ney) + v] += d;
         }
      }
   }
}

void Image::ReadBMP(const char* filename, bool single)
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
   //int cm = *(int*)&info[30]; compression method
   int bb = bits/8;

   vdim = single ? 1 : std::min(bb,3);
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
          if (!single)
          {
             for (int b = 0; b < vdim; b++)
             {
                data[vdim*ind + b] = (double)bmpdata[bb*j+b];
             }
          }
          else
          {
             for (int b = 0; b < vdim; b++)
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
   if (vdim > 1)
   {
      gf.ProjectDiscCoefficient(*imgcoeffv,GridFunction::ARITHMETIC);
   }
   else
   {
      gf.ProjectDiscCoefficient(*imgcoeff,GridFunction::ARITHMETIC);
   }
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

   Array<ParGridFunction*> gfs;
   for (int i = 0; i < vdim; i++)
   {
      gfs.Append(new ParGridFunction(sfes));
   }
   if (vdim > 1)
   {
      for (int i = 0; i < vdim; i++)
      {
         ComponentCoefficient coeff(*imgcoeffv,i);
         gfs[i]->ProjectDiscCoefficient(coeff,GridFunction::ARITHMETIC);
      }
   }
   else
   {
      gfs[0]->ProjectDiscCoefficient(*imgcoeff,GridFunction::ARITHMETIC);
   }

   for (int i = 0; i < vdim; i++)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sock(vishost, visport);
      sock << "parallel " << size << " " << rank << "\n";
      sock.precision(8);
      sock << "solution\n" << *pmesh << *(gfs[i]) << std::flush;
      sock << "window_size 800 800\n";
      if (vdim > 1)
      {
         sock << "window_title '" << sname << "-" << i << "'\n";
      }
      else
      {
         sock << "window_title '" << sname << "'\n";
      }
      sock << "keys RJlc\n" << std::flush;
   }
   for (int i = 0; i < vdim; i++)
   {
      delete gfs[i];
   }
}

PDCoefficient* Image::CreatePDCoefficient()
{
   if (vdim > 1)
   {
      return new PDCoefficient(*imgcoeffv,pmesh,fec);
   }
   else
   {
      return new PDCoefficient(*imgcoeff,pmesh,fec);
   }
}

double ImageCoefficient::Eval(ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);
   int ix = std::floor(x[0]/img->hx);
   int iy = std::floor(x[1]/img->hy);
   MFEM_VERIFY(0 <= ix && ix < img->nex,"Wrong ix " << ix);
   MFEM_VERIFY(0 <= iy && iy < img->ney,"Wrong iy " << iy);
   return img->data[iy + ix*img->ney];
}

ImageVectorCoefficient::ImageVectorCoefficient(Image *_img) : VectorCoefficient(0)
{
   img  = _img;
   vdim = img->vdim;
   time = 0;
}

void ImageVectorCoefficient::Eval(Vector &V,
                                  ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);
   int ix = std::floor(x[0]/img->hx);
   int iy = std::floor(x[1]/img->hy);
   MFEM_VERIFY(0 <= ix && ix < img->nex,"Wrong ix " << ix);
   MFEM_VERIFY(0 <= iy && iy < img->ney,"Wrong iy " << iy);
   for (int v = 0; v < vdim; v++) V[v] = img->data[vdim*(iy + ix*img->ney)+v];
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
   virtual void Init(const Vector&);
   virtual void Update(int,const Vector&,const Vector&,const Vector&,const Vector&);
   //virtual void PostCheck(const Vector&,Vector&,Vector&,bool&,bool&) const;
};

ImageFunctional::ImageFunctional(int _lsize, TikhonovRegularizer *_tk, TVRegularizer *_tv) : H(), tk(_tk), tv(_tv)
{
   height = width = _lsize;
}

void ImageFunctional::ComputeObjective(const Vector& u, double *f) const
{
   Vector dummy;
   double f1 = 0.0,f2 = 0.0;
   if (tk) tk->Eval(dummy,u,0.,&f1);
   if (tv) tv->Eval(dummy,u,0.,&f2);
   *f = f1 + f2;
}

void ImageFunctional::ComputeGradient(const Vector& u, Vector& g) const
{
   g = 0.0;

   Vector dummy,g1;
   g1.SetSize(g.Size());
   g1 = 0.0;
   if (tk) tk->EvalGradient_M(dummy,u,0.,g);
   if (tv) tv->EvalGradient_M(dummy,u,0.,g1);
   g += g1;
}

Operator& ImageFunctional::GetHessian(const Vector& u) const
{
   Vector dummy;
   if (tk) tk->SetUpHessian_MM(dummy,u,0.);
   if (tv) tv->SetUpHessian_MM(dummy,u,0.);
   Operator *Htk = tk ? tk->GetHessianOperator_MM() : NULL;
   Operator *Htv = tv ? tv->GetHessianOperator_MM() : NULL;
   PetscParMatrix *pHtk = Htk ? dynamic_cast<mfem::PetscParMatrix *>(Htk) : NULL;
   PetscParMatrix *pHtv = Htv ? dynamic_cast<mfem::PetscParMatrix *>(Htv) : NULL;
   if (Htk) MFEM_VERIFY(pHtk,"Unsupported operator type");
   if (Htv) MFEM_VERIFY(pHtv,"Unsupported operator type");

   /* Pointers to operators returned by the ObjectiveFunction class should not be
      deleted or modified */
   if (pHtv) H = *pHtv;
   else if (pHtk) H = *pHtk;

   /* These matrices have the same (or possibly a subset, if using vector TV with fully coupled channels) pattern,
      the MFEM overloaded += operator uses DIFFERENT_NONZERO_PATTERN */
   if (pHtv && pHtk) {
      PetscErrorCode ierr;

      ierr = MatAXPY(H,1.0,*pHtk,SUBSET_NONZERO_PATTERN); PCHKERRQ(pHtv,ierr);
   }
   return H;
}

/* This method is called at the beginning of each nonlinear step
   We use it to update the dual variables for the TV regularizer */
void ImageFunctional::Update(int it, const Vector& F, const Vector& X,
                             const Vector& dX, const Vector& pX)
{
   if (!tv) return;
   if (!it)
   {
      tv->UpdateDual(X);
   }
   else
   {
      double lambda = pX.Size() ? (pX[0] - X[0])/dX[0] : 0.0;
      tv->UpdateDual(pX,dX,lambda);
   }
}

void ImageFunctional::Init(const Vector& X)
{
   if (tv) tv->UpdateDual(X);
}

/* This method is called after a successful line search
   We use it to update the dual variable for the TV regularizer */
#if 0
void ImageFunctional::PostCheck(const Vector& X, Vector& Y, Vector &W, bool& cy, bool& cw) const
{
   /* we don't change the step (Y) or the updated solution (W = X - lambda*Y) */
   cy = false;
   cw = false;
   double lambda = X.Size() ? (X[0] - W[0])/Y[0] : 0.0;
   tv->UpdateDual(X,Y,lambda);
}
#endif

/* the main routine */
int main(int argc, char* argv[])
{
   MFEMInitializePetsc(&argc,&argv,NULL,help);

   /* process options */
   PetscErrorCode ierr;
   char imgfile[PETSC_MAX_PATH_LEN] = "../../../../share/petscopt/data/logo_noise.txt";
   PetscBool save = PETSC_FALSE, viz = PETSC_TRUE, mon = PETSC_TRUE, visit = PETSC_FALSE;
   PetscBool primal_dual = PETSC_FALSE, symmetrize = PETSC_FALSE, project = PETSC_FALSE, vector = PETSC_FALSE, coupled = PETSC_TRUE;
   PetscBool quad = PETSC_FALSE, normalize = PETSC_TRUE, mataij = PETSC_FALSE;
   PetscReal noise = 0.0, tv_alpha = 0.0007, tv_beta = 0.1, tk_alpha = 1.0;
   PetscInt  ord = 1, randseed = 1; /* make the tests reproducible on a given machine */
   PetscBool test = PETSC_FALSE, test_progress = PETSC_FALSE, test_part = PETSC_FALSE, test_taylor = PETSC_FALSE;

   ierr = PetscOptionsGetBool(NULL,NULL,"-normalize",&normalize,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-visit",&visit,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-save",&save,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-glvis",&viz,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&mon,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-primaldual",&primal_dual,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-symmetrize",&symmetrize,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-project",&project,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-vector",&vector,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-coupled",&coupled,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-mataij",&mataij,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-quad",&quad,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-order",&ord,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-noise",&noise,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-tk_alpha",&tk_alpha,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-tv_alpha",&tv_alpha,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-tv_beta",&tv_beta,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetString(NULL,NULL,"-image",imgfile,sizeof(imgfile),NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-test",&test,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-test_seed",&randseed,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-test_taylor",&test_taylor,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-test_progress",&test_progress,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-test_partitioning",&test_part,NULL);CHKERRQ(ierr);

   /* indent to have stacked objects destroyed before PetscFinalize() is called */
   {
      Image img(PETSC_COMM_WORLD,imgfile,ord,quad,vector,test_part);
      if (normalize) img.Normalize();

      /* The optimization variable */
      PDCoefficient* imgpd = img.CreatePDCoefficient();

      if (viz)
      {
         img.Visualize("Image to be reconstructed");
      }
      img.AddNoise(noise);
      if (viz && noise != 0.0)
      {
         img.Visualize("Image to be reconstructed (noisy)");
      }

      /* Regularizers : obj = tk_alpha * 0.5 * || u - u0 || ^2 + tv_alpha * TV(u,tv_beta) */
      TikhonovRegularizer tk(imgpd);
      tk.SetScale(tk_alpha);

      TVRegularizer tv(imgpd,tv_beta,primal_dual,coupled);
      tv.SetScale(tv_alpha);
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
         u.Randomize(randseed); /* make the tests reproducible on a given machine */
         if (randseed) randseed++;
         ierr = PetscPrintf(PETSC_COMM_WORLD,"---------------------------------------\n");CHKERRQ(ierr);
         ierr = PetscPrintf(PETSC_COMM_WORLD,"TV tests\n");CHKERRQ(ierr);
         tv.Eval(dummy,u,0.,&f1);
         tv.TestFDGradient(PETSC_COMM_WORLD,dummy,u,0.0,1.e-6,test_progress);
         tv.UpdateDual(u);
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

      if (test_taylor)
      {
	 u.Randomize(randseed);
         tv.UpdateDual(u);
         objective.TestTaylor(PETSC_COMM_WORLD,u,true);
      }
      /* Test newton solver */
      PetscNonlinearSolverOpt newton(PETSC_COMM_WORLD,objective);

      NewtonMonitor mymonitor;
      if (mon) newton.SetMonitor(&mymonitor);

      /* force automatic conversion after having computed the Jacobian */
      if (!mataij) newton.SetJacobianType(Operator::ANY_TYPE);
      else newton.SetJacobianType(Operator::PETSC_MATAIJ);

      /* solve via SNES */
      u = 0.;
      newton.Mult(dummy,u);

      if (viz || save || visit)
      {
         imgpd->Distribute(u);
         if (save) imgpd->Save("reconstructed_image");
         if (viz) imgpd->Visualize("RJlc");
         if (visit) imgpd->SaveVisIt("reconstructed_image_visit");
      }

      /* Test optimization solver */
      PetscOptimizationSolver opt(PETSC_COMM_WORLD,objective,"opt_");

      OptimizationMonitor myoptmonitor;
      if (mon) opt.SetMonitor(&myoptmonitor);

      /* force automatic conversion after having computed the Hessian */
      if (!mataij) opt.SetHessianType(Operator::ANY_TYPE);
      else opt.SetHessianType(Operator::PETSC_MATAIJ);

      /* solve via TAO */
      u = 0.;
      opt.Solve(u);

      if (viz || save || visit)
      {
         imgpd->Distribute(u);
         if (save) imgpd->Save("reconstructed_image_opt");
         if (viz) imgpd->Visualize("RJlc");
         if (visit) imgpd->SaveVisIt("reconstructed_image_opt_visit");
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
    args: -glvis 0 -test_partitioning -test -test_progress 0 -image ${petscopt_dir}/share/petscopt/data/img_small.bmp -monitor 0 -snes_converged_reason -quad 0 -order 2 -opt_tao_converged_reason -opt_tao_type bnls -test_taylor -taylor_seed 2

  test:
    filter: sed -e "s/-nan/nan/g"
    nsize: {{1 2}}
    suffix: test_pd
    args: -glvis 0 -test_partitioning -test -test_progress 0 -image ${petscopt_dir}/share/petscopt/data/img_small.bmp -monitor 0 -snes_converged_reason -quad 0 -order 2 -primaldual -opt_tao_converged_reason -opt_tao_type ntr -opt_tao_ntr_pc_type gamg

  test:
    timeoutfactor: 3
    nsize: {{1 2}}
    suffix: tv
    args: -glvis 0 -test_partitioning -image ${petscopt_dir}/share/petscopt/data/logo_noise.txt -quad 1 -order 1 -snes_converged_reason -snes_rtol 1.e-10 -snes_atol 1.e-10 -ksp_rtol 1.e-10 -ksp_atol 1.e-10 -primaldual 0 -symmetrize 0 -monitor 0 -snes_converged_reason -opt_tao_converged_reason -opt_tao_type nls -opt_tao_nls_ksp_type gmres

  test:
    timeoutfactor: 3
    nsize: {{1 2}}
    requires: hypre
    suffix: tv_pd
    filter: sed -e "s/CONVERGED_FNORM_ABS iterations 17/CONVERGED_FNORM_ABS iterations 16/g"
    args: -glvis 0 -test_partitioning -image ${petscopt_dir}/share/petscopt/data/logo_noise.txt -quad 1 -order 1 -snes_converged_reason -snes_rtol 1.e-10 -snes_atol 1.e-10 -ksp_rtol 1.e-10 -ksp_atol 1.e-10 -ksp_type cg -pc_type gamg -primaldual 1 -symmetrize 0 -monitor 0 -snes_converged_reason -snes_type {{newtonls newtontr}separate output} -opt_tao_type nls -opt_tao_converged_reason -opt_tao_converged_reason -opt_tao_gatol 1.e-10 -opt_tao_nls_pc_type hypre -opt_tao_nls_ksp_type cg

  test:
    timeoutfactor: 3
    nsize: {{1 2}}
    suffix: tv_pd_project
    args: -glvis 0 -test_partitioning -image ${petscopt_dir}/share/petscopt/data/logo_noise.txt -quad 1 -order 1 -snes_converged_reason -snes_rtol 1.e-10 -snes_atol 1.e-10 -ksp_rtol 1.e-10 -ksp_atol 1.e-10 -ksp_type cg -pc_type gamg -primaldual 1 -symmetrize 1 -project -monitor 0 -snes_converged_reason -opt_tao_converged_reason -opt_tao_type nls -opt_tao_nls_ksp_type cg -opt_tao_nls_pc_type gamg -opt_tao_gatol 1.e-10 -opt_tao_nls_ksp_rtol 1.e-10

  test:
    nsize: {{1 2}}
    suffix: vtv_test
    args: -glvis 0 -test_partitioning -test -test_progress 0 -image ${petscopt_dir}/share/petscopt/data/img_small.bmp -monitor -quad -order 2 -opt_tao_max_it 0 -snes_max_it 0 -vector 1 -primaldual 0 -symmetrize {{0 1}} -coupled {{0 1}}

  test:
    nsize: {{1 2}}
    suffix: vtv_pd_test
    args: -glvis 0 -test_partitioning -test -test_progress 0 -image ${petscopt_dir}/share/petscopt/data/img_small.bmp -monitor -quad -order 2 -opt_tao_max_it 0 -snes_max_it 0 -vector 1 -primaldual 1 -symmetrize {{0 1}} -project {{0 1}} -coupled {{0 1}}

  testset:
    nsize: 1
    args: -glvis 0 -image ${petscopt_dir}/share/petscopt/data/hearts.txt -monitor 0 -noise 0.2 -quad -order 1 -vector 1 -opt_tao_type nls -snes_type newtonls -mataij -snes_converged_reason -opt_tao_converged_reason -snes_rtol 1.e-10 -snes_atol 1.e-10 -opt_tao_gatol 1.e-10 -ksp_type preonly -opt_tao_nls_ksp_type preonly -tv_beta 1.e-4 -snes_max_it 30 -opt_tao_max_it 30
    test:
      suffix: vtv
      args: -pc_type lu -pc_factor_mat_ordering_type nd -opt_tao_nls_pc_type lu -opt_tao_nls_pc_factor_mat_ordering_type nd -coupled {{0 1}separate output}
    test:
      suffix: vtv_pd
      args: -primaldual -pc_type lu -pc_factor_mat_ordering_type nd -opt_tao_nls_pc_type lu -opt_tao_nls_pc_factor_mat_ordering_type nd -coupled {{0 1}separate output}
    test:
      suffix: vtv_pd_spd
      args: -primaldual -pc_type cholesky -pc_factor_mat_ordering_type nd -opt_tao_nls_pc_type cholesky -opt_tao_nls_pc_factor_mat_ordering_type nd -coupled {{0 1}separate output} -symmetrize -project {{0 1}separate output}

TEST*/

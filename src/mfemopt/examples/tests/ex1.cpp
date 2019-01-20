static const char help[] = "A simple total-variation based, primal-dual image regularization.";

#include <mfemopt.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>

#include <mfem.hpp>
#include <petscdm.h>

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
   Image() : data(NULL), fec(NULL), pfes(NULL), pmesh(NULL) {}
   Image(MPI_Comm,const char*,int=1,bool=true);

   virtual double Eval(ElementTransformation&,
                       const IntegrationPoint&);

   PDCoefficient* CreatePDCoefficient();
   ParFiniteElementSpace* ParFESpace() { return pfes; }

   void Save(const char*);
   void Visualize(const char*);

   virtual ~Image() { delete[] data; delete fec; delete pfes; delete pmesh; }
};

Image::Image(MPI_Comm comm, const char* filename, int ord, bool quad) : Image()
{
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
   pmesh = new ParMesh(comm,*mesh);
   delete mesh;
   fec = new H1_FECollection(std::max(ord,1), 2);
   pfes = new ParFiniteElementSpace(pmesh, fec, 1, Ordering::byVDIM);
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
   MFEM_VERIFY(!input.fail(),"Missing filename " << filename);

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
   gf.ProjectDiscCoefficient(*this,GridFunction::AvgType::ARITHMETIC);
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
   gf.ProjectDiscCoefficient(*this,GridFunction::AvgType::ARITHMETIC);

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
class ImageFunctional : public ReducedFunctional, public Operator
{
private:
   Image *img;
   mutable PetscParMatrix *H;

public:
   mutable TikhonovRegularizer *tk;
   mutable TVRegularizer *tv;

   ImageFunctional(Image*,TikhonovRegularizer*,TVRegularizer*);

   virtual int GetParameterSize() {return height;}
   virtual void ComputeObjective(const Vector&, double*);
   virtual void ComputeGradient(const Vector&, Vector&);
   virtual Operator* GetHessian(const Vector&);

   virtual void Mult(const Vector&, Vector&) const;
   virtual Operator& GetGradient(const Vector&) const;
   virtual ~ImageFunctional() { delete H; }
};

ImageFunctional::ImageFunctional(Image* _img, TikhonovRegularizer *_tk, TVRegularizer *_tv) : img(_img), H(NULL), tk(_tk), tv(_tv)
{
   height = width = img->ParFESpace()->GetTrueVSize();
}

/* interface for MFEMOPT */
void ImageFunctional::ComputeObjective(const Vector& u, double *f)
{
   Vector dummy;
   double f1,f2;
   tk->Eval(dummy,u,0.,&f1);
   tv->Eval(dummy,u,0.,&f2);
   *f = f1 + f2;
}

void ImageFunctional::ComputeGradient(const Vector& u, Vector& g)
{
   Vector dummy,g1;
   g1.SetSize(g.Size());
   tk->EvalGradient_M(dummy,u,0.,g);
   tv->EvalGradient_M(dummy,u,0.,g1);
   g += g1;
}

Operator* ImageFunctional::GetHessian(const Vector& u)
{
   Operator& tH = GetGradient(u); /* MFEM use Gradient for the Jacobian of the PDE */
   return &tH;
}

/* interface for Newton based solvers provided by mfem::PetscNonlinearSolver */
void ImageFunctional::Mult(const Vector& u, Vector& g) const
{
   Vector dummy,g1;

   g1.SetSize(g.Size());
   tk->EvalGradient_M(dummy,u,0.,g);
   tv->EvalGradient_M(dummy,u,0.,g1);
   g += g1;
}

Operator& ImageFunctional::GetGradient(const Vector& u) const
{
   MPI_Comm comm = img->ParFESpace()->GetParMesh()->GetComm();
   Vector dummy;
   tk->SetUpHessian_MM(dummy,u,0.);
   tv->SetUpHessian_MM(dummy,u,0.);
   Operator* Htk = tk->GetHessianOperator_MM();
   Operator* Htv = tv->GetHessianOperator_MM();
   PetscParMatrix *pHtk = new PetscParMatrix(comm,Htk,Operator::PETSC_MATAIJ);
   PetscParMatrix *pHtv = new PetscParMatrix(comm,Htv,Operator::PETSC_MATAIJ);

   PetscErrorCode ierr;
   if (!H)
   {
      H = new PetscParMatrix();
   }
   *H = *pHtv;

   /* matrices have the same pattern, the MFEM overloaded += operator uses DIFFERENT_NONZERO_PATTERN */
   ierr = MatAXPY(*H,1.0,*pHtk,SAME_NONZERO_PATTERN); CCHKERRQ(comm,ierr);
   delete pHtk;
   delete pHtv;
   return *H;
}

/* auxiliary class to monitor the objective function inside a Newton solver */
class ObjectiveMonitor : public PetscSolverMonitor
{
private:
   Array<ObjectiveFunction*> objs;
public:
   ObjectiveMonitor(Array<ObjectiveFunction*>);
   virtual void MonitorSolver(PetscSolver*);
   virtual ~ObjectiveMonitor() {};
};

ObjectiveMonitor::ObjectiveMonitor(Array<ObjectiveFunction*> _objs) : PetscSolverMonitor(false,false)
{
   objs.SetSize(_objs.Size());
   objs.Assign(_objs);
}

void ObjectiveMonitor::MonitorSolver(PetscSolver *solver)
{
   PetscNonlinearSolver *newton = dynamic_cast<PetscNonlinearSolver *>(solver);
   MFEM_VERIFY(newton,"Not a nonlinear solver");

   SNES snes = (SNES)(*newton);

   DM dm;
   KSP ksp;
   PetscErrorCode ierr;
   SNESLineSearch ls;
   Vec X,dX,G,pG;
   PetscErrorCode (*snesobj)(SNES,Vec,PetscReal*,void*);
   PetscReal lambda,normg;
   PetscScalar inn;
   PetscInt it,lit;

   ierr = SNESGetDM(snes,&dm); PCHKERRQ(snes,ierr);
   ierr = DMGetNamedGlobalVector(dm,"prev_gradient",&pG); PCHKERRQ(snes,ierr);
   ierr = SNESGetKSP(snes,&ksp); PCHKERRQ(snes,ierr);
   ierr = KSPGetIterationNumber(ksp,&lit); PCHKERRQ(snes,ierr);
   ierr = SNESGetSolution(snes,&X); PCHKERRQ(snes,ierr);
   ierr = SNESGetSolutionUpdate(snes,&dX); PCHKERRQ(snes,ierr);
   ierr = SNESGetObjective(snes,&snesobj,NULL); PCHKERRQ(snes,ierr);
   ierr = SNESGetIterationNumber(snes,&it); PCHKERRQ(snes,ierr);
   ierr = SNESGetLineSearch(snes,&ls); PCHKERRQ(snes,ierr);
   ierr = SNESLineSearchGetLambda(ls,&lambda); PCHKERRQ(snes,ierr);
   ierr = VecDot(pG,dX,&inn); PCHKERRQ(snes,ierr);
   ierr = VecNorm(pG,NORM_2,&normg); PCHKERRQ(snes,ierr);
   if (!it) {
     PetscInt dofs;

     ierr = VecGetSize(X,&dofs); PCHKERRQ(snes,ierr);
     ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Number of dofs %D\n",dofs); PCHKERRQ(snes,ierr);
     ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"it\tlit\tenergy\t\t(g,du)\t\t||g||_l2\tstep\n"); PCHKERRQ(snes,ierr);
     lambda = 0.0;
   }
   ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"%D\t%D\t",it,lit); PCHKERRQ(snes,ierr);
   if (snesobj)
   {
      PetscReal f;

      ierr = SNESComputeObjective(snes,X,&f); PCHKERRQ(snes,ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"%1.6e\t",(double)f); PCHKERRQ(snes,ierr);
   }
   else
   {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"------------\t"); PCHKERRQ(snes,ierr);
   }
   ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"%1.6e\t%1.6e\t%1.6e\n",-(double)PetscRealPart(inn),(double)normg,(double)lambda); PCHKERRQ(snes,ierr);
#if 0
   PetscParVector x(X,true);
   for (int i = 0; i < objs.Size(); i++)
   {
      double f;
      Vector dummy; /* XXX */
      objs[i]->Eval(dummy,x,0.0,&f);
      if (!i)
      {
         ierr = PetscPrintf(PetscObjectComm((PetscObject)snes)," ("); PCHKERRQ(snes,ierr);
      }
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes)," %g",f); PCHKERRQ(snes,ierr);
      if (i == objs.Size() -1)
      {
         ierr = PetscPrintf(PetscObjectComm((PetscObject)snes)," )\n"); PCHKERRQ(snes,ierr);
      }
   }
#endif
   ierr = SNESGetFunction(snes,&G,NULL,NULL); PCHKERRQ(snes,ierr);
   ierr = VecCopy(G,pG); PCHKERRQ(snes,ierr);
   ierr = DMRestoreNamedGlobalVector(dm,"prev_gradient",&pG); PCHKERRQ(snes,ierr);
}

/* we need a couple of ugly callbacks to pass objective functions and post-primal-update hooks to PETSc,
   since the interface of mfem::Operator does not allow to do so */
void UglyObjFn(Operator *UglyOp, const Vector& u, double *f)
{
   ImageFunctional *UglyObj = dynamic_cast<ImageFunctional*>(UglyOp);
   MFEM_VERIFY(UglyObj,"Missing Ugly operator");
   UglyObj->ComputeObjective(u,f);
}

void UglyPostCheckFn(Operator *UglyOp, const Vector& X, Vector& Y, Vector &W, bool& cy, bool& cw)
{
   ImageFunctional *UglyObj = dynamic_cast<ImageFunctional*>(UglyOp);
   MFEM_VERIFY(UglyObj,"Missing Ugly operator");
   /* we don't change the step (Y) or the updated solution (W = X - lambda*Y) */
   cy = false;
   cw = false;
   double lambda = X.Size() ? (X[0] - W[0])/Y[0] : 0.0;
   UglyObj->tv->UpdateDual(X,Y,lambda);
}

/* the main routine */
int main(int argc, char* argv[])
{
   MFEMInitializePetsc(&argc,&argv,NULL,NULL);

   /* process options */
   PetscErrorCode ierr;
   char imgfile[PETSC_MAX_PATH_LEN] = "./logo_noise.txt";
   char trueimgfile[PETSC_MAX_PATH_LEN] = "./logo.txt";
   PetscBool flg1,flg2;
   PetscBool save = PETSC_FALSE, viz = PETSC_FALSE, mon = PETSC_TRUE;
   PetscBool primal_dual = PETSC_FALSE, symmetrize = PETSC_FALSE, project = PETSC_FALSE, quad = PETSC_FALSE;
   PetscInt ord = 1;

   ierr = PetscOptionsGetBool(NULL,NULL,"-save",&save,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-glvis",&viz,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&mon,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-primaldual",&primal_dual,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-symmetrize",&symmetrize,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-project",&project,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-quad",&quad,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-order",&ord,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetString(NULL,NULL,"-image",imgfile,sizeof(imgfile),&flg1);CHKERRQ(ierr);
   ierr = PetscOptionsGetString(NULL,NULL,"-trueimage",trueimgfile,sizeof(imgfile),&flg2);CHKERRQ(ierr);

   /* indent to have stacked objects destroyed before PetscFinalize() is called */
   {
      Image img(PETSC_COMM_WORLD,imgfile,ord,quad);
      if (!flg1 && !flg2)
      {
         Image trueimg(PETSC_COMM_WORLD,trueimgfile,1,PETSC_TRUE);
         if (viz)
         {
            trueimg.Visualize("True image");
            img.Visualize("Noise image");
         }
      }
      PDCoefficient* imgpd = img.CreatePDCoefficient();

      TikhonovRegularizer tk(imgpd);
      TVRegularizer tv(imgpd,0.0007,0.1,primal_dual);
      tv.Symmetrize(symmetrize);
      tv.Project(project);
      ImageFunctional objective(&img,&tk,&tv);

      Vector dummy;
      Vector u(imgpd->GetLocalSize());

#if 0
      //u.Randomize();
      //u = 1.0;
      //imgpd->Save("test_init_image");
      imgpd->GetCurrentVector(u);
      double f;
      std::cout << "---------------------------------------" << std::endl;
      //tv.PrimalToDual(u);
      tv.Eval(dummy,u,0.,&f);
      std::cout << "TV tests" << std::endl;
      std::cout << "TV EVAL: " << f << std::endl;
      std::cout << "TV FD TESTS" << std::endl;
      tv.TestFDGradient(PETSC_COMM_WORLD,dummy,u,0.0,1.e-6);
      tv.TestFDHessian(PETSC_COMM_WORLD,dummy,u,0.0);
      std::cout << "---------------------------------------" << std::endl;

      std::cout << "---------------------------------------" << std::endl;
      tk.Eval(dummy,u,0.,&f);
      std::cout << "Tikhonov tests" << std::endl;
      std::cout << "Tikhonov EVAL: " << f << std::endl;
      std::cout << "Tikhonov FD TESTS" << std::endl;
      tk.TestFDGradient(PETSC_COMM_WORLD,dummy,u,0.0,1.e-6);
      tk.TestFDHessian(PETSC_COMM_WORLD,dummy,u,0.0);
      std::cout << "---------------------------------------" << std::endl;

      std::cout << "---------------------------------------" << std::endl;
      objective.ComputeObjective(u,&f);
      std::cout << "Image tests" << std::endl;
      std::cout << "Image EVAL: " << f << std::endl;
      std::cout << "Image FD TESTS" << std::endl;
      objective.TestFDGradient(PETSC_COMM_WORLD,u,1.e-6);
      objective.TestFDHessian(PETSC_COMM_WORLD,u);
      std::cout << "---------------------------------------" << std::endl;
#endif
      PetscNonlinearSolver newton(PETSC_COMM_WORLD,objective);

      newton.SetObjective(UglyObjFn);
      newton.SetPostCheck(UglyPostCheckFn);

      Array<ObjectiveFunction*> objs;
      objs.Append(&tk);
      objs.Append(&tv);
      ObjectiveMonitor mymonitor(objs);
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
    localrunfiles: logo_noise.txt logo.txt
    suffix: tv
    args: -quad 1 -order 1 -snes_converged_reason -snes_max_it 500 -snes_rtol 1.e-10 -snes_atol 1.e-10 -primaldual 0 -symmetrize 0 -monitor 0 -snes_converged_reason

  test:
    localrunfiles: logo_noise.txt logo.txt
    suffix: tv_pd
    args: -quad 1 -order 1 -snes_converged_reason -snes_max_it 500 -snes_rtol 1.e-10 -snes_atol 1.e-10 -ksp_type cg -pc_type gamg -primaldual 1 -symmetrize 1 -monitor 0 -snes_converged_reason

TEST*/

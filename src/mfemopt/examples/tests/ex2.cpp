static const char help[] = "Tests a parameter dependent one-dimensional diffusion.";

#include <petscopt.h>
#include <mfemopt.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>

#include <mfem.hpp>


#include <iostream>

using namespace mfem;
using namespace mfemopt;

typedef enum {FEC_L2, FEC_H1} FECType;
static const char *FECTypes[] = {"L2","H1","FecType","FEC_",0};
// TODO
//typedef enum {SIGMA_NONE, SIGMA_SCALAR, SIGMA_DIAG, SIGMA_FULL} SIGMAType;
//static const char *SIGMATypes[] = {"NONE","SCALAR","DIAG","FULL","SigmaType","SIGMA_",0};

/* auxiliary functions to perform refinement */
static int refine_fn(const Vector &x)
{
   for (int d = 0; d < x.Size(); d++) if (x(d) < -1.0 || x(d) > 1.0) return 0;
   return 1;
}

void NCRefinement(ParMesh *mesh, int (*fn)(const Vector&))
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
   for (int d = 0; d < x.Size(); d++) if (x(d) < -1.0 || x(d) > 1.0) return true;
   return false;
}

/* callbacks for the ParameterMap */
static double mu_m(double m)
{
   return std::exp(m);
}

static double dmu_dm(double m)
{
   return std::exp(m);
}

static double m_mu(double mu)
{
   return std::log(mu);
}

/* callbacks for parameter functions */
static double mu_exact_jump(const Vector &x)
{
   double m = 0;
   if (-0.7 <= x(0) && x(0) <= -0.2) m = -2.0;
   else if (0.3 <= x(0) && x(0) <= 0.7) m = -1.0;
   return mu_m(m);
}

static double mu_exact_const(const Vector &x)
{
   return 1.0;
}

static void sigma_exact(const Vector& x,DenseMatrix& K)
{
   K = 0.0;
   for (int i = 0; i < x.Size(); i++) K(i,i) = 1.e0;
}

/* The classes needed to define the objective function we want to minimize */

class MultiSourceMisfit;
class MultiSourceMisfitHessian: public Operator
{
private:
   Array<PetscParMatrix*> arrayH;
   Array<PetscODESolver*> arrayS;

public:
   MultiSourceMisfitHessian(const MultiSourceMisfit*,const Vector&);
   virtual void Mult(const Vector&,Vector&) const;
   ~MultiSourceMisfitHessian();
};

/*
  Misfit function : f(x,t) = 1/2 \int^tn_t0 \sum^{nreceivers}_{i=1} ||x(t,r_i) - s_i(t)||^2
   - x solution of the PDE
   - t time
   - r_i location of i-th receiver
   - s_i signal at i-th receiver
*/
class MultiSourceMisfit : public ReducedFunctional
{
private:
   ModelHeat *heat;

   Array<TDLeastSquares*> lsobj;
   Array<Coefficient*> sources;

   mutable ParGridFunction *u;
   mutable PetscParVector *U, *G;

   double t0, dt, tf;
   PetscBCHandler *bchandler;
   PetscODESolver *odesolver;

   mutable Operator *H;

   MPI_Comm comm;

protected:
   friend class MultiSourceMisfitHessian;
   mutable PetscParVector *M;

public:
   MultiSourceMisfit(ParFiniteElementSpace*,PDCoefficient*,MatrixCoefficient*,PetscBCHandler*,DenseMatrix&,DenseMatrix&,double,double,double,const char*);

   MPI_Comm GetComm() const { return comm; }

   virtual void ComputeObjective(const Vector&,double*) const;
   virtual void ComputeGradient(const Vector&,Vector&) const;
   virtual void ComputeObjectiveAndGradient(const Vector&,double*,Vector&) const;

   virtual Operator& GetHessian(const Vector&) const;

   virtual void ComputeGuess(Vector&) const;

   virtual ~MultiSourceMisfit();
};

/* Misfit function plus total variation regularizer */
class RegularizedMultiSourceMisfit : public ReducedFunctional
{
public:
   mutable Operator *H;
   mutable MultiSourceMisfit *obj;
   mutable TVRegularizer *reg;
   ParameterMap *pmap;

   RegularizedMultiSourceMisfit(MultiSourceMisfit*,TVRegularizer*,ParameterMap* = NULL);
   virtual void ComputeObjective(const Vector&,double*) const;
   virtual void ComputeGradient(const Vector&,Vector&) const;
   virtual Operator& GetHessian(const Vector&) const;
   virtual void PostCheck(const Vector&,Vector&,Vector&,bool&,bool&) const;
   virtual ~RegularizedMultiSourceMisfit() { delete H; }
};

class RegularizedMultiSourceMisfitHessian: public Operator
{
private:
   Operator *Hobj,*Hreg;
   ParameterMap *pmap;

public:
   RegularizedMultiSourceMisfitHessian(const RegularizedMultiSourceMisfit*,const Vector&);
   virtual void Mult(const Vector&,Vector&) const;
   virtual void MultTranspose(const Vector&,Vector&) const;
};

MultiSourceMisfit::MultiSourceMisfit(ParFiniteElementSpace* _fes, PDCoefficient* mu, MatrixCoefficient* sigma, PetscBCHandler* _bchandler,
                                     DenseMatrix& srcpoints, DenseMatrix& recpoints,
                                     double _t0, double _dt, double _tf,
                                     const char *scratch)
{
   /* store time limits */
   t0 = _t0;
   dt = _dt;
   tf = _tf;

   comm = _fes->GetParMesh()->GetComm();
   bchandler = _bchandler;

   /* mfem::TimeDependent operator (as PDOperator) */
   heat = new ModelHeat(mu,sigma,_fes,Operator::PETSC_MATAIJ);
   heat->SetBCHandler(*_bchandler);

   /* ReducedFunctional base class is mfem::Operator */
   height = width = heat->GetParameterSize();

   /* We use the same solver for objective and gradient computations */
   odesolver = new PetscODESolver(comm,"worker_");
   odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);
   odesolver->SetBCHandler(_bchandler); /* XXX fix inconsistency */
   odesolver->Customize();

   u = new ParGridFunction(_fes);
   U = new PetscParVector(_fes);

   /* Setup sources and receivers */
   for (int i = 0; i < srcpoints.Width(); i++)
   {
      Array<Receiver*> receivers;
      for (int j = 0; j < recpoints.Width(); j++)
      {
         receivers.Append(new Receiver(std::string(scratch) + "/src-" + std::to_string(i) + "-rec-" + std::to_string(j) + ".txt"));
      }
      lsobj.Append(new TDLeastSquares(receivers,_fes,true));

      Vector x;
      srcpoints.GetColumn(i,x);
      sources.Append(new RickerSource(x,10,1.1,1.0e3));
   }

   /* XXX No constructor with given local size */
   Vector g;
   g.SetSize(heat->GetParameterSize());
   G = new PetscParVector(comm,g);
   M = new PetscParVector(comm,g);

   H = NULL;
}

MultiSourceMisfit::~MultiSourceMisfit()
{
   delete heat;
   for (int i = 0; i < lsobj.Size(); i++) delete lsobj[i];
   for (int i = 0; i < sources.Size(); i++) delete sources[i];
   delete u;
   delete U;
   delete G;
   delete M;
   delete odesolver;
   delete H;
}

void MultiSourceMisfit::ComputeGuess(Vector& m)
{
   m.SetSize(heat->GetParameterSize());
   m = 1.0;
}

void MultiSourceMisfit::ComputeObjective(const Vector& m, double *f) const
{
   odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);

   /* Set callbacks for setup
      ModelHeat specific callbacks are handled in mfemopt_setupts */
   PetscErrorCode ierr;
   ierr = TSSetSetUpFromDesign(*odesolver,mfemopt_setupts,heat);PCHKERRQ(*odesolver,ierr);
   ierr = TSSetFromOptions(*odesolver);PCHKERRQ(*odesolver,ierr);

   *f = 0.0;
   M->PlaceArray(m.GetData());

   /* Loop over least-squares objectives
      TDLeastSquares specific callbacks are handled in mfemopt_eval_tdobj* */
   for (int i = 0; i < lsobj.Size(); i++)
   {
      ierr = TSResetObjective(*odesolver);PCHKERRQ(*odesolver,ierr);
      ierr = TSAddObjective(*odesolver,PETSC_MIN_REAL,
                            mfemopt_eval_tdobj,
                            mfemopt_eval_tdobj_x,NULL,
                            NULL,NULL,NULL,NULL,NULL,NULL,lsobj[i]);PCHKERRQ(*odesolver,ierr);

      heat->SetRHS(sources[i]);

      PetscReal rf;
      ierr = TSComputeObjectiveAndGradient(*odesolver,t0,dt,tf,*U,*M,NULL,&rf);PCHKERRQ(*odesolver,ierr);
      *f += rf;
   }

   M->ResetArray();
}

void MultiSourceMisfit::ComputeGradient(const Vector& m, Vector& g) const
{
   odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);

   /* Specify callbacks for the purpose of computing the gradient (wrt the model parameters) of the residual function */
   PetscErrorCode ierr;
   PetscParMatrix *A = new PetscParMatrix(comm,heat->GetGradientOperator(),Operator::PETSC_MATSHELL);
   ierr = TSSetGradientDAE(*odesolver,*A,mfemopt_gradientdae,NULL);PCHKERRQ(*odesolver,ierr);
   delete A;

   ierr = TSSetSetUpFromDesign(*odesolver,mfemopt_setupts,heat);PCHKERRQ(*odesolver,ierr);
   ierr = TSSetFromOptions(*odesolver);PCHKERRQ(*odesolver,ierr);

   g.SetSize(heat->GetParameterSize());
   g = 0.0;
   M->PlaceArray(m.GetData());

   /* Loop over least-squares objectives */
   for (int i = 0; i < lsobj.Size(); i++)
   {
      ierr = TSResetObjective(*odesolver);PCHKERRQ(*odesolver,ierr);
      ierr = TSAddObjective(*odesolver,PETSC_MIN_REAL,
                            mfemopt_eval_tdobj,
                            mfemopt_eval_tdobj_x,NULL,
                            NULL,NULL,NULL,NULL,NULL,NULL,lsobj[i]);PCHKERRQ(*odesolver,ierr);

      heat->SetRHS(sources[i]);

      ierr = TSComputeObjectiveAndGradient(*odesolver,t0,dt,tf,*U,*M,*G,NULL);PCHKERRQ(*odesolver,ierr);
      g += *G;
   }

   M->ResetArray();
}

void MultiSourceMisfit::ComputeObjectiveAndGradient(const Vector& m, double *f, Vector& g) const
{
   odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);

   PetscErrorCode ierr;
   PetscParMatrix *A = new PetscParMatrix(comm,heat->GetGradientOperator(),Operator::PETSC_MATSHELL);
   ierr = TSSetGradientDAE(*odesolver,*A,mfemopt_gradientdae,NULL);PCHKERRQ(*odesolver,ierr);
   delete A;

   ierr = TSSetSetUpFromDesign(*odesolver,mfemopt_setupts,heat);PCHKERRQ(*odesolver,ierr);
   ierr = TSSetFromOptions(*odesolver);PCHKERRQ(*odesolver,ierr);

   *f = 0.0;

   g.SetSize(heat->GetParameterSize());
   g = 0.0;

   M->PlaceArray(m.GetData());

   /* Loop over least-squares objectives */
   for (int i = 0; i < lsobj.Size(); i++)
   {
      ierr = TSResetObjective(*odesolver);PCHKERRQ(*odesolver,ierr);
      ierr = TSAddObjective(*odesolver,PETSC_MIN_REAL,
                            mfemopt_eval_tdobj,
                            mfemopt_eval_tdobj_x,NULL,
                            NULL,NULL,NULL,NULL,NULL,NULL,lsobj[i]);PCHKERRQ(*odesolver,ierr);

      heat->SetRHS(sources[i]);

      PetscReal rf;
      ierr = TSComputeObjectiveAndGradient(*odesolver,t0,dt,tf,*U,*M,*G,&rf);PCHKERRQ(*odesolver,ierr);
      *f += rf;
      g += *G;
   }

   M->ResetArray();
}

Operator& MultiSourceMisfit::GetHessian(const Vector& m) const
{
   delete H;
   H = new MultiSourceMisfitHessian(this,m);
   return *H;
}

MultiSourceMisfitHessian::MultiSourceMisfitHessian(const MultiSourceMisfit* _msobj, const Vector& _m) : Operator(_m.Size(),_m.Size())
{
   MPI_Comm comm = _msobj->GetComm();

   arrayH.SetSize(_msobj->lsobj.Size());
   arrayS.SetSize(_msobj->lsobj.Size());

   PetscParMatrix *A = new PetscParMatrix(comm,_msobj->heat->GetGradientOperator(),Operator::PETSC_MATSHELL);

   _msobj->M->PlaceArray(_m.GetData());

   /* Loop over least-squares objectives */
   for (int i = 0; i < _msobj->lsobj.Size(); i++)
   {
      /* we create new solvers to not interfere with the gradient solver */
      PetscODESolver *odesolver = new PetscODESolver(comm,"worker_");
      odesolver->Init(*(_msobj->heat),PetscODESolver::ODE_SOLVER_LINEAR);
      odesolver->SetBCHandler(_msobj->bchandler);

      PetscErrorCode ierr;
      ierr = TSSetGradientDAE(*odesolver,*A,mfemopt_gradientdae,NULL);CCHKERRQ(comm,ierr);
      ierr = TSSetHessianDAE(*odesolver,NULL,NULL,NULL,
                                        NULL,NULL,mfemopt_hessiandae_xtm,
                                        NULL,mfemopt_hessiandae_mxt,NULL,_msobj->heat);CCHKERRQ(comm,ierr);
      ierr = TSSetSetUpFromDesign(*odesolver,mfemopt_setupts,_msobj->heat);CCHKERRQ(comm,ierr);


      PetscParMatrix *Hls = new PetscParMatrix(comm,_msobj->lsobj[i]->GetHessianOperator_XX(),Operator::PETSC_MATSHELL);
      ierr = TSAddObjective(*odesolver,PETSC_MIN_REAL,
                            mfemopt_eval_tdobj,
                            mfemopt_eval_tdobj_x,NULL,
                            *Hls,NULL,NULL,NULL,NULL,NULL,_msobj->lsobj[i]);CCHKERRQ(comm,ierr);
      delete Hls;

      _msobj->heat->SetRHS(_msobj->sources[i]);

      odesolver->Customize();

      Mat pH;
      ierr = MatCreate(comm,&pH);CCHKERRQ(comm,ierr);
      ierr = TSComputeHessian(*odesolver,_msobj->t0,_msobj->dt,_msobj->tf,*(_msobj->U),*(_msobj->M),pH);CCHKERRQ(comm,ierr);
      arrayH[i] = new PetscParMatrix(pH,false);
      arrayS[i] = odesolver;
   }
   delete A;
   _msobj->M->ResetArray();
}

void MultiSourceMisfitHessian::Mult(const Vector& x, Vector& y) const
{
   y.SetSize(x.Size());
   y = 0.0;
   for (int i = 0; i < arrayH.Size(); i++)
   {
      arrayH[i]->Mult(1.0,x,1.0,y);
   }
}

MultiSourceMisfitHessian::~MultiSourceMisfitHessian()
{
   for (int i = 0; i < arrayH.Size(); i++) { delete arrayH[i]; }
   for (int i = 0; i < arrayS.Size(); i++) { delete arrayS[i]; }
}

RegularizedMultiSourceMisfit::RegularizedMultiSourceMisfit(MultiSourceMisfit *_obj, TVRegularizer *_reg, ParameterMap *_pmap)
{
   obj  = _obj;
   reg  = _reg;
   pmap = _pmap;
   H    = NULL;

   height = width = _obj->Height();
}

void RegularizedMultiSourceMisfit::ComputeObjective(const Vector& m, double *f) const
{
   Vector pm;
   if (pmap) pmap->Map(m,pm);
   else pm = m;

   double f1,f2;
   Vector dummy;
   obj->ComputeObjective(pm,&f1);
   reg->Eval(dummy,pm,0.,&f2);
   *f = f1 + f2;
}

void RegularizedMultiSourceMisfit::ComputeGradient(const Vector& m, Vector& g) const
{
   Vector pm;
   if (pmap) pmap->Map(m,pm);
   else pm = m;

   Vector dummy;
   Vector g1,g2;
   g1.SetSize(pm.Size());
   g2.SetSize(pm.Size());

   obj->ComputeGradient(pm,g1);
   reg->EvalGradient_M(dummy,pm,0.,g2);
   g1 += g2;
   if (pmap) pmap->GradientMap(m,g1,true,g);
   else g = g1;
}

Operator& RegularizedMultiSourceMisfit::GetHessian(const Vector& m) const
{
   delete H;
   H = new RegularizedMultiSourceMisfitHessian(this,m);
   return *H;
}

void RegularizedMultiSourceMisfit::PostCheck(const Vector& X, Vector& Y, Vector &W, bool& cy, bool& cw) const
{
   /* we don't change the step (Y) or the updated solution (W = X - lambda*Y) */
   cy = false;
   cw = false;
   double lambda = X.Size() ? (X[0] - W[0])/Y[0] : 0.0;
   reg->UpdateDual(X,Y,lambda);
}

/*
   The hessian of the full objective
   - H = H_map + J(mu(m))^T * ( H_tv + H_misfit ) J(mu(m))
*/
RegularizedMultiSourceMisfitHessian::RegularizedMultiSourceMisfitHessian(const RegularizedMultiSourceMisfit* _rmsobj, const Vector& _m)
{
   height = width = _m.Size();
   pmap = _rmsobj->pmap;

   Vector pm;
   if (pmap) pmap->Map(_m,pm);
   else pm = _m;

   Hobj = &( _rmsobj->obj->GetHessian(pm));

   Vector dummy;
   _rmsobj->reg->SetUpHessian_MM(dummy,pm,0.);
   Hreg = _rmsobj->reg->GetHessianOperator_MM();

   if (pmap && pmap->SecondOrder())
   {
      Vector g1,g2;
      g1.SetSize(pm.Size());
      g2.SetSize(pm.Size());

      _rmsobj->obj->ComputeGradient(pm,g1);
      _rmsobj->reg->EvalGradient_M(dummy,pm,0.,g2);
      g1 += g2;
      pmap->SetUpHessianMap(_m,g1);
   }
}

void RegularizedMultiSourceMisfitHessian::Mult(const Vector& x, Vector& y) const
{
   Vector px,py1,py2;

   if (pmap)
   {
      const Vector& m = pmap->GetParameter();
      pmap->GradientMap(m,x,false,px);
   }
   else px = x;
   py1.SetSize(px.Size());
   py2.SetSize(px.Size());
   Hobj->Mult(px,py1);
   Hreg->Mult(px,py2);
   py1 += py2;
   if (pmap)
   {
      const Vector& m = pmap->GetParameter();
      pmap->GradientMap(m,py1,true,y);
      if (pmap->SecondOrder())
      {
         Vector y2;

         pmap->HessianMult(x,y2);
         y += y2;
      }
   }
   else
   {
      y = py1;
   }
}

/* We need the transpose callback just in case the Newton solver fails, and PETSc requests it */
void RegularizedMultiSourceMisfitHessian::MultTranspose(const Vector& x, Vector& y) const
{
   Vector px,py1,py2;

   if (pmap)
   {
      const Vector& m = pmap->GetParameter();
      pmap->GradientMap(m,x,false,px);
   }
   else px = x;
   py1.SetSize(px.Size());
   py2.SetSize(px.Size());
   Hobj->Mult(px,py1); /* the Hessian of the misfit function is symmetric */
   Hreg->MultTranspose(px,py2);
   py1 += py2;
   if (pmap)
   {
      const Vector& m = pmap->GetParameter();
      pmap->GradientMap(m,py1,true,y);
      if (pmap->SecondOrder())
      {
         Vector y2;

         pmap->HessianMult(x,y2);
         y += y2;
      }
   }
   else
   {
      y = py1;
   }
}

#if 0
PetscParMatrix* RegularizedMultiSourceMisfitHessian::CreatePmat() const
{
   MPI_Comm comm = PETSC_COMM_WORLD/*XXX*/;

   PetscParMatrix *pH = new PetscParMatrix();
   PetscParMatrix *pHreg = new PetscParMatrix(comm,Hreg,Operator::PETSC_MATAIJ);
   *pH = *pHreg;
   delete pHreg;

   if (pmap) /* XXX valid only for diagonal maps */
   {
      const Vector& m = pmap->GetParameter();
      Vector x((*this).Height()),y((*this).Height()),px;
      x = 1.0;
      pmap->GradientMap(m,x,false,px);
      y = 0.0;
      if (pmap->SecondOrder())
      {
         pmap->HessianMult(x,y);
      }
      pH->ScaleRows(px);
      pH->ScaleCols(px);
      pH->Shift(y);
   }
   return pH;
}

typedef struct
{
   mfem::Operator *op;
} __mfem_mat_shell_ctx;


class RMSHPFactory : public PetscPreconditionerFactory
{
public:
   RMSHPFactory() : PetscPreconditionerFactory("Hessian preconditioner") { }
   virtual Solver *NewPreconditioner(const OperatorHandle& oh);
};

Solver* RMSHPFactory::NewPreconditioner(const OperatorHandle& oh)
{
  Solver *solver = NULL;
  PetscParMatrix *pA;
  oh.Get(pA);
  std::cout << "HEY" << std::endl;
  if (oh.Type() == Operator::PETSC_MATSHELL)
  {
      Mat A = *pA;
      __mfem_mat_shell_ctx *ctx;
      PetscErrorCode       ierr;

      ierr = MatShellGetContext(A,(void **)&ctx); PCHKERRQ(A,ierr);
      RegularizedMultiSourceMisfitHessian *H = dynamic_cast<RegularizedMultiSourceMisfitHessian*>(ctx->op);
      MFEM_VERIFY(H,"expected RMSOH");
      PetscParMatrix *pH = H->CreatePmat();
      solver = new PetscPreconditioner(*pH,"factory_");
      delete pH;
  }
  else
  {
     MFEM_VERIFY(0,"Unhandled type " << oh.Type());
  }
  return solver;
}
#endif

/* Monitors for real time graphics */
class UserMonitor : public PetscSolverMonitor
{
private:
   ParGridFunction *u;
   ParameterMap    *pmap;
   PDCoefficient   *m;
   int             vt;
   std::string     name;
   bool            pause;
   socketstream    sout;

public:
   UserMonitor(ParGridFunction* _u, int _vt = 1, const std::string &_name = "Solution") :
      PetscSolverMonitor(true,false), u(_u), pmap(NULL), m(NULL), vt(_vt), name(_name), pause(true)
   {
      if (vt > 0)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         sout.open(vishost, visport);
         if (!sout)
         {
            if (!PetscGlobalRank)
               std::cout << "Unable to connect to GLVis server at "
                    << vishost << ':' << visport << std::endl;
            if (!PetscGlobalRank)
            {
               std::cout << "GLVis visualization disabled." << std::endl;
            }
         }
         else
         {
            sout.precision(6);
         }
      }
   }

   UserMonitor(PDCoefficient* _m, ParameterMap* _pmap, int _vt = 1, const std::string &_name = "Solution") :
      PetscSolverMonitor(true,false)
   {
      m = _m;
      Array<ParGridFunction*> gf = m->GetCoeffs();
      pmap = _pmap;
      u = gf[0];
      vt = _vt;
      name = _name;
      pause = true;
      if (vt > 0)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         sout.open(vishost, visport);
         if (!sout)
         {
            if (!PetscGlobalRank)
               std::cout << "Unable to connect to GLVis server at "
                    << vishost << ':' << visport << std::endl;
            if (!PetscGlobalRank)
            {
               std::cout << "GLVis visualization disabled." << std::endl;
            }
         }
         else
         {
            sout.precision(6);
         }
      }
   }

   ~UserMonitor() {}

   void MonitorSolution(PetscInt step, PetscReal time, const Vector &X)
   {
      if (vt <= 0) return;
      ParFiniteElementSpace *pfes = u->ParFESpace();
      if (!pmap && !m)
      {
         HypreParMatrix &P = *pfes->Dof_TrueDof_Matrix();
         P.Mult(X,*u);
      }
      else
      {
         Vector pu;
         if (pmap) pmap->Map(X,pu);
         else pu = X;
         if (m)
         {
            m->UpdateCoefficient(pu); /* XXX WITH GF! */
         }
         else
         {
            HypreParMatrix &P = *pfes->Dof_TrueDof_Matrix();
            P.Mult(pu,*u);
         }
      }

      if (sout && step % vt == 0)
      {
         int  num_procs, myid;

         ParMesh * pmesh = pfes->GetParMesh();
         MPI_Comm_size(pmesh->GetComm(),&num_procs);
         MPI_Comm_rank(pmesh->GetComm(),&myid);
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << *pmesh << *u;
         if (pause)
         {
           sout << "window_size 800 800\n";
           sout << "window_title '" << name << "'\n";
           sout << "keys cm\n";
           sout << "pause\n";
         }
         sout << std::flush;
         if (pause)
         {
            pause = false;
            if (myid == 0)
            {
               std::cout << "GLVis visualization paused. Press space (in the GLVis window) to resume it." << std::endl;
            }
         }
      }
   }
};

int main(int argc, char *argv[])
{
   MFEMInitializePetsc(&argc,&argv,NULL,help);

   PetscInt ord = 1, srl = 0, prl = 0, viz = 0, ncrl = 0;

   PetscInt  gridr[3] = {1,1,1};
   PetscInt  grids[3] = {1,1,1};
   PetscReal gridr_bb[6] = {-1.0,1.0,-1.0,1.0,-1.0,1.0};
   PetscReal grids_bb[6] = {-1.0,1.0,-1.0,1.0,-1.0,1.0};

   char scratchdir[PETSC_MAX_PATH_LEN] = "/tmp";
   char meshfile[PETSC_MAX_PATH_LEN] = "../../../../share/petscopt/meshes/segment-m5-5.mesh";
   PetscReal t0 = 0.0, dt = 1.e-3, tf = 0.1;

   FECType   mu_fec_type = FEC_H1;
   PetscInt  mu_ord = 0;
   PetscInt  n_mu_excl = 1024;
   PetscInt  mu_excl[1024];
   PetscBool mu_excl_fn = PETSC_FALSE, mu_with_jumps = PETSC_FALSE;

   PetscReal tva = 1.0, tvb = 0.1;
   PetscBool tvpd = PETSC_TRUE, tvsy = PETSC_FALSE, tvpr = PETSC_FALSE;

   PetscBool test_null = PETSC_FALSE, test_newton = PETSC_TRUE, test_progress = PETSC_TRUE;
   PetscBool test_misfit[3] = {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}, test_misfit_reg[2] = {PETSC_FALSE,PETSC_FALSE};
   PetscReal test_newton_noise = 0.0;
   PetscBool glvis = PETSC_TRUE;

   /* Process options */
   {
      PetscBool      flg;
      PetscInt       i,j;
      PetscErrorCode ierr;

      ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for Heat equation",NULL);CHKERRQ(ierr);

      /* Simulation parameters */
      ierr = PetscOptionsString("-scratch","Location where to put temporary data (must be present)",NULL,scratchdir,scratchdir,sizeof(scratchdir),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsString("-meshfile","Mesh filename",NULL,meshfile,meshfile,sizeof(meshfile),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-ord","FEM order",NULL,ord,&ord,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-srl","Number of sequential refinements",NULL,srl,&srl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-prl","Number of parallel refinements",NULL,prl,&prl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-ncrl","Number of non-conforming refinements (refines element with center in [-1,1]^d)",NULL,ncrl,&ncrl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-t0","Initial time",NULL,t0,&t0,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-dt","Initial time step",NULL,dt,&dt,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-tf","Final time",NULL,tf,&tf,NULL);CHKERRQ(ierr);

      /* GLVis */
      ierr = PetscOptionsInt("-viz","Visualization steps for model sampling",NULL,viz,&viz,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-glvis","Activate GLVis monitoring of Newton process",NULL,glvis,&glvis,NULL);CHKERRQ(ierr);

      /* Sources and receivers */
      ierr = PetscOptionsIntArray("-grid_rcv_n","Grid receivers: points per direction",NULL,gridr,(i=3,&i),NULL);CHKERRQ(ierr);
      for (j=i;j<3;j++) gridr[j] = gridr[i > 0 ? i-1 : 0];
      ierr = PetscOptionsRealArray("-grid_rcv_bb","Grid receivers: bounding box",NULL,gridr_bb,(i=6,&i),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray("-grid_src_n","Grid sources: points per direction",NULL,grids,(i=3,&i),NULL);CHKERRQ(ierr);
      for (j=i;j<3;j++) grids[j] = grids[i > 0 ? i-1 : 0];
      ierr = PetscOptionsRealArray("-grid_src_bb","Grid sources: bounding box",NULL,grids_bb,(i=6,&i),NULL);CHKERRQ(ierr);

      /* Parameter space */
      ierr = PetscOptionsInt("-mu_ord","Polynomial order approximation for mu",NULL,mu_ord,&mu_ord,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-mu_jumps","Use jumping target for mu",NULL,mu_with_jumps,&mu_with_jumps,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnum("-mu_fec_type","FEC for mu","",FECTypes,(PetscEnum)mu_fec_type,(PetscEnum*)&mu_fec_type,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-mu_exclude_fn","Excludes elements outside [-1,1]^d for mu optimization",NULL,mu_excl_fn,&mu_excl_fn,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray("-mu_exclude","Elements' tag to exclude for mu optimization",NULL,mu_excl,&n_mu_excl,&flg);CHKERRQ(ierr);
      if (!flg) n_mu_excl = 0;

      /* TV options */
      ierr = PetscOptionsReal("-tv_alpha","",NULL,tva,&tva,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-tv_beta","",NULL,tvb,&tvb,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-tv_pd","",NULL,tvpd,&tvpd,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-tv_symm","",NULL,tvsy,&tvsy,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-tv_proj","",NULL,tvpr,&tvpr,NULL);CHKERRQ(ierr);

      ierr = PetscOptionsBool("-test_newton","Test Newton solver",NULL,test_newton,&test_newton,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-test_newton_noise","Test Newton solver: noise level",NULL,test_newton_noise,&test_newton_noise,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-test_null","Use exact solution when testing",NULL,test_null,&test_null,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-test_progress","Report progress when testing",NULL,test_progress,&test_progress,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBoolArray("-test_misfit","Test misfit function callbacks",NULL,test_misfit,(i=3,&i),NULL);CHKERRQ(ierr);
      for (int j=i; j<3; j++) test_misfit[j] = test_misfit[i > 0 ? i-1 : 0];
      ierr = PetscOptionsBoolArray("-test_misfit_reg","Test regularized misfit function callbacks",NULL,test_misfit_reg,(i=2,&i),NULL);CHKERRQ(ierr);
      for (int j=i; j<2; j++) test_misfit_reg[j] = test_misfit_reg[i > 0 ? i-1 : 0];

      ierr = PetscOptionsEnd();CHKERRQ(ierr);
   }

   if (mu_fec_type == FEC_H1 && !mu_ord) mu_ord = 1;
   Array<int> mu_excl_a(mu_excl,n_mu_excl);

   /* Create mesh and finite element space for the independent variable */
   ParMesh *pmesh = NULL;
   {
      Mesh *mesh = new Mesh(meshfile, 1, 1);
      MFEM_VERIFY(mesh->SpaceDimension() == mesh->Dimension(),"Embedded meshes not supported")
      for (int lev = 0; lev < srl; lev++)
      {
         mesh->UniformRefinement();
      }
      mesh->EnsureNCMesh();

      pmesh = new ParMesh(PETSC_COMM_WORLD, *mesh);
      delete mesh;
      for (int lev = 0; lev < prl; lev++)
      {
         pmesh->UniformRefinement();
      }
      for (int lev = 0; lev < ncrl; lev++)
      {
         NCRefinement(pmesh,refine_fn);
      }
   }
   FiniteElementCollection *fec = new H1_FECollection(ord, pmesh->Dimension());
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, fec);

   /* Boundary conditions handler */
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   PetscBCHandler *bchandler = new PetscBCHandler(ess_tdof_list);

   /* Sources */
   PetscInt ns = 1;
   for (PetscInt i = pmesh->SpaceDimension(); i < 3; i++) grids[i] = 1;
   for (PetscInt i = 0; i < pmesh->SpaceDimension(); i++) ns *= grids[i];
   DenseMatrix srcpoints(pmesh->SpaceDimension(),ns);
   ns = 0;
   for (PetscInt k = 0 ; k < grids[2]; k++)
   {
      for (PetscInt j = 0 ; j < grids[1]; j++)
      {
         for (PetscInt i = 0 ; i < grids[0]; i++)
         {
            Vector    point;
            PetscReal parpoint[3];

            parpoint[0] = (i + 0.5)/grids[0];
            parpoint[1] = (j + 0.5)/grids[1];
            parpoint[2] = (k + 0.5)/grids[2];

            srcpoints.GetColumnReference(ns,point);
            for (int d = 0; d < pmesh->SpaceDimension(); d++) point(d) = grids_bb[2*d] + (grids_bb[2*d+1]-grids_bb[2*d])*parpoint[d];
            ns++;
         }
      }
   }

   /* Receivers */
   PetscInt nr = 1;
   for (PetscInt i = pmesh->SpaceDimension(); i < 3; i++) gridr[i] = 1;
   for (PetscInt i = 0; i < pmesh->SpaceDimension(); i++) nr *= gridr[i];
   DenseMatrix recpoints(pmesh->SpaceDimension(),nr);
   nr = 0;
   for (PetscInt k = 0 ; k < gridr[2]; k++)
   {
      for (PetscInt j = 0 ; j < gridr[1]; j++)
      {
         for (PetscInt i = 0 ; i < gridr[0]; i++)
         {
            Vector    point;
            PetscReal parpoint[3];

            parpoint[0] = (i + 0.5)/gridr[0];
            parpoint[1] = (j + 0.5)/gridr[1];
            parpoint[2] = (k + 0.5)/gridr[2];

            recpoints.GetColumnReference(nr,point);
            for (int d = 0; d < pmesh->SpaceDimension(); d++) point(d) = gridr_bb[2*d] + (gridr_bb[2*d+1]-gridr_bb[2*d])*parpoint[d];
            nr++;
         }
      }
   }
   if (!PetscGlobalRank)
   {
      std::cout << "Sources matrix" << std::endl;
      srcpoints.Print();
      std::cout << "Receivers matrix" << std::endl;
      recpoints.Print();
   }

   /* exact PDE coefficients */
   MatrixFunctionCoefficient *sigma = new MatrixFunctionCoefficient(pmesh->SpaceDimension(),sigma_exact);
   FunctionCoefficient *mu;
   if (mu_with_jumps) mu = new FunctionCoefficient(mu_exact_jump);
   else mu = new FunctionCoefficient(mu_exact_const);

   /* Multi-source sampling */
   for (int i = 0; i < srcpoints.Width(); i++)
   {
      ParGridFunction *u = new ParGridFunction(fes);
      HypreParVector *U = u->GetTrueDofs();
      *U = 0.0;

      UserMonitor *monitor = new UserMonitor(u,viz,"Fwd source - " + std::to_string(i));
      ReceiverMonitor *rmonitor = new ReceiverMonitor(u,recpoints,std::string(scratchdir) + "/src-" + std::to_string(i) + "-rec");

      ModelHeat *heat = new ModelHeat(mu,sigma,fes,Operator::PETSC_MATAIJ);

      Vector srcctr;
      srcpoints.GetColumn(i,srcctr);
      RickerSource rick(srcctr,10,1.1,1.0e3);
      heat->SetRHS(&rick);

      PetscODESolver *odesolver = new PetscODESolver(pmesh->GetComm(),"model_");
      odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);
      odesolver->SetBCHandler(bchandler);
      odesolver->SetMonitor(rmonitor);
      odesolver->SetMonitor(monitor);

      double tt0 = t0, tdt = dt, ttf = tf;
      odesolver->Run(*U,tt0,tdt,ttf);

      delete monitor;
      delete rmonitor; /* this triggers the dumping of the data */
      delete odesolver;
      delete heat;
      delete u;
      delete U;
   }

   /* Optimization space */
   FiniteElementCollection *mu_fec = NULL;
   switch (mu_fec_type)
   {
      case FEC_L2:
         mu_fec = new L2_FECollection(mu_ord,pmesh->Dimension());
         break;
      case FEC_H1:
         mu_fec = new H1_FECollection(mu_ord,pmesh->Dimension());
         break;
      default:
         MFEM_ABORT("Unhandled FEC Type");
         break;
   }
   ParFiniteElementSpace *mu_fes = new ParFiniteElementSpace(pmesh,mu_fec);

   /* The parameter dependent coefficient for mu
      We construct the guess from the exact solution just for testing purposes */
   PDCoefficient *mu_pd;
   if (mu_excl_fn) mu_pd = new PDCoefficient(*mu,mu_fes,excl_fn);
   else mu_pd = new PDCoefficient(*mu,mu_fes,mu_excl_a);

   /* Exact solution */
   Vector muv_exact;
   mu_pd->GetCurrentVector(muv_exact);

   /* The misfit function */
   MultiSourceMisfit *obj = new MultiSourceMisfit(fes,mu_pd,sigma,bchandler,srcpoints,recpoints,t0,dt,tf,scratchdir);

   /* Test misfit callbacks */
   if (test_misfit[0])
   {
      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      double f1,f2;
      PetscParVector g1(PETSC_COMM_WORLD,muv), g2(PETSC_COMM_WORLD,muv);
      obj->ComputeObjectiveAndGradient(muv,&f1,g1);
      obj->ComputeObjective(muv,&f2);
      obj->ComputeGradient(muv,g2);
      MFEM_VERIFY(std::abs(f1-f2) < PETSC_SMALL,"Error on misfit computations! " << f1 << ", " << f2)
      g1 -= g2;
      f1 = ParNormlp(g1,infinity(),PETSC_COMM_WORLD);
      MFEM_VERIFY(f1 < PETSC_SMALL,"Error on misfit computations! gradient error " << f1)
      if (!PetscGlobalRank) std::cout << "Misfit objective " << f2 << std::endl;
   }

   /* FD test for misfit function gradient */
   if (test_misfit[1])
   {
      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      obj->TestFDGradient(PETSC_COMM_WORLD,muv,1.e-8,test_progress);
   }

   /* matrix-free FD test for misfit function hessian */
   if (test_misfit[2])
   {
      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      obj->TestFDHessian(PETSC_COMM_WORLD,muv);
   }

   /* Total variation regularizer */
   TVRegularizer *tv = new TVRegularizer(mu_pd,tva,tvb,tvpd);
   tv->Symmetrize(tvsy);
   tv->Project(tvpr);

   /* Map from optimization variables to model variables */
   PointwiseMap pmap(mu_m,m_mu,dmu_dm,dmu_dm);

   /* The full objective: misfit + regularization */
   RegularizedMultiSourceMisfit *robj = new RegularizedMultiSourceMisfit(obj,tv,&pmap);

   /* Test callbacks for full objective */
   if (test_misfit_reg[0])
   {
      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      /* Misfit in terms of mu, the optimization variable is the inverse of the map */
      Vector m;
      pmap.InverseMap(muv,m);

      double f;
      robj->ComputeObjective(m,&f);
      if (!PetscGlobalRank) std::cout << "Regularized objective " << f << std::endl;
   }

   /* Test callbacks for full objective gradient */
   if (test_misfit_reg[1])
   {
      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      /* Misfit in terms of mu, the optimization variable is the inverse of the map */
      Vector m;
      pmap.InverseMap(muv,m);

      Vector y;
      robj->Mult(m,y);
   }

   /* Test Newton solver */
   if (test_newton) {
      PetscNonlinearSolverOpt newton(PETSC_COMM_WORLD,*robj,"newton_");

      newton.SetJacobianType(Operator::PETSC_MATSHELL);
      newton.iterative_mode = true; /* we always use an initial guess, it can be zero */

#if 0
      RMSHPFactory myfactory;
      newton.SetPreconditionerFactory(&myfactory);
#endif
      NewtonMonitor mymonitor;
      newton.SetMonitor(&mymonitor);
      UserMonitor solmonitor(mu_pd,&pmap,glvis ? 1 : 0,"Newton solution");
      if (glvis) newton.SetMonitor(&solmonitor);

      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      Vector dummy,u;
      pmap.InverseMap(muv,u);

      GaussianNoise nnoise;
      Vector vnoise;
      nnoise.random(vnoise,u.Size());
      vnoise *= test_newton_noise;
      u += vnoise;

      newton.Mult(dummy,u);
      //mu_pd->UpdateCoefficient(u);
      //mu_pd->Save("reconstructed_sigma");
      //mu_pd->Visualize("RJlc");
   }

   delete robj;
   delete tv;
   delete mu_fec;
   delete mu_fes;
   delete mu_pd;
   delete obj;

   delete bchandler;
   delete sigma;
   delete mu;
   delete fec;
   delete fes;
   delete pmesh;

   MFEMFinalizePetsc();
   return 0;
}

/*TEST

   build:
     requires: mfemopt

   testset:
     args: -meshfile ${petscopt_dir}/share/petscopt/meshes/segment-m5-5.mesh -ts_trajectory_type memory -ts_trajectory_reconstruction_order 2 -mfem_use_splitjac -model_ts_type cn -model_ksp_type cg -worker_ts_max_snes_failures -1 -worker_ts_type cn -worker_ksp_type cg
     test:
       suffix: null_test
       args: -test_misfit 1 -test_misfit_reg 1 -test_newton -test_progress 0 -tv_alpha 0 -newton_pc_type none -newton_snes_atol 1.e-8 -glvis 0
     test:
       suffix: newton_test
       args: -glvis 0 -newton_snes_converged_reason -newton_snes_max_it 1 -newton_snes_test_jacobian -newton_snes_test_jacobian_view -newton_snes_rtol 1.e-6 -newton_snes_atol 1.e-6 -newton_snes_view -newton_ksp_type fgmres -newton_pc_type none -mu_jumps -tv_alpha 0.01 -mu_exclude_fn -ncrl 1
     test:
       suffix: newton_full
       args: -glvis 0 -newton_snes_converged_reason -newton_snes_max_it 10 -newton_snes_rtol 1.e-6 -newton_snes_atol 1.e-6 -newton_snes_view -newton_ksp_type fgmres -newton_pc_type none -mu_jumps -tv_alpha 0.01 -mu_exclude 2 -ncrl 1
TEST*/

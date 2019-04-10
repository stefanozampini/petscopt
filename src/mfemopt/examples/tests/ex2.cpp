static const char help[] = "Tests a parameter dependent time-dependent diffusion.";

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
typedef enum {OID_MATAIJ, OID_MATIS, OID_MATHYPRE, OID_HYPRE, OID_ANY} OIDType;
static const char *OIDTypes[] = {"PETSC_MATAIJ",
                                 "PETSC_MATIS",
                                 "PETSC_MATHYPRE",
                                 "Hypre_ParCSR",
                                 "ANY_TYPE",
                                 "OIDType",
                                 "OID_",0};

// TODO
//typedef enum {SIGMA_NONE, SIGMA_SCALAR, SIGMA_DIAG, SIGMA_FULL} SIGMAType;
//static const char *SIGMATypes[] = {"NONE","SCALAR","DIAG","FULL","SigmaType","SIGMA_",0};

/* auxiliary functions to perform refinement */
PetscReal refine_fn_bb[6] = {-1.0,1.0,-1.0,1.0,-1.0,1.0};

static int refine_fn(const Vector &x)
{
   int r = 1;
   for (int d = 0; d < x.Size(); d++) if (x(d) < refine_fn_bb[2*d] || x(d) > refine_fn_bb[2*d+1]) r = 0;
   return r;
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
   mesh->GeneralRefinement(el_to_refine,-1);
#if 0
   if (mesh->Nonconforming())
   {
      mesh->Rebalance();
   }
#endif
}

/* experimental */
static double mesh_scal = 1.0;
static double signal_scal = 1.0e3;
static double mu_guess = 1.0;
static bool oil_test = false; // YYY

PetscReal excl_fn_bb[6] = {-1.0,1.0,-1.0,1.0,-1.0,1.0};

static bool excl_fn(const Vector &x)
{
   for (int d = 0; d < x.Size(); d++) if (x(d) <= excl_fn_bb[2*d] || x(d) >= excl_fn_bb[2*d+1]) return true;
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

#if 0
static double mu_exact_jump(const Vector &x)
{
   double m = 0;
   if (1000 <= x(0) && x(0) <= 2000) m = -2.0;
   else if (3000 <= x(0) && x(0) <= 6000) m = -1.0;
   return mu_m(m);
}
#endif

static double mu_const_val = 1.0;

static double mu_exact_const(const Vector &x)
{
   return mu_const_val;
}

static double mu_exact_jump_test_em_2d(const Vector &x)
{
   double val;
   double xx = x(0)*mesh_scal, yy=x(1)*mesh_scal;

   double water_v = 3.0;
   double back_v = 1.e-1;
   double oil_v = 1.e-3;
   double salt_v = 1.e-2;

   if (yy < 600) val = back_v;
   else val = water_v;

   double oil_1[2][2] = {{300,450},{1000,550}};
   double oil_2[2][2] = {{550,350},{750,450}};
   //double oil_1[2][2] = {{300,350},{1000,450}};
   //double oil_2[2][2] = {{550,250},{750,350}};
   if (oil_1[0][0] <= xx && xx <= oil_1[1][0] &&
       oil_1[0][1] <= yy && yy <= oil_1[1][1]) val = oil_v;
   if (oil_2[0][0] <= xx && xx <= oil_2[1][0] &&
       oil_2[0][1] <= yy && yy <= oil_2[1][1]) val = oil_v;

   double dome_r = 250.0, dome_c[2] = {1250.0,0.0};
   if (std::sqrt((xx - dome_c[0])*(xx - dome_c[0]) + (yy - dome_c[1])*(yy - dome_c[1])) < dome_r) val = salt_v;
   if (yy < 0) val = back_v;
   return val*mesh_scal;
}

/* inverse of magnetic permeability of free space (meters) */
static double sigma_scal_muinv_0 = 1.0/(4 * M_PI * 1.0e-7);

static double sigma_scal = 1.0;

static double sigma_exact(const Vector& x)
{
   return sigma_scal;
}

/* The classes needed to define the objective function we want to minimize */

class MultiSourceMisfit;
class MultiSourceMisfitHessian: public Operator
{
private:
   Array<PetscParMatrix*> arrayH;
   Array<PetscODESolver*> arrayS;
   const MultiSourceMisfit *msobj;

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
   Array<VectorCoefficient*> vsources;

   mutable ParGridFunction *u;
   mutable PetscParVector *U, *G;

   double t0, dt, tf;
   PetscBCHandler *bchandler;
   PetscODESolver *odesolver;

   mutable Operator *H;

   double scale_ls;

   MPI_Comm comm;

protected:
   friend class MultiSourceMisfitHessian;
   mutable PetscParVector *M;

public:
   MultiSourceMisfit(ParFiniteElementSpace*,PDCoefficient*,Coefficient*,Operator::Type,Operator::Type,PetscBCHandler*,DenseMatrix&,int,DenseMatrix&,bool,double,double,double,double,const char*);

   MPI_Comm GetComm() const { return comm; }
   void RunTests(bool=true);

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
   DataReplicator *drep;
   ParameterMap *pmap;
   bool tvopt;

   RegularizedMultiSourceMisfit(MultiSourceMisfit*,TVRegularizer*,DataReplicator*,ParameterMap*,bool tvopt = false);
   virtual void ComputeObjective(const Vector&,double*) const;
   virtual void ComputeGradient(const Vector&,Vector&) const;
   virtual Operator& GetHessian(const Vector&) const;
   virtual void ComputeGuess(Vector&) const;

   virtual void Update(int,const Vector&,const Vector&,const Vector&,const Vector&) const;
   //virtual void PostCheck(const Vector&,Vector&,Vector&,bool&,bool&) const;
   virtual ~RegularizedMultiSourceMisfit() { delete H; }
};

class RegularizedMultiSourceMisfitHessian: public Operator
{
private:
   Operator *Hobj,*Hreg;
   ParameterMap *pmap;
   DataReplicator *drep;
   bool tvopt;

public:
   RegularizedMultiSourceMisfitHessian(const RegularizedMultiSourceMisfit*,const Vector&);
   virtual void Mult(const Vector&,Vector&) const;
   virtual void MultTranspose(const Vector&,Vector&) const;
   PetscParMatrix* CreatePmat() const;
};

MultiSourceMisfit::MultiSourceMisfit(ParFiniteElementSpace* _fes, PDCoefficient* mu, Coefficient* sigma,
                                     Operator::Type oid, Operator::Type jid,
                                     PetscBCHandler* _bchandler,
                                     DenseMatrix& srcpoints, int srcgid,
                                     DenseMatrix& recpoints,
                                     bool scalar, double _scale_ls,
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
   heat = new ModelHeat(mu,sigma,_fes,oid);
   heat->SetBCHandler(bchandler);

   /* ReducedFunctional base class is an mfem::Operator */
   height = width = heat->GetParameterSize();

   /* We use the same solver for objective and gradient computations */
   odesolver = new PetscODESolver(comm,"worker_");
   odesolver->SetBCHandler(bchandler);
   odesolver->SetPreconditionerFactory(heat->GetPreconditionerFactory());
   odesolver->SetJacobianType(jid);
   odesolver->Customize();

   u = new ParGridFunction(_fes);
   U = new PetscParVector(_fes);

   scale_ls = 1.0;

   /* Uncomment the following two lines to scale outside of the callbacks */
   //scale_ls = _scale_ls;
   // _scale_ls = 1.0;

   /* Setup sources and receivers */
   for (int i = 0; i < srcpoints.Width(); i++)
   {
      Array<Receiver*> receivers;
      for (int j = 0; j < recpoints.Width(); j++)
      {
         std::stringstream tmp;
         tmp << scratch << "/src-" << i + srcgid << "-rec-" << j << ".txt";
         receivers.Append(new Receiver(tmp.str()));
      }
      lsobj.Append(new TDLeastSquares(receivers,_fes,true));
      lsobj[i]->SetScale(_scale_ls);

      Vector x;
      srcpoints.GetColumn(i,x);
      if (scalar)
      {
         sources.Append(new RickerSource(x,10,1.1,signal_scal));
      }
      else
      {
         Vector dir(_fes->GetParMesh()->SpaceDimension());
         dir = 0.0;
         dir(0) = 1.0;
         vsources.Append(new VectorRickerSource(x,dir,10,1.4,signal_scal/mesh_scal));
      }
   }

   /* XXX No PetscParVector constructor with a given local size */
   Vector g;
   g.SetSize(heat->GetParameterSize());
   G = new PetscParVector(comm,g);
   M = new PetscParVector(comm,g);

   H = NULL;
}

void MultiSourceMisfit::RunTests(bool progress)
{
   PetscErrorCode ierr;
   double t = t0 + 0.5*(tf - t0);

   /* Test least-squares objective */
   if (lsobj.Size())
   {
      Vector dummy;
      double f;

      U->Randomize();

      ierr = PetscPrintf(comm,"---------------------------------------\n");CCHKERRQ(comm,ierr);
      ierr = PetscPrintf(comm,"TDLeastSquares tests\n");CCHKERRQ(comm,ierr);
      lsobj[0]->Eval(*U,dummy,t,&f);
      lsobj[0]->TestFDGradient(comm,*U,dummy,t,1.e-6,progress);
      lsobj[0]->TestFDHessian(comm,*U,dummy,t);
   }

   /* Test PDOperator */
   ierr = PetscPrintf(comm,"---------------------------------------\n");CCHKERRQ(comm,ierr);
   PetscParVector Xdot(*U);
   PetscParVector X(*U);
   Xdot.Randomize();
   X.Randomize();
   heat->GetCurrentVector(*M);
   heat->TestFDGradient(comm,Xdot,X,*M,t);
   ierr = PetscPrintf(comm,"---------------------------------------\n");CCHKERRQ(comm,ierr);
}

MultiSourceMisfit::~MultiSourceMisfit()
{
   delete heat;
   for (int i = 0; i < lsobj.Size(); i++) delete lsobj[i];
   for (int i = 0; i < sources.Size(); i++) delete sources[i];
   for (int i = 0; i < vsources.Size(); i++) delete vsources[i];
   delete u;
   delete U;
   delete G;
   delete M;
   delete odesolver;
   delete H;
}

void MultiSourceMisfit::ComputeGuess(Vector& m) const
{
   m.SetSize(heat->GetParameterSize());
   m = mu_guess*mesh_scal;
}

PetscLogStage stages[5];

void MultiSourceMisfit::ComputeObjective(const Vector& m, double *f) const
{
   PetscErrorCode ierr;
   ierr = PetscLogStagePush(stages[0]); CCHKERRQ(PETSC_COMM_SELF,ierr);

   odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);

   /* Set callbacks for setup
      ModelHeat specific callbacks are handled in mfemopt_setupts */
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
                            NULL,NULL,
                            NULL,NULL,NULL,NULL,NULL,NULL,lsobj[i]);PCHKERRQ(*odesolver,ierr);

      if (sources.Size())
      {
         heat->SetRHS(sources[i]);
      }
      else if (vsources.Size())
      {
         heat->SetRHS(vsources[i]);
      }

      PetscReal rf;
      ierr = TSComputeObjectiveAndGradient(*odesolver,t0,dt,tf,*U,*M,NULL,&rf);PCHKERRQ(*odesolver,ierr);
      *f += rf;
   }
   *f *= scale_ls;

   M->ResetArray();
   ierr = PetscLogStagePop(); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

void MultiSourceMisfit::ComputeGradient(const Vector& m, Vector& g) const
{
   PetscErrorCode ierr;
   ierr = PetscLogStagePush(stages[1]); CCHKERRQ(PETSC_COMM_SELF,ierr);
   odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);

   /* Specify callbacks for the purpose of computing the gradient (wrt the model parameters) of the residual function */
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

      if (sources.Size())
      {
         heat->SetRHS(sources[i]);
      }
      else if (vsources.Size())
      {
         heat->SetRHS(vsources[i]);
      }

      ierr = TSComputeObjectiveAndGradient(*odesolver,t0,dt,tf,*U,*M,*G,NULL);PCHKERRQ(*odesolver,ierr);
      g += *G;
   }

   g *= scale_ls;

   M->ResetArray();
   ierr = PetscLogStagePop(); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

void MultiSourceMisfit::ComputeObjectiveAndGradient(const Vector& m, double *f, Vector& g) const
{
   PetscErrorCode ierr;
   ierr = PetscLogStagePush(stages[2]); CCHKERRQ(PETSC_COMM_SELF,ierr);

   odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);

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

      if (sources.Size())
      {
         heat->SetRHS(sources[i]);
      }
      else if (vsources.Size())
      {
         heat->SetRHS(vsources[i]);
      }

      PetscReal rf;
      ierr = TSComputeObjectiveAndGradient(*odesolver,t0,dt,tf,*U,*M,*G,&rf);PCHKERRQ(*odesolver,ierr);
      *f += rf;
      g += *G;
   }

   *f *= scale_ls;
   g *= scale_ls;

   M->ResetArray();
   ierr = PetscLogStagePop(); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

Operator& MultiSourceMisfit::GetHessian(const Vector& m) const
{
   delete H;
   H = new MultiSourceMisfitHessian(this,m);
   return *H;
}

MultiSourceMisfitHessian::MultiSourceMisfitHessian(const MultiSourceMisfit* _msobj, const Vector& _m) : Operator(_m.Size(),_m.Size())
{
   PetscErrorCode ierr;
   ierr = PetscLogStagePush(stages[3]); CCHKERRQ(PETSC_COMM_SELF,ierr);

   MPI_Comm comm = _msobj->GetComm();

   msobj = _msobj;

   PetscParMatrix *A = new PetscParMatrix(comm,msobj->heat->GetGradientOperator(),Operator::PETSC_MATSHELL);

   msobj->M->PlaceArray(_m.GetData());

   /* Loop over least-squares objectives */
   for (int i = 0; i < msobj->lsobj.Size(); i++)
   {
      /* we create new solvers to not interfere with the gradient solver */
      PetscODESolver *odesolver = new PetscODESolver(comm,"worker_");
      odesolver->Init(*(msobj->heat),PetscODESolver::ODE_SOLVER_LINEAR);
      odesolver->SetBCHandler(msobj->bchandler);
      odesolver->SetPreconditionerFactory(msobj->heat->GetPreconditionerFactory());

      ierr = TSSetGradientDAE(*odesolver,*A,mfemopt_gradientdae,NULL);CCHKERRQ(comm,ierr);
      ierr = TSSetHessianDAE(*odesolver,NULL,NULL,NULL,
                                        NULL,NULL,mfemopt_hessiandae_xtm,
                                        NULL,mfemopt_hessiandae_mxt,NULL,msobj->heat);CCHKERRQ(comm,ierr);
      ierr = TSSetSetUpFromDesign(*odesolver,mfemopt_setupts,msobj->heat);CCHKERRQ(comm,ierr);


      PetscParMatrix *Hls = new PetscParMatrix(comm,msobj->lsobj[i]->GetHessianOperator_XX(),Operator::PETSC_MATSHELL);
      ierr = TSAddObjective(*odesolver,PETSC_MIN_REAL,
                            mfemopt_eval_tdobj,
                            mfemopt_eval_tdobj_x,NULL,
                            *Hls,mfemopt_eval_tdobj_xx,NULL,NULL,NULL,NULL,msobj->lsobj[i]);CCHKERRQ(comm,ierr);
      delete Hls;

      if (msobj->sources.Size())
      {
         msobj->heat->SetRHS(msobj->sources[i]);
      }
      else if (msobj->vsources.Size())
      {
         msobj->heat->SetRHS(msobj->vsources[i]);
      }

      odesolver->Customize();

      Mat pH;
      ierr = MatCreate(comm,&pH);CCHKERRQ(comm,ierr);
      ierr = TSComputeHessian(*odesolver,msobj->t0,msobj->dt,msobj->tf,*(msobj->U),*(msobj->M),pH);CCHKERRQ(comm,ierr);
      arrayH.Append(new PetscParMatrix(pH,false));
      arrayS.Append(odesolver);
   }
   delete A;

   msobj->M->ResetArray();

   ierr = PetscLogStagePop(); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

void MultiSourceMisfitHessian::Mult(const Vector& x, Vector& y) const
{
   y.SetSize(x.Size());
   PetscErrorCode ierr;
   ierr = PetscLogStagePush(stages[4]); CCHKERRQ(PETSC_COMM_SELF,ierr);
   y = 0.0;

   Vector yt(y);
   for (int i = 0; i < arrayH.Size(); i++)
   {
      /* In case we use MFFD, we need to set the sources, as we need to recompute forward and backward for F(u+h*dx) */
      if (msobj->sources.Size())
      {
         msobj->heat->SetRHS(msobj->sources[i]);
      }
      else if (msobj->vsources.Size())
      {
         msobj->heat->SetRHS(msobj->vsources[i]);
      }

      // MFFD does not support MatMultAdd
      //arrayH[i]->Mult(1.0,x,1.0,y);

      arrayH[i]->Mult(x,yt);
      y += yt;
   }
   y *= msobj->scale_ls;
   ierr = PetscLogStagePop(); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

MultiSourceMisfitHessian::~MultiSourceMisfitHessian()
{
   for (int i = 0; i < arrayH.Size(); i++) { delete arrayH[i]; }
   for (int i = 0; i < arrayS.Size(); i++) { delete arrayS[i]; }
}

RegularizedMultiSourceMisfit::RegularizedMultiSourceMisfit(MultiSourceMisfit *_obj, TVRegularizer *_reg, DataReplicator *_drep, ParameterMap *_pmap, bool _tvopt)
{
   obj   = _obj;
   reg   = _reg;
   drep  = _drep;
   pmap  = _pmap;
   tvopt = _tvopt;
   H     = NULL;

   height = width = drep->IsMaster() ? _obj->Height() : 0.0;
}

void RegularizedMultiSourceMisfit::ComputeGuess(Vector& m) const
{
   Vector dummy(obj->Height());
   dummy = 0.0;
   if (drep->IsMaster())
   {
      obj->ComputeGuess(dummy);
   }

   m.SetSize(Height());
   m = 0.0;
   drep->Reduce("opt_data",dummy,m);
   pmap->InverseMap(m,m);
}

void RegularizedMultiSourceMisfit::ComputeObjective(const Vector& m, double *f) const
{
   Vector pm(m.Size());
   pmap->Map(m,pm);

   double f1 = 0.0,f2 = 0.0;
   Vector dummy(obj->Height());
   drep->Broadcast("opt_data",pm,dummy);
   obj->ComputeObjective(dummy,&f1);
   f2 = 0.0;
   drep->Reduce(f1,&f2);
   drep->Broadcast(f2,&f1);
   if (tvopt)
   {
      reg->Eval(dummy,m,0.,&f2);
   }
   else
   {
      reg->Eval(dummy,pm,0.,&f2);
   }
   *f = f1 + f2;
}

void RegularizedMultiSourceMisfit::ComputeGradient(const Vector& m, Vector& g) const
{
   Vector pm(m.Size());
   pmap->Map(m,pm);

   Vector dummy(obj->Height());
   Vector g1(g.Size()),g2(dummy.Size());

   drep->Broadcast("opt_data",pm,dummy);
   obj->ComputeGradient(dummy,g2);
   g1 = 0.0;
   drep->Reduce("opt_data",g2,g1);

   if (tvopt)
   {
      pmap->GradientMap(m,g1,true,g);
      reg->EvalGradient_M(dummy,m,0.,g1);
      g += g1;
   }
   else
   {
      Vector g3(g.Size());
      reg->EvalGradient_M(dummy,pm,0.,g3);
      g1 += g3;
      pmap->GradientMap(m,g1,true,g);
   }
}

Operator& RegularizedMultiSourceMisfit::GetHessian(const Vector& m) const
{
   delete H;
   H = new RegularizedMultiSourceMisfitHessian(this,m);
   return *H;
}

void RegularizedMultiSourceMisfit::Update(int it, const Vector& F, const Vector& X,
                                   const Vector& dX, const Vector &pX) const
{
   if (!it)
   {
      reg->UpdateDual(X);
   }
   else
   {
      double lambda = pX.Size() ? (pX[0] - X[0])/dX[0] : 0.0;
      reg->UpdateDual(pX,dX,lambda);
   }
}

//void RegularizedMultiSourceMisfit::PostCheck(const Vector& X, Vector& Y, Vector &W, bool& cy, bool& cw) const
//{
//   /* we don't change the step (Y) or the updated solution (W = X - lambda*Y) */
//   cy = false;
//   cw = false;
//   double lambda = X.Size() ? (X[0] - W[0])/Y[0] : 0.0;
//   reg->UpdateDual(X,Y,lambda);
//}

/*
   The hessian of the full objective

   - TV on model parameter mu:

      H = H_map(_m) + J(mu(_m))^T * ( H_tv(mu(_m)) + H_misfit(mu(_m)) ) J(mu(_m))

   - TV on optimization parameter _m:

      H = H_map(_m) + J(mu(_m))^T * ( H_misfit(mu(_m)) ) J(mu(_m)) + H_tv(_m)
*/
RegularizedMultiSourceMisfitHessian::RegularizedMultiSourceMisfitHessian(const RegularizedMultiSourceMisfit* _rmsobj, const Vector& _m)
{
   height = width = _m.Size();
   pmap = _rmsobj->pmap;
   tvopt = _rmsobj->tvopt;
   drep = _rmsobj->drep;

   Vector pm(_m.Size());
   pmap->Map(_m,pm);

   Vector pmrep(_rmsobj->obj->Height());
   drep->Broadcast("opt_data",pm,pmrep);
   Hobj = &( _rmsobj->obj->GetHessian(pmrep));

   if (tvopt)
   {
      Vector dummy;
      _rmsobj->reg->SetUpHessian_MM(dummy,_m,0.);
      Hreg = _rmsobj->reg->GetHessianOperator_MM();

      if (pmap->SecondOrder())
      {
         Vector g1(pm.Size()),g1rep(_rmsobj->obj->Height());

         _rmsobj->obj->ComputeGradient(pmrep,g1rep);
         g1 = 0.0;
         drep->Reduce("opt_data",g1rep,g1);

         pmap->SetUpHessianMap(_m,g1);
      }
   }
   else
   {
      Vector dummy;
      _rmsobj->reg->SetUpHessian_MM(dummy,pm,0.);
      Hreg = _rmsobj->reg->GetHessianOperator_MM();

      if (pmap->SecondOrder())
      {
         Vector g1(pm.Size()),g1rep(_rmsobj->obj->Height()),g2(pm.Size());

         _rmsobj->obj->ComputeGradient(pmrep,g1rep);
         g1 = 0.0;
         drep->Reduce("opt_data",g1rep,g1);
         _rmsobj->reg->EvalGradient_M(dummy,pm,0.,g2);
         g1 += g2;
         pmap->SetUpHessianMap(_m,g1);
      }
   }
}

void RegularizedMultiSourceMisfitHessian::Mult(const Vector& x, Vector& y) const
{
   Vector px(x.Size());

   const Vector& m = pmap->GetParameter();
   pmap->GradientMap(m,x,false,px);

   Vector py1(px.Size()),py2(px.Size());

   Vector pxrep(Hobj->Width()),py1rep(Hobj->Height());
   drep->Broadcast("opt_data",px,pxrep);
   Hobj->Mult(pxrep,py1rep);
   py1 = 0.0;
   drep->Reduce("opt_data",py1rep,py1);

   if (tvopt)
   {
      Vector y2(x.Size());

      const Vector& m = pmap->GetParameter();
      pmap->GradientMap(m,py1,true,y);
      if (pmap->SecondOrder())
      {
         pmap->HessianMult(x,y2);
         y += y2;
      }

      y2 = 0.0;
      Hreg->Mult(x,y2);
      y += y2;
   }
   else
   {
      Hreg->Mult(px,py2);
      py1 += py2;

      const Vector& m = pmap->GetParameter();
      pmap->GradientMap(m,py1,true,y);
      if (pmap->SecondOrder())
      {
         Vector y2(x.Size());

         pmap->HessianMult(x,y2);
         y += y2;
      }
   }
}

/* We need the transpose callback just in case the Newton solver fails, and PETSc requests it */
void RegularizedMultiSourceMisfitHessian::MultTranspose(const Vector& x, Vector& y) const
{
   Vector px(x.Size());

   const Vector& m = pmap->GetParameter();
   pmap->GradientMap(m,x,false,px);

   Vector py1(px.Size()),py2(px.Size());

   /* the Hessian of the misfit function is symmetric */
   Vector pxrep(Hobj->Width()),py1rep(Hobj->Height());
   drep->Broadcast("opt_data",px,pxrep);
   Hobj->Mult(pxrep,py1rep);
   py1 = 0.0;
   drep->Reduce("opt_data",py1rep,py1);

   if (tvopt)
   {
      Vector y2(x.Size());

      const Vector& m = pmap->GetParameter();
      pmap->GradientMap(m,py1,true,y);
      if (pmap->SecondOrder())
      {
         pmap->HessianMult(x,y2);
         y += y2;
      }

      y2 = 0.0;
      Hreg->MultTranspose(x,y2);
      y += y2;
   }
   else
   {
      Hreg->MultTranspose(px,py2);
      py1 += py2;

      const Vector& m = pmap->GetParameter();
      pmap->GradientMap(m,py1,true,y);
      if (pmap->SecondOrder())
      {
         Vector y2(x.Size());

         pmap->HessianMult(x,y2);
         y += y2;
      }
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

   if (pmap && !tvopt) /* XXX valid only for diagonal maps */
   {
      const Vector& m = pmap->GetParameter();
      Vector x((*this).Height()),y((*this).Height()),px((*this).Height());
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
      vt =  _vt;
      name = _name;
      pause = true;
      if (vt > 0)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         sout.open(vishost, visport);
         if (!sout)
         {
            if (!PetscGlobalRank) /* XXX */
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
         Vector pu(X.Size());
         if (pmap) pmap->Map(X,pu);
         else pu = X;
         if (m)
         {
            m->Distribute(pu); /* XXX WITH GF! */
         }
         else
         {
            HypreParMatrix &P = *pfes->Dof_TrueDof_Matrix();
            P.Mult(pu,*u);
         }
         /* visualize optimization space */
         if (pmap) pmap->InverseMap(*u,*u);
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
   {
      PetscErrorCode ierr;
      ierr = PetscLogDefaultBegin(); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscLogStageRegister("MSObj",&stages[0]); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscLogStageRegister("MSGrad",&stages[1]); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscLogStageRegister("MSObjAndGrad",&stages[2]); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscLogStageRegister("MSHessSetUp",&stages[3]); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscLogStageRegister("MSHessMult",&stages[4]); CCHKERRQ(PETSC_COMM_SELF,ierr);
   }
   PetscInt srl = 0, prl = 0, viz = 0, ncrl = 0;

   PetscInt  gridr[3] = {1,1,1};
   PetscInt  grids[3] = {1,1,1};
   PetscReal gridr_bb[6] = {-1.0,1.0,-1.0,1.0,-1.0,1.0};
   PetscReal grids_bb[6] = {-1.0,1.0,-1.0,1.0,-1.0,1.0};

   char scratchdir[PETSC_MAX_PATH_LEN] = "/tmp";
   char meshfile[PETSC_MAX_PATH_LEN] = "../../../../share/petscopt/meshes/segment-m5-5.mesh";
   PetscReal t0 = 0.0, dt = 1.e-3, tf = 0.1;
   PetscReal scale_ls = 1.0;
   PetscBool exact_sample = PETSC_TRUE;

   PetscInt  nrep = 1;
   PetscBool contig = PETSC_TRUE;
   bool master = true;

   FECType   s_fec_type = FEC_H1;
   PetscInt  s_ord = 1;
   OIDType   s_oid_type = OID_MATAIJ, s_jid_type = OID_MATAIJ;

   FECType   mu_fec_type = FEC_H1;
   PetscInt  mu_ord = 1;
   PetscInt  n_mu_excl = 1024;
   PetscInt  mu_excl[1024];
   PetscBool mu_excl_fn = PETSC_FALSE, mu_with_jumps = PETSC_FALSE;

   PetscReal tva = 1.0, tvb = 0.1;
   PetscBool tvopt = PETSC_FALSE, tvpd = PETSC_TRUE, tvsy = PETSC_FALSE, tvpr = PETSC_FALSE;

   PetscBool test_part = PETSC_FALSE, test_null = PETSC_FALSE, test_progress = PETSC_TRUE;
   PetscBool test_newton = PETSC_FALSE, test_opt = PETSC_FALSE;
   PetscBool test_misfit[3] = {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE}, test_misfit_reg[2] = {PETSC_FALSE,PETSC_FALSE}, test_misfit_internal = PETSC_FALSE;
   PetscReal test_newton_noise = 0.0;
   PetscBool glvis = PETSC_TRUE, save = PETSC_FALSE;

   /* Process options */
   {
      PetscBool      flg;
      PetscInt       i,j;
      PetscErrorCode ierr;

      ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for Heat equation",NULL);CHKERRQ(ierr);

      ierr = PetscOptionsInt("-nrep","Number of replicas",NULL,nrep,&nrep,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-contig","Contiguous replicas",NULL,contig,&contig,NULL);CHKERRQ(ierr);

      /* Simulation parameters */
      ierr = PetscOptionsEnum("-state_oid_type","Operator::Type for state",NULL,OIDTypes,(PetscEnum)s_oid_type,(PetscEnum*)&s_oid_type,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnum("-state_jid_type","Operator::Type for state jacobian",NULL,OIDTypes,(PetscEnum)s_jid_type,(PetscEnum*)&s_jid_type,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-state_ord","Polynomial order approximation for state variables",NULL,s_ord,&s_ord,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnum("-state_fec_type","FEC for state","",FECTypes,(PetscEnum)s_fec_type,(PetscEnum*)&s_fec_type,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsString("-scratch","Location where to put temporary data (must be present)",NULL,scratchdir,scratchdir,sizeof(scratchdir),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsString("-meshfile","Mesh filename",NULL,meshfile,meshfile,sizeof(meshfile),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-srl","Number of sequential refinements",NULL,srl,&srl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-prl","Number of parallel refinements",NULL,prl,&prl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-ncrl","Number of non-conforming refinements (refines element with center in [-1,1]^d)",NULL,ncrl,&ncrl,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsRealArray("-ncrl_fn_bb","Bounding box for non-conforming refinement (defaults to [-1,1]^d)",NULL,refine_fn_bb,(i=6,&i),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-t0","Initial time",NULL,t0,&t0,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-dt","Initial time step",NULL,dt,&dt,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-tf","Final time",NULL,tf,&tf,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-exact_sampling","Use exact coefficients when generating the data",NULL,exact_sample,&exact_sample,NULL);CHKERRQ(ierr);

      /* GLVis */
      ierr = PetscOptionsInt("-viz","Visualization steps for model sampling",NULL,viz,&viz,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-glvis","Activate GLVis monitoring of Newton process",NULL,glvis,&glvis,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-save","Save final result",NULL,save,&save,NULL);CHKERRQ(ierr);

      /* Sources and receivers */
      ierr = PetscOptionsIntArray("-grid_rcv_n","Grid receivers: points per direction",NULL,gridr,(i=3,&i),NULL);CHKERRQ(ierr);
      for (j=i;j<3;j++) gridr[j] = gridr[i > 0 ? i-1 : 0];
      ierr = PetscOptionsRealArray("-grid_rcv_bb","Grid receivers: bounding box",NULL,gridr_bb,(i=6,&i),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray("-grid_src_n","Grid sources: points per direction",NULL,grids,(i=3,&i),NULL);CHKERRQ(ierr);
      for (j=i;j<3;j++) grids[j] = grids[i > 0 ? i-1 : 0];
      ierr = PetscOptionsRealArray("-grid_src_bb","Grid sources: bounding box",NULL,grids_bb,(i=6,&i),NULL);CHKERRQ(ierr);

      ierr = PetscOptionsReal("-sigma_scal","Diffusion scaling factor",NULL,sigma_scal,&sigma_scal,NULL);CHKERRQ(ierr);
      /* Parameter space */
      ierr = PetscOptionsInt("-mu_ord","Polynomial order approximation for mu",NULL,mu_ord,&mu_ord,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-mu_const_val","Costant mu value",NULL,mu_const_val,&mu_const_val,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-mu_jumps","Use jumping target for mu",NULL,mu_with_jumps,&mu_with_jumps,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnum("-mu_fec_type","FEC for mu","",FECTypes,(PetscEnum)mu_fec_type,(PetscEnum*)&mu_fec_type,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-mu_exclude_fn","Excludes elements outside a given bounding box for mu optimization",NULL,mu_excl_fn,&mu_excl_fn,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsRealArray("-mu_exclude_fn_bb","Excludes elements outside the specified bounding box (defaults to [-1,1]^d)",NULL,excl_fn_bb,(i=6,&i),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray("-mu_exclude","Elements' tag to exclude for mu optimization",NULL,mu_excl,&n_mu_excl,&flg);CHKERRQ(ierr);
      if (!flg) n_mu_excl = 0;

      ierr = PetscOptionsReal("-mesh_scale","Scaling factor Mesh",NULL,mesh_scal,&mesh_scal,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-signal_scale","Scaling factor signal",NULL,signal_scal,&signal_scal,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-mu_guess","Constant mu guess",NULL,mu_guess,&mu_guess,NULL);CHKERRQ(ierr);

      /* Objectives' options */
      ierr = PetscOptionsReal("-ls_scale","Scaling factor for least-squares objective",NULL,scale_ls,&scale_ls,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-tv_alpha","Scaling factor for TV regularizer",NULL,tva,&tva,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-tv_beta","Gradient norm perturbation for TV regularizer",NULL,tvb,&tvb,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-tv_pd","Use Primal-Dual TV regularizer",NULL,tvpd,&tvpd,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-tv_symm","Symmetrize TV Hessian",NULL,tvsy,&tvsy,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-tv_proj","Project on the unit ball instead of performing line search for the dual TV variables",NULL,tvpr,&tvpr,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-tv_opt","TV on optimization variable or model variable",NULL,tvopt,&tvopt,NULL);CHKERRQ(ierr);

      ierr = PetscOptionsBool("-test_partitioning","Test with a fixed element partition",NULL,test_part,&test_part,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-test_newton","Test Newton solver",NULL,test_newton,&test_newton,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-test_opt","Test Optimization solver",NULL,test_opt,&test_opt,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-test_newton_noise","Test Newton solver: noise level",NULL,test_newton_noise,&test_newton_noise,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-test_null","Use exact solution when testing",NULL,test_null,&test_null,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-test_progress","Report progress when testing",NULL,test_progress,&test_progress,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBoolArray("-test_misfit","Test misfit function callbacks",NULL,test_misfit,(i=3,&i),NULL);CHKERRQ(ierr);
      for (int j=i; j<3; j++) test_misfit[j] = test_misfit[i > 0 ? i-1 : 0];
      ierr = PetscOptionsBoolArray("-test_misfit_reg","Test regularized misfit function callbacks",NULL,test_misfit_reg,(i=2,&i),NULL);CHKERRQ(ierr);
      for (int j=i; j<2; j++) test_misfit_reg[j] = test_misfit_reg[i > 0 ? i-1 : 0];
      ierr = PetscOptionsBool("-test_misfit_internal","Tests internal objects inside misfit function",NULL,test_misfit_internal,&test_misfit_internal,NULL);CHKERRQ(ierr);

      ierr = PetscOptionsEnd();CHKERRQ(ierr);
   }
   Operator::Type oid,jid;
   switch (s_oid_type)
   {
      case OID_MATAIJ:
         oid = Operator::PETSC_MATAIJ;
         break;
      case OID_MATIS:
         oid = Operator::PETSC_MATIS;
         break;
      case OID_MATHYPRE:
         oid = Operator::PETSC_MATHYPRE;
         break;
      case OID_HYPRE:
         oid = Operator::Hypre_ParCSR;
         break;
      case OID_ANY:
      default:
         oid = Operator::ANY_TYPE;
         break;
   }
   switch (s_jid_type)
   {
      case OID_MATAIJ:
         jid = Operator::PETSC_MATAIJ;
         break;
      case OID_MATIS:
         jid = Operator::PETSC_MATIS;
         break;
      case OID_MATHYPRE:
         jid = Operator::PETSC_MATHYPRE;
         break;
      case OID_HYPRE:
         jid = Operator::Hypre_ParCSR;
         break;
      case OID_ANY:
      default:
         jid = Operator::ANY_TYPE;
         break;
   }

   Array<int> mu_excl_a((int)n_mu_excl);
   for (int i = 0; i < n_mu_excl; i++) mu_excl_a[i] = (int)mu_excl[i];

   /* Create mesh and finite element space for the independent variable */
   ParMesh *pmesh = NULL;
   ReplicatedParMesh *rpmesh = NULL;
   {
      Mesh *mesh = new Mesh(meshfile, 1, 1);
      MFEM_VERIFY(mesh->SpaceDimension() == mesh->Dimension(),"Embedded meshes not supported")
      for (int lev = 0; lev < srl; lev++)
      {
         mesh->UniformRefinement();
      }
      mesh->EnsureNCMesh();

      if (test_part)
      {
         pmesh = ParMeshTest(PETSC_COMM_WORLD, *mesh);
      }
      else
      {
         rpmesh = new ReplicatedParMesh(PETSC_COMM_WORLD, *mesh, nrep, contig);
         pmesh = rpmesh->GetChild();
      }
      delete mesh;
      for (int lev = 0; lev < prl; lev++)
      {
         pmesh->UniformRefinement();
         if (rpmesh)
         {
            rpmesh->GetParent()->UniformRefinement();
         }
      }
      for (int lev = 0; lev < ncrl; lev++)
      {
         NCRefinement(pmesh,refine_fn);
         if (rpmesh)
         {
            NCRefinement(rpmesh->GetParent(),refine_fn);
         }
      }
   }

   DataReplicator *drep = new DataReplicator(PETSC_COMM_WORLD, rpmesh ? nrep : 1, contig);
   master = drep->IsMaster();

   /* Simulation space */
   bool scalar = true;
   FiniteElementCollection *s_fec = NULL;
   switch (s_fec_type)
   {
      case FEC_HCURL:
         /* magnetic -> assumes the mesh in meters */
         sigma_scal *= sigma_scal_muinv_0/mesh_scal;
         scalar = false;
         s_fec = new ND_FECollection(s_ord,pmesh->Dimension());
         break;
      case FEC_H1:
         s_fec = new H1_FECollection(s_ord,pmesh->Dimension());
         break;
      default:
         MFEM_ABORT("Unhandled FEC Type");
         break;
   }
   ParFiniteElementSpace *s_fes = new ParFiniteElementSpace(pmesh, s_fec);

   /* Boundary conditions handler */
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      s_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
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
   FunctionCoefficient *sigma = new FunctionCoefficient(sigma_exact);
   FunctionCoefficient *mu;

   if (!oil_test)
   {
      if (mu_with_jumps) mu = new FunctionCoefficient(mu_exact_jump);
      else mu = new FunctionCoefficient(mu_exact_const);
   }
   else
   {
      mu = new FunctionCoefficient(mu_exact_jump_test_em_2d);
   }

   /* Source terms */
   RickerSource       *rick = NULL;
   VectorRickerSource *vrick = NULL;
   if (srcpoints.Width())
   {
      Vector t(pmesh->SpaceDimension());
      if (scalar)
      {
         rick = new RickerSource(t,10,1.1,signal_scal);
      }
      else
      {
         Vector dir(pmesh->SpaceDimension());
         dir = 0.0;
         dir(0) = 1.0;
         vrick = new VectorRickerSource(t,dir,10,1.4,signal_scal/mesh_scal);
      }
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

   /* The parameter dependent coefficient for mu
      We construct the guess from the exact solution just for testing purposes */
   PDCoefficient *mu_pd;
   if (mu_excl_fn) mu_pd = new PDCoefficient(*mu,pmesh,mu_fec,excl_fn);
   else mu_pd = new PDCoefficient(*mu,pmesh,mu_fec,mu_excl_a);

   /* Exact solution */
   Vector muv_exact(mu_pd->GetLocalSize());
   mu_pd->GetCurrentVector(muv_exact);

   /* Multi-source sampling */
   if (master)
   {
      for (int i = 0; i < srcpoints.Width(); i++)
      {
         ParGridFunction *u = new ParGridFunction(s_fes);
         HypreParVector *U = u->GetTrueDofs();
         *U = 0.0;

         std::stringstream tmp1,tmp2;
         tmp1 << "Fwd source - " << i;
         tmp2 << scratchdir << "/src-" << i << "-rec";
         UserMonitor *monitor = new UserMonitor(u,viz,tmp1.str());
         ReceiverMonitor *rmonitor = new ReceiverMonitor(u,recpoints,tmp2.str());

         ModelHeat *heat;
         if (exact_sample) heat =new ModelHeat(mu,sigma,s_fes,oid);
         else heat = new ModelHeat(mu_pd,sigma,s_fes,oid);
         heat->SetBCHandler(bchandler); /* XXX so many times setting the object -> add specialized constructor for odesolver? */

         Vector srcctr;
         srcpoints.GetColumn(i,srcctr);
         if (vrick)
         {
            vrick->SetDeltaCenter(srcctr);
            heat->SetRHS(vrick);
         }
         else if (rick)
         {
            rick->SetDeltaCenter(srcctr);
            heat->SetRHS(rick);
         }

         PetscODESolver *odesolver = new PetscODESolver(pmesh->GetComm(),"model_");
         odesolver->Init(*heat,PetscODESolver::ODE_SOLVER_LINEAR);
         odesolver->SetBCHandler(bchandler);
         odesolver->SetPreconditionerFactory(heat->GetPreconditionerFactory());
         odesolver->SetJacobianType(jid);
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
   }
   /* wait for completion */
   MPI_Barrier(PETSC_COMM_WORLD);

   /* The misfit function */
   int srcgid;
   drep->Split(srcpoints,srcpoints,&srcgid);
   MultiSourceMisfit *obj = new MultiSourceMisfit(s_fes,mu_pd,sigma,oid,jid,bchandler,srcpoints,srcgid,recpoints,scalar,scale_ls,t0,dt,tf,scratchdir);

   /* Tests internal objects inside the misfit function */
   if (test_misfit_internal && master)
   {
      obj->RunTests(test_progress);
   }

   /* Test misfit callbacks */
   if (test_misfit[0])
   {
      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      double f1,f2;
      PetscParVector g1(pmesh->GetComm(),muv), g2(pmesh->GetComm(),muv);
      obj->ComputeObjectiveAndGradient(muv,&f1,g1);
      obj->ComputeObjective(muv,&f2);
      obj->ComputeGradient(muv,g2);
      MFEM_VERIFY(std::abs(f1-f2) < PETSC_SMALL,"Error on misfit computations! " << f1 << ", " << f2)
      g1 -= g2;
      f1 = ParNormlp(g1,infinity(),pmesh->GetComm());
      MFEM_VERIFY(f1 < PETSC_SMALL,"Error on misfit computations! gradient error " << f1)

      f1 = 0.0;
      drep->Reduce(f2,&f1);
      PetscPrintf(PETSC_COMM_WORLD,"Misfit objective %g\n",f1);
#if 0
      g1 = 0.0;
      drep->Reduce("test",g2,g1);
      if (drep->IsMaster()) g1.Print();
#endif
   }

   /* FD test for misfit function gradient */
   if (master && test_misfit[1])
   {
      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      obj->TestFDGradient(pmesh->GetComm(),muv,1.e-8,test_progress);
   }

   /* matrix-free FD test for misfit function hessian */
   if (master && test_misfit[2])
   {
      Vector muv;
      if (!test_null) obj->ComputeGuess(muv);
      else muv = muv_exact;

      obj->TestFDHessian(pmesh->GetComm(),muv);
   }

   /* Map from optimization variables to model variables */
   PointwiseMap pmap(mu_m,m_mu,dmu_dm,dmu_dm);

   /* Coefficient to be regularized */
   PDCoefficient *mu_inv_pd = NULL;
   FunctionOfCoefficient *mu_inv = NULL;
   mu_inv = new FunctionOfCoefficient(tvopt ? m_mu : NULL,*mu);
   if (mu_excl_fn) mu_inv_pd = new PDCoefficient(*mu_inv,rpmesh ? rpmesh->GetParent() : pmesh,mu_fec,excl_fn);
   else mu_inv_pd = new PDCoefficient(*mu_inv,rpmesh ? rpmesh->GetParent() : pmesh,mu_fec,mu_excl_a);
   if (glvis) mu_inv_pd->Visualize();

   /* Total variation regularizer */
   TVRegularizer *tv = new TVRegularizer(mu_inv_pd,tva,tvb,tvpd);
   tv->Symmetrize(tvsy);
   tv->Project(tvpr);

   /* The full objective: misfit + regularization */
   RegularizedMultiSourceMisfit *robj = new RegularizedMultiSourceMisfit(obj,tv,drep,&pmap,tvopt);

   if (test_misfit_internal)
   {
      Vector muv;
      if (!test_null) robj->ComputeGuess(muv);
      else muv = muv_exact;
      Vector dummy;
      double f;

      /* Misfit in terms of mu, the optimization variable is the inverse of the map */
      Vector m(muv);
      if (tvopt) pmap.InverseMap(muv,m);
      PetscErrorCode ierr;

      MPI_Comm comm = PETSC_COMM_WORLD;
      ierr = PetscPrintf(comm,"---------------------------------------\n");CCHKERRQ(comm,ierr);
      ierr = PetscPrintf(comm,"TV tests\n");CCHKERRQ(comm,ierr);
      tv->Eval(dummy,m,0.0,&f);
      tv->TestFDGradient(comm,dummy,m,0.0,1.e-6,test_progress);
      tv->TestFDHessian(comm,dummy,m,0.0);
   }

   /* Test callbacks for full objective */
   if (test_misfit_reg[0])
   {
      Vector m(muv_exact.Size());
      if (!test_null) robj->ComputeGuess(m);
      else
      {
         pmap.InverseMap(muv_exact,m);
      }
      if (!master) m.SetSize(0);

      double f;
      robj->ComputeObjective(m,&f);
      PetscPrintf(PETSC_COMM_WORLD,"Regularized objective %g\n",f);
   }

   /* Test callbacks for full objective gradient */
   if (test_misfit_reg[1])
   {
      Vector m(muv_exact.Size());
      if (!test_null) robj->ComputeGuess(m);
      else
      {
         pmap.InverseMap(muv_exact,m);
      }
      if (!master) m.SetSize(0);

      Vector y(m.Size());
      robj->Mult(m,y);
   }

   /* Test Newton solver */
   if (test_newton)
   {
      PetscNonlinearSolverOpt newton(PETSC_COMM_WORLD,*robj,"newton_");

      newton.SetJacobianType(Operator::PETSC_MATSHELL);
      newton.iterative_mode = true; /* we always use an initial guess, it can be zero */

//YYY
#if 0
      RMSHPFactory myfactory;
      newton.SetPreconditionerFactory(&myfactory);
#endif
      NewtonMonitor mymonitor;
      newton.SetMonitor(&mymonitor);
      UserMonitor solmonitor(mu_inv_pd,NULL,glvis ? 1 : 0,"Newton solution");
      if (glvis) newton.SetMonitor(&solmonitor);

      Vector u(muv_exact.Size());
      if (!test_null) robj->ComputeGuess(u);
      else
      {
         pmap.InverseMap(muv_exact,u);
      }
      if (!master) u.SetSize(0);

      GaussianNoise nnoise;
      Vector vnoise;
      nnoise.random(vnoise,u.Size());
      vnoise *= test_newton_noise;
      u += vnoise;

      double f1,f2;
      robj->ComputeObjective(u,&f1);
      tv->SetScale(0.0);
      robj->ComputeObjective(u,&f2);
      PetscPrintf(PETSC_COMM_WORLD,"Initial objective %g (LS %g, TV %g)\n",f1,f2,f1-f2);
      tv->SetScale(tva);

      Vector dummy;
      newton.Mult(dummy,u);

      robj->ComputeObjective(u,&f1);
      tv->SetScale(0.0);
      robj->ComputeObjective(u,&f2);
      PetscPrintf(PETSC_COMM_WORLD,"Final objective %g (LS %g, TV %g)\n",f1,f2,f1-f2);

      // TODO HANG
      //u = 0.0;
      //u[1] = 1e6;
      //u[2] = -1e6;
      //mu_pd->UpdateCoefficient(u);
      //mu_pd->Visualize("RJlc");
      if (save)
      {
         mu_inv_pd->Distribute(u);
         mu_inv_pd->Save("solution");
      }
   }

   /* Test Optimization solver */
   if (test_opt)
   {
      PetscOptimizationSolver opt(PETSC_COMM_WORLD,*robj,"opt_");

      OptimizationMonitor mymonitor;
      opt.SetMonitor(&mymonitor);
      UserMonitor solmonitor(mu_inv_pd,NULL,glvis ? 1 : 0,"TAO solution");
      if (glvis) opt.SetMonitor(&solmonitor);

      // TODO check guess with TAO
      Vector u(muv_exact.Size());
      if (!test_null) robj->ComputeGuess(u);
      else
      {
         pmap.InverseMap(muv_exact,u);
      }
      if (!master) u.SetSize(0);

      GaussianNoise nnoise;
      Vector vnoise;
      nnoise.random(vnoise,u.Size());
      vnoise *= test_newton_noise;
      u += vnoise;

      double f1,f2;
      robj->ComputeObjective(u,&f1);
      tv->SetScale(0.0);
      robj->ComputeObjective(u,&f2);
      PetscPrintf(PETSC_COMM_WORLD,"OPT: Initial objective %g (LS %g, TV %g)\n",f1,f2,f1-f2);
      tv->SetScale(tva);

      opt.Solve(u);

      robj->ComputeObjective(u,&f1);
      tv->SetScale(0.0);
      robj->ComputeObjective(u,&f2);
      PetscPrintf(PETSC_COMM_WORLD,"OPT: Final objective %g (LS %g, TV %g)\n",f1,f2,f1-f2);

      if (save)
      {
         mu_inv_pd->Distribute(u);
         mu_inv_pd->Save("tao_solution");
      }
   }
   if (rpmesh) pmesh = NULL;
   delete rpmesh;
   delete drep;

   delete robj;
   delete tv;
   delete mu_fec;
   delete mu_inv_pd;
   delete mu_pd;
   delete obj;

   delete rick;
   delete vrick;

   delete bchandler;
   delete sigma;
   delete mu;
   delete mu_inv;
   delete s_fec;
   delete s_fes;
   delete pmesh;

   MFEMFinalizePetsc();
   return 0;
}

/*TEST

   build:
     requires: mfemopt

   testset:
     filter: sed -e "s/-nan/nan/g"
     timeoutfactor: 3
     nsize: {{1 2}}
     args: -scratch ./ -test_partitioning -meshfile ${petscopt_dir}/share/petscopt/meshes/segment-m5-5.mesh -ts_trajectory_type memory -ts_trajectory_reconstruction_order 2 -mfem_use_splitjac -model_ts_type cn -model_ksp_type cg -worker_ts_max_snes_failures -1 -worker_ts_type cn -worker_ksp_type cg -test_newton
     test:
       suffix: null_test
       args: -test_null -test_misfit_internal -test_misfit 1 -test_misfit_reg 1 -test_progress 0 -tv_alpha 0 -newton_pc_type none -newton_snes_atol 1.e-8 -glvis 0
     test:
       suffix: newton_test
       args: -glvis 0 -newton_snes_converged_reason -newton_snes_max_it 1 -newton_snes_test_jacobian -newton_snes_rtol 1.e-6 -newton_snes_atol 1.e-6 -newton_ksp_type fgmres -newton_pc_type none -mu_jumps -tv_alpha 0.01 -mu_exclude_fn -ncrl 1
     test:
       suffix: newton_full
       args: -glvis 0 -newton_snes_converged_reason -newton_snes_max_it 10 -newton_snes_rtol 1.e-6 -newton_snes_atol 1.e-6 -newton_ksp_type fgmres -newton_pc_type none -mu_jumps -tv_alpha 0.01 -mu_exclude 2 -ncrl 1
     test:
       suffix: taonewton_full
       args: -glvis 0 -test_newton 0 -test_opt -opt_tao_converged_reason -opt_tao_max_it 10 -opt_tao_gttol 1.e-6 -opt_tao_gatol 1.e-6 -opt_tao_type nls -opt_tao_nls_ksp_type fgmres -opt_tao_nls_pc_type none -mu_jumps -tv_alpha 0.01 -mu_exclude 2 -ncrl 1

   test:
      suffix: em_test
      timeoutfactor: 3
      nsize: 1
      args: -newton_snes_max_it 4 -signal_scale 1.0 -scratch ./ -meshfile ${petscopt_dir}/share/petscopt/meshes/inline_quad_testem.mesh -dt 0.01 -tf 1 -ts_trajectory_type memory -ts_trajectory_reconstruction_order 2 -mfem_use_splitjac -model_ts_type cn -model_ksp_type cg -worker_ts_max_snes_failures -1 -worker_ts_type cn -worker_ksp_type cg -state_fec_type HCURL -test_progress 0 -test_misfit_internal -test_misfit 1,0,0 -ls_scale 1.e12 -tv_alpha 1.e-12 -tv_beta 1.e-6 -test_newton -test_newton_noise 0.0 -newton_pc_type none -newton_snes_atol 1.e-6 -newton_snes_converged_reason -test_null 0 -mu_const_val 0.9 -mu_exclude_fn_bb 3000,7000,4000,8000 -mu_exclude_fn  -grid_src_n 1 -grid_src_bb 5000,7000,5000,7000 -grid_rcv_n 4,1 -grid_rcv_bb 5500,6500,5800,5800 -glvis 0

TEST*/

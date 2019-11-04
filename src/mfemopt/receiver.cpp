#include <mfemopt/receiver.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <petsc/private/tsimpl.h>
#include <sstream>
#include <algorithm>
#include <limits>

namespace mfemopt
{
using namespace mfem;

void Receiver::Init()
{
   W = NULL;
   ts = NULL;
}

Receiver::Receiver() : center(), idata(), idata_isfinalized(false) { }

Receiver::Receiver(double _x) : center(), idata(), idata_isfinalized(false)
{
   Init();
   center.SetSize(1);
   center[0] = _x;
}

Receiver::Receiver(double _x, double _y) : center(), idata(), idata_isfinalized(false)
{
   Init();
   center.SetSize(2);
   center[0] = _x;
   center[1] = _y;
}

Receiver::Receiver(double _x, double _y, double _z) : center(), idata(), idata_isfinalized(false)
{
   Init();
   center.SetSize(3);
   center[0] = _x;
   center[1] = _y;
   center[2] = _z;
}

Receiver::Receiver(const std::string& filename) : center(), idata(), idata_isfinalized(false)
{
   Init();
   Load(filename);
}

Receiver::Receiver(const Vector& _center, const std::vector<double> & T, const std::vector<double> & X, const std::vector<double> & Y, const std::vector<double> & Z) :
          center(), idata(), idata_isfinalized(false)
{
   Init();
   MFEM_ASSERT(X.size() && T.size() <= X.size(),"X data insufficient");
   MFEM_ASSERT(!Y.size() || T.size() <= Y.size(),"Y data insufficient");
   MFEM_ASSERT(!Z.size() || T.size() <= Z.size(),"Z data insufficient");
   center = _center;

   // Time history
   idata.reserve(T.size());
   for (unsigned int i = 0; i < T.size(); i++)
   {
      struct signal_data sig;
      sig.t = T[i];
      sig.v_x = i < X.size() ? X[i] : 0.0;
      sig.v_y = i < Y.size() ? Y[i] : 0.0;
      sig.v_z = i < Z.size() ? Z[i] : 0.0;
      idata.push_back(sig);
   }
   FinalizeIData();

   SetUpData();
}

void Receiver::SetUpData()
{
   PetscErrorCode ierr;
   TSTrajectory tj;

   ierr = VecDestroy(&W);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSDestroy(&ts);CCHKERRQ(PETSC_COMM_SELF,ierr);

   if (!idata.size()) return;

   ierr = VecCreateSeq(PETSC_COMM_SELF,3,&W);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSCreate(PETSC_COMM_SELF,&ts);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSSetSolution(ts,W);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSSetMaxSteps(ts,idata.size());CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSSetSaveTrajectory(ts);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSGetTrajectory(ts,&tj);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = PetscObjectSetOptionsPrefix((PetscObject)tj,"receiver_");CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSTrajectorySetType(tj,ts,TSTRAJECTORYMEMORY);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSTrajectorySetFromOptions(tj,ts);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSTrajectorySetSolutionOnly(tj,PETSC_TRUE);CCHKERRQ(PETSC_COMM_SELF,ierr);
   tj->adjoint_solve_mode = PETSC_FALSE;
   ierr = TSTrajectorySetUp(tj,ts);CCHKERRQ(PETSC_COMM_SELF,ierr);

   /* populate trajectory */
   for (unsigned int i = 0; i < idata.size(); i++) {
     PetscScalar *w;

     ierr = VecGetArray(W,&w);CCHKERRQ(PETSC_COMM_SELF,ierr);
     w[0] = idata[i].v_x;
     w[1] = idata[i].v_y;
     w[2] = idata[i].v_z;
     ierr = VecRestoreArray(W,&w);CCHKERRQ(PETSC_COMM_SELF,ierr);
     ierr = TSSetStepNumber(ts,i);CCHKERRQ(PETSC_COMM_SELF,ierr);
     ierr = TSTrajectorySet(tj,ts,i,idata[i].t,W);CCHKERRQ(PETSC_COMM_SELF,ierr);
   }
}

void Receiver::FinalizeIData()
{
   if (idata_isfinalized) return;
   std::sort(idata.begin(),idata.end(),ltcompare);
   idata_isfinalized = true;
}

void Receiver::GetIData(double time, Vector& vals)
{
   if (!vals.Size() || !idata.size())
   {
      vals = 0.0;
      return;
   }
   PetscErrorCode ierr;

   TSTrajectory tj;
   ierr = TSGetTrajectory(ts,&tj);CCHKERRQ(PETSC_COMM_SELF,ierr);

   PetscScalar ptime = time;
   ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&ptime,W,NULL);CCHKERRQ(PETSC_COMM_SELF,ierr);

   vals = 0.0;
   PetscScalar *w;
   ierr = VecGetArray(W,&w);CCHKERRQ(PETSC_COMM_SELF,ierr);
   for (int i = 0; i < std::min(vals.Size(),3); i++) vals[i] = w[i];
   ierr = VecRestoreArray(W,&w);CCHKERRQ(PETSC_COMM_SELF,ierr);
}

void Receiver::Dump(const std::string& filename, bool binary)
{
   MFEM_VERIFY(!binary,"binary dump not yet implemented");

   if (filename.size())
   {
      std::ofstream f(filename.c_str());
      MFEM_VERIFY(f.is_open(),"Error opening (w) " << filename);
      f << std::setprecision(12);
      ASCIIDump(f);
   }
   else
   {
      ASCIIDump();
   }
}

void Receiver::ASCIIDump(std::ostream& f)
{
   // Receiver location
   f << center.Size() << ' ';
   for (int i = 0; i < center.Size(); i++) f << center[i] << ' ';
   f << '\n';

   // Time history
   f << idata.size() << '\n';
   for (unsigned int i = 0; i < idata.size(); i++)
      f << idata[i].t << ' ' << idata[i].v_x << ' ' << idata[i].v_y << ' ' << idata[i].v_z << '\n';
}

void Receiver::Load(const std::string& filename, bool binary)
{
   MFEM_VERIFY(!binary,"binary load not yet implemented");

   std::ifstream f(filename.c_str());
   MFEM_VERIFY(f.is_open(),"Error opening (r) " << filename);

   ASCIILoad(f);
   FinalizeIData();
   SetUpData();
}

void Receiver::ASCIILoad(std::istream& f)
{
   MFEM_VERIFY(!idata.size(),"Input data already present");

   // Receiver location
   int sdim;
   f >> sdim;
   MFEM_VERIFY(sdim <=3,"Unhandled dimension " << sdim);
   center.SetSize(sdim);
   for (int i = 0; i < sdim; i++) f >> center[i];

   // Time history
   int nt;
   f >> nt;
   idata.reserve(nt);
   for (int i = 0; i < nt; i++)
   {
      struct signal_data sig;
      f >> sig.t;
      f >> sig.v_x;
      f >> sig.v_y;
      f >> sig.v_z;
      idata.push_back(sig);
   }
}

Receiver::~Receiver()
{
   PetscErrorCode ierr;

   ierr = VecDestroy(&W);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = TSDestroy(&ts);CCHKERRQ(PETSC_COMM_SELF,ierr);
}

ReceiverMonitor::ReceiverMonitor(ParGridFunction* _u, const DenseMatrix& _points, const std::string& _filename) :
      PetscSolverMonitor(true,false), u(_u), points(_points), filename(_filename)
{
   ParMesh *pmesh = u->ParFESpace()->GetParMesh();
   pmesh->FindPoints(points,eids,ips,true);

   T.reserve(1000);
   int np = 0;
   int vsize = -1;
   for (int i = 0; i<points.Width(); i++)
   {
      if (eids[i] < 0) continue;
      Vector R;
      u->GetVectorValue(eids[i],ips[i],R);
      vsize = R.Size();
      np++;
   }
   Xd.resize(np);
   Yd.resize(np);
   Zd.resize(np);
   for (int i = 0; i < np; i++) Xd[i].reserve(1000);
   if (vsize > 1)
   {
      for (int i = 0; i < np; i++) Yd[i].reserve(1000);
   }
   if (vsize > 2)
   {
      for (int i = 0; i < np; i++) Zd[i].reserve(1000);
   }
}

ReceiverMonitor::~ReceiverMonitor()
{
   for (int i = 0, np = 0; i<points.Width(); i++)
   {
      if (eids[i] < 0) continue;
      Vector c;
      points.GetColumn(i,c);
      Receiver r(c,T,Xd[np],Yd[np],Zd[np]);
      std::stringstream tmp;
      tmp << filename << "-" << i << ".txt";
      r.Dump(tmp.str());
      np++;
   }
   MPI_Barrier(u->ParFESpace()->GetParMesh()->GetComm());
}

void ReceiverMonitor::MonitorSolution(PetscInt step, PetscReal time, const Vector &X)
{
   ParFiniteElementSpace *pfes = u->ParFESpace();
   HypreParMatrix &P = *pfes->Dof_TrueDof_Matrix();
   P.Mult(X,*u);

   T.push_back(time);
   for (int i = 0, np = 0; i<points.Width(); i++)
   {
      if (eids[i] < 0) continue;
      Vector R;
      u->GetVectorValue(eids[i],ips[i],R);
      if (R.Size() > 2)
      {
         Zd[np].push_back(R[2]);
      }
      if (R.Size() > 1)
      {
         Yd[np].push_back(R[1]);
      }
      if (R.Size() > 0)
      {
         Xd[np].push_back(R[0]);
      }
      np++;
   }
}

}

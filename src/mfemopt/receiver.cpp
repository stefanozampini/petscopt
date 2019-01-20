#include <mfemopt/receiver.hpp>
#include <algorithm>
#include <limits>

namespace mfemopt
{
using namespace mfem;

Receiver::Receiver() : center(), idata(), idata_isfinalized(false) { }

Receiver::Receiver(double _x) : center(), idata(), idata_isfinalized(false)
{
   center.SetSize(1);
   center[0] = _x;
}

Receiver::Receiver(double _x, double _y) : center(), idata(), idata_isfinalized(false)
{
   center.SetSize(2);
   center[0] = _x;
   center[1] = _y;
}

Receiver::Receiver(double _x, double _y, double _z) : center(), idata(), idata_isfinalized(false)
{
   center.SetSize(3);
   center[0] = _x;
   center[1] = _y;
   center[2] = _z;
}

Receiver::Receiver(const std::string& filename) : center(), idata(), idata_isfinalized(false)
{
   Load(filename);
}

Receiver::Receiver(const Vector& _center, const std::vector<double> & T, const std::vector<double> & X, const std::vector<double> & Y, const std::vector<double> & Z) :
          center(), idata(), idata_isfinalized(false)
{
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
}

bool Receiver::idataIsSorted()
{
   if (!idata_isfinalized) return std::is_sorted(idata.begin(),idata.end(),ltcompare);
   return true;
}

void Receiver::FinalizeIdata()
{
   if (idata_isfinalized) return;
   std::sort(idata.begin(),idata.end(),ltcompare);
   idata_isfinalized = true;
}

void Receiver::GetIData(double time, Vector& vals)
{
   signal_data data;
   data.t = time;
   InterpolateLinear(data);
   switch (vals.Size())
   {
      case 3:
         vals(2) = data.v_z;
      case 2:
         vals(1) = data.v_y;
      case 1:
         vals(0) = data.v_x;
         break;
      default:
         MFEM_ABORT("Invalid dimension " << vals.Size());
         break;
   }
}

// interpolates input data at a given time stored in interp.t by using a binary_search
// results are placed in interp.v_x, interp.v_y and interp.v_z
void Receiver::InterpolateLinear(struct signal_data& interp)
{
  interp.v_x = interp.v_y = interp.v_z = std::numeric_limits<double>::min();
  double lx,ly,lz;
  double ux,uy,uz;
  double t0,t1,t = interp.t,tt;
  std::vector<struct signal_data>::iterator it1,it2,lb;

  MFEM_VERIFY(idata.size(),"Missing input data");
  MFEM_VERIFY(idataIsSorted(),"Input data not sorted!");
  /* XXX */ MFEM_VERIFY(interp.t >= idata[0].t,"Cannot extrapolate input data! " << interp.t << " < " << idata[0].t);
  if (interp.t < idata[0].t)
  {
     MFEM_WARNING("Cannot extrapolate input data! " << interp.t << " < " << idata[0].t);
     interp.t = idata[0].t;
  }
  it1 = idata.begin();
  it2 = idata.end();
  lb = std::lower_bound(it1,it2,interp,ltcompare);
  /* XXX */ MFEM_VERIFY(lb != it2,"Cannot extrapolate! " << interp.t << " > " << (*(lb-1)).t);
  if (lb == it2)
  {
     MFEM_WARNING("Cannot extrapolate! " << interp.t << " > " << (*(lb-1)).t);
     lb--;
  }
  t1 = lb->t;
  ux = lb->v_x;
  uy = lb->v_y;
  uz = lb->v_z;
  if (lb == it1) {       t0 = t1;    tt = 1.0;            lx = ux;      ly = uy;      lz = uz; }
  else           { lb--; t0 = lb->t; tt = (t-t0)/(t1-t0); lx = lb->v_x; ly = lb->v_y; lz = lb->v_z; }
  interp.v_x = lx * (1.0 - tt) + ux * tt;
  interp.v_y = ly * (1.0 - tt) + uy * tt;
  interp.v_z = lz * (1.0 - tt) + uz * tt;
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
   for (int i = 0; i < center.Size(); i++) f << center(i) << ' ';
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

ReceiverMonitor::ReceiverMonitor(ParGridFunction* _u, const DenseMatrix& _points, const std::string& _filename) :
      PetscSolverMonitor(true,false), u(_u), points(_points), filename(_filename)
{
   ParMesh *pmesh = u->ParFESpace()->GetParMesh();
   pmesh->FindPoints(points,eids,ips,false);

   T.reserve(1000);
   int np = 0;
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
      r.Dump(filename + "-" + std::to_string(i) + ".txt");
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
      if (vsize > 2)
      {
         Zd[np].push_back(R(2));
      }
      if (vsize > 1)
      {
         Yd[np].push_back(R(1));
      }
      Xd[np].push_back(R(0));
      np++;
   }
}

}

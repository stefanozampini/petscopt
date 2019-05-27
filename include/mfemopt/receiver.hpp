#if !defined(_MFEMOPT_RECEIVER_HPP)
#define _MFEMOPT_RECEIVER_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/linalg/vector.hpp>
#include <mfem/linalg/petsc.hpp>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/fem/intrules.hpp>
#include <mfem/general/array.hpp>
#include <fstream>
#include <ostream>
#include <vector>
#include <string>

namespace mfemopt
{

class Receiver
{
protected:
   // data format for the signal
   struct signal_data {
      double t;
      double v_x;
      double v_y;
      double v_z;
   };

   void InterpolateLinear(struct signal_data&);
   // comparison routine for signal_data
   static inline bool ltcompare(const struct signal_data& d1, const struct signal_data& d2)
   { return d1.t < d2.t; }

private:
   mfem::Vector                    center;
   std::vector<struct signal_data> idata;             // input data
   bool                            idata_isfinalized; // if true, time is sorted

   void ASCIILoad(std::istream&);
   void ASCIIDump(std::ostream& = std::cout);
   void FinalizeIData();

public:
   Receiver();
   Receiver(double);               // 1D receiver, no input (XXX vdim == sdim)
   Receiver(double,double);        // 2D receiver, no input (XXX vdim == sdim)
   Receiver(double,double,double); // 3D receiver, no input (XXX vdim == sdim)
   Receiver(const std::string&);   // load receiver data from filename
   Receiver(const mfem::Vector&,const std::vector<double>&,const std::vector<double>&,const std::vector<double>& = std::vector<double>(),const std::vector<double>& = std::vector<double>());

   void GetIData(double,mfem::Vector&); // access input data at a given time
   mfem::Vector& Center() { return center; }
   void Dump(const std::string&,bool = false); // dump data to filename
   void Load(const std::string&,bool = false); // load data from filename
};

class ReceiverMonitor : public mfem::PetscSolverMonitor
{
private:
   mfem::ParGridFunction* u;
   mfem::DenseMatrix      points;
   std::string            filename;

   /* receiver element location */
   mfem::Array<int>                    eids;
   mfem::Array<mfem::IntegrationPoint> ips;

   /* Signal data */
   std::vector<double> T;
   std::vector< std::vector<double> > Xd;
   std::vector< std::vector<double> > Yd;
   std::vector< std::vector<double> > Zd;

public:
   ReceiverMonitor(mfem::ParGridFunction*,const mfem::DenseMatrix&,const std::string&);
   virtual void MonitorSolution(PetscInt,PetscReal,const mfem::Vector&);
   virtual ~ReceiverMonitor();

};

}
#endif

#endif

#if !defined(_MFEMOPT_MONITOR_HPP)
#define _MFEMOPT_MONITOR_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/linalg/petsc.hpp>

namespace mfemopt
{

class NewtonMonitor : public mfem::PetscSolverMonitor
{
public:
   NewtonMonitor() : mfem::PetscSolverMonitor(false,false) {}
   virtual void MonitorSolver(mfem::PetscSolver*);
   virtual ~NewtonMonitor() {};
};

class OptimizationMonitor : public mfem::PetscSolverMonitor
{
public:
   OptimizationMonitor() : mfem::PetscSolverMonitor(false,false) {}
   virtual void MonitorSolver(mfem::PetscSolver*);
   virtual ~OptimizationMonitor() {};
};

}
#endif

#endif

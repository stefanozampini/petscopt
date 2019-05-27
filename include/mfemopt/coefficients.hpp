#if !defined(_MFEMOPT_COEFFICIENTS_HPP)
#define _MFEMOPT_COEFFICIENTS_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/fem/coefficient.hpp>
#include <mfem/linalg/vector.hpp>
#include <mfem/fem/eltrans.hpp>
#include <mfem/fem/intrules.hpp>

namespace mfemopt
{

class RickerSource : public mfem::DeltaCoefficient
{
private:
   double freq, tps;

public:
   RickerSource(const mfem::Vector&,
                double = 10. /* freq */,double = 1.4/* tps */,double = 1.0);
   virtual double EvalDelta(mfem::ElementTransformation&,
                            const mfem::IntegrationPoint&);
};

class VectorRickerSource : public mfem::VectorDeltaCoefficient
{
private:
   double freq, tps;

public:
   VectorRickerSource(const mfem::Vector&,const mfem::Vector& = mfem::Vector(),
                      double = 10. /* freq */,double = 1.4/* tps */,double = 1.0);
   virtual void EvalDelta(mfem::Vector&,mfem::ElementTransformation&,
                          const mfem::IntegrationPoint&);
};

}
#endif

#endif

#if !defined(_MFEMOPT_RANDOM_HPP)
#define _MFEMOPT_RANDOM_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/linalg/vector.hpp>

namespace mfemopt
{

class GaussianNoise
{
private:
   static double epsilon,two_pi;

   double mu,sigma;

   mutable bool   reuse;
   mutable double y0;

public:
   GaussianNoise(double = 0.0, double = 1.0);
   double random() const;
   void random(mfem::Vector&,int = -1) const;
   ~GaussianNoise() {}
};

}
#endif

#endif

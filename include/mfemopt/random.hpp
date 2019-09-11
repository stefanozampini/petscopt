#if !defined(_MFEMOPT_RANDOM_HPP)
#define _MFEMOPT_RANDOM_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/linalg/vector.hpp>

namespace mfemopt
{

class Noise
{
public:
   Noise() {}
   virtual double Random() const = 0;
   virtual void Randomize(mfem::Vector&,int = -1) const;
   virtual ~Noise() {}
};

class UniformNoise : public Noise
{
private:
   double A,B;

public:
   UniformNoise(double = 0.0, double = 1.0);
   virtual double Random() const;
};

class GaussianNoise : public Noise
{
private:
   static double epsilon,two_pi;

   double mu,sigma;

   mutable bool   reuse;
   mutable double y0;

public:
   GaussianNoise(double = 0.0, double = 1.0);
   virtual double Random() const;
};

}
#endif

#endif

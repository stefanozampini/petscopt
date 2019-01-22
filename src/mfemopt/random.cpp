#include <mfemopt/random.hpp>

#include <cstdlib>
#include <cmath>
#include <limits>

namespace mfemopt
{
using namespace mfem;

double GaussianNoise::epsilon = std::numeric_limits<double>::min();
double GaussianNoise::two_pi = 2.0*3.14159265358979323846;

GaussianNoise::GaussianNoise(double _mu, double _sigma)
{
   mu    = _mu;
   sigma = _sigma;
   reuse = true;
}

double GaussianNoise::random() const
{
   reuse = !reuse;
   if (reuse) return y0*sigma + mu;
   double r1,r2;
   do
   {
      r1 = std::rand()*(1.0/RAND_MAX);
      r2 = std::rand()*(1.0/RAND_MAX);
   } while (r1 <= epsilon);

   double y1,t;
   t  = std::sqrt(-2.0*std::log(r1));
   y0 = t*std::cos(two_pi*r2);
   y1 = t*std::sin(two_pi*r2);
   return y1 * sigma + mu;
}

void GaussianNoise::random(Vector& v, int n) const
{
   if (n >= 0) v.SetSize(n);
   for (int i = 0; i < v.Size(); i++) v(i) = (*this).random();
}

}

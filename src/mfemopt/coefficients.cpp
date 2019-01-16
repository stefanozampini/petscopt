#include <mfemopt/coefficients.hpp>

namespace mfemopt
{
using namespace mfem;

static double _ricker_freq = 1.;
static double _ricker_tps  = 1.;

static double _ricker(double t)
{
   //Ricker Wavelet
   double tp = 1/_ricker_freq;
   double ts = _ricker_tps*tp;
   double a = pow((M_PI*(t-ts)/tp),2);
   double ht = -2*(a-0.5)*exp(-a);
   return ht;
}

static double _ricker_v(double t)
{
   //Ricker Wavelet
   double tp = 1/_ricker_freq;
   double ts = _ricker_tps*tp;
   double a = pow((M_PI*(t-ts)/tp),2);
   double dhdt = (1.5-a)*(exp(-a))*(2 * pow(M_PI,2)*(t-ts)/(tp*tp));
   return dhdt;
}

VectorRickerSource::VectorRickerSource(const Vector& _loc, const Vector& _dir,
                                       double _freq, double _tps, double _s) : VectorDeltaCoefficient(0)
{
   freq = _freq;
   tps  = _tps;
   dir  = _dir;
   d.SetDeltaCenter(_loc);
   d.SetScale(_s);
   d.SetFunction(_ricker_v);
}

void VectorRickerSource::EvalDelta(Vector &V, ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   _ricker_freq = freq;
   _ricker_tps  = tps;
   VectorDeltaCoefficient::EvalDelta(V,T,ip);
}

RickerSource::RickerSource(const Vector& _loc,
                           double _freq, double _tps, double _s) : DeltaCoefficient()
{
   freq = _freq;
   tps  = _tps;
   (*this).SetDeltaCenter(_loc);
   (*this).SetScale(_s);
   (*this).SetFunction(_ricker);
}

double RickerSource::EvalDelta(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   _ricker_freq = freq;
   _ricker_tps  = tps;
   return DeltaCoefficient::EvalDelta(T,ip);
}

}

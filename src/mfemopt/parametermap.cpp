#include <mfemopt/parametermap.hpp>

namespace mfemopt
{
using namespace mfem;

void ParameterMap::SetUpHessianMap(const Vector& _m,const Vector& _cg)
{
   m  = _m;
   cg = _cg;
}

PointwiseMap::PointwiseMap(double (*_p)(double), double (*_pinv)(double), double (*_dp_dm)(double), double (*_d2p_dm2)(double))
{
   MFEM_VERIFY(_p,"Need to pass the mapping function!");
   MFEM_VERIFY(_dp_dm,"Need to pass the derivative of the mapping function!");
   p       = _p;
   pinv    = _pinv;
   dp_dm   = _dp_dm;
   d2p_dm2 = _d2p_dm2;
   if (d2p_dm2) second_order = true;
}

void PointwiseMap::Map(const Vector& m, Vector& model_m)
{
   model_m.SetSize(m.Size());
   for (int i = 0; i < m.Size(); i++) model_m(i) = (*p)(m(i));
}

void PointwiseMap::InverseMap(const Vector& model_m, Vector& m)
{
   m.SetSize(model_m.Size());
   MFEM_VERIFY(pinv,"Missing the inverse of the mapping function!");
   for (int i = 0; i < model_m.Size(); i++) m(i) = (*pinv)(model_m(i));
}

void PointwiseMap::GradientMap(const Vector& m, const Vector& gin, bool transpose, Vector& g)
{
   MFEM_VERIFY(gin.Size() == m.Size(),"Wrong sizes! " << m.Size() << " != " << gin.Size());
   g.SetSize(m.Size());
   for (int i = 0; i < m.Size(); i++) g(i) = (*dp_dm)(m(i))*gin(i);
}

void PointwiseMap::HessianMult(const Vector& x, Vector& y)
{
   const Vector& g = (*this).GetContractingGradient();
   const Vector& m = (*this).GetParameter();
   MFEM_VERIFY(m.Size() == x.Size(),"Wrong sizes! " << x.Size() << " != " << m.Size());
   MFEM_VERIFY(g.Size() == m.Size(),"Wrong sizes! " << m.Size() << " != " << g.Size());
   MFEM_VERIFY(d2p_dm2,"Missing the second derivative of the mapping function!");
   y.SetSize(x.Size());
   for (int i = 0; i < x.Size(); i++) y(i) = (*d2p_dm2)(m(i))*g(i)*x(i);
}

}

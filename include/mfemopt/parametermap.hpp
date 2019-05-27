#if !defined(_MFEMOPT_PARAMETERMAP_HPP)
#define _MFEMOPT_PARAMETERMAP_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <mfem/linalg/vector.hpp>

namespace mfemopt
{

/* Abstract class to map from optimization parameters to model parameters */
class ParameterMap
{
protected:
   mfem::Vector m,cg;
   bool second_order;

public:
   ParameterMap(bool so=false) : m(0), cg(0), second_order(so) { }
   virtual void Map(const mfem::Vector&,mfem::Vector&) = 0;
   virtual void InverseMap(const mfem::Vector&,mfem::Vector&)
   { mfem::mfem_error("ParameterMap::InverseMap not overloaded!"); }
   /*
      Jacobian of the mapping N x M matrix
      M #optimization parameters, N #model parameters, the boolean indicates the direction, i.e.
      true  : apply J
      false : apply J^T
   */
   virtual void GradientMap(const mfem::Vector&,const mfem::Vector&,bool,mfem::Vector&)
   { mfem::mfem_error("ParameterMap::GradientMap not overloaded!"); }
   /*
      The Hessian of the mapping: HessianMult(X,Y) should compute

      Y = (df/dmodel \otimes I_M)*(d2model/dparam2)*X
      with df/dmodel the gradient wrt the model parameters and d2model/dparam2 the Hessian of the parameter mapping.
      \otimes is the Kronecker product
      The SetUpHessianMap method can be overloaded: the first argument represent the current linearization point of
      the model parameters, while the second is df/dmodel. The default implementation just stores these two vectors.
   */
   virtual void SetUpHessianMap(const mfem::Vector&,const mfem::Vector&);
   virtual void HessianMult(const mfem::Vector&, mfem::Vector&)
   { mfem::mfem_error("ParameterMap::HessianMult not overloaded!"); }

   /* The method to retrieve the current df/dmodel */
   const mfem::Vector& GetContractingGradient() { return cg; };
   /* The method to retrieve the current linearization point in parameter space */
   const mfem::Vector& GetParameter() { return m; };
   bool SecondOrder() { return second_order; }
   virtual ~ParameterMap() { }
};

/* Map using a pointwise scalar function */
class PointwiseMap : public ParameterMap
{
private:
   double (*p)(double);
   double (*pinv)(double);
   double (*dp_dm)(double);
   double (*d2p_dm2)(double);

public:
   PointwiseMap(double (*)(double),double (*)(double),double (*)(double),double (*)(double));
   virtual void Map(const mfem::Vector&,mfem::Vector&);
   virtual void InverseMap(const mfem::Vector&,mfem::Vector&);
   virtual void GradientMap(const mfem::Vector&,const mfem::Vector&,bool,mfem::Vector&);
   virtual void HessianMult(const mfem::Vector&,mfem::Vector&);
   virtual ~PointwiseMap() { }
};

}
#endif

#endif

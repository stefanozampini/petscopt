#if !defined(_MFEMOPTUTILS_HPP)
#define _MFEMOPTUTILS_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <cstdlib>
#if defined(__GNUC__)
#include <cxxabi.h>
#endif
#include <string>
#include <iostream>

namespace mfemopt
{
template<typename T> inline
std::string type_name()
{
    int status;
    std::string tname = typeid(T).name();
#if defined(__GNUC__)
    char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
    if (!status)
    {
       tname = demangled_name;
       std::free(demangled_name);
    }
#endif
    return tname;
}

/* Safely returns a pointer to class R from a void* pointer */
template <class R,class A> inline R* mi_void_safe_cast(void *ctx)
{

  R* r = dynamic_cast<R*>(static_cast<A*>(ctx));
  A* a = dynamic_cast<A*>(static_cast<R*>(ctx));
  if (!a && !r) {
    std::string errstr;
    errstr = "Unsupported void_safe_cast to *" + type_name<R>() + ".\n";
    errstr += "  This is a known issue with void* contexts and C++ multiple inheritance.\n";
    errstr += "  Pass a void* context that first inherits from " + type_name<R>() + " or (alternatively) " + type_name<A>() + ".";
    std::cerr << errstr << std::endl;
    return NULL;
  }
  if (!r) r = static_cast<R*>(ctx);
  return r;
}

}
#endif

#endif

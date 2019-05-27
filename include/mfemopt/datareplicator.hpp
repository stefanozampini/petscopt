#if !defined(_MFEMOPT_DATAREPLICATOR_HPP)
#define _MFEMOPT_DATAREPLICATOR_HPP

#include <petscoptconf.h>

#if defined(PETSCOPT_HAVE_MFEMOPT)
#include <mfemoptconf.h>
#include <map>
#include <string>
#include <mfem/linalg/densemat.hpp>
#include <mfem/linalg/vector.hpp>
#include <petscsystypes.h> /* sftypes does not include systypes */
#include <petscsftypes.h>

namespace mfemopt
{

class DataReplicator
{
private:
   MPI_Comm parent_comm;
   MPI_Comm child_comm;
   MPI_Comm red_comm;
   int      nrep;
   int      color;

   std::map<std::string,PetscSF> sfmap;

public:
   DataReplicator(MPI_Comm,int,bool=true);
   inline int GetColor() { return color; }
   bool IsMaster() { return color ? false : true; }

   void Broadcast(const std::string&,int,const void*,int,void*,MPI_Datatype);
   void Broadcast(const std::string&,const mfem::Array<bool>&,mfem::Array<bool>&);
   void Broadcast(const mfem::Array<bool>&,mfem::Array<bool>&);

   void Broadcast(const std::string&,const mfem::Array<int>&,mfem::Array<int>&);
   void Broadcast(const mfem::Array<int>&,mfem::Array<int>&);

   void Broadcast(const std::string&,const mfem::Vector&,mfem::Vector&);
   void Broadcast(const mfem::Vector&,mfem::Vector&);

   void Broadcast(double,double*);
   void Broadcast(int,int*);

   void Reduce(const std::string&,int,const void*,int,void*,MPI_Datatype,MPI_Op);

   void Reduce(const std::string&,const mfem::Array<bool>&,mfem::Array<bool>&,MPI_Op = MPI_LOR);
   void Reduce(const mfem::Array<bool>&,mfem::Array<bool>&,MPI_Op = MPI_LOR);

   void Reduce(const std::string&,const mfem::Array<int>&,mfem::Array<int>&,MPI_Op = MPI_SUM);
   void Reduce(const mfem::Array<int>&,mfem::Array<int>&,MPI_Op = MPI_SUM);

   void Reduce(const std::string&,const mfem::Vector&,mfem::Vector&,MPI_Op = MPI_SUM);
   void Reduce(const mfem::Vector&,mfem::Vector&,MPI_Op = MPI_SUM);

   void Reduce(double,double*,MPI_Op = MPI_SUM);

   void Split(int,int*,int*);
   void Split(const mfem::DenseMatrix&,mfem::DenseMatrix&,int* = NULL);
   virtual ~DataReplicator();
};

}
#endif

#endif

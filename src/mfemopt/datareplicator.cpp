#include <mfemopt/datareplicator.hpp>
#include <mfemopt/private/mfemoptpetscmacros.h>
#include <petscsf.h>

namespace mfemopt
{
using namespace mfem;

DataReplicator::DataReplicator(MPI_Comm comm, int _nrep, bool contig)
{
   PetscErrorCode ierr;
   PetscSubcomm   subcomm;
   MPI_Comm       rcomm;
   PetscMPIInt    size,crank;

   nrep = _nrep;

   MFEM_VERIFY(nrep > 0,"Number of replicas should be positive");
   ierr = MPI_Comm_size(comm,&size); CCHKERRQ(comm,ierr);
   MFEM_VERIFY(!(size%nrep),"Size of comm must be a multiple of the number of replicas");

   ierr = PetscSubcommCreate(comm, &subcomm); CCHKERRQ(comm,ierr);
   ierr = PetscSubcommSetNumber(subcomm, (PetscInt)nrep); CCHKERRQ(comm,ierr);
   ierr = PetscSubcommSetType(subcomm, contig ? PETSC_SUBCOMM_CONTIGUOUS : PETSC_SUBCOMM_INTERLACED); CCHKERRQ(comm,ierr);

   color = subcomm->color;

   /* original comm */
   ierr = PetscCommDuplicate(comm,&parent_comm,NULL); CCHKERRQ(comm,ierr);
   /* comm for replicated mesh */
   ierr = PetscCommDuplicate(subcomm->child,&child_comm,NULL); CCHKERRQ(subcomm->child,ierr);
   /* reduction comm */
   ierr = MPI_Comm_rank(child_comm,&crank);CCHKERRQ(child_comm,ierr);
   ierr = MPI_Comm_split(parent_comm,crank,color,&rcomm);CCHKERRQ(comm,ierr);
   ierr = PetscCommDuplicate(rcomm,&red_comm,NULL); CCHKERRQ(rcomm,ierr);
   ierr = MPI_Comm_free(&rcomm);CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = PetscSubcommDestroy(&subcomm); CCHKERRQ(comm,ierr);
}

DataReplicator::~DataReplicator()
{
   PetscErrorCode ierr;
   std::map<std::string,PetscSF>::iterator it;
   for (it=sfmap.begin(); it!=sfmap.end(); ++it)
   {
      ierr = PetscSFDestroy(&it->second); CCHKERRQ(PETSC_COMM_SELF,ierr);
   }
   ierr = PetscCommDestroy(&parent_comm); CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = PetscCommDestroy(&child_comm); CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = PetscCommDestroy(&red_comm); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

void DataReplicator::Split(int tot, int *n, int *c)
{
   if (tot < 0) { *n = 0; *c = 0; return; }

   int nl = tot/nrep;
   int rm = tot%nrep;
   int cl = color*nl;
   if (color < rm) nl++;
   *n = nl;
   *c = cl + std::min(color,rm);
}

void DataReplicator::Split(const DenseMatrix& work, DenseMatrix &swork, int *oc)
{
   int n,c;
   Split(work.Width(),&n,&c);
   DenseMatrix t;
   t.CopyCols(work,c,c+n-1);
   swork = t;
   if (oc) *oc = c;
}

static PetscSF NewSF(MPI_Comm comm, const std::string& name, PetscInt nroots, PetscInt nleaves)
{
   PetscErrorCode ierr;
   PetscSFNode *iremote;
   ierr = PetscMalloc1(nleaves,&iremote); CCHKERRQ(PETSC_COMM_SELF,ierr);
   for (PetscInt i = 0; i < nleaves; i++)
   {
      iremote[i].rank  = 0;
      iremote[i].index = i;
   }
   PetscSF sf;
   ierr = PetscSFCreate(comm,&sf); CCHKERRQ(comm,ierr);
   ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER); PCHKERRQ(sf,ierr);
   ierr = PetscSFSetUp(sf); PCHKERRQ(sf,ierr);
   return sf;
}

void DataReplicator::Broadcast(const Vector& x, Vector &y)
{
   std::string s;
   Broadcast(s,x,y);
}

void DataReplicator::Broadcast(const std::string& name, const Vector& x, Vector &y)
{
   Broadcast(name,x.Size(),x.GetData(),y.Size(),y.GetData(),MPI_DOUBLE_PRECISION);
}

void DataReplicator::Broadcast(const Array<bool>& x, Array<bool> &y)
{
   std::string s;
   Broadcast(s,x,y);
}

void DataReplicator::Broadcast(const std::string& name, const Array<bool>& x, Array<bool> &y)
{
   Broadcast(name,x.Size(),x.GetData(),y.Size(),y.GetData(),MPI::BOOL);
}

void DataReplicator::Broadcast(const Array<int>& x, Array<int> &y)
{
   std::string s;
   Broadcast(s,x,y);
}

void DataReplicator::Broadcast(const std::string& name, const Array<int>& x, Array<int> &y)
{
   Broadcast(name,x.Size(),x.GetData(),y.Size(),y.GetData(),MPI_INT);
}

void DataReplicator::Broadcast(const std::string& name, int nin, const void* datain, int nout, void *dataout, MPI_Datatype unit)
{
   PetscErrorCode ierr;
   PetscInt nleaves,nroots;
   PetscSF sf;

   std::map<std::string,PetscSF>::iterator it;
   it = sfmap.find(name);
   if (it == sfmap.end())
   {

      nroots = IsMaster() ? nin : 0;
      nleaves = nout;
      sf = NewSF(red_comm,name,nroots,nleaves);
      if (name.length()) sfmap.insert(std::pair<std::string,PetscSF>(name,sf));
   }
   else
   {
      sf = it->second;
   }

   ierr = PetscSFGetGraph(sf,&nroots,&nleaves,NULL,NULL); PCHKERRQ(sf,ierr);
   MFEM_VERIFY(nin >= nroots,"Invalid size for input: " << nin << " < " << nroots);
   MFEM_VERIFY(nout >= nleaves,"Invalid size for output: " << nout << " < " << nleaves);

   ierr = PetscSFBcastBegin(sf,unit,datain,dataout); PCHKERRQ(sf,ierr);
   ierr = PetscSFBcastEnd(sf,unit,datain,dataout); PCHKERRQ(sf,ierr);
   if (!name.length())
   {
      ierr = PetscSFDestroy(&sf); CCHKERRQ(PETSC_COMM_SELF,ierr);
   }
}

void DataReplicator::Reduce(const Vector& x, Vector &y, MPI_Op op)
{
   std::string s;
   Reduce(s,x,y,op);
}

void DataReplicator::Reduce(const std::string& name, const Vector& x, Vector &y, MPI_Op op)
{
   Reduce(name,x.Size(),x.GetData(),y.Size(),y.GetData(),MPI_DOUBLE_PRECISION,op);
}

void DataReplicator::Reduce(const Array<bool>& x, Array<bool> &y, MPI_Op op)
{
   std::string s;
   Reduce(s,x,y,op);
}

void DataReplicator::Reduce(const std::string& name, const Array<bool>& x, Array<bool> &y, MPI_Op op)
{
   Reduce(name,x.Size(),x.GetData(),y.Size(),y.GetData(),MPI::BOOL,op);
}

void DataReplicator::Reduce(const Array<int>& x, Array<int> &y, MPI_Op op)
{
   std::string s;
   Reduce(s,x,y,op);
}

void DataReplicator::Reduce(const std::string& name, const Array<int>& x, Array<int> &y, MPI_Op op)
{
   Reduce(name,x.Size(),x.GetData(),y.Size(),y.GetData(),MPI_INT,op);
}

void DataReplicator::Reduce(const std::string& name, int nin, const void* datain, int nout, void *dataout, MPI_Datatype unit, MPI_Op op)
{
   PetscErrorCode ierr;
   PetscInt nleaves,nroots;
   PetscSF sf;

   std::map<std::string,PetscSF>::iterator it;
   it = sfmap.find(name);
   if (it == sfmap.end())
   {
      nroots = IsMaster() ? nout : 0;
      nleaves = nin;
      sf = NewSF(red_comm,name,nroots,nleaves);
      if (name.length()) sfmap.insert(std::pair<std::string,PetscSF>(name,sf));
   }
   else
   {
      sf = it->second;
   }

   ierr = PetscSFGetGraph(sf,&nroots,&nleaves,NULL,NULL); PCHKERRQ(sf,ierr);
   MFEM_VERIFY(nin >= nleaves,"Invalid size for input: " << nin << " < " << nleaves);
   MFEM_VERIFY(nout >= nroots,"Invalid size for output: " << nout << " < " << nroots);

   ierr = PetscSFReduceBegin(sf,unit,datain,dataout,op); PCHKERRQ(sf,ierr);
   ierr = PetscSFReduceEnd(sf,unit,datain,dataout,op); PCHKERRQ(sf,ierr);
   if (!name.length())
   {
      ierr = PetscSFDestroy(&sf); CCHKERRQ(PETSC_COMM_SELF,ierr);
   }
}

void DataReplicator::Broadcast(double in, double *out)
{
   double tout;
   Broadcast("_mfemopt_data_repl_scalar",1,&in,1,&tout,MPI_DOUBLE_PRECISION);
   *out = tout;
}

void DataReplicator::Broadcast(int in, int *out)
{
   int tout;
   Broadcast("_mfemopt_data_repl_scalar",1,&in,1,&tout,MPI_INT);
   *out = tout;
}

void DataReplicator::Reduce(double in, double *out, MPI_Op op)
{
   Vector X(1),Y(1);
   X[0] = in;
   Y[0] = *out;
   Reduce("_mfemopt_data_repl_scalar",X,Y,op);
   *out = Y[0];
}

}

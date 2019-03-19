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

void DataReplicator::Broadcast(std::string name, const Vector& x, Vector &y)
{
   PetscErrorCode ierr;
   PetscInt nleaves,nroots;
   PetscSF sf;

   std::map<std::string,PetscSF>::iterator it;
   it = sfmap.find(name);
   if (it == sfmap.end())
   {
      PetscSFNode *iremote;
      nroots = IsMaster() ? x.Size() : 0;
      nleaves = y.Size();
      ierr = PetscMalloc1(nleaves,&iremote); CCHKERRQ(PETSC_COMM_SELF,ierr);
      for (PetscInt i = 0; i < nleaves; i++)
      {
         iremote[i].rank  = 0;
         iremote[i].index = i;
      }
      ierr = PetscSFCreate(red_comm,&sf); CCHKERRQ(red_comm,ierr);
      ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER); PCHKERRQ(sf,ierr);
      ierr = PetscSFSetUp(sf); PCHKERRQ(sf,ierr);
      sfmap.insert(std::pair<std::string,PetscSF>(name,sf));
   }
   else
   {
      sf = it->second;
   }

   ierr = PetscSFGetGraph(sf,&nroots,&nleaves,NULL,NULL); PCHKERRQ(sf,ierr);
   MFEM_VERIFY(x.Size() >= nroots,"Invalid size for x: " << x.Size() << " < " << nroots);
   MFEM_VERIFY(y.Size() >= nleaves,"Invalid size for y: " << y.Size() << " < " << nleaves);

   double *xd,*yd;
   xd = x.GetData();
   yd = y.GetData();
   ierr = PetscSFBcastBegin(sf,MPI_DOUBLE_PRECISION,xd,yd); PCHKERRQ(sf,ierr);
   ierr = PetscSFBcastEnd(sf,MPI_DOUBLE_PRECISION,xd,yd); PCHKERRQ(sf,ierr);
}

void DataReplicator::Reduce(std::string name, const Vector& x, Vector &y, MPI_Op op)
{
   PetscErrorCode ierr;
   PetscInt nleaves,nroots;
   PetscSF sf;

   std::map<std::string,PetscSF>::iterator it;
   it = sfmap.find(name);
   if (it == sfmap.end())
   {
      PetscSFNode *iremote;
      nroots = IsMaster() ? y.Size() : 0;
      nleaves = x.Size();
      ierr = PetscMalloc1(nleaves,&iremote); CCHKERRQ(PETSC_COMM_SELF,ierr);
      for (PetscInt i = 0; i < nleaves; i++)
      {
         iremote[i].rank  = 0;
         iremote[i].index = i;
      }
      ierr = PetscSFCreate(red_comm,&sf); CCHKERRQ(red_comm,ierr);
      ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER); PCHKERRQ(sf,ierr);
      ierr = PetscSFSetUp(sf); PCHKERRQ(sf,ierr);
      sfmap.insert(std::pair<std::string,PetscSF>(name,sf));
   }
   else
   {
      sf = it->second;
   }

   ierr = PetscSFGetGraph(sf,&nroots,&nleaves,NULL,NULL); PCHKERRQ(sf,ierr);
   MFEM_VERIFY(x.Size() >= nleaves,"Invalid size for x: " << x.Size() << " < " << nleaves);
   MFEM_VERIFY(y.Size() >= nroots,"Invalid size for y: " << y.Size() << " < " << nroots);

   double *xd,*yd;
   xd = x.GetData();
   yd = y.GetData();
   ierr = PetscSFReduceBegin(sf,MPI_DOUBLE_PRECISION,xd,yd,op); PCHKERRQ(sf,ierr);
   ierr = PetscSFReduceEnd(sf,MPI_DOUBLE_PRECISION,xd,yd,op); PCHKERRQ(sf,ierr);
}

void DataReplicator::Broadcast(double in, double *out)
{
   Vector X(1),Y(1);
   X[0] = in;
   Broadcast("_mfemopt_data_repl_scalar",X,Y);
   *out = Y[0];
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

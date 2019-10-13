static const char help[] = "Mesh-dependent coefficient map";

#include <mfemopt.hpp>
//#include <mfemopt/private/mfemoptpetscmacros.h>
#include <mfem.hpp>
#include <petscsf.h>

#define DUMP_DEBUG 1

using namespace mfem;
using namespace mfemopt;

static double slow_init(const Vector &x)
{
   double s = 1.0;
   for (int i = 0; i < x.Size(); i++) s += 1./(x[i]*x[i] + 1.0);
   return s;
}

static double sigma_init(const Vector &x)
{
   double s = 1.0;
   for (int i = 0; i < x.Size(); i++) s += x[i]*x[i];
   return s;
}

/************************************************************************************/
/* callbacks for the ParameterMap */
static double _mubase = std::exp(1.0);

/* optimization to model */
static double mu_of_m(double m)
{
   return std::pow(_mubase,m);
}

/* first derivative of optimization to model */
static double dmu_dm(double m)
{
   return std::pow(_mubase,m)*std::log(_mubase);
}

/* second derivative of optimization to model */
static double d2mu_dm2(double m)
{
   return std::pow(_mubase,m)*std::log(_mubase)*std::log(_mubase);
}

/* model to optimization */
static double m_of_mu(double mu)
{
   return std::log(mu)/std::log(_mubase);
}

/************************************************************************************/
/* optimization mesh class */
class PizzaBoxMesh : public Mesh
{
private:
   int _rx,_ry,_rz;
   double _sx0,_sy0,_sz0;
   Array<int> _refs;

public:
   PizzaBoxMesh(int,int,int,double,double,double,int,int,int,double,double* =NULL,double* =NULL);
   int GetAnisoRefines();
   Array<int>& GetAnisoRefinements() { return _refs; }
   ~PizzaBoxMesh();
};

PizzaBoxMesh::PizzaBoxMesh(int rx,int ry,int rz,double sx0,double sy0,double sz0,int nx,int ny,int nz,double h,
                           double *ril, double *riu)
{
   _rx = rx;
   _ry = ry;
   _rz = rz;
   _sx0 = sx0;
   _sy0 = sy0;
   _sz0 = sz0;
   double hx = h * std::pow(2,_rx);
   double hy = h * std::pow(2,_ry);
   double hz = h * std::pow(2,_rz);

   int NVert, NElem, NBdrElem;

   NVert = (nx+1) * (ny+1) * (nz+1);
   NElem = nx * ny * nz;
   NBdrElem = 2*(nx*ny+nx*nz+ny*nz);

   InitMesh(3, 3, NVert, NElem, NBdrElem);

   /* Region of interest bounding box */
   double bb_ll[6] = {PETSC_MIN_REAL,PETSC_MIN_REAL,PETSC_MIN_REAL};
   double bb_ur[6] = {PETSC_MAX_REAL,PETSC_MAX_REAL,PETSC_MAX_REAL};
   if (ril)
   {
      bb_ll[0] = ril[0];
      bb_ll[1] = ril[1];
      bb_ll[2] = ril[2];
   }
   if (riu)
   {
      bb_ur[0] = riu[0];
      bb_ur[1] = riu[1];
      bb_ur[2] = riu[2];
   }

   double coord[3];
   int x, y, z;
   // Sets vertices and the corresponding coordinates
   for (z = 0; z <= nz; z++)
   {
      coord[2] = _sz0 + z*hz;
      for (y = 0; y <= ny; y++)
      {
         coord[1] = _sy0 + y*hy;
         for (x = 0; x <= nx; x++)
         {
            coord[0] = _sx0 + x*hx;
            AddVertex(coord);
         }
      }
   }

#define VTX(XC, YC, ZC) ((XC)+((YC)+(ZC)*(ny+1))*(nx+1))

   int ind[8];
   for (z = 0; z < nz; z++)
   {
      for (y = 0; y < ny; y++)
      {
         for (x = 0; x < nx; x++)
         {
            ind[0] = VTX(x  , y  , z  );
            ind[1] = VTX(x+1, y  , z  );
            ind[2] = VTX(x+1, y+1, z  );
            ind[3] = VTX(x  , y+1, z  );
            ind[4] = VTX(x  , y  , z+1);
            ind[5] = VTX(x+1, y  , z+1);
            ind[6] = VTX(x+1, y+1, z+1);
            ind[7] = VTX(x  , y+1, z+1);

            // mark region of interest
            const double *el_ll = GetVertex(ind[0]);
            const double *el_ur = GetVertex(ind[6]);
            bool flg1 = el_ur[0] > bb_ll[0];
            bool flg2 = el_ll[0] < bb_ur[0];
            bool flg3 = el_ur[1] > bb_ll[1];
            bool flg4 = el_ll[1] < bb_ur[1];
            bool flg5 = el_ur[2] > bb_ll[2];
            bool flg6 = el_ll[2] < bb_ur[2];
            //printf("CHECKING [%g,%g] x [%g,%g] x [%g,%g] against [%g,%g] x [%g,%g] x [%g,%g]\n",el_ll[0],el_ur[0],el_ll[1],el_ur[1],el_ll[2],el_ur[2],bb_ll[0],bb_ur[0],bb_ll[1],bb_ur[1],bb_ll[2],bb_ur[2]);
            //printf(" -> %d %d %d %d %d %d\n",flg1,flg2,flg3,flg4,flg5,flg6);
            bool intersect = (flg1 && flg2 && flg3 && flg4 && flg5 && flg6);
            AddHex(ind, intersect ? 1 : 2);
         }
      }
   }

   // Sets boundary elements and the corresponding indices of vertices
   // bottom, bdr. attribute 1
   for (y = 0; y < ny; y++)
   {
      for (x = 0; x < nx; x++)
      {
         ind[0] = VTX(x  , y  , 0);
         ind[1] = VTX(x  , y+1, 0);
         ind[2] = VTX(x+1, y+1, 0);
         ind[3] = VTX(x+1, y  , 0);
         AddBdrQuad(ind, 1);
      }
   }
   // top, bdr. attribute 6
   for (y = 0; y < ny; y++)
   {
      for (x = 0; x < nx; x++)
      {
         ind[0] = VTX(x  , y  , nz);
         ind[1] = VTX(x+1, y  , nz);
         ind[2] = VTX(x+1, y+1, nz);
         ind[3] = VTX(x  , y+1, nz);
         AddBdrQuad(ind, 6);
      }
   }
   // left, bdr. attribute 5
   for (z = 0; z < nz; z++)
   {
      for (y = 0; y < ny; y++)
      {
         ind[0] = VTX(0  , y  , z  );
         ind[1] = VTX(0  , y  , z+1);
         ind[2] = VTX(0  , y+1, z+1);
         ind[3] = VTX(0  , y+1, z  );
         AddBdrQuad(ind, 5);
      }
   }
   // right, bdr. attribute 3
   for (z = 0; z < nz; z++)
   {
      for (y = 0; y < ny; y++)
      {
         ind[0] = VTX(nx, y  , z  );
         ind[1] = VTX(nx, y+1, z  );
         ind[2] = VTX(nx, y+1, z+1);
         ind[3] = VTX(nx, y  , z+1);
         AddBdrQuad(ind, 3);
      }
   }

   // front, bdr. attribute 2
   for (x = 0; x < nx; x++)
   {
      for (z = 0; z < nz; z++)
      {
         ind[0] = VTX(x  , 0, z  );
         ind[1] = VTX(x+1, 0, z  );
         ind[2] = VTX(x+1, 0, z+1);
         ind[3] = VTX(x  , 0, z+1);
         AddBdrQuad(ind, 2);
      }
   }
   // back, bdr. attribute 4
   for (x = 0; x < nx; x++)
   {
      for (z = 0; z < nz; z++)
      {
         ind[0] = VTX(x  , ny, z  );
         ind[1] = VTX(x  , ny, z+1);
         ind[2] = VTX(x+1, ny, z+1);
         ind[3] = VTX(x+1, ny, z  );
         AddBdrQuad(ind, 4);
      }
   }

#undef VTX

   FinalizeTopology();
   EnsureNCMesh();

   // array containing a sequence of anisotropic refinements to reach the hx = hy = hz limit the seismic code desires
   int maxr = _rx > _ry ? _rx : _ry;
   maxr = maxr > _rz ? maxr : _rz;
   _refs.Reserve(maxr);
   for (int i = 0; i < maxr; i++)
   {
      int ref_type = 0;
      if (i < _rx) ref_type += 1;
      if (i < _ry) ref_type += 2;
      if (i < _rz) ref_type += 4;
      _refs.Append(ref_type);
   }
}

int PizzaBoxMesh::GetAnisoRefines()
{
   return _refs.Size();
}

PizzaBoxMesh::~PizzaBoxMesh() {}

class FakeObj : public ReducedFunctional
{
private:
   MPI_Comm _comm;
   Operator *H;

public:
   FakeObj(MPI_Comm,int);
   virtual void ComputeObjective(const Vector&,double*) const;
   virtual void ComputeGradient(const Vector&,Vector&) const;
   virtual Operator& GetHessian(const Vector&) const;
   ~FakeObj() { delete H; };
};

FakeObj::FakeObj(MPI_Comm comm, int size)
{
   height = width = size;
   _comm = comm;

   Mat M,MT;
   MatCreateDense(comm,height,width,PETSC_DECIDE,PETSC_DECIDE,NULL,&M);
   MatSetRandom(M,NULL);
   MatTranspose(M,MAT_INITIAL_MATRIX,&MT);
   MatAXPY(M,1.0,MT,SAME_NONZERO_PATTERN);
   MatDestroy(&MT);
   H = new PetscParMatrix(M,false);
}

void FakeObj::ComputeObjective(const Vector& m, double *obj) const
{
   Vector t(m.Size());
   H->Mult(m,t);

   *obj = 0.5*InnerProduct(_comm,t,m);
}

void FakeObj::ComputeGradient(const Vector& m, Vector& g) const
{
   H->Mult(m,g);
}

Operator& FakeObj::GetHessian(const Vector& m) const
{
   return *H;
}

/************************************************************************************/
class OptHandler : public ReducedFunctional
{
private:
   PizzaBoxMesh *_optmesh;

   VectorCoefficient *_vc;
   FiniteElementCollection *_vcfec;
   MPI_Comm _vccomm;
   ParMesh *_vcmesh;
   PDCoefficient *_vcpd;
   PetscSF _vcsf;
   Array<int> _vcexcl_tag;

   TVRegularizer *_vcobj;
   double _vcw;

   MPI_Comm _emcomm;
   ReplicatedParMesh *_emrpmesh;
   PDCoefficient *_empd;
   PetscSF _emsf;
   PetscParMatrix *_emtrans;
   mutable Vector _emin;
   PetscSF _emobjsf;
   PointwiseMap *_emmap;
   double _emw;
   ReducedFunctional *_emobj;

   MPI_Comm _secomm;
   double _sew;
   PetscSF _sesf;
   PetscParMatrix *_setrans;
   PetscSF _seobjsf;
   ReducedFunctional *_seobj;
   mutable Vector _sein;

   MPI_Comm _worldcomm;
   ParMesh* _worldmesh; /* serial, replicated */
   ParFiniteElementSpace *_worldfes; /* serial, replicated */

   void GetInitialSF(PDCoefficient*,int[],PetscSF*);

   // this should be a member of the base class
   mutable Operator* _H;

public:
   OptHandler(MPI_Comm,PizzaBoxMesh*);
   void SetUpCoefficient(MPI_Group,VectorCoefficient*,double,bool,bool);
   void SetUpEM(MPI_Group,int=1,int=1);
   void SetUpSeismic(MPI_Group,int=0);
   void SetWeights(double,double,double);

   virtual void ComputeObjective(const Vector&,double*) const;
   virtual void ComputeGradient(const Vector&,Vector&) const;
   virtual Operator& GetHessian(const Vector&) const;

   class Hessian : public Operator
   {
   private:
      const OptHandler& _opt;
      Operator *_vcH,*_emH,*_seH;
   public:
      Hessian(const OptHandler&,const Vector&);
      virtual void Mult(const Vector&,Vector&) const;
      virtual void MultTranspose(const Vector&,Vector&) const;
   };
   ~OptHandler();
};

OptHandler::OptHandler(MPI_Comm comm, PizzaBoxMesh *optmesh)
{
   _optmesh = optmesh;

   _vc = NULL;
   _vcfec = new H1_FECollection(1,_optmesh->Dimension());
   _vccomm = MPI_COMM_NULL;
   _vcmesh = NULL;
   _vcpd = NULL;

   _vcsf = NULL;

   _vcobj = NULL;
   _vcw = 1.0;

   _emcomm = MPI_COMM_NULL;
   _emrpmesh = NULL;
   _empd = NULL;
   _emtrans = NULL;
   _emsf = NULL;
   _emobjsf = NULL;
   _emmap = NULL;
   _emw = 1.0;
   _emobj = NULL;

   _secomm = MPI_COMM_NULL;
   _sew = 1.0;
   _setrans = NULL;
   _sesf = NULL;
   _seobj = NULL;
   _seobjsf = NULL;

   PetscCommDuplicate(comm,&_worldcomm,NULL);
   _worldmesh = new ParMesh(PETSC_COMM_SELF,*_optmesh);
   _worldfes = new ParFiniteElementSpace(_worldmesh, _vcfec);

   _H = NULL;
}

OptHandler::~OptHandler()
{
  delete _vcfec;
  delete _vcmesh;
  delete _vcpd;
  delete _vcobj;
  PetscSFDestroy(&_vcsf);
  MPI_Comm_free(&_vccomm);

  delete _emrpmesh;
  delete _empd;
  delete _emtrans;
  delete _emmap;
  delete _emobj;
  PetscSFDestroy(&_emsf);
  PetscSFDestroy(&_emobjsf);
  MPI_Comm_free(&_emcomm);

  delete _setrans;
  delete _seobj;
  PetscSFDestroy(&_sesf);
  PetscSFDestroy(&_seobjsf);
  MPI_Comm_free(&_secomm);

  delete _worldmesh;
  delete _worldfes;
  PetscCommDestroy(&_worldcomm);

  delete _H;
}

void OptHandler::GetInitialSF(PDCoefficient *pd, int part[], PetscSF *sf)
{
   Array<PetscSFNode> lp2wg;
   if (pd)
   {
      Array<ParGridFunction*>& pgf = pd->GetCoeffs();
      if (pgf.Size())
      {
         Array<PetscInt> lcols = pd->GetLocalCols();

         ParFiniteElementSpace *pfes = pgf[0]->ParFESpace();
         Array<int> wdofs, pdofs;
         int rank, elcnt = 0;

         lp2wg.SetSize(pfes->GetTrueVSize());
         for (int i = 0; i < lp2wg.Size(); i++)
         {
            lp2wg[i].index = -1;
            lp2wg[i].rank = -1;
         }

         Array<int> *map = NULL;
         if (lcols.Size() < lp2wg.Size())
         {
            map = new Array<int>(lp2wg.Size());
            *map = -1;
            for (int i = 0; i < lcols.Size(); i++) (*map)[lcols[i]] = i;
         }

         MPI_Comm_rank(pfes->GetParMesh()->GetComm(),&rank);
         for (int i = 0; i < _optmesh->GetNE(); i++)
         {
            if (part ? part[i] == rank : 1)
            {
               pfes->GetElementVDofs(elcnt, pdofs);
               _worldfes->GetElementVDofs(i, wdofs);
               MFEM_VERIFY(pdofs.Size() == wdofs.Size(),"Mismatch local dofs");
               for (int d = 0; d < pdofs.Size(); d++)
               {
                  int pdof = pdofs[d] < 0 ? -1-pdofs[d] : pdofs[d];
                  int wdof = wdofs[d] < 0 ? -1-wdofs[d] : wdofs[d];
                  int lpdof = pfes->GetLocalTDofNumber(pdof);
                  int lwdof = _worldfes->GetLocalTDofNumber(wdof);
                  if (lpdof > -1) /* owned by the PDCoefficient */
                  {
                     MFEM_VERIFY(lwdof > -1,"This should not happen");
                     int mlpdof = map ? (*map)[lpdof] : lpdof;
                     if (mlpdof > -1)
                     {
                        lp2wg[mlpdof].index = _worldfes->GetGlobalTDofNumber(lwdof);
                        lp2wg[mlpdof].rank = 0;
                     }
                  }
               }
               elcnt++;
            }
         }
         delete map;
      }
   }

   // PetscSF
   PetscMPIInt wrank;
   PetscInt nroots = 0, nleaves = 0;
   for (int i = 0; i < lp2wg.Size(); i++)
   {
      if (lp2wg[i].index == -1) continue;
      lp2wg[nleaves].index = lp2wg[i].index;
      lp2wg[nleaves].rank = lp2wg[i].rank;
      nleaves++;
   }
   MPI_Comm_rank(_worldcomm, &wrank);
   if (!wrank)
   {
      nroots = _worldfes->GlobalTrueVSize();
   }
   PetscSFCreate(_worldcomm,sf);
   PetscSFSetGraph(*sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,lp2wg.GetData(),PETSC_COPY_VALUES);
   PetscSFSetUp(*sf);
}

void OptHandler::SetUpCoefficient(MPI_Group group, VectorCoefficient* vc, double beta, bool pd, bool uncoupled)
{
   height = width = 0;

   int *vcpart = NULL;

   _vcexcl_tag.SetSize(1);
   _vcexcl_tag[0] = 2;

   int nel = 0;
   _vc = vc;
   MPI_Comm_create(_worldcomm,group,&_vccomm);
   if (_vccomm != MPI_COMM_NULL)
   {
     int size;
     MPI_Comm_size(_vccomm,&size);
      vcpart = _optmesh->GeneratePartitioning(size, 1);
      _vcmesh = new ParMesh(_vccomm,*_optmesh,vcpart,1);
      _vcpd = new PDCoefficient(*_vc,_vcmesh,_vcfec,_vcexcl_tag);
      _vcobj = new TVRegularizer(_vcpd,beta,pd,uncoupled);
      height = width = _vcpd->GetLocalSize();
      Array<bool> exel = _vcpd->GetExcludedElements();
      nel = exel.Size();
      for (int i = 0; i < exel.Size(); i++) if (exel[i]) nel--;
   }

   /* stats */
   PetscInt ls[2],gs[2];
   ls[0] = height;
   ls[1] = nel;
   MPI_Allreduce(&ls,&gs,2,MPIU_INT,MPI_SUM,_worldcomm);
   PetscPrintf(_worldcomm,"Number of optimization dofs %D, elements %d\n",gs[0],gs[1]);

   GetInitialSF(_vcpd,vcpart,&_vcsf);
   delete [] vcpart;

#if DUMP_DEBUG
   {
      int wrank;
      MPI_Comm_rank(_worldcomm,&wrank);
      PDCoefficient mpd(*_vc,_worldmesh,_vcfec);
      if (!wrank)
      {
         mpd.Save("masterpd");
      }

      Vector mv(mpd.GetLocalSize());
      Vector sv(_vcpd ? _vcpd->GetLocalSize() : 0);
      mpd.GetCurrentVector(mv);
      if (_vcpd) _vcpd->GetCurrentVector(sv);
      PetscScalar *rdata = mv.GetData();
      PetscScalar *ldata = sv.GetData();
      PetscInt rsize = mpd.GetLocalCols().Size();
      PetscInt lsize = _vcpd ? _vcpd->GetLocalCols().Size() : 0;
      for (int i = 0; i < mpd.GetCoeffs().Size(); i++)
      {
         PetscSFBcastBegin(_vcsf,MPIU_SCALAR,rdata,ldata);
         PetscSFBcastEnd(_vcsf,MPIU_SCALAR,rdata,ldata);
         rdata += rsize;
         ldata += lsize;
      }
      if (_vcpd)
      {
         _vcpd->Distribute(sv);
         _vcpd->Save("vcpd");
         _vcpd->SaveExcl("vcpd_ex");
      }
   }
#endif
}

void OptHandler::SetUpEM(MPI_Group group, int nrep, int nref)
{
   MPI_Comm_create(_worldcomm,group,&_emcomm);
   int *empart = NULL;
   if (_emcomm != MPI_COMM_NULL)
   {
      bool contig = true; /* broken with contig = false */
      /* cannot use anisotropic refinement (ParNCMesh::Prune() not implemented for anisotropic refinements (happens when distributing)) */
      _emrpmesh = new ReplicatedParMesh(_emcomm,*_optmesh,nrep,contig,&empart);

      ComponentCoefficient coeff(*_vc,0);
      _empd = new PDCoefficient(coeff,_emrpmesh->GetChild(),_vcfec,_vcexcl_tag);
      _emin.SetSize(_emrpmesh->IsMaster() ? _empd->GetLocalSize() : 0);
      _emmap = new PointwiseMap(mu_of_m,m_of_mu,dmu_dm,d2mu_dm2);
   }
   GetInitialSF(_emrpmesh ? (_emrpmesh->IsMaster() ? _empd : NULL) : NULL,empart,&_emsf);

   /* from input vector in callbacks to master in em */
   PetscSF tsf;
   PetscSFCreateInverseSF(_vcsf,&tsf);
   PetscSFSetUp(tsf);
   PetscSFCompose(tsf,_emsf,&_emobjsf);
   PetscSFDestroy(&tsf);

   int anel = 0, nel = 0;
   if (_empd)
   {
      for (int i = 0; i < nref; i++)
      {
         _emrpmesh->GetChild()->UniformRefinement();
         _empd->Update(true);
         if (!i)
         {
            _emtrans = new PetscParMatrix(_empd->GetTrueTransferOperator(),_empd->GetTrueTransferOperator()->GetType());
         }
         else
         {
            PetscParMatrix *t = ParMult(_empd->GetTrueTransferOperator(),_emtrans);
            delete _emtrans;
            _emtrans = t;
         }
      }
      _emrpmesh->GetChild()->Rebalance();
      _empd->Update(true);
      if (_emtrans)
      {
         PetscParMatrix *t = ParMult(_empd->GetTrueTransferOperator(),_emtrans);
         delete _emtrans;
         _emtrans = t;
      }
      else
      {
         _emtrans = new PetscParMatrix(_empd->GetTrueTransferOperator(),_empd->GetTrueTransferOperator()->GetType());
      }

      _emobj = new FakeObj(_emrpmesh->GetReplicator().GetChildComm(),_empd->GetLocalSize());

      Array<bool> exel = _empd->GetExcludedElements();
      nel = anel = exel.Size();
      for (int i = 0; i < exel.Size(); i++) if (exel[i]) anel--;
      if (!_emrpmesh->IsMaster()) nel = anel = 0;
   }

   /* stats */
   PetscInt ls[3],gs[3];
   ls[0] = _emrpmesh ? (_emrpmesh->IsMaster() ? _emobj->Height() : 0) : 0;
   ls[1] = anel;
   ls[2] = nel;
   MPI_Allreduce(&ls,&gs,3,MPIU_INT,MPI_SUM,_worldcomm);
   PetscPrintf(_worldcomm,"Number of mapped optimization dofs %D for EM, active elements %D, total elements %D\n",gs[0],gs[1],gs[2]);

#if DUMP_DEBUG
   {
      Vector mv(_vcpd ? _vcpd->GetLocalSize() : 0);
      if (_vcpd) _vcpd->GetCurrentVector(mv);
      PetscSFBcastBegin(_emobjsf,MPIU_SCALAR,mv.GetData(),_emin.GetData());
      PetscSFBcastEnd(_emobjsf,MPIU_SCALAR,mv.GetData(),_emin.GetData());
      if (_empd)
      {
         DataReplicator& drep = _emrpmesh->GetReplicator();
         Vector eminrep(_emtrans->Width());
         drep.Broadcast("em_in",_emin,eminrep);
         std::ostringstream name;
         name << "empd_" << _emrpmesh->GetColor();
         Vector sv(_empd->GetLocalSize());
         _emtrans->Mult(eminrep,sv);
         sv -= (double)_emrpmesh->GetColor();
         _empd->Distribute(sv);
         _empd->Save(name.str().c_str());
         name << "_ex";
         _empd->SaveExcl(name.str().c_str());
      }
   }
#endif
   delete [] empart;
}

void OptHandler::SetUpSeismic(MPI_Group group, int nref)
{
   MPI_Comm_create(_worldcomm,group,&_secomm);

   ParMesh *roptmesh = NULL;
   PDCoefficient *_sepd = NULL;
   if (_secomm != MPI_COMM_NULL)
   {
      int rank;
      MPI_Comm_rank(_secomm,&rank);
      if (!rank)
      {
         ComponentCoefficient coeff(*_vc,1);
         roptmesh = new ParMesh(PETSC_COMM_SELF,*_optmesh);
         _sepd = new PDCoefficient(coeff,roptmesh,_vcfec,_vcexcl_tag);
         _sein.SetSize(_sepd->GetLocalSize());
      }
   }
   GetInitialSF(_sepd,NULL,&_sesf);

   /* from input vector in callbacks to master in seisimic */
   PetscSF tsf;
   PetscSFCreateInverseSF(_vcsf,&tsf);
   PetscSFSetUp(tsf);
   PetscSFCompose(tsf,_sesf,&_seobjsf);
   PetscSFDestroy(&tsf);

   int nel = 0;
   if (_sepd)
   {
      Array<int>& refs = _optmesh->GetAnisoRefinements();
      for (int i = 0; i < refs.Size() + nref; i++)
      {
         if (i < refs.Size())
         {
            int nel = roptmesh->GetNE();
            int ref_type = refs[i];
            Array<Refinement> elrefs(nel);
            for (int e = 0; e < nel; e++)
            {
               elrefs[e].index = e;
               elrefs[e].ref_type = ref_type;
            }

            roptmesh->GeneralRefinement(elrefs);
         }
         else
         {
            roptmesh->UniformRefinement();
         }
         _sepd->Update(true);
         if (!i)
         {
            _setrans = new PetscParMatrix(_sepd->GetTrueTransferOperator(),_sepd->GetTrueTransferOperator()->GetType());
         }
         else
         {
            PetscParMatrix *t = ParMult(_sepd->GetTrueTransferOperator(),_setrans);
            delete _setrans;
            _setrans = t;
         }
      }

      if (!_setrans)
      {
         _setrans = ParMult(_sepd->GetR(),_sepd->GetP());
      }
      Array<bool> exel = _sepd->GetExcludedElements();
      nel = exel.Size();
      for (int i = 0; i < exel.Size(); i++) if (exel[i]) nel--;
   }
   if (_secomm != MPI_COMM_NULL)
   {
      _seobj = new FakeObj(_secomm,_sepd ? _sepd->GetLocalSize() : 0);
   }

#if DUMP_DEBUG
   {
      Vector mv(_vcpd ? _vcpd->GetLocalSize() : 0);
      if (_vcpd) _vcpd->GetCurrentVector(mv);
      PetscSFBcastBegin(_seobjsf,MPIU_SCALAR,mv.GetData()+mv.Size()/2,_sein.GetData());
      PetscSFBcastEnd(_seobjsf,MPIU_SCALAR,mv.GetData()+mv.Size()/2,_sein.GetData());
      if (_sepd)
      {
         Vector sv(_setrans ? _setrans->Height() : 0);
         if (_setrans) _setrans->Mult(_sein,sv);
         _sepd->Distribute(sv);
         _sepd->Save("sepd");
         _sepd->SaveExcl("sepd_ex");
      }
   }
#endif
   delete _sepd;
   delete roptmesh;

   /* stats */
   PetscInt ls[2],gs[2];
   ls[0] = _seobj ? _seobj->Height() : 0;
   ls[1] = nel;
   MPI_Allreduce(&ls,&gs,2,MPIU_INT,MPI_SUM,_worldcomm);
   PetscPrintf(_worldcomm,"Number of mapped optimization dofs for SEISMIC %D, elements %D\n",gs[0],gs[1]);
}

void OptHandler::SetWeights(double tvw, double emw, double sew)
{
   _vcw = tvw;
   _emw = emw;
   _sew = sew;
}

void OptHandler::ComputeObjective(const Vector& m, double *obj) const
{
   double tvobj = 0.0;
   double emobj = 0.0;
   double seobj = 0.0;

   MPI_Request *tvreq = NULL,*emreq = NULL,*sereq = NULL;
   int emtag,tvtag,setag,wrank;
   MPI_Comm_rank(_worldcomm,&wrank);

   PetscCommGetNewTag(_worldcomm,&tvtag);
   PetscCommGetNewTag(_worldcomm,&emtag);
   PetscCommGetNewTag(_worldcomm,&setag);
   if (!wrank)
   {
      tvreq = new MPI_Request;
      emreq = new MPI_Request;
      sereq = new MPI_Request;
      MPI_Irecv(&tvobj,1,MPIU_SCALAR,MPI_ANY_SOURCE,tvtag,_worldcomm,tvreq);
      MPI_Irecv(&emobj,1,MPIU_SCALAR,MPI_ANY_SOURCE,emtag,_worldcomm,emreq);
      MPI_Irecv(&seobj,1,MPIU_SCALAR,MPI_ANY_SOURCE,setag,_worldcomm,sereq);
   }
   if (_vcobj)
   {
      double f;
      Vector dummy;
      _vcobj->Eval(dummy,m,0.,&f);

      int rank;
      MPI_Comm_rank(_vccomm,&rank);
      if (!rank) MPI_Send(&f,1,MPIU_SCALAR,0,tvtag,_worldcomm);
   }

   // from opt to em master comm (em coefficient ordered first)
   PetscSFBcastBegin(_emobjsf,MPIU_SCALAR,m.GetData(),_emin.GetData());
   PetscSFBcastEnd(_emobjsf,MPIU_SCALAR,m.GetData(),_emin.GetData());
   if (_empd)
   {
      // From log to physical
      Vector emin_mapped(_emin.Size());
      _emmap->Map(_emin,emin_mapped);

      // From em master to children
      DataReplicator& drep = _emrpmesh->GetReplicator();
      Vector emin_rep(_emtrans->Width());
      drep.Broadcast("em_in",emin_mapped,emin_rep);

      // From coarse to fine
      Vector emin_fine(_emtrans->Height());
      _emtrans->Mult(emin_rep,emin_fine);

      // compute EM objective
      double emobjrep;
      _emobj->ComputeObjective(emin_fine,&emobjrep);

      // Sum contributions from children to em master
      double f = 0.0;
      drep.Reduce(emobjrep,&f);

      // Send to opt
      int rank;
      MPI_Comm_rank(drep.GetChildComm(),&rank);
      if (!rank && drep.IsMaster()) MPI_Send(&f,1,MPIU_SCALAR,0,emtag,_worldcomm);
   }

   // from opt to se master
   PetscSFBcastBegin(_seobjsf,MPIU_SCALAR,m.GetData() + m.Size()/2,_sein.GetData());
   PetscSFBcastEnd(_seobjsf,MPIU_SCALAR,m.GetData() + m.Size()/2,_sein.GetData());
   if (_seobj)
   {
      // From coarse to fine
      Vector sein_fine(_setrans ? _setrans->Height() : 0);
      if (_setrans) _setrans->Mult(_sein,sein_fine);

      // compute SE objective
      double f;
      _seobj->ComputeObjective(sein_fine,&f);

      // Send to opt
      int rank;
      MPI_Comm_rank(_secomm,&rank);
      if (!rank) MPI_Send(&f,1,MPIU_SCALAR,0,setag,_worldcomm);
   }

   if (!wrank)
   {
      MPI_Wait(tvreq,MPI_STATUSES_IGNORE);
      MPI_Wait(emreq,MPI_STATUSES_IGNORE);
      MPI_Wait(sereq,MPI_STATUSES_IGNORE);
   }
   delete tvreq;
   delete emreq;
   delete sereq;
   *obj = _vcw*tvobj;
   *obj+= _emw*emobj;
   *obj+= _sew*seobj;

   // All processors need to know the value of the objective
   MPI_Bcast(obj,1,MPIU_SCALAR,0,_worldcomm);
}

void OptHandler::ComputeGradient(const Vector& m, Vector& g) const
{
   Vector tvgrad(m.Size());
   Vector emgrad(m.Size());
   Vector segrad(m.Size());

   /**** TV ****/
   tvgrad = 0.0;
   if (_vcobj)
   {
      Vector dummy;
      _vcobj->EvalGradient_M(dummy,m,0.,tvgrad);
   }
   tvgrad *= _vcw;

   /**** EM ****/
   // from opt to em master comm
   PetscSFBcastBegin(_emobjsf,MPIU_SCALAR,m.GetData(),_emin.GetData());
   PetscSFBcastEnd(_emobjsf,MPIU_SCALAR,m.GetData(),_emin.GetData());
   if (_empd)
   {
      // save linearization point
      Vector m_save(_emin);

      // From log to physical
      Vector emin_mapped(_emin.Size());
      _emmap->Map(_emin,emin_mapped);

      // From master to children
      DataReplicator& drep = _emrpmesh->GetReplicator();
      Vector emin_rep(_emtrans->Width());
      drep.Broadcast("em_in",emin_mapped,emin_rep);

      // From coarse to fine
      Vector emin_fine(_emtrans->Height());
      _emtrans->Mult(emin_rep,emin_fine);

      // compute EM gradient
      Vector emin_fine_out(emin_fine.Size());
      _emobj->ComputeGradient(emin_fine,emin_fine_out);

      // From fine to coarse
      _emtrans->MultTranspose(emin_fine_out,emin_rep);

      // Sum contributions from children to em master
      emin_mapped = 0.0;
      drep.Reduce("em_in",emin_rep,emin_mapped);

      // From physical to log (apply J^T(mu(m)))
      _emmap->GradientMap(m_save,emin_mapped,true,_emin);
   }
   // from em master to opt
   emgrad = 0.0;
   PetscSFReduceBegin(_emobjsf,MPIU_SCALAR,_emin.GetData(),emgrad.GetData(),MPIU_REPLACE);
   PetscSFReduceEnd(_emobjsf,MPIU_SCALAR,_emin.GetData(),emgrad.GetData(),MPIU_REPLACE);
   emgrad *= _emw;

   /**** SEISMIC ****/
   // from opt to se master
   PetscSFBcastBegin(_seobjsf,MPIU_SCALAR,m.GetData() + m.Size()/2,_sein.GetData());
   PetscSFBcastEnd(_seobjsf,MPIU_SCALAR,m.GetData() + m.Size()/2,_sein.GetData());
   if (_seobj)
   {
      // From coarse to fine
      Vector sein_fine(_setrans ? _setrans->Height() : 0);
      if (_setrans) _setrans->Mult(_sein,sein_fine);

      // compute SE gradient
      Vector sein_fine_out(sein_fine.Size());
      _seobj->ComputeGradient(sein_fine,sein_fine_out);

      // From fine to coarse
      if (_setrans) _setrans->MultTranspose(sein_fine_out,_sein);
   }
   // from em master to opt
   segrad = 0.0;
   PetscSFReduceBegin(_seobjsf,MPIU_SCALAR,_sein.GetData(),segrad.GetData() + m.Size()/2,MPIU_REPLACE);
   PetscSFReduceEnd(_seobjsf,MPIU_SCALAR,_sein.GetData(),segrad.GetData() + m.Size()/2,MPIU_REPLACE);
   segrad *= _sew;

   g = tvgrad;
   g+= emgrad;
   g+= segrad;
}

Operator& OptHandler::GetHessian(const Vector& m) const
{
   delete _H;
   _H = new OptHandler::Hessian(*this,m);
   return *_H;
}

//H = H_map(_m) + J(mu(_m)) * ( H_em(mu(_m)) ) J(mu(_m))^T + H_tv(_m) + J(slow(_m)) * ( H_se(slow(_m)) ) J(slow(_m))^T
OptHandler::Hessian::Hessian(const OptHandler& opt,const Vector& m) : _opt(opt)
{
   height = width = m.Size();
   _vcH = NULL;
   _emH = NULL;
   _seH = NULL;
   if (_opt._vcobj)
   {
      Vector dummy;
      _opt._vcobj->SetUpHessian_MM(dummy,m,0.);
      _vcH = _opt._vcobj->GetHessianOperator_MM();
   }

   // from opt to em master comm
   PetscSFBcastBegin(_opt._emobjsf,MPIU_SCALAR,m.GetData(),_opt._emin.GetData());
   PetscSFBcastEnd(_opt._emobjsf,MPIU_SCALAR,m.GetData(),_opt._emin.GetData());
   if (_opt._empd)
   {
      // save linearization point
      Vector m_save(_opt._emin);

      // From log to physical
      Vector emin_mapped(_opt._emin.Size());
      _opt._emmap->Map(_opt._emin,emin_mapped);

      // From em master to children
      DataReplicator& drep = _opt._emrpmesh->GetReplicator();
      Vector emin_rep(_opt._emtrans->Width());
      drep.Broadcast("em_in",emin_mapped,emin_rep);

      // From coarse to fine
      Vector emin_fine(_opt._emtrans->Height());
      _opt._emtrans->Mult(emin_rep,emin_fine);

      _emH = &(_opt._emobj->GetHessian(emin_fine));

      if (_opt._emmap->SecondOrder())
      {
         Vector emin_fine_out(emin_fine.Size());
         _opt._emobj->ComputeGradient(emin_fine,emin_fine_out);

         // From fine to coarse
         _opt._emtrans->MultTranspose(emin_fine_out,emin_rep);

         // From em children to master
         emin_mapped = 0.0;
         drep.Reduce("em_in",emin_rep,emin_mapped);

         // call SetUp to apply the Hessian of the map later
         _opt._emmap->SetUpHessianMap(m_save,emin_mapped);
      }
   }

   // from opt to se master
   PetscSFBcastBegin(_opt._seobjsf,MPIU_SCALAR,m.GetData() + m.Size()/2,_opt._sein.GetData());
   PetscSFBcastEnd(_opt._seobjsf,MPIU_SCALAR,m.GetData() + m.Size()/2,_opt._sein.GetData());
   if (_opt._seobj)
   {
      // From coarse to fine
      Vector sein_fine(_opt._setrans ? _opt._setrans->Height() : 0);
      if (_opt._setrans) _opt._setrans->Mult(_opt._sein,sein_fine);

      _seH = &(_opt._seobj->GetHessian(sein_fine));
   }
}

void OptHandler::Hessian::Mult(const Vector& x,Vector& y) const
{
   Vector ytv(y.Size());
   Vector yem(y.Size());
   Vector yse(y.Size());

   ytv = 0.0;
   if (_vcH)
   {
      _vcH->Mult(x,ytv);
   }
   ytv *= _opt._vcw;

   // from opt to em master comm
   PetscSFBcastBegin(_opt._emobjsf,MPIU_SCALAR,x.GetData(),_opt._emin.GetData());
   PetscSFBcastEnd(_opt._emobjsf,MPIU_SCALAR,x.GetData(),_opt._emin.GetData());
   if (_emH)
   {
      // save direction
      Vector x_save(_opt._emin);

      // From log to physical (apply J(mu(m)))
      const Vector& m_save = _opt._emmap->GetParameter();
      Vector emin_mapped(_opt._emin.Size());
      _opt._emmap->GradientMap(m_save,_opt._emin,false,emin_mapped);

      // From master to children
      DataReplicator& drep = _opt._emrpmesh->GetReplicator();
      Vector emin_rep(_opt._emtrans->Width());
      drep.Broadcast("em_in",emin_mapped,emin_rep);

      // From coarse to fine
      Vector emin_fine(_opt._emtrans->Height());
      _opt._emtrans->Mult(emin_rep,emin_fine);

      // apply EM Hessian
      Vector emin_fine_out(emin_fine.Size());
      _emH->Mult(emin_fine,emin_fine_out);

      // From fine to coarse
      _opt._emtrans->MultTranspose(emin_fine_out,emin_rep);

      // From children to master
      emin_mapped = 0.0;
      drep.Reduce("em_in",emin_rep,emin_mapped);

      // From physical to log (apply J^T(mu(m)))
      _opt._emmap->GradientMap(m_save,emin_mapped,true,_opt._emin);

      // Sum action of Hessian of the map
      if (_opt._emmap->SecondOrder())
      {
         _opt._emmap->HessianMult(x_save,emin_mapped);
         _opt._emin += emin_mapped;
      }
   }

   // from em master to opt
   yem = 0.0;
   PetscSFReduceBegin(_opt._emobjsf,MPIU_SCALAR,_opt._emin.GetData(),yem.GetData(),MPIU_REPLACE);
   PetscSFReduceEnd(_opt._emobjsf,MPIU_SCALAR,_opt._emin.GetData(),yem.GetData(),MPIU_REPLACE);
   yem *= _opt._emw;

   // from opt to em master comm
   PetscSFBcastBegin(_opt._seobjsf,MPIU_SCALAR,x.GetData() + x.Size()/2,_opt._sein.GetData());
   PetscSFBcastEnd(_opt._seobjsf,MPIU_SCALAR,x.GetData() + x.Size()/2,_opt._sein.GetData());
   if (_seH)
   {
      // From coarse to fine
      Vector sein_fine(_opt._setrans ? _opt._setrans->Height() : 0);
      if (_opt._setrans) _opt._setrans->Mult(_opt._sein,sein_fine);

      // apply SEISMIC Hessian
      Vector sein_fine_out(sein_fine.Size());
      _seH->Mult(sein_fine,sein_fine_out);

      // From fine to coarse
      if (_opt._setrans) _opt._setrans->MultTranspose(sein_fine_out,_opt._sein);
   }

   // from seismic master to opt
   yse = 0.0;
   PetscSFReduceBegin(_opt._seobjsf,MPIU_SCALAR,_opt._sein.GetData(),yse.GetData() + x.Size()/2,MPIU_REPLACE);
   PetscSFReduceEnd(_opt._seobjsf,MPIU_SCALAR,_opt._sein.GetData(),yse.GetData() + x.Size()/2,MPIU_REPLACE);
   yse *= _opt._sew;

   y = ytv;
   y+= yem;
   y+= yse;
}

void OptHandler::Hessian::MultTranspose(const Vector& x,Vector& y) const
{
   // Not always true?
   OptHandler::Hessian::Mult(x,y);
}

/* the main routine */
int main(int argc, char* argv[])
{
   MFEMInitializePetsc(&argc,&argv,NULL,help);

   /* process options */
   PetscErrorCode ierr;
   PetscInt       rx = 2,ry = 3,rz = 1, i;
   PetscInt       nex = 4,ney = 2,nez = 3;
   PetscReal      sx0 = -10., sy0 = -2., sz0 = -3.;
   PetscReal      h = 1.e-2;
   PetscBool      flg;
   PetscReal      optbb[3],*ril = NULL, *riu = NULL;

   PetscInt       tv_size = 1;
   PetscScalar    tv_w = 1.0;
   PetscScalar    tv_beta = 1.e-3;
   PetscBool      tv_pd = PETSC_FALSE, tv_uncoupled = PETSC_FALSE;

   PetscInt       em_size = 1;
   PetscScalar    em_w = 1.0;
   PetscInt       em_nref = 0, em_nrep = 1;

   PetscInt       se_size = 1;
   PetscScalar    se_w = 1.0;
   PetscInt       se_nref = 0;

   ierr = PetscOptionsGetReal(NULL,NULL,"-final_h",&h,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-rx",&rx,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-ry",&ry,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-rz",&rz,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-sx0",&sx0,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-sy0",&sy0,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-sz0",&sz0,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-nex",&nex,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-ney",&ney,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-nez",&nez,NULL);CHKERRQ(ierr);

   ierr = PetscOptionsGetInt(NULL,NULL,"-tv_size",&tv_size,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetScalar(NULL,NULL,"-tv_w",&tv_w,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetScalar(NULL,NULL,"-tv_beta",&tv_beta,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-tv_pd",&tv_pd,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetBool(NULL,NULL,"-tv_uncoupled",&tv_uncoupled,NULL);CHKERRQ(ierr);

   ierr = PetscOptionsGetInt(NULL,NULL,"-em_size",&em_size,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-em_nref",&em_nref,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-em_nrep",&em_nrep,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-em_w",&em_w,NULL);CHKERRQ(ierr);

   ierr = PetscOptionsGetInt(NULL,NULL,"-se_size",&se_size,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetReal(NULL,NULL,"-se_w",&se_w,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-se_nref",&se_nref,NULL);CHKERRQ(ierr);

   ierr = PetscOptionsGetRealArray(NULL,NULL,"-opt_bb_ll",optbb,(i=3,&i),&flg);CHKERRQ(ierr);
   if (flg)
   {
      ierr = PetscMalloc1(3,&ril);CHKERRQ(ierr);
      for (PetscInt j=0;j<i;j++) ril[j] = optbb[j];
      for (PetscInt j=i;j<3;j++) ril[j] = PETSC_MIN_REAL;
   }
   ierr = PetscOptionsGetRealArray(NULL,NULL,"-opt_bb_ur",optbb,(i=3,&i),&flg);CHKERRQ(ierr);
   if (flg)
   {
      ierr = PetscMalloc1(3,&riu);CHKERRQ(ierr);
      for (PetscInt j=0;j<i;j++) riu[j] = optbb[j];
      for (PetscInt j=i;j<3;j++) riu[j] = PETSC_MAX_REAL;
   }

   PizzaBoxMesh pbmesh(rx,ry,rz,sx0,sy0,sz0,nex,ney,nez,h,ril,riu);
   ierr = PetscFree(ril);CHKERRQ(ierr);
   ierr = PetscFree(riu);CHKERRQ(ierr);

   VectorArrayCoefficient *optcoeff = new VectorArrayCoefficient(2);
   optcoeff->Set(0,new FunctionCoefficient(sigma_init));
   optcoeff->Set(1,new FunctionCoefficient(slow_init));

   OptHandler *opt = new OptHandler(PETSC_COMM_WORLD,&pbmesh);

   MPI_Group master_group, em_group, se_group;

   {
      int size;
      MPI_Comm_size(PETSC_COMM_WORLD,&size);
      if (tv_size > size) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"TV comm size %D not allowed to be larger than world size %d\n",tv_size,size);
      if (em_size > size) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"EM comm size %D not allowed to be larger than world size %d\n",em_size,size);
      if (se_size > size) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Seismic comm size %D not allowed to be larger than world size %d\n",se_size,size);

      int ranges[1][3],tv_start,em_start,se_start;
      MPI_Group world_group;
      MPI_Comm_group(PETSC_COMM_WORLD,&world_group);
      if (tv_size + em_size + se_size <= size) /* overlap all */
      {
         tv_start = 0;
         em_start = tv_start + tv_size;
         se_start = em_start + em_size;
      }
      else if (em_size + se_size <= size) /* overlap em and seisimic */
      {
         tv_start = 0;
         em_start = size - em_size;
         se_start = size - em_size - se_size;
      }
      else /* no overlap, testing */
      {
         tv_start = size - tv_size;
         em_start = size - em_size;
         se_start = size - se_size;
      }

      /* TV */
      ranges[0][0] = tv_start;
      ranges[0][1] = tv_start + tv_size-1;
      ranges[0][2] = 1;
      PetscPrintf(PETSC_COMM_WORLD,"Running TV on processes [%d,%d]\n",ranges[0][0],ranges[0][1]);
      MPI_Group_range_incl(world_group,1,ranges,&master_group);

      /* EM */
      ranges[0][0] = em_start;
      ranges[0][1] = em_start + em_size-1;
      ranges[0][2] = 1;
      PetscPrintf(PETSC_COMM_WORLD,"Running EM on processes [%d,%d]\n",ranges[0][0],ranges[0][1]);
      MPI_Group_range_incl(world_group,1,ranges,&em_group);

      /* SEISMIC */
      ranges[0][0] = se_start;
      ranges[0][1] = se_start + se_size-1;
      ranges[0][2] = 1;
      PetscPrintf(PETSC_COMM_WORLD,"Running SEISMIC on processes [%d,%d]\n",ranges[0][0],ranges[0][1]);
      MPI_Group_range_incl(world_group,1,ranges,&se_group);
      MPI_Group_free(&world_group);
   }

   opt->SetUpCoefficient(master_group,optcoeff,tv_beta,tv_pd,tv_uncoupled);
   opt->SetUpEM(em_group,em_nrep,em_nref);
   opt->SetUpSeismic(se_group,se_nref);
   MPI_Group_free(&master_group);
   MPI_Group_free(&em_group);
   MPI_Group_free(&se_group);

   opt->SetWeights(tv_w,em_w,se_w);

   /* tests */
   {
      double obj;
      Vector dummy(opt->Height());
      PetscParVector m(PETSC_COMM_WORLD,dummy,false);
      PetscParVector g(PETSC_COMM_WORLD,dummy,false);
      m.Randomize();
      opt->ComputeObjective(m,&obj);
      opt->ComputeGradient(m,g);
      opt->TestTaylor(PETSC_COMM_WORLD,m,true);
      opt->TestFDGradient(PETSC_COMM_WORLD,m,1.e-8,true);
      opt->TestFDHessian(PETSC_COMM_WORLD,m);
   }

   delete opt;
   delete optcoeff;

   MFEMFinalizePetsc();
   return 0;
}

/*TEST

  build:
    requires: mfemopt

TEST*/

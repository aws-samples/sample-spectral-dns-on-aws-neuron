/* Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0 */
/* The collectives bootstrap address is taken from NEURON_RT_ROOT_COMM_ID when the caller sets it. */
/* tgv_fused.c - fused-stage driver: ONE NEFF execute (two collectives) per RK3 stage (kernels/tgv_fused_wide_nki.py).
 * R-rank device-resident Taylor-Green Re=1600 run through libnrt collectives, rank r pinned to NeuronCore r.
 * The parent forks every rank and supervises: a rank that exits non-zero, or no step 20 within 240 s of the start,
 * kills every remaining rank. Ranks barrier once after the initial transform so the first stage starts in step.
 * Each rank loads dist_fwd3 and tgv_fused, holds its state/accumulators in two ping-pong sets of 12 blocks, and
 * every <every> steps reads back its 6 spectral state blocks and writes partial sums to <prefix>_r<r>.csv:
 *   step,t,E_part,Omega_part,step_wall_ms   with E = 0.5 sum |u_hat|^2 / N^6, Omega = 0.5 sum K2 |u_hat|^2 / N^6
 * Usage: tgv_fused <N> <R> <steps> <dt> <nu> <every> <prefix> [ckpt_every]
 * With ckpt_every > 0, at step 0 and every ckpt_every steps each rank dumps its six (N, C2) fp32 state blocks to
 * <prefix>_ckpt_<t:.4f>_r<rank>.bin. */
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
static int rank;
static int arg_int(const char* s,int lo,int hi){ char* e; errno=0; long v=strtol(s,&e,10); if(errno||*e||e==s||v<lo||v>hi){ fprintf(stderr,"bad integer argument: %s (want %d..%d)\n",s,lo,hi); exit(2);} return (int)v; }
static void* xmalloc(size_t n){ void* p=malloc(n); if(!p){ perror("malloc"); exit(2);} return p; }
static FILE* xfopen(const char* p,const char* mode){ FILE* f=fopen(p,mode); if(!f){ perror(p); exit(2);} return f; }
static double arg_double(const char* s){ char* e; errno=0; double v=strtod(s,&e); if(errno||*e||e==s){ fprintf(stderr,"bad number argument: %s\n",s); exit(2);} return v; }

typedef struct { volatile int progress; volatile int ready[64]; } shm_t;
static shm_t* shm;
#define CK(r,msg) do{ if((r)!=NRT_SUCCESS){ fprintf(stderr,"rank%d %s (%d)\n",rank,msg,(int)(r)); exit(2);} }while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
/* Map a whole input file read-only; want > 0 requires exactly that many bytes, want == 0 any non-empty size. */
static void* map_file(const char*p,size_t*sz,size_t want){struct stat sb;int fd=open(p,O_RDONLY);if(fd<0||fstat(fd,&sb)!=0){perror(p);exit(2);}
  if(sb.st_size<=0||(want&&(size_t)sb.st_size!=want)){fprintf(stderr,"%s: %lld bytes, expected %zu\n",p,(long long)sb.st_size,want);exit(2);}
  *sz=sb.st_size;void*m=mmap(NULL,sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);if(m==MAP_FAILED){perror(p);exit(2);}return m;}
static nrt_tensor_t* dev(size_t b,const char*n){nrt_tensor_t*t;CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE,0,b,n,&t),"alloc");return t;}
static nrt_model_t* load(const char*name,int N,int R){char p[128];snprintf(p,sizeof p,"%s_%d_%d.neff",name,N,R);size_t sz;void*b=map_file(p,&sz,0);nrt_model_t*m;CK(nrt_load_collectives(b,sz,0,1,(uint32_t)rank,(uint32_t)R,&m),p);munmap(b,sz);return m;}
static nrt_tensor_set_t* tset(void){nrt_tensor_set_t*s;CK(nrt_allocate_tensor_set(&s),"tset");return s;}
static void add(nrt_tensor_set_t*s,const char*n,nrt_tensor_t*t){CK(nrt_add_tensor_to_tensor_set(s,n,t),n);}
static void addout(nrt_tensor_set_t*s,int i,nrt_tensor_t*t){char n[16];snprintf(n,sizeof n,"output_%d",i);add(s,n,t);}
static nrt_tensor_t* filled(size_t b,const char*n,const float* src){nrt_tensor_t*t=dev(b,n);CK(nrt_tensor_write(t,(void*)src,0,b),n);return t;}
static const char* STATE[12]={"uh_re","uh_im","vh_re","vh_im","wh_re","wh_im","du_re","du_im","dv_re","dv_im","dw_re","dw_im"};
static const char* MAT[3]={"C","S","nS"};
static const char* RC[6]={"KX","KY","KZ","IK2","DEAL","DECAY"};
static const char* RK[3]={"RKA","RKB","RKDT"};
static const double RK_A[3]={1.0/3.0,15.0/16.0,8.0/15.0}, RK_B[3]={0.0,-5.0/9.0,-153.0/128.0};
static double wn(int i,int N){return i<N/2?i:i-N;}
#define KB (128*512)

static int ckpt_every=0;
static int run_rank(int N,int R,int steps,double dt,double nu,int every,const char*prefix){
  char vc[8]; snprintf(vc,sizeof vc,"%d",rank); setenv("NEURON_RT_VISIBLE_CORES",vc,1); setenv("NEURON_RT_ROOT_COMM_ID","127.0.0.1:46820",0);   /* the caller may pick the port: a killed run leaves its port in TIME_WAIT for 60 s */
  size_t L=N/R, C2=(size_t)N*N/R, NC=(size_t)N*C2, spec=NC*4, slab=L*(size_t)N*N*4, mat=(size_t)N*N*4; char path[256];
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW,"",""),"nrt_init");
  nrt_model_t *mf=load("dist_fwd3",N,R), *ms=load("tgv_fused",N,R);
  /* constants: DFT matrices from files, rank-layout wavenumber arrays computed here, RK constants as (128, 512) tiles */
  nrt_tensor_t *tm[3], *rc[6], *ones, *rk[3][3];
  for(int i=0;i<3;i++){ snprintf(path,sizeof path,"dist_%s_%d.bin",MAT[i],N); size_t sz; void*b=map_file(path,&sz,mat); tm[i]=dev(mat,MAT[i]); CK(nrt_tensor_write(tm[i],b,0,mat),"write mat"); munmap(b,sz); }
  float* cb=xmalloc(spec*6); double* k2=xmalloc(NC*sizeof(double));
  for(size_t i=0;i<(size_t)N;i++) for(size_t n=0;n<C2;n++){ size_t g=rank*C2+n, j=i*C2+n; double kx=wn((int)i,N), ky=wn((int)(g/N),N), kz=wn((int)(g%N),N); double K2=kx*kx+ky*ky+kz*kz; int zero=(i==0&&g==0); if(zero) K2=1.0;
    cb[0*NC+j]=(float)kx; cb[1*NC+j]=(float)ky; cb[2*NC+j]=(float)kz; cb[3*NC+j]=1.0f/(float)K2; cb[4*NC+j]=(fabs(kx)<N/3.0&&fabs(ky)<N/3.0&&fabs(kz)<N/3.0)?1.0f:0.0f; cb[5*NC+j]=zero?1.0f:(float)exp(-nu*K2*dt); k2[j]=K2; }
  for(int i=0;i<6;i++) rc[i]=filled(spec,RC[i],cb+i*NC);
  for(size_t j=0;j<NC;j++) cb[j]=1.0f;
  ones=filled(spec,"ONES",cb);
  float* kb=xmalloc(KB*4);
  for(int s=0;s<3;s++){ double v[3]={RK_A[s],RK_B[s],dt}; for(int c=0;c<3;c++){ for(int j=0;j<KB;j++) kb[j]=(float)v[c]; rk[s][c]=filled(KB*4,RK[c],kb); } }
  /* device tensors: 2 x 12 state/accumulator blocks, 3 physical slabs for the initial field, the guard output */
  nrt_tensor_t *st[2][12], *ph[3], *guard=dev(KB*4,"guard");
  for(int k=0;k<2;k++) for(int i=0;i<12;i++){ char nm[32]; snprintf(nm,sizeof nm,"%s_%d",STATE[i],k); st[k][i]=dev(spec,nm); }
  for(int i=0;i<3;i++) ph[i]=dev(slab,"a");
  /* initial field: u = sin x cos y cos z, v = -cos x sin y cos z, w = 0 on the rank's x-slab; accumulators zero */
  float* fb=xmalloc(slab); double tp=2*M_PI/N;
  for(int c=0;c<3;c++){ for(size_t p=0;p<L;p++){ double x=tp*(rank*L+p); for(int j=0;j<N;j++){ double y=tp*j; for(int k=0;k<N;k++){ double z=tp*k; double val=c==0? sin(x)*cos(y)*cos(z) : c==1? -cos(x)*sin(y)*cos(z) : 0.0; fb[(p*N+j)*N+k]=(float)val; } } } CK(nrt_tensor_write(ph[c],fb,0,slab),"write init"); }
  float* zeros=calloc(NC,sizeof(float)); if(!zeros){ perror("calloc"); exit(2);} for(int k=0;k<2;k++) for(int i=0;i<12;i++) CK(nrt_tensor_write(st[k][i],zeros,0,spec),"zero"); free(zeros);
  /* tensor sets: init (a0..a2 + mats -> 6 state), stage s from set k (12 state + mats + rc + stage constants -> set 1-k + guard) */
  static const char* A3[3]={"a0","a1","a2"};
  nrt_tensor_set_t *ini=tset(),*ino=tset(), *si[2][3],*so[2];
  for(int c=0;c<3;c++) add(ini,A3[c],ph[c]);
  for(int i=0;i<3;i++) add(ini,MAT[i],tm[i]);
  for(int i=0;i<6;i++) addout(ino,i,st[0][i]);
  for(int k=0;k<2;k++){ for(int s=0;s<3;s++){ si[k][s]=tset(); for(int i=0;i<12;i++) add(si[k][s],STATE[i],st[k][i]); for(int i=0;i<3;i++) add(si[k][s],MAT[i],tm[i]);
      for(int i=0;i<5;i++) add(si[k][s],RC[i],rc[i]);
      add(si[k][s],RC[5],s==2?rc[5]:ones);
      for(int c=0;c<3;c++) add(si[k][s],RK[c],rk[s][c]); }
    so[k]=tset(); for(int i=0;i<12;i++) addout(so[k],i,st[1-k][i]); addout(so[k],12,guard); }
  CK(nrt_execute(mf,ini,ino),"init fwd3");
  snprintf(path,sizeof path,"%s_r%d.csv",prefix,rank); FILE* f=xfopen(path,"w"); fprintf(f,"step,t,E_part,Omega_part,step_wall_ms\n");
  double N6=(double)N*N*N*(double)N*N*N, t=0; int cur=0;
#define STATS(stepno,ms) do{ double E=0,Om=0; for(int i=0;i<6;i++) CK(nrt_tensor_read(st[cur][i],cb+i*NC,0,spec),"read"); \
    for(size_t j=0;j<NC;j++){ double e=0; for(int i=0;i<6;i++) e+=(double)cb[i*NC+j]*cb[i*NC+j]; E+=e; Om+=e*k2[j]; } E*=0.5/N6; Om*=0.5/N6; \
    fprintf(f,"%d,%.6f,%.12g,%.12g,%.3f\n",stepno,t,E,Om,ms); if(rank==0&&((stepno)%20==0)){ fflush(f); fprintf(stderr,"step %d t=%.3f E_r0=%.6g step_ms=%.2f\n",stepno,t,E,ms);} }while(0)
#define DUMP(tt) do{ char cp[300]; snprintf(cp,sizeof cp,"%s_ckpt_%.4f_r%d.bin",prefix,(tt),rank); FILE* cf=xfopen(cp,"wb"); \
    for(int i=0;i<6;i++){ CK(nrt_tensor_read(st[cur][i],cb,0,spec),"read ckpt"); fwrite(cb,1,spec,cf); } fclose(cf); }while(0)
  STATS(0,0.0); if(ckpt_every>0) DUMP(0.0);
  /* barrier: every rank has finished the initial transform before the first stage's collective can start */
  shm->ready[rank]=1; for(;;){ int n=0; for(int r=0;r<R;r++) n+=shm->ready[r]; if(n>=R) break; usleep(200); }
  double t_all=now();
  for(int step=1;step<=steps;step++){
    double t0=now();
    for(int s=0;s<3;s++){ CK(nrt_execute(ms,si[cur][s],so[cur]),"stage"); cur=1-cur; }
    double ms_=(now()-t0)*1e3; t+=dt;
    if(step%every==0||step==steps) STATS(step,ms_);
    if(ckpt_every>0&&step%ckpt_every==0) DUMP(step*dt);
    if(rank==0) shm->progress=step;
  }
  double wall=now()-t_all; fclose(f);
  if(rank==0) printf("TGVFUSED N=%d R=%d steps=%d dt=%.6f t_end=%.4f wall_s=%.2f ms_per_step_incl_stats=%.2f\n",N,R,steps,dt,t,wall,1e3*wall/steps);
  fflush(stdout); nrt_close(); return 0;   /* ranks leave through _exit, which does not flush stdio */
}
int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: tgv_fused <N> <R> <steps> <dt> <nu> <every> <prefix> [ckpt_every]\n");return 2;}
  int N=arg_int(argv[1],1,512),R=arg_int(argv[2],1,64),steps=arg_int(argv[3],1,1000000); double dt=arg_double(argv[4]),nu=arg_double(argv[5]); int every=arg_int(argv[6],1,1000000); if(argc>8) ckpt_every=arg_int(argv[8],0,1000000);
  if(R<1||R>64||N<R||N%R||(N*N)%R||N>512||steps<1||every<1||dt<=0||nu<=0){fprintf(stderr,"need 1 <= R <= 64 dividing N and N*N, N <= 512, steps >= 1, every >= 1, dt > 0, nu > 0\n");return 2;}
  shm=mmap(NULL,sizeof(shm_t),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0); if(shm==MAP_FAILED){ perror("mmap"); return 2; }   /* anonymous mappings are zero-filled */
  pid_t pids[64]; int done[64]={0}, left=R, rc=0; const char* why="";
  for(int r=0;r<R;r++){ pid_t p=fork(); if(p==0){ rank=r; _exit(run_rank(N,R,steps,dt,nu,every,argv[7])); } pids[r]=p; }
  double t0=now();
  while(left>0){
    usleep(100000);
    for(int r=0;r<R;r++){ if(done[r]) continue; int s; pid_t w=waitpid(pids[r],&s,WNOHANG); if(w==pids[r]){ done[r]=1; left--; if(!WIFEXITED(s)||WEXITSTATUS(s)){ rc=1; why="a rank failed"; } } }
    if(!rc&&shm->progress<20&&now()-t0>240.0){ rc=1; why="watchdog: no step 20 within 240 s"; }
    if(rc) break;
  }
  if(rc){ fprintf(stderr,"TGVFUSED FAIL: %s; killing %d live rank(s)\n",why,left); for(int r=0;r<R;r++) if(!done[r]) kill(pids[r],SIGKILL); for(int r=0;r<R;r++) if(!done[r]){ int s; waitpid(pids[r],&s,0); } }
  printf("TGVFUSED %s\n",rc==0?"ALL RANKS OK":"FAIL"); return rc;
}

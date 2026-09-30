/* Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0 */
/* tgv_run.c - device-resident Taylor-Green run through libnrt (Re=1600 validation).
 * Loads the three RK3 stage NEFFs, keeps the 12 state/accumulator tensors on the device
 * (ping-pong between two sets), executes stage0->stage1->stage2 per step, and reads back the
 * six spectral state tensors each step to compute on the host:
 *   E(t) = 0.5 sum |u_hat|^2 / N^6,  Omega(t) = 0.5 sum K2 |u_hat|^2 / N^6,  eps = 2 nu Omega
 * Writes a CSV: step,t,E,Omega,eps_enstrophy,eps_dEdt,step_wall_ms.
 *   tgv_run <N> <steps> <dt> <nu> <out.csv>   (needs tgv_stage{0,1,2}_<N>.neff and stage_*_<N>.bin) */
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
static int arg_int(const char* s,int lo,int hi){ char* e; errno=0; long v=strtol(s,&e,10); if(errno||*e||e==s||v<lo||v>hi){ fprintf(stderr,"bad integer argument: %s (want %d..%d)\n",s,lo,hi); exit(2);} return (int)v; }
static void* xmalloc(size_t n){ void* p=malloc(n); if(!p){ perror("malloc"); exit(2);} return p; }
static FILE* xfopen(const char* p,const char* mode){ FILE* f=fopen(p,mode); if(!f){ perror(p); exit(2);} return f; }
static double arg_double(const char* s){ char* e; errno=0; double v=strtod(s,&e); if(errno||*e||e==s){ fprintf(stderr,"bad number argument: %s\n",s); exit(2);} return v; }
#define CK(r,msg) do{ if((r)!=NRT_SUCCESS){ fprintf(stderr,"%s (%d)\n",msg,(int)(r)); exit(2);} }while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
/* Map a whole input file read-only; want > 0 requires exactly that many bytes, want == 0 any non-empty size. */
static void* map_file(const char*p,size_t*sz,size_t want){struct stat sb;int fd=open(p,O_RDONLY);if(fd<0||fstat(fd,&sb)!=0){perror(p);exit(2);}
  if(sb.st_size<=0||(want&&(size_t)sb.st_size!=want)){fprintf(stderr,"%s: %lld bytes, expected %zu\n",p,(long long)sb.st_size,want);exit(2);}
  *sz=sb.st_size;void*m=mmap(NULL,sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);if(m==MAP_FAILED){perror(p);exit(2);}return m;}
static const char* STATE[12]={"uh_re","uh_im","vh_re","vh_im","wh_re","wh_im","du_re","du_im","dv_re","dv_im","dw_re","dw_im"};
static const char* CONST[9]={"C","S","nS","KX","KY","KZ","IK2","DEAL","DECAY"};

int main(int argc,char**argv){
  if(argc<6){fprintf(stderr,"usage: tgv_run <N> <steps> <dt> <nu> <out.csv>\n");return 2;}
  int N=arg_int(argv[1],32,128), steps=arg_int(argv[2],1,1000000); double dt=arg_double(argv[3]), nu=arg_double(argv[4]); const char* csv=argv[5];
  if(N<32||N>128||N%32||steps<1||dt<=0||nu<=0){fprintf(stderr,"N in 32..128 (multiple of 32), steps >= 1, dt > 0, nu > 0\n");return 2;}
  size_t NN3=(size_t)N*N*N, fb=NN3*4; char path[256];
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW,"",""),"nrt_init");
  nrt_model_t* m[3];
  for(int s=0;s<3;s++){ snprintf(path,sizeof path,"tgv_stage%d_%d.neff",s,N); size_t sz; void* b=map_file(path,&sz,0); CK(nrt_load(b,sz,-1,-1,&m[s]),"nrt_load"); munmap(b,sz); }
  /* tensors: two sets of 12 state tensors, plus 9 constants */
  nrt_tensor_t *st[2][12], *cst[9]; nrt_tensor_set_t *in[2], *out[2];
  for(int k=0;k<2;k++) for(int i=0;i<12;i++){ char nm[32]; snprintf(nm,sizeof nm,"%s_%d",STATE[i],k); CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE,0,fb,nm,&st[k][i]),"alloc state"); }
  for(int i=0;i<9;i++){ snprintf(path,sizeof path,"stage_%s_%d.bin",CONST[i],N); size_t sz; void* b=map_file(path,&sz,0); CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE,0,sz,CONST[i],&cst[i]),"alloc const"); CK(nrt_tensor_write(cst[i],b,0,sz),"write const"); munmap(b,sz); }
  /* initial state into set 0 */
  for(int i=0;i<12;i++){ snprintf(path,sizeof path,"stage_%s_%d.bin",STATE[i],N); size_t sz; void* b=map_file(path,&sz,fb); CK(nrt_tensor_write(st[0][i],b,0,fb),"write state"); munmap(b,sz); }
  /* tensor sets: in[k] reads set k, out[k] writes set 1-k (outputs output_0..11 in STATE order) */
  for(int k=0;k<2;k++){ CK(nrt_allocate_tensor_set(&in[k]),"tset"); CK(nrt_allocate_tensor_set(&out[k]),"tset");
    for(int i=0;i<12;i++){ CK(nrt_add_tensor_to_tensor_set(in[k],STATE[i],st[k][i]),"add in"); char on[16]; snprintf(on,sizeof on,"output_%d",i); CK(nrt_add_tensor_to_tensor_set(out[k],on,st[1-k][i]),"add out"); }
    for(int i=0;i<9;i++) CK(nrt_add_tensor_to_tensor_set(in[k],CONST[i],cst[i]),"add const"); }
  /* K2 for enstrophy from IK2 (K2 = 1/IK2; DC mode has K2 set to 1 in the oracle, its energy is 0) */
  float* ik2=xmalloc(fb); { snprintf(path,sizeof path,"stage_IK2_%d.bin",N); size_t sz; void* b=map_file(path,&sz,fb); memcpy(ik2,b,fb); munmap(b,sz); }
  float* buf=xmalloc(fb*6);
  FILE* f=xfopen(csv,"w"); fprintf(f,"step,t,E,Omega,eps_enstrophy,eps_dEdt,step_wall_ms\n");
  double N6=(double)NN3*(double)NN3, Eprev=0, t=0; int cur=0;
  /* energy at t=0 */
  { double E=0,Om=0; for(int i=0;i<6;i++) CK(nrt_tensor_read(st[cur][i],buf+i*NN3,0,fb),"read"); for(size_t j=0;j<NN3;j++){ double e=0; for(int i=0;i<6;i++) e+=(double)buf[i*NN3+j]*buf[i*NN3+j]; E+=e; Om+=e/(double)ik2[j]; } E*=0.5/N6; Om*=0.5/N6; fprintf(f,"0,0,%.9g,%.9g,%.9g,,\n",E,Om,2*nu*Om); Eprev=E; }
  double t_all=now();
  for(int step=1;step<=steps;step++){
    double t0=now();
    for(int s=0;s<3;s++){ CK(nrt_execute(m[s],in[cur],out[cur]),"execute"); cur=1-cur; }
    double t1=now(); t+=dt;
    double E=0,Om=0; for(int i=0;i<6;i++) CK(nrt_tensor_read(st[cur][i],buf+i*NN3,0,fb),"read");
    for(size_t j=0;j<NN3;j++){ double e=0; for(int i=0;i<6;i++) e+=(double)buf[i*NN3+j]*buf[i*NN3+j]; E+=e; Om+=e/(double)ik2[j]; }
    E*=0.5/N6; Om*=0.5/N6;
    fprintf(f,"%d,%.6f,%.9g,%.9g,%.9g,%.9g,%.3f\n",step,t,E,Om,2*nu*Om,-(E-Eprev)/dt,(t1-t0)*1e3); Eprev=E;
    if(step%20==0){ fflush(f); fprintf(stderr,"step %d t=%.3f E=%.6f eps=%.6f step_ms=%.2f\n",step,t,E,2*nu*Om,(t1-t0)*1e3); }
  }
  double wall=now()-t_all; fclose(f);
  printf("TGVRUN N=%d steps=%d dt=%.5f t_end=%.3f device_steps_wall_s=%.3f ms_per_step_incl_readback=%.3f\n",N,steps,dt,t,wall,1e3*wall/steps);
  for(int s=0;s<3;s++) nrt_unload(m[s]);
  nrt_close(); return 0;
}

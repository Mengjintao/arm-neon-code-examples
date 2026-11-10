#include "../../src/gemm.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dlfcn.h>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using Fn = void(*)(int,int,int,const float*,int,const float*,int,float*,int,float*,float*);
static double gettime() { timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main() {
    std::ifstream list(".ablation/cases.txt");
    std::vector<std::string> names; std::vector<Fn> fn;
    std::string name,pack;
    while (list>>name>>pack) {
        void* h=dlopen((".ablation/"+name+".so").c_str(),RTLD_NOW|RTLD_LOCAL);
        if (!h) { puts(dlerror()); return 1; }
        names.push_back(name); fn.push_back((Fn)dlsym(h,"run"));
        if (!fn.back()) return 1;
    }
    auto alloc=[](size_t n) { return (float*)aligned_alloc(64,(n*4+63)/64*64); };
    const int N=1024;
    float *a=alloc(N*N), *b=alloc(N*N), *c=alloc(N*N), *pa=alloc(MC*KC), *pb=alloc(KC*N);
    srand(1);
    for(int i=0;i<N*N;++i) { a[i]=(float)rand()/RAND_MAX-0.5f; b[i]=(float)rand()/RAND_MAX-0.5f; }
    // Both full size and small remainder cases use the same double reference.
    for(auto dims : {std::vector<int>{1024,1024,1024}, {91,53,257}, {5,17,1}, {5,17,2}, {5,17,3}, {5,17,5}}) {
        int m=dims[0],n=dims[1],k=dims[2];
        std::vector<double> ref((size_t)m*n,0);
        for(int i=0;i<m;++i) for(int p=0;p<k;++p) for(int j=0;j<n;++j) ref[i*n+j]+=(double)a[i*k+p]*b[p*n+j];
        for(size_t v=0;v<fn.size();++v) {
            std::fill_n(c,(size_t)m*n,NAN);
            fn[v](m,n,k,a,k,b,n,c,n,pa,pb);
            for(int i=0;i<m*n;++i) if (!(std::abs(c[i]-ref[i])<=1e-4*(1+std::abs(ref[i])))) { printf("FAIL %s %d %d %d idx=%d\n",names[v].c_str(),m,n,k,i); return 2; }
        }
        printf("CHECK PASS all variants %d %d %d\n",m,n,k); fflush(stdout);
    }
    std::vector<int> order; for(size_t v=0;v<fn.size();++v) order.push_back(v);
    for(int round=1;round<=3;++round) {
        std::mt19937 rng(round); std::vector<std::vector<double>> times(fn.size());
        for(int sample=0;sample<15;++sample) {
            std::shuffle(order.begin(),order.end(),rng);
            for(int v:order) {
                for(int w=0;w<2;++w) fn[v](N,N,N,a,N,b,N,c,N,pa,pb);
                double t=gettime(); fn[v](N,N,N,a,N,b,N,c,N,pa,pb); double ms=(gettime()-t)*1000;
                times[v].push_back(ms);
                printf("SAMPLE round=%d sample=%d variant=%s ms=%.6f\n",round,sample+1,names[v].c_str(),ms);
            }
            fflush(stdout);
        }
        for(size_t v=0;v<fn.size();++v) {
            auto t=times[v]; std::sort(t.begin(),t.end());
            printf("RESULT round=%d variant=%s median_ms=%.6f GFLOPS=%.6f min_ms=%.6f max_ms=%.6f\n",round,names[v].c_str(),t[7],2.0*N*N*N/(t[7]*1e6),t.front(),t.back());
        }
        fflush(stdout);
    }
    free(a);free(b);free(c);free(pa);free(pb);
}

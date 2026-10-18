// Lossless NF4 layout permutation for direct mma.m16n8k16 BF16 operands.
// Accumulate K in original sequential groups of 16, with no split reduction.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__global__ void pack_registers(const uint8_t* w,uint32_t* out,int rows,int cols) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i>=rows*cols/8)return;
    int lane=i%32,tile=(i/32)%(cols/16),rowblock=(i/32)/(cols/16);
    int r=rowblock*16+lane/4,k=tile*16+(lane%4)*2;
    int base=(r*cols+k)/2;
    out[i]=(uint32_t)w[base] | ((uint32_t)w[base+4*cols]<<8)
          | ((uint32_t)w[base+4]<<16) | ((uint32_t)w[base+4*cols+4]<<24);
}

__device__ __forceinline__ uint32_t expand_pair(uint32_t byte,const uint16_t* palette) {
    uint32_t pair=(uint32_t)palette[(byte>>4)&15] | ((uint32_t)palette[byte&15]<<16);
    return pair;
}

template<int WARPS,int UNROLL>
__global__ void register_mma(const __nv_bfloat16* x,const uint32_t* w,
    const float* scales,const float* code,__nv_bfloat16* y,int rows,int cols) {
    int lane=threadIdx.x%32;
    int blockrow=blockIdx.x*WARPS+threadIdx.x/32;
    int group=lane/4,l=lane%4,r=blockrow*16+group;
    float d0=0,d1=0,d2=0,d3=0;
    int nt=cols/16;
    for(int tilebase=0;tilebase<nt;tilebase+=UNROLL) {
        uint32_t q[UNROLL],b0[UNROLL],b1[UNROLL];
        const uint16_t* s0[UNROLL]; const uint16_t* s1[UNROLL];
        #pragma unroll
        for(int j=0;j<UNROLL;j++) {
            int tile=tilebase+j,k=tile*16+l*2;
            q[j]=w[(blockrow*nt+tile)*32+lane];
            s0[j]=(const uint16_t*)scales+(r*(cols/64)+tile/4)*16;
            s1[j]=(const uint16_t*)scales+((r+8)*(cols/64)+tile/4)*16;
            b0[j]=group<4?*(const uint32_t*)(x+group*cols+k):0;
            b1[j]=group<4?*(const uint32_t*)(x+group*cols+k+8):0;
        }
        #pragma unroll
        for(int j=0;j<UNROLL;j++) {
            uint32_t a0=expand_pair(q[j],s0[j]),a1=expand_pair(q[j]>>8,s1[j]);
            uint32_t a2=expand_pair(q[j]>>16,s0[j]),a3=expand_pair(q[j]>>24,s1[j]);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0[j]),"r"(b1[j]));
        }
    }
    if(l<2) {
        y[(l*2)*rows+r]=__float2bfloat16_rn(d0);
        y[(l*2+1)*rows+r]=__float2bfloat16_rn(d1);
        y[(l*2)*rows+r+8]=__float2bfloat16_rn(d2);
        y[(l*2+1)*rows+r+8]=__float2bfloat16_rn(d3);
    }
}

extern "C" __declspec(dllexport) int pack_nf4(void* w,void* out,int rows,int cols,void* stream) {
    pack_registers<<<(rows*cols/8+255)/256,256,0,(cudaStream_t)stream>>>((uint8_t*)w,(uint32_t*)out,rows,cols);
    return (int)cudaGetLastError();
}
extern "C" __declspec(dllexport) int launch_register(void* x,void* w,void* scales,void* code,void* y,int rows,int cols,void* stream,int warps,int unroll) {
    #define RUN(W,U) register_mma<W,U><<<rows/(16*W),32*W,0,(cudaStream_t)stream>>>((__nv_bfloat16*)x,(uint32_t*)w,(float*)scales,(float*)code,(__nv_bfloat16*)y,rows,cols)
    #define CHOOSE(W) if(warps==W) { if(unroll==1){RUN(W,1);}else if(unroll==4){RUN(W,4);}else if(unroll==8){RUN(W,8);}else if(unroll==16){RUN(W,16);}else return -1; }
    CHOOSE(1) else CHOOSE(2) else CHOOSE(4) else CHOOSE(8) else return -2;
    return (int)cudaGetLastError();
}

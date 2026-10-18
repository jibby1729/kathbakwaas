// Exact BF16 expansion palettes for NF4 blocks, with sequential WMMA accumulation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>

__global__ void build_palette(const float* scales,const float* code,__nv_bfloat16* palette,int blocks) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<blocks*16) palette[i]=__float2bfloat16_rn(code[i%16]*scales[i/16]);
}

template<int WARPS,int TILEK,bool PALETTE>
__global__ void palette_mma(const __nv_bfloat16* x,const uint8_t* w,
    const __nv_bfloat16* palette,const float* scales,const float* code,
    __nv_bfloat16* y,int rows,int cols) {
    constexpr int ROWS=16*WARPS,STRIDE=TILEK+8,THREADS=32*WARPS;
    __shared__ __align__(32) __nv_bfloat16 a[ROWS*STRIDE];
    __shared__ __align__(32) __nv_bfloat16 b[16*STRIDE];
    __shared__ __align__(32) float out[WARPS*256];
    int warp=threadIdx.x/32,rowbase=blockIdx.x*ROWS;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> af;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> bf;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> cf;
    nvcuda::wmma::fill_fragment(cf,0.f);
    for(int base=0;base<cols;base+=TILEK) {
        for(int i=threadIdx.x;i<ROWS*TILEK/2;i+=THREADS) {
            int r=i/(TILEK/2),k=(i%(TILEK/2))*2;
            int wi=(rowbase+r)*cols+base+k;
            uint8_t packed=w[wi/2];
            if constexpr(PALETTE) {
                a[r*STRIDE+k]=palette[(wi/64)*16+(packed>>4)];
                a[r*STRIDE+k+1]=palette[(wi/64)*16+(packed&15)];
            } else {
                float scale=scales[wi/64];
                a[r*STRIDE+k]=__float2bfloat16_rn(code[packed>>4]*scale);
                a[r*STRIDE+k+1]=__float2bfloat16_rn(code[packed&15]*scale);
            }
        }
        for(int i=threadIdx.x;i<16*TILEK;i+=THREADS) {
            int r=i/TILEK,k=i%TILEK;
            b[r*STRIDE+k]=r<4?x[r*cols+base+k]:__float2bfloat16_rn(0.f);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILEK;k+=16) {
            nvcuda::wmma::load_matrix_sync(af,a+warp*16*STRIDE+k,STRIDE);
            nvcuda::wmma::load_matrix_sync(bf,b+k,STRIDE);
            nvcuda::wmma::mma_sync(cf,af,bf,cf);
        }
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(out+warp*256,cf,16,nvcuda::wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<ROWS*4;i+=THREADS) {
        int row=i/4,beam=i%4;
        y[beam*rows+rowbase+row]=__float2bfloat16_rn(out[(row/16)*256+(row%16)*16+beam]);
    }
}

extern "C" __declspec(dllexport) int make_palette(void* scales,void* code,void* out,int blocks,void* stream) {
    build_palette<<<(blocks*16+255)/256,256,0,(cudaStream_t)stream>>>((float*)scales,(float*)code,(__nv_bfloat16*)out,blocks);
    return (int)cudaGetLastError();
}

template<bool P> int dispatch(void* x,void* w,void* palette,void* scales,void* code,void* y,int rows,int cols,void* stream,int warps,int tile) {
    #define RUN(W,K) palette_mma<W,K,P><<<rows/(16*W),32*W,0,(cudaStream_t)stream>>>((__nv_bfloat16*)x,(uint8_t*)w,(__nv_bfloat16*)palette,(float*)scales,(float*)code,(__nv_bfloat16*)y,rows,cols)
    #define CHOOSE(W) if(warps==W) { if(tile==32){RUN(W,32);}else if(tile==64){RUN(W,64);}else if(tile==128){RUN(W,128);}else if(tile==256){RUN(W,256);}else return -1; }
    CHOOSE(1) else CHOOSE(2) else CHOOSE(4) else return -2;
    return (int)cudaGetLastError();
}
extern "C" __declspec(dllexport) int launch_palette(void* x,void* w,void* palette,void* scales,void* code,void* y,int rows,int cols,void* stream,int warps,int tile,int use_palette) {
    if(use_palette)return dispatch<true>(x,w,palette,scales,code,y,rows,cols,stream,warps,tile);
    return dispatch<false>(x,w,palette,scales,code,y,rows,cols,stream,warps,tile);
}

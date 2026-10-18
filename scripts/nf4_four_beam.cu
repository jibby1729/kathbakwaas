// Experimental fused NF4 matrix-vector products for four beams.
// Same quantized weights and BF16 operands; reduction equivalence must be tested.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <mma.h>

__global__ void nf4_four_beam(const __nv_bfloat16* x, const uint8_t* w,
    const float* scales, const float* code, __nv_bfloat16* y, int rows, int cols) {
    int row = blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;
    float sums[4] = {0,0,0,0};
    for (int k=lane; k<cols; k+=32) {
        int index=row*cols+k;
        uint8_t packed=w[index/2];
        int q=(index&1)?(packed&15):(packed>>4);
        float weight=__bfloat162float(__float2bfloat16_rn(code[q]*scales[index/64]));
        #pragma unroll
        for (int b=0;b<4;b++) sums[b]=fmaf(weight,__bfloat162float(x[b*cols+k]),sums[b]);
    }
    #pragma unroll
    for(int b=0;b<4;b++) {
        for(int offset=16;offset>0;offset/=2) sums[b]+=__shfl_down_sync(0xffffffff,sums[b],offset);
        if(lane==0) y[b*rows+row]=__float2bfloat16_rn(sums[b]);
    }
}

extern "C" __declspec(dllexport) int launch_nf4_four_beam(
    void* x,void* w,void* scales,void* code,void* y,int rows,int cols,void* stream) {
    nf4_four_beam<<<(rows+3)/4,128,0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x,(const uint8_t*)w,(const float*)scales,
        (const float*)code,(__nv_bfloat16*)y,rows,cols);
    return (int)cudaGetLastError();
}

// Probe tensor-core accumulation layouts against cuBLAS before use.
template<int SPLITS,int GROUP>
__global__ void nf4_wmma(const __nv_bfloat16* x,const uint8_t* w,
    const float* scales,const float* code,__nv_bfloat16* y,int rows,int cols) {
    constexpr int TILEK=SPLITS*GROUP;
    __shared__ __align__(32) __nv_bfloat16 a[16*TILEK];
    __shared__ __align__(32) __nv_bfloat16 b[16*TILEK];
    __shared__ __align__(32) float partial[SPLITS*256];
    int warp=threadIdx.x/32;
    int rowbase=blockIdx.x*16;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> af;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> bf;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> cf;
    nvcuda::wmma::fill_fragment(cf,0.f);
    for(int base=0;base<cols;base+=TILEK) {
        for(int i=threadIdx.x;i<16*TILEK;i+=32*SPLITS) {
            int r=i/TILEK,k=i%TILEK;
            int wi=(rowbase+r)*cols+base+k;
            uint8_t packed=w[wi/2];
            a[i]=__float2bfloat16_rn(code[(wi&1)?(packed&15):(packed>>4)]*scales[wi/64]);
            b[i]=r<4?x[r*cols+base+k]:__float2bfloat16_rn(0.f);
        }
        __syncthreads();
        for(int k=warp*GROUP;k<(warp+1)*GROUP;k+=16) {
            nvcuda::wmma::load_matrix_sync(af,a+k,TILEK);
            nvcuda::wmma::load_matrix_sync(bf,b+k,TILEK);
            nvcuda::wmma::mma_sync(cf,af,bf,cf);
        }
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(partial+warp*256,cf,16,nvcuda::wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<64;i+=32*SPLITS) {
        int row=i/4,beam=i%4;
        float sum=partial[row*16+beam];
        #pragma unroll
        for(int s=1;s<SPLITS;s++)sum+=partial[s*256+row*16+beam];
        y[beam*rows+rowbase+row]=__float2bfloat16_rn(sum);
    }
}

template<int TILEK>
__global__ void nf4_wmma_tiled(const __nv_bfloat16* x,const uint8_t* w,
    const float* scales,const float* code,__nv_bfloat16* y,int rows,int cols) {
    constexpr int STRIDE=TILEK+8;
    __shared__ __align__(32) __nv_bfloat16 a[64*STRIDE];
    __shared__ __align__(32) __nv_bfloat16 b[16*STRIDE];
    __shared__ __align__(32) float out[4*256];
    int warp=threadIdx.x/32;
    int rowbase=blockIdx.x*64;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> af;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> bf;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> cf;
    nvcuda::wmma::fill_fragment(cf,0.f);
    for(int base=0;base<cols;base+=TILEK) {
        for(int i=threadIdx.x;i<64*TILEK/2;i+=128) {
            int r=i/(TILEK/2),k=(i%(TILEK/2))*2;
            int wi=(rowbase+r)*cols+base+k;
            uint8_t packed=w[wi/2];
            float scale=scales[wi/64];
            a[r*STRIDE+k]=__float2bfloat16_rn(code[packed>>4]*scale);
            a[r*STRIDE+k+1]=__float2bfloat16_rn(code[packed&15]*scale);
        }
        for(int i=threadIdx.x;i<16*TILEK;i+=128) {
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
    for(int i=threadIdx.x;i<256;i+=128) {
        int row=i/4,beam=i%4;
        y[beam*rows+rowbase+row]=__float2bfloat16_rn(out[(row/16)*256+(row%16)*16+beam]);
    }
}

extern "C" __declspec(dllexport) int launch_nf4_wmma(
    void* x,void* w,void* scales,void* code,void* y,int rows,int cols,void* stream,int splits,int group) {
    #define LAUNCH(S,G) nf4_wmma<S,G><<<rows/16,32*S,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)x,(const uint8_t*)w,(const float*)scales,(const float*)code,(__nv_bfloat16*)y,rows,cols)
    #define TILED(K) nf4_wmma_tiled<K><<<rows/64,128,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)x,(const uint8_t*)w,(const float*)scales,(const float*)code,(__nv_bfloat16*)y,rows,cols)
    if(splits==0 && group==64) { TILED(64); }
    else if(splits==0 && group==128) { TILED(128); }
    else if(splits==0 && group==256) { TILED(256); }
    else if(splits==1) { LAUNCH(1,16); }
    else if(splits==2) { LAUNCH(2,32); }
    else if(splits==4 && group==16) { LAUNCH(4,16); }
    else if(splits==4 && group==32) { LAUNCH(4,32); }
    else if(splits==4 && group==64) { LAUNCH(4,64); }
    else if(splits==8) { LAUNCH(8,16); }
    else return -1;
    return (int)cudaGetLastError();
}

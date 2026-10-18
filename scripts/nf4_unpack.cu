// Invert the register layout permutation, preserving every quantized byte.
#include <cuda_runtime.h>
#include <stdint.h>
__global__ void unpack_registers(const uint32_t* packed,uint8_t* out,int rows,int cols) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i>=rows*cols/8)return;
    int lane=i%32,tile=(i/32)%(cols/16),rowblock=(i/32)/(cols/16);
    int r=rowblock*16+lane/4,k=tile*16+(lane%4)*2;
    int base=(r*cols+k)/2;
    uint32_t value=packed[i];
    out[base]=value&255;out[base+4*cols]=(value>>8)&255;
    out[base+4]=(value>>16)&255;out[base+4*cols+4]=(value>>24)&255;
}
extern "C" __declspec(dllexport) int unpack_nf4(void* packed,void* out,int rows,int cols,void* stream) {
    unpack_registers<<<(rows*cols/8+255)/256,256,0,(cudaStream_t)stream>>>((uint32_t*)packed,(uint8_t*)out,rows,cols);
    return (int)cudaGetLastError();
}

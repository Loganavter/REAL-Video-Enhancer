// Minimal LSU-cost probes for the hop's data movement: C=64 int8 codes, 3x3 taps, one thread per output
// pixel, 16-byte vector loads. Compute is deliberately trivial (dp4a accumulate) - this counts the
// load/store instruction class, which is the resource the staging claim is about.
#include <cstdint>
#include <cuda_runtime.h>

#define C 64
#define TILE 128

extern "C" __global__ void today(const int8_t* __restrict__ x, const int8_t* __restrict__ w,
                                 int32_t* __restrict__ out, int H, int W) {
    int p = blockIdx.x * TILE + threadIdx.x;
    int h = p / W, wc = p % W;
    int32_t acc = 0;
    const int4* wp = (const int4*)w;
    for (int tap = 0; tap < 9; ++tap) {
        int dy = tap / 3 - 1, dx = tap % 3 - 1;
        int yy = min(max(h + dy, 0), H - 1), xx = min(max(wc + dx, 0), W - 1);
        const int4* rp = (const int4*)(x + ((size_t)yy * W + xx) * C);
#pragma unroll
        for (int v = 0; v < C / 16; ++v) {           // 4 vector loads per tap = 64 bytes
            int4 a = rp[v];
            int4 b = wp[tap * (C / 16) + v];
            acc += __dp4a(a.x, b.x, 0) + __dp4a(a.y, b.y, 0) + __dp4a(a.z, b.z, 0) + __dp4a(a.w, b.w, 0);
        }
    }
    out[p] = acc;
}

extern "C" __global__ void staged(const int8_t* __restrict__ x, const int8_t* __restrict__ w,
                                  int32_t* __restrict__ out, int H, int W) {
    __shared__ int8_t band[3][TILE + 2][C];
    int base = blockIdx.x * TILE;
    int h = base / W, wc = base % W;
    // stage three rows: (TILE+2) pixels x C bytes each, 16 bytes at a time
    for (int t = threadIdx.x; t < 3 * (TILE + 2) * (C / 16); t += blockDim.x) {
        int tap = t / ((TILE + 2) * (C / 16));
        int rem = t % ((TILE + 2) * (C / 16));
        int col = rem / (C / 16), vec = rem % (C / 16);
        int yy = min(max(h + tap - 1, 0), H - 1), xx = min(max(wc + col - 1, 0), W - 1);
        ((int4*)band[tap][col])[vec] = ((const int4*)(x + ((size_t)yy * W + xx) * C))[vec];
    }
    __syncthreads();
    int p = base + threadIdx.x;
    if (p >= (int)((size_t)H * W)) return;
    int32_t acc = 0;
    const int4* wp = (const int4*)w;
#pragma unroll
    for (int tap = 0; tap < 9; ++tap) {
        int dy = tap / 3, dx = tap % 3;               // row dy, column dx - 1 inside the band
        const int4* rp = (const int4*)band[dy][threadIdx.x + dx];
#pragma unroll
        for (int v = 0; v < C / 16; ++v) {
            int4 a = rp[v];
            int4 b = wp[tap * (C / 16) + v];
            acc += __dp4a(a.x, b.x, 0) + __dp4a(a.y, b.y, 0) + __dp4a(a.z, b.z, 0) + __dp4a(a.w, b.w, 0);
        }
    }
    out[p] = acc;
}

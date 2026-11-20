// Does a shared-memory window remove the dependent round trip? A pure memory-mechanism microbenchmark:
// no tensor cores, no MMA, just the address pattern of the hop expressed two ways.
#include <cuda_runtime.h>
#include <cstdint>

// A: nine dependent gathers from global, one per tap (today's pattern, no MMA)
__global__ void gather9(const int8_t* __restrict__ x, int32_t* __restrict__ out, int H, int W, int C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;         // one thread per pixel
    if (row >= H * W) return;
    int h = row / W, w = row % W;
    int32_t acc = 0;
    for (int tap = 0; tap < 9; ++tap) {
        int dy = tap / 3 - 1, dx = tap % 3 - 1;
        int yy = h + dy, xx = w + dx;
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
        const int8_t* p = x + (size_t)(yy * W + xx) * C;
        for (int c = 0; c < C; ++c) acc += p[c];               // 64 B = 2 sectors
    }
    out[row] = acc;
}

// B: one row block staged in shared memory (3 rows x (BM+2) x C), taps read from smem
__global__ void smem9(const int8_t* __restrict__ x, int32_t* __restrict__ out, int H, int W, int C,
                      int BM) {
    extern __shared__ int8_t tile[];                          // 3 rows x (BM+2) x C
    int base = blockIdx.x * BM;                               // first pixel of this block
    int h0 = base / W, w0 = base % W;
    int TW = BM + 2;
    // one cooperative load of the three-row window
    for (int dy = 0; dy < 3; ++dy) {
        for (int i = threadIdx.x; i < TW * C; i += blockDim.x) {
            int px = i / C, c = i % C;
            int yy = h0 + dy - 1, xx = w0 + px - 1;
            int8_t v = 0;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) v = x[(size_t)(yy * W + xx) * C + c];
            tile[(dy * TW + px) * C + c] = v;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BM; i += blockDim.x) {
        int row = base + i;
        if (row >= H * W) continue;
        int32_t acc = 0;
        for (int tap = 0; tap < 9; ++tap) {
            int dy = tap / 3, dx = tap % 3;
            const int8_t* p = tile + ((dy * TW + i + dx) * C);
            for (int c = 0; c < C; ++c) acc += p[c];
        }
        out[row] = acc;
    }
}

extern "C" {
void run_gather9(const void* x, void* out, int H, int W, int C, int threads, int iters) {
    for (int i = 0; i < iters; ++i)
        gather9<<<(H * W + threads - 1) / threads, threads>>>((const int8_t*)x, (int32_t*)out, H, W, C);
}
void run_smem9(const void* x, void* out, int H, int W, int C, int BM, int threads, int iters) {
    size_t smem = (size_t)3 * (BM + 2) * C;
    for (int i = 0; i < iters; ++i)
        smem9<<<(H * W + BM - 1) / BM, threads, smem>>>((const int8_t*)x, (int32_t*)out, H, W, C, BM);
}
}

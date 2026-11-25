// Native Blackwell CUDA Vision Tower for Inkling-2-Mini (Inkling2VisionModel).
// Vision kernels & image preprocessing pipeline adapted from
// satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0).
//
// Pure C host runtime (zero libstdc++ ABI dependencies, 100% compatible with Zig linking) +
// Zero-Permanent-VRAM architecture:
//   - All 422 model.visual.* BF16 tensors (~607 MiB) are loaded once into host RAM (0 MiB VRAM).
//   - On image encode, a single temporary GPU workspace (~34 MiB layer staging + ~80-150 MiB
//     activations) streams each of the 23 ViT layers through cuBLAS BF16 Tensor Cores (FP32 accum)
//     and releases all GPU memory immediately on return.
//   - A 16-entry host-RAM LRU cache keyed by 64-bit pixel hash eliminates redundant encoding across
//     multi-turn conversations while negative synthetic token IDs give automatic Prefix KV Cache hits.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#pragma nv_diag_suppress 550
#include "third_party/stb/stb_image.h"
#pragma nv_diag_default 550

namespace {

constexpr int V_LAYERS = 23;
constexpr int V_HIDDEN = 1024;
constexpr int V_HEADS = 16;
constexpr int V_HEAD_DIM = 64;
constexpr int V_PATCH = 20;
constexpr int V_TILE = 40;
constexpr int V_PATCH_DIM = 1200; // 20 * 20 * 3
constexpr int V_POS_GRID = 32;
constexpr int V_MLP_DIM = 4096;
constexpr int V_MERGED = 4096;
constexpr int V_OUT_DIM = 2304;
constexpr int V_ROPE_FREQS = 16;
constexpr float V_LN_EPS = 1e-5f;
constexpr int V_ROW_T = 256;
constexpr int V_ELEM_T = 256;
constexpr int V_MAX_SIDE = 16384;
constexpr int64_t V_MAX_PIXELS = 64LL * 1024 * 1024;
constexpr int V_DEFAULT_MAX_TOKENS = 1024;
constexpr int V_ABS_MAX_TOKENS = 4096;
constexpr size_t V_SCORE_BYTES = 64ULL << 20;
constexpr int V_CACHE_MAX = 16;
constexpr uint8_t V_HASH_VERSION = 1;

static inline int v_min_i(int a, int b) { return a < b ? a : b; }
static inline int v_max_i(int a, int b) { return a > b ? a : b; }
static inline int v_clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int round_up(int v, int to) { return (v + to - 1) / to * to; }

static inline uint64_t splitmix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t hash_pixels(int width, int height, const uint8_t* rgb, size_t n) {
    uint64_t h = splitmix64((uint64_t(uint32_t(width)) << 32) | uint32_t(height)) ^ V_HASH_VERSION;
    const uint8_t* p = rgb;
    size_t rem = n;
    for (; rem >= 8; p += 8, rem -= 8) {
        uint64_t word;
        std::memcpy(&word, p, 8);
        h = splitmix64(h + 0x9E3779B97F4A7C15ULL + word);
    }
    if (rem > 0) {
        uint64_t word = 0;
        std::memcpy(&word, p, rem);
        h = splitmix64(h + 0x9E3779B97F4A7C15ULL + word);
    }
    return splitmix64(h ^ uint64_t(n));
}

static inline int32_t synth_token_id(uint64_t hash, int k) {
    const uint64_t v = splitmix64(hash ^ (uint64_t(uint32_t(k)) + 1ULL) * 0x9E3779B97F4A7C15ULL) >> 33;
    return int32_t(-1 - int64_t(v));
}

static inline int calc_image_tokens(int height, int width) {
    return ((height + V_TILE - 1) / V_TILE) * ((width + V_TILE - 1) / V_TILE);
}

static void fit_image_dims(int height, int width, int max_tokens, int* out_h, int* out_w) {
    if (calc_image_tokens(height, width) <= max_tokens) {
        *out_h = height;
        *out_w = width;
        return;
    }
    const double scale = std::sqrt(double(max_tokens) * V_TILE * V_TILE / (double(height) * double(width)));
    const bool wide = width >= height;
    int longer = v_max_i(1, int(std::floor((wide ? width : height) * scale)));
    for (;; --longer) {
        const double ratio = double(wide ? height : width) / double(wide ? width : height);
        const int shorter = v_max_i(1, int(std::lround(longer * ratio)));
        const int h = wide ? shorter : longer;
        const int w = wide ? longer : shorter;
        if (calc_image_tokens(h, w) <= max_tokens || longer == 1) {
            *out_h = h;
            *out_w = w;
            return;
        }
    }
}

static uint8_t* reduce_area_rgb(const uint8_t* rgb, int h, int w, int oh, int ow) {
    float* rows = static_cast<float*>(std::malloc(size_t(h) * ow * 3 * sizeof(float)));
    uint8_t* out = static_cast<uint8_t*>(std::malloc(size_t(oh) * ow * 3));
    float* acc = static_cast<float*>(std::malloc(size_t(ow) * 3 * sizeof(float)));
    if (!rows || !out || !acc) {
        std::free(rows); std::free(out); std::free(acc);
        return nullptr;
    }

    const double step_x = double(w) / double(ow);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgb + size_t(y) * w * 3;
        float* dst = rows + size_t(y) * ow * 3;
        for (int ox = 0; ox < ow; ++ox) {
            const double lo = ox * step_x, hi = (ox + 1) * step_x;
            const int first = v_min_i(w - 1, int(std::floor(lo)));
            const int last = v_min_i(w - 1, v_max_i(first, int(std::ceil(hi)) - 1));
            double sum_w = 0.0, a0 = 0.0, a1 = 0.0, a2 = 0.0;
            for (int ix = first; ix <= last; ++ix) {
                double wt = std::min(hi, double(ix + 1)) - std::max(lo, double(ix));
                if (wt < 0.0) wt = 0.0;
                const uint8_t* p = src + size_t(ix) * 3;
                a0 += wt * p[0];
                a1 += wt * p[1];
                a2 += wt * p[2];
                sum_w += wt;
            }
            const double inv = sum_w > 0.0 ? (1.0 / sum_w) : 0.0;
            dst[ox * 3 + 0] = float(a0 * inv);
            dst[ox * 3 + 1] = float(a1 * inv);
            dst[ox * 3 + 2] = float(a2 * inv);
        }
    }

    const double step_y = double(h) / double(oh);
    for (int oy = 0; oy < oh; ++oy) {
        const double lo = oy * step_y, hi = (oy + 1) * step_y;
        const int first = v_min_i(h - 1, int(std::floor(lo)));
        const int last = v_min_i(h - 1, v_max_i(first, int(std::ceil(hi)) - 1));
        std::memset(acc, 0, size_t(ow) * 3 * sizeof(float));
        double sum_w = 0.0;
        for (int iy = first; iy <= last; ++iy) {
            double wt = std::min(hi, double(iy + 1)) - std::max(lo, double(iy));
            if (wt < 0.0) wt = 0.0;
            sum_w += wt;
        }
        const double inv = sum_w > 0.0 ? (1.0 / sum_w) : 0.0;
        for (int iy = first; iy <= last; ++iy) {
            double wt = std::min(hi, double(iy + 1)) - std::max(lo, double(iy));
            if (wt < 0.0) wt = 0.0;
            const float fw = float(wt * inv);
            const float* src = rows + size_t(iy) * ow * 3;
            for (size_t j = 0; j < size_t(ow) * 3; ++j) acc[j] += fw * src[j];
        }
        uint8_t* dst = out + size_t(oy) * ow * 3;
        for (size_t j = 0; j < size_t(ow) * 3; ++j) {
            long val = std::lround(acc[j]);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            dst[j] = uint8_t(val);
        }
    }

    std::free(rows);
    std::free(acc);
    return out;
}

static bool starts_with_nocase(const char* text, size_t len, const char* prefix) {
    size_t plen = std::strlen(prefix);
    if (len < plen) return false;
    for (size_t i = 0; i < plen; ++i) {
        char c = text[i];
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        if (c != prefix[i]) return false;
    }
    return true;
}

static uint8_t* decode_base64(const char* text, size_t len, size_t* out_len) {
    static int8_t table[256];
    static bool inited = false;
    if (!inited) {
        for (int i = 0; i < 256; ++i) table[i] = -1;
        const char* alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) table[static_cast<uint8_t>(alpha[i])] = int8_t(i);
        inited = true;
    }
    size_t cap = (len / 4) * 3 + 8;
    uint8_t* out = static_cast<uint8_t*>(std::malloc(cap));
    if (!out) return nullptr;
    size_t n = 0;
    uint32_t buf = 0;
    int bits = 0;
    size_t syms = 0;
    bool padded = false;
    for (size_t i = 0; i < len; ++i) {
        uint8_t c = static_cast<uint8_t>(text[i]);
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        if (c == '=') { padded = true; continue; }
        int v = table[c];
        if (v < 0 || padded) { std::free(out); return nullptr; }
        buf = (buf << 6) | uint32_t(v);
        bits += 6;
        ++syms;
        if (bits >= 8) {
            bits -= 8;
            out[n++] = uint8_t((buf >> bits) & 0xFF);
        }
    }
    if ((syms % 4) == 1) { std::free(out); return nullptr; }
    *out_len = n;
    return out;
}

// --- CUDA Vision Kernels (Apache-2.0, adapted from satellitedown/inkling-2-mini-abliterated-ingine) ---

__device__ __forceinline__ uint16_t bf16_bits(float v) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

__device__ __forceinline__ float v_block_sum(float v, float* smem) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < V_ROW_T / 32; ++w) total += smem[w];
    __syncthreads();
    return total;
}

__device__ __forceinline__ void patch_position(int p, int cols, int& row, int& col) {
    const int tile = p >> 2, in = p & 3;
    row = (tile / cols) * 2 + (in >> 1);
    col = (tile % cols) * 2 + (in & 1);
}

__global__ void v_patchify_kernel(const uint8_t* __restrict__ rgb, int height, int width, int cols, int patches,
                                  uint16_t* __restrict__ out, int64_t total) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int p = int(i / V_PATCH_DIM), e = int(i % V_PATCH_DIM);
    if (p >= patches) { out[i] = 0; return; }
    int row, col;
    patch_position(p, cols, row, col);
    const int y = row * V_PATCH + e / (V_PATCH * 3);
    const int x = col * V_PATCH + (e / 3) % V_PATCH;
    const int c = e % 3;
    const float v = (y < height && x < width) ? float(rgb[(size_t(y) * width + x) * 3 + c]) : -1.f;
    out[i] = bf16_bits(__fdiv_rn(__fsub_rn(v, 127.5f), 127.5f));
}

__device__ __forceinline__ void axis_taps(int index, int size, int& t0, int& t1, float& w1) {
    const float src = __fsub_rn(__fdiv_rn(__fmul_rn(__fadd_rn(float(index), 0.5f), float(V_POS_GRID)), float(size)), 0.5f);
    const float low = floorf(src);
    w1 = __fsub_rn(src, low);
    const int l = int(low);
    t0 = min(max(l, 0), V_POS_GRID - 1);
    t1 = min(max(l + 1, 0), V_POS_GRID - 1);
}

__global__ void __launch_bounds__(V_ROW_T) v_embed_kernel(
    const float* __restrict__ patches, const float* __restrict__ pos_table,
    const float* __restrict__ cls, int rows, int cols, float* __restrict__ h) {
    const int token = blockIdx.x, c = threadIdx.x * 4;
    const int S = 1 + 4 * rows * cols;
    float4* out = reinterpret_cast<float4*>(h + size_t(token) * V_HIDDEN + c);
    if (token >= S) { *out = make_float4(0.f, 0.f, 0.f, 0.f); return; }
    if (token == 0) { *out = *reinterpret_cast<const float4*>(cls + c); return; }
    const int p = token - 1;
    int row, col;
    patch_position(p, cols, row, col);
    int r0, r1, c0, c1;
    float wr, wc;
    axis_taps(row, rows * 2, r0, r1, wr);
    axis_taps(col, cols * 2, c0, c1, wc);
    const float w00 = __fmul_rn(1.f - wr, 1.f - wc), w01 = __fmul_rn(1.f - wr, wc);
    const float w10 = __fmul_rn(wr, 1.f - wc), w11 = __fmul_rn(wr, wc);
    auto entry = [&](int r, int cc) {
        return *reinterpret_cast<const float4*>(pos_table + size_t(r * V_POS_GRID + cc) * V_HIDDEN + c);
    };
    const float4 a = entry(r0, c0), b = entry(r0, c1), d = entry(r1, c0), e = entry(r1, c1);
    const float4 x = *reinterpret_cast<const float4*>(patches + size_t(p) * V_HIDDEN + c);
    auto blend = [&](float xa, float pa, float pb, float pd, float pe) {
        return xa + (((pa * w00 + pb * w01) + pd * w10) + pe * w11);
    };
    *out = make_float4(blend(x.x, a.x, b.x, d.x, e.x), blend(x.y, a.y, b.y, d.y, e.y),
                       blend(x.z, a.z, b.z, d.z, e.z), blend(x.w, a.w, b.w, d.w, e.w));
}

__global__ void __launch_bounds__(V_ROW_T) v_layernorm_kernel(
    const float* x, const float* __restrict__ w, const float* __restrict__ b, void* out, bool out_fp32) {
    __shared__ float smem[V_ROW_T / 32];
    const int c = threadIdx.x * 4;
    const size_t off = size_t(blockIdx.x) * V_HIDDEN + c;
    const float4 v = *reinterpret_cast<const float4*>(x + off);
    const float mean = v_block_sum((v.x + v.y) + (v.z + v.w), smem) * (1.f / V_HIDDEN);
    const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
    const float var = v_block_sum((d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3), smem) * (1.f / V_HIDDEN);
    const float r = 1.f / sqrtf(var + V_LN_EPS);
    const float4 w4 = *reinterpret_cast<const float4*>(w + c), b4 = *reinterpret_cast<const float4*>(b + c);
    const float y0 = d0 * r * w4.x + b4.x, y1 = d1 * r * w4.y + b4.y;
    const float y2 = d2 * r * w4.z + b4.z, y3 = d3 * r * w4.w + b4.w;
    if (out_fp32) {
        *reinterpret_cast<float4*>(static_cast<float*>(out) + off) = make_float4(y0, y1, y2, y3);
    } else {
        uint16_t* o = static_cast<uint16_t*>(out) + off;
        o[0] = bf16_bits(y0); o[1] = bf16_bits(y1); o[2] = bf16_bits(y2); o[3] = bf16_bits(y3);
    }
}

__global__ void __launch_bounds__(V_ROW_T) v_qkv_post_kernel(
    const float* __restrict__ qkv, const float* __restrict__ bias, const float* __restrict__ inv_freq,
    int cols, int S, int Sp, uint16_t* __restrict__ q, uint16_t* __restrict__ k, uint16_t* __restrict__ vt) {
    const int token = blockIdx.x;
    const bool valid = token < S;
    float pos_row = 0.f, pos_col = 0.f;
    if (valid && token > 0) {
        int row, col;
        patch_position(token - 1, cols, row, col);
        pos_row = float(row + 1);
        pos_col = float(col + 1);
    }
    const float* x = qkv + size_t(token) * (3 * V_HIDDEN);
    for (int pair = threadIdx.x; pair < V_HIDDEN / 2; pair += V_ROW_T) {
        const int head = pair / (V_HEAD_DIM / 2), j = pair % (V_HEAD_DIM / 2);
        const int c = head * V_HEAD_DIM + 2 * j;
        const size_t qk = (size_t(head) * Sp + token) * V_HEAD_DIM + 2 * j;
        const size_t v0 = (size_t(head) * V_HEAD_DIM + 2 * j) * Sp + token;
        if (!valid) {
            q[qk] = q[qk + 1] = k[qk] = k[qk + 1] = 0;
            vt[v0] = vt[v0 + Sp] = 0;
            continue;
        }
        const float angle = __fmul_rn(j < V_ROPE_FREQS ? pos_col : pos_row, inv_freq[j % V_ROPE_FREQS]);
        float sn, cs;
        sincosf(angle, &sn, &cs);
        const float qa = x[c] + bias[c], qb = x[c + 1] + bias[c + 1];
        const float ka = x[V_HIDDEN + c] + bias[V_HIDDEN + c], kb = x[V_HIDDEN + c + 1] + bias[V_HIDDEN + c + 1];
        q[qk] = bf16_bits(__fadd_rn(__fmul_rn(qa, cs), __fmul_rn(-qb, sn)));
        q[qk + 1] = bf16_bits(__fadd_rn(__fmul_rn(qb, cs), __fmul_rn(qa, sn)));
        k[qk] = bf16_bits(__fadd_rn(__fmul_rn(ka, cs), __fmul_rn(-kb, sn)));
        k[qk + 1] = bf16_bits(__fadd_rn(__fmul_rn(kb, cs), __fmul_rn(ka, sn)));
        vt[v0] = bf16_bits(x[2 * V_HIDDEN + c] + bias[2 * V_HIDDEN + c]);
        vt[v0 + Sp] = bf16_bits(x[2 * V_HIDDEN + c + 1] + bias[2 * V_HIDDEN + c + 1]);
    }
}

__global__ void __launch_bounds__(V_ROW_T) v_softmax_kernel(
    const float* __restrict__ scores, int ld, int S, float scale, uint16_t* __restrict__ p) {
    __shared__ float smem[V_ROW_T / 32];
    __shared__ float s_max[V_ROW_T / 32];
    const float* row = scores + size_t(blockIdx.x) * ld;
    uint16_t* out = p + size_t(blockIdx.x) * ld;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < S; c += V_ROW_T) m = fmaxf(m, row[c] * scale);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) s_max[threadIdx.x >> 5] = m;
    __syncthreads();
    m = s_max[0];
#pragma unroll
    for (int w = 1; w < V_ROW_T / 32; ++w) m = fmaxf(m, s_max[w]);
    __syncthreads();
    float sum = 0.f;
    for (int c = threadIdx.x; c < S; c += V_ROW_T) sum += expf(row[c] * scale - m);
    const float inv = 1.f / v_block_sum(sum, smem);
    for (int c = threadIdx.x; c < ld; c += V_ROW_T) {
        out[c] = c < S ? bf16_bits(expf(row[c] * scale - m) * inv) : 0;
    }
}

__global__ void v_bias_act_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                  int64_t total, int N, bool gelu, uint16_t* __restrict__ out) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = x[i];
    if (bias) v += bias[i % N];
    if (gelu) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752440f));
    out[i] = bf16_bits(v);
}

__global__ void __launch_bounds__(V_ROW_T) v_residual_kernel(
    const float* __restrict__ y, const float* __restrict__ bias, const float* __restrict__ gamma, float* __restrict__ h) {
    const int c = threadIdx.x * 4;
    const size_t off = size_t(blockIdx.x) * V_HIDDEN + c;
    const float4 v = *reinterpret_cast<const float4*>(y + off);
    const float4 b = *reinterpret_cast<const float4*>(bias + c);
    const float4 g = *reinterpret_cast<const float4*>(gamma + c);
    float4* out = reinterpret_cast<float4*>(h + off);
    const float4 r = *out;
    *out = make_float4(r.x + g.x * (v.x + b.x), r.y + g.y * (v.y + b.y),
                       r.z + g.z * (v.z + b.z), r.w + g.w * (v.w + b.w));
}

__global__ void v_merge_kernel(const float* __restrict__ h, int64_t valid, int64_t total, uint16_t* __restrict__ out) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = i < valid ? bf16_bits(h[V_HIDDEN + i]) : 0;
}

static inline unsigned v_blocks(int64_t n) { return unsigned((n + V_ELEM_T - 1) / V_ELEM_T); }

// --- Minimal Pure-C Safetensors Index for model.visual.* ---

constexpr int MAX_VIS_ENTRIES = 512;

struct VisTensorEntry {
    char name[128];
    uint64_t off_start;
    uint64_t off_end;
};

struct VisIndex {
    VisTensorEntry entries[MAX_VIS_ENTRIES];
    int count;
    uint64_t data_base;
};

static const VisTensorEntry* find_vis_entry(const VisIndex* idx, const char* name) {
    for (int i = 0; i < idx->count; ++i) {
        if (std::strcmp(idx->entries[i].name, name) == 0) return &idx->entries[i];
    }
    return nullptr;
}

static bool parse_safetensors_visual_index(FILE* f, VisIndex* idx) {
    idx->count = 0;
    idx->data_base = 0;
    uint64_t hlen = 0;
    if (std::fread(&hlen, 8, 1, f) != 1 || hlen == 0 || hlen > (64ULL << 20)) return false;
    char* hdr = static_cast<char*>(std::malloc(size_t(hlen) + 1));
    if (!hdr) return false;
    if (std::fread(hdr, 1, size_t(hlen), f) != size_t(hlen)) { std::free(hdr); return false; }
    hdr[hlen] = '\0';
    idx->data_base = 8 + hlen;

    size_t pos = 0;
    const size_t n = size_t(hlen);
    auto skip_ws = [&]() {
        while (pos < n && (hdr[pos] == ' ' || hdr[pos] == '\n' || hdr[pos] == '\r' || hdr[pos] == '\t')) ++pos;
    };
    auto parse_str = [&](char* dst, size_t cap) -> bool {
        skip_ws();
        if (pos >= n || hdr[pos] != '"') return false;
        ++pos;
        size_t out = 0;
        while (pos < n && hdr[pos] != '"') {
            if (hdr[pos] == '\\' && pos + 1 < n) ++pos;
            if (dst && out + 1 < cap) dst[out++] = hdr[pos];
            ++pos;
        }
        if (pos >= n) return false;
        ++pos;
        if (dst && cap > 0) dst[out] = '\0';
        return true;
    };
    auto parse_u64 = [&](uint64_t* val) -> bool {
        skip_ws();
        if (pos >= n || hdr[pos] < '0' || hdr[pos] > '9') return false;
        uint64_t v = 0;
        while (pos < n && hdr[pos] >= '0' && hdr[pos] <= '9') v = v * 10 + uint64_t(hdr[pos++] - '0');
        *val = v;
        return true;
    };
    auto skip_val = [&]() {
        skip_ws();
        if (pos >= n) return;
        char c = hdr[pos];
        if (c == '"') { parse_str(nullptr, 0); return; }
        if (c == '{' || c == '[') {
            char open = c, close = (c == '{') ? '}' : ']';
            int depth = 1;
            ++pos;
            while (pos < n && depth > 0) {
                if (hdr[pos] == '"') { parse_str(nullptr, 0); continue; }
                if (hdr[pos] == open) ++depth;
                else if (hdr[pos] == close) --depth;
                ++pos;
            }
            return;
        }
        while (pos < n && hdr[pos] != ',' && hdr[pos] != '}' && hdr[pos] != ']') ++pos;
    };

    skip_ws();
    if (pos < n && hdr[pos] == '{') ++pos;
    char key[256];
    char fkey[64];
    while (pos < n) {
        skip_ws();
        if (pos < n && hdr[pos] == '}') break;
        if (!parse_str(key, sizeof(key))) break;
        skip_ws();
        if (pos < n && hdr[pos] == ':') ++pos;
        skip_ws();
        if (std::strncmp(key, "model.visual.", 13) == 0 && pos < n && hdr[pos] == '{') {
            ++pos;
            uint64_t a = 0, b = 0;
            bool got = false;
            while (pos < n) {
                skip_ws();
                if (pos < n && hdr[pos] == '}') { ++pos; break; }
                if (!parse_str(fkey, sizeof(fkey))) break;
                skip_ws();
                if (pos < n && hdr[pos] == ':') ++pos;
                if (std::strcmp(fkey, "data_offsets") == 0) {
                    skip_ws();
                    if (pos < n && hdr[pos] == '[') {
                        ++pos;
                        bool ok = parse_u64(&a);
                        skip_ws();
                        if (pos < n && hdr[pos] == ',') ++pos;
                        ok = ok && parse_u64(&b);
                        while (pos < n && hdr[pos] != ']') ++pos;
                        if (pos < n) ++pos;
                        if (ok) got = true;
                    }
                } else {
                    skip_val();
                }
                skip_ws();
                if (pos < n && hdr[pos] == ',') ++pos;
            }
            if (got && idx->count < MAX_VIS_ENTRIES) {
                VisTensorEntry& e = idx->entries[idx->count++];
                std::strncpy(e.name, key, sizeof(e.name) - 1);
                e.name[sizeof(e.name) - 1] = '\0';
                e.off_start = a;
                e.off_end = b;
            }
        } else {
            skip_val();
        }
        skip_ws();
        if (pos < n && hdr[pos] == ',') ++pos;
    }

    std::free(hdr);
    return find_vis_entry(idx, "model.visual.proj_out.weight") != nullptr;
}

// --- Pure-C Host-RAM Vision Weights & LRU Cache ---

struct HostVisionLayer {
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b, *ls1, *ls2;
    uint16_t* wqkv; // [3072, 1024]
    float* bqkv;    // [3072]
    uint16_t* wo;   // [1024, 1024]
    float* bo;      // [1024]
    uint16_t* fc;   // [4096, 1024]
    float* bfc;     // [4096]
    uint16_t* proj; // [1024, 4096]
    float* bproj;   // [1024]
};

struct CachedImage {
    bool valid;
    uint64_t hash;
    uint64_t last_use;
    int num_tokens;
    int32_t* token_ids;
    float* embeds_f32;
};

struct VisionContext {
    uint16_t* patch_w;
    float* pos;
    float* cls;
    float *ln_pre_w, *ln_pre_b;
    HostVisionLayer layers[V_LAYERS];
    uint16_t* proj_in;
    uint16_t* proj_out;
    float inv_freq[V_ROPE_FREQS];

    cublasHandle_t cublas;
    pthread_mutex_t mu;
    uint64_t lru_clock;
    CachedImage cache[V_CACHE_MAX];
};

static uint16_t* load_bf16(FILE* f, const VisIndex* idx, const char* name, size_t elems) {
    const VisTensorEntry* e = find_vis_entry(idx, name);
    if (!e || size_t(e->off_end - e->off_start) != elems * 2) return nullptr;
    uint16_t* buf = static_cast<uint16_t*>(std::malloc(elems * 2));
    if (!buf) return nullptr;
    if (fseeko(f, off_t(idx->data_base + e->off_start), SEEK_SET) != 0 ||
        std::fread(buf, 2, elems, f) != elems) {
        std::free(buf);
        return nullptr;
    }
    return buf;
}

static bool load_bf16_into(FILE* f, const VisIndex* idx, const char* name, size_t elems, uint16_t* dst) {
    const VisTensorEntry* e = find_vis_entry(idx, name);
    if (!e || size_t(e->off_end - e->off_start) != elems * 2) return false;
    return fseeko(f, off_t(idx->data_base + e->off_start), SEEK_SET) == 0 &&
           std::fread(dst, 2, elems, f) == elems;
}

static float* load_bf16_as_f32(FILE* f, const VisIndex* idx, const char* name, size_t elems) {
    uint16_t* raw = load_bf16(f, idx, name, elems);
    if (!raw) return nullptr;
    float* out = static_cast<float*>(std::malloc(elems * sizeof(float)));
    if (!out) { std::free(raw); return nullptr; }
    for (size_t i = 0; i < elems; ++i) {
        uint32_t w = uint32_t(raw[i]) << 16;
        std::memcpy(&out[i], &w, 4);
    }
    std::free(raw);
    return out;
}

static void free_vision_ctx(VisionContext* ctx) {
    if (!ctx) return;
    if (ctx->cublas) cublasDestroy(ctx->cublas);
    std::free(ctx->patch_w);
    std::free(ctx->pos);
    std::free(ctx->cls);
    std::free(ctx->ln_pre_w);
    std::free(ctx->ln_pre_b);
    for (int i = 0; i < V_LAYERS; ++i) {
        HostVisionLayer& L = ctx->layers[i];
        std::free(L.ln1_w); std::free(L.ln1_b);
        std::free(L.ln2_w); std::free(L.ln2_b);
        std::free(L.ls1);   std::free(L.ls2);
        std::free(L.wqkv);  std::free(L.bqkv);
        std::free(L.wo);    std::free(L.bo);
        std::free(L.fc);    std::free(L.bfc);
        std::free(L.proj);  std::free(L.bproj);
    }
    std::free(ctx->proj_in);
    std::free(ctx->proj_out);
    for (int i = 0; i < V_CACHE_MAX; ++i) {
        std::free(ctx->cache[i].token_ids);
        std::free(ctx->cache[i].embeds_f32);
    }
    pthread_mutex_destroy(&ctx->mu);
    std::free(ctx);
}

static void run_gemm_bf16(cublasHandle_t handle, const __nv_bfloat16* A, int lda,
                          const __nv_bfloat16* B, int ldb, void* C, int ldc,
                          bool out_fp32, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, ldb,
        A, CUDA_R_16BF, lda,
        &beta,
        C, out_fp32 ? CUDA_R_32F : CUDA_R_16BF, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

static bool encode_rgb_on_gpu(VisionContext* ctx, const uint8_t* rgb, int height, int width,
                              cudaStream_t stream, float* out_embeds_host) {
    const int rows = (height + V_TILE - 1) / V_TILE;
    const int cols = (width + V_TILE - 1) / V_TILE;
    const int N = rows * cols;
    const int S = 4 * N + 1;
    const int Sp = round_up(S, 64);
    const int Np = round_up(N, 16);
    int Qc = int((V_SCORE_BYTES / (size_t(Sp) * 6)) / 64 * 64);
    Qc = v_clamp_i(Qc, 64, Sp);

    constexpr size_t STAGE_BYTES = 34ULL << 20;
    auto align256 = [](size_t& cur, size_t bytes) -> size_t {
        size_t off = cur;
        cur += (bytes + 255) & ~size_t(255);
        return off;
    };

    size_t at = 0;
    const size_t off_stage   = align256(at, STAGE_BYTES);
    const size_t off_invfreq = align256(at, V_ROPE_FREQS * 4);
    const size_t off_pixels  = align256(at, size_t(height) * width * 3);
    const size_t off_h       = align256(at, size_t(Sp) * V_HIDDEN * 4);
    const size_t off_xb      = align256(at, size_t(Sp) * V_HIDDEN * 2);
    const size_t off_big32   = align256(at, size_t(Sp) * V_MLP_DIM * 4);
    const size_t off_big16   = align256(at, size_t(Sp) * V_MLP_DIM * 2);
    const size_t off_attn    = align256(at, size_t(Sp) * V_HIDDEN * 2);
    const size_t off_y       = align256(at, size_t(Sp) * V_HIDDEN * 4);
    const size_t off_scores  = align256(at, size_t(Qc) * Sp * 4);
    const size_t off_probs   = align256(at, size_t(Qc) * Sp * 2);
    const size_t off_out     = align256(at, size_t(Np) * V_OUT_DIM * 4);

    void* work = nullptr;
    if (cudaMalloc(&work, at) != cudaSuccess) {
        std::fprintf(stderr, "[vision] cudaMalloc failed for %.1f MiB temporary workspace\n", double(at) / (1024.0 * 1024.0));
        return false;
    }

    if (!ctx->cublas) {
        if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) { cudaFree(work); return false; }
        cublasSetMathMode(ctx->cublas, CUBLAS_DEFAULT_MATH);
    }
    cublasSetStream(ctx->cublas, stream);

    uint8_t* base = static_cast<uint8_t*>(work);
    uint8_t* stage_base = base + off_stage;
    float* d_inv_freq = reinterpret_cast<float*>(base + off_invfreq);
    uint8_t* d_pixels = base + off_pixels;
    float* d_h = reinterpret_cast<float*>(base + off_h);
    __nv_bfloat16* d_xb = reinterpret_cast<__nv_bfloat16*>(base + off_xb);
    float* d_big32 = reinterpret_cast<float*>(base + off_big32);
    __nv_bfloat16* d_big16 = reinterpret_cast<__nv_bfloat16*>(base + off_big16);
    __nv_bfloat16* d_attn = reinterpret_cast<__nv_bfloat16*>(base + off_attn);
    float* d_y = reinterpret_cast<float*>(base + off_y);
    float* d_scores = reinterpret_cast<float*>(base + off_scores);
    __nv_bfloat16* d_probs = reinterpret_cast<__nv_bfloat16*>(base + off_probs);
    float* d_out = reinterpret_cast<float*>(base + off_out);

    cudaMemcpyAsync(d_inv_freq, ctx->inv_freq, V_ROPE_FREQS * 4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pixels, rgb, size_t(height) * width * 3, cudaMemcpyHostToDevice, stream);

    // Stage 0: Patch embedding + pos + cls + ln_pre
    {
        size_t s = 0;
        __nv_bfloat16* d_patch_w = reinterpret_cast<__nv_bfloat16*>(stage_base + align256(s, size_t(V_HIDDEN) * V_PATCH_DIM * 2));
        float* d_pos = reinterpret_cast<float*>(stage_base + align256(s, size_t(V_POS_GRID) * V_POS_GRID * V_HIDDEN * 4));
        float* d_cls = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_lnw = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_lnb = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        cudaMemcpyAsync(d_patch_w, ctx->patch_w, size_t(V_HIDDEN) * V_PATCH_DIM * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_pos, ctx->pos, size_t(V_POS_GRID) * V_POS_GRID * V_HIDDEN * 4, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_cls, ctx->cls, V_HIDDEN * 4, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_lnw, ctx->ln_pre_w, V_HIDDEN * 4, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_lnb, ctx->ln_pre_b, V_HIDDEN * 4, cudaMemcpyHostToDevice, stream);

        const int64_t total_p = int64_t(Sp) * V_PATCH_DIM;
        v_patchify_kernel<<<v_blocks(total_p), V_ELEM_T, 0, stream>>>(
            d_pixels, height, width, cols, 4 * N, reinterpret_cast<uint16_t*>(d_big16), total_p);
        run_gemm_bf16(ctx->cublas, d_big16, V_PATCH_DIM, d_patch_w, V_PATCH_DIM, d_y, V_HIDDEN, true, Sp, V_HIDDEN, V_PATCH_DIM);
        v_embed_kernel<<<Sp, V_ROW_T, 0, stream>>>(d_y, d_pos, d_cls, rows, cols, d_h);
        v_layernorm_kernel<<<Sp, V_ROW_T, 0, stream>>>(d_h, d_lnw, d_lnb, d_h, true);
    }

    // Stages 1..23: Stream each ViT layer through the 34 MiB staging buffer
    __nv_bfloat16* d_q = d_big16;
    __nv_bfloat16* d_k = d_q + size_t(V_HEADS) * Sp * V_HEAD_DIM;
    __nv_bfloat16* d_vt = d_k + size_t(V_HEADS) * Sp * V_HEAD_DIM;
    const float scale = 1.0f / std::sqrt(float(V_HEAD_DIM));

    for (int li = 0; li < V_LAYERS; ++li) {
        const HostVisionLayer& L = ctx->layers[li];
        size_t s = 0;
        __nv_bfloat16* d_wqkv = reinterpret_cast<__nv_bfloat16*>(stage_base + align256(s, size_t(3) * V_HIDDEN * V_HIDDEN * 2));
        __nv_bfloat16* d_wo   = reinterpret_cast<__nv_bfloat16*>(stage_base + align256(s, size_t(V_HIDDEN) * V_HIDDEN * 2));
        __nv_bfloat16* d_fc   = reinterpret_cast<__nv_bfloat16*>(stage_base + align256(s, size_t(V_MLP_DIM) * V_HIDDEN * 2));
        __nv_bfloat16* d_proj = reinterpret_cast<__nv_bfloat16*>(stage_base + align256(s, size_t(V_HIDDEN) * V_MLP_DIM * 2));
        float* d_ln1_w = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_ln1_b = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_ln2_w = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_ln2_b = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_ls1   = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_ls2   = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_bqkv  = reinterpret_cast<float*>(stage_base + align256(s, 3 * V_HIDDEN * 4));
        float* d_bo    = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));
        float* d_bfc   = reinterpret_cast<float*>(stage_base + align256(s, V_MLP_DIM * 4));
        float* d_bproj = reinterpret_cast<float*>(stage_base + align256(s, V_HIDDEN * 4));

        cudaMemcpyAsync(d_wqkv,  L.wqkv,  size_t(3) * V_HIDDEN * V_HIDDEN * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_wo,    L.wo,    size_t(V_HIDDEN) * V_HIDDEN * 2,     cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_fc,    L.fc,    size_t(V_MLP_DIM) * V_HIDDEN * 2,    cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_proj,  L.proj,  size_t(V_HIDDEN) * V_MLP_DIM * 2,    cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ln1_w, L.ln1_w, V_HIDDEN * 4,  cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ln1_b, L.ln1_b, V_HIDDEN * 4,  cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ln2_w, L.ln2_w, V_HIDDEN * 4,  cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ln2_b, L.ln2_b, V_HIDDEN * 4,  cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ls1,   L.ls1,   V_HIDDEN * 4,  cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ls2,   L.ls2,   V_HIDDEN * 4,  cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_bqkv,  L.bqkv,  3 * V_HIDDEN * 4, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_bo,    L.bo,    V_HIDDEN * 4,  cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_bfc,   L.bfc,   V_MLP_DIM * 4, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_bproj, L.bproj, V_HIDDEN * 4,  cudaMemcpyHostToDevice, stream);

        v_layernorm_kernel<<<Sp, V_ROW_T, 0, stream>>>(d_h, d_ln1_w, d_ln1_b, d_xb, false);
        run_gemm_bf16(ctx->cublas, d_xb, V_HIDDEN, d_wqkv, V_HIDDEN, d_big32, 3 * V_HIDDEN, true, Sp, 3 * V_HIDDEN, V_HIDDEN);
        v_qkv_post_kernel<<<Sp, V_ROW_T, 0, stream>>>(
            d_big32, d_bqkv, d_inv_freq, cols, S, Sp,
            reinterpret_cast<uint16_t*>(d_q), reinterpret_cast<uint16_t*>(d_k), reinterpret_cast<uint16_t*>(d_vt));

        for (int head = 0; head < V_HEADS; ++head) {
            const __nv_bfloat16* qh = d_q + size_t(head) * Sp * V_HEAD_DIM;
            const __nv_bfloat16* kh = d_k + size_t(head) * Sp * V_HEAD_DIM;
            const __nv_bfloat16* vth = d_vt + size_t(head) * V_HEAD_DIM * Sp;
            for (int q0 = 0; q0 < Sp; q0 += Qc) {
                const int Q = v_min_i(Qc, Sp - q0);
                run_gemm_bf16(ctx->cublas, qh + size_t(q0) * V_HEAD_DIM, V_HEAD_DIM, kh, V_HEAD_DIM,
                              d_scores, Sp, true, Q, Sp, V_HEAD_DIM);
                v_softmax_kernel<<<Q, V_ROW_T, 0, stream>>>(d_scores, Sp, S, scale, reinterpret_cast<uint16_t*>(d_probs));
                run_gemm_bf16(ctx->cublas, d_probs, Sp, vth, Sp,
                              d_attn + size_t(q0) * V_HIDDEN + head * V_HEAD_DIM, V_HIDDEN, false, Q, V_HEAD_DIM, Sp);
            }
        }
        run_gemm_bf16(ctx->cublas, d_attn, V_HIDDEN, d_wo, V_HIDDEN, d_y, V_HIDDEN, true, Sp, V_HIDDEN, V_HIDDEN);
        v_residual_kernel<<<Sp, V_ROW_T, 0, stream>>>(d_y, d_bo, d_ls1, d_h);

        v_layernorm_kernel<<<Sp, V_ROW_T, 0, stream>>>(d_h, d_ln2_w, d_ln2_b, d_xb, false);
        run_gemm_bf16(ctx->cublas, d_xb, V_HIDDEN, d_fc, V_HIDDEN, d_big32, V_MLP_DIM, true, Sp, V_MLP_DIM, V_HIDDEN);
        const int64_t total_mlp = int64_t(Sp) * V_MLP_DIM;
        v_bias_act_kernel<<<v_blocks(total_mlp), V_ELEM_T, 0, stream>>>(
            d_big32, d_bfc, total_mlp, V_MLP_DIM, true, reinterpret_cast<uint16_t*>(d_big16));
        run_gemm_bf16(ctx->cublas, d_big16, V_MLP_DIM, d_proj, V_MLP_DIM, d_y, V_HIDDEN, true, Sp, V_HIDDEN, V_MLP_DIM);
        v_residual_kernel<<<Sp, V_ROW_T, 0, stream>>>(d_y, d_bproj, d_ls2, d_h);
    }

    // Stage 24: 2x2 Merger -> proj_in -> GELU -> proj_out (FP32)
    {
        const int64_t total_m = int64_t(Np) * V_MERGED;
        v_merge_kernel<<<v_blocks(total_m), V_ELEM_T, 0, stream>>>(
            d_h, int64_t(N) * V_MERGED, total_m, reinterpret_cast<uint16_t*>(d_big16));

        __nv_bfloat16* d_pin = reinterpret_cast<__nv_bfloat16*>(stage_base);
        cudaMemcpyAsync(d_pin, ctx->proj_in, size_t(V_MERGED) * V_MERGED * 2, cudaMemcpyHostToDevice, stream);
        run_gemm_bf16(ctx->cublas, d_big16, V_MERGED, d_pin, V_MERGED, d_big32, V_MERGED, true, Np, V_MERGED, V_MERGED);
        v_bias_act_kernel<<<v_blocks(total_m), V_ELEM_T, 0, stream>>>(
            d_big32, nullptr, total_m, V_MERGED, true, reinterpret_cast<uint16_t*>(d_big16));

        __nv_bfloat16* d_pout = reinterpret_cast<__nv_bfloat16*>(stage_base);
        cudaMemcpyAsync(d_pout, ctx->proj_out, size_t(V_OUT_DIM) * V_MERGED * 2, cudaMemcpyHostToDevice, stream);
        run_gemm_bf16(ctx->cublas, d_big16, V_MERGED, d_pout, V_MERGED, d_out, V_OUT_DIM, true, Np, V_OUT_DIM, V_MERGED);
    }

    cudaMemcpyAsync(out_embeds_host, d_out, size_t(N) * V_OUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    cudaFree(work);
    if (sync_err != cudaSuccess) {
        std::fprintf(stderr, "[vision] CUDA error during vision encode: %s\n", cudaGetErrorString(sync_err));
        return false;
    }
    return true;
}

static bool decode_and_encode_bytes(VisionContext* ctx, const uint8_t* raw_bytes, size_t raw_len,
                                    int max_tokens, cudaStream_t stream,
                                    int32_t* out_token_ids, int max_out_ids,
                                    const float** out_embeds_ptr, int* out_num_tokens) {
    if (!ctx || !raw_bytes || raw_len == 0 || raw_len > size_t(0x7FFFFFFF)) return false;
    if (max_tokens <= 0) max_tokens = V_DEFAULT_MAX_TOKENS;
    if (max_tokens > V_ABS_MAX_TOKENS) max_tokens = V_ABS_MAX_TOKENS;

    int w = 0, h = 0, ch = 0;
    if (!stbi_info_from_memory(raw_bytes, int(raw_len), &w, &h, &ch)) return false;
    if (w < 1 || h < 1 || w > V_MAX_SIDE || h > V_MAX_SIDE || int64_t(w) * h > V_MAX_PIXELS) return false;

    stbi_uc* pixels = stbi_load_from_memory(raw_bytes, int(raw_len), &w, &h, &ch, 0);
    if (!pixels) return false;

    uint8_t* rgb = static_cast<uint8_t*>(std::malloc(size_t(w) * h * 3));
    if (!rgb) { stbi_image_free(pixels); return false; }

    const bool grey = ch <= 2;
    const bool alpha = (ch == 2 || ch == 4);
    const uint8_t* src = pixels;
    for (int y = 0; y < h; ++y) {
        uint8_t* dst = rgb + size_t(y) * w * 3;
        for (int x = 0; x < w; ++x, src += ch, dst += 3) {
            int c[3] = {src[0], grey ? src[0] : src[1], grey ? src[0] : src[2]};
            if (alpha) {
                const int a = src[ch - 1];
                const int bg = ((y >> 3) + (x >> 3)) % 2 == 0 ? 204 : 153;
                for (int k = 0; k < 3; ++k) c[k] = (c[k] * a + bg * (255 - a) + 127) / 255;
            }
            dst[0] = uint8_t(c[0]);
            dst[1] = uint8_t(c[1]);
            dst[2] = uint8_t(c[2]);
        }
    }
    stbi_image_free(pixels);

    int oh = h, ow = w;
    fit_image_dims(h, w, max_tokens, &oh, &ow);
    if (oh != h || ow != w) {
        uint8_t* small_rgb = reduce_area_rgb(rgb, h, w, oh, ow);
        std::free(rgb);
        if (!small_rgb) return false;
        rgb = small_rgb;
        h = oh; w = ow;
    }

    const int num_tokens = calc_image_tokens(h, w);
    if (num_tokens <= 0 || num_tokens > max_out_ids) { std::free(rgb); return false; }
    const uint64_t hash = hash_pixels(w, h, rgb, size_t(w) * h * 3);

    pthread_mutex_lock(&ctx->mu);
    for (int i = 0; i < V_CACHE_MAX; ++i) {
        if (ctx->cache[i].valid && ctx->cache[i].hash == hash && ctx->cache[i].num_tokens == num_tokens) {
            ctx->cache[i].last_use = ++ctx->lru_clock;
            std::memcpy(out_token_ids, ctx->cache[i].token_ids, size_t(num_tokens) * sizeof(int32_t));
            *out_embeds_ptr = ctx->cache[i].embeds_f32;
            *out_num_tokens = num_tokens;
            pthread_mutex_unlock(&ctx->mu);
            std::free(rgb);
            return true;
        }
    }

    // Pick empty or LRU slot
    int slot = 0;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < V_CACHE_MAX; ++i) {
        if (!ctx->cache[i].valid) { slot = i; break; }
        if (ctx->cache[i].last_use < oldest) { oldest = ctx->cache[i].last_use; slot = i; }
    }

    float* embeds = static_cast<float*>(std::malloc(size_t(num_tokens) * V_OUT_DIM * sizeof(float)));
    int32_t* tids = static_cast<int32_t*>(std::malloc(size_t(num_tokens) * sizeof(int32_t)));
    if (!embeds || !tids) {
        std::free(embeds); std::free(tids); std::free(rgb);
        pthread_mutex_unlock(&ctx->mu);
        return false;
    }
    for (int k = 0; k < num_tokens; ++k) tids[k] = synth_token_id(hash, k);

    bool ok = encode_rgb_on_gpu(ctx, rgb, h, w, stream, embeds);
    std::free(rgb);
    if (!ok) {
        std::free(embeds); std::free(tids);
        pthread_mutex_unlock(&ctx->mu);
        return false;
    }

    std::free(ctx->cache[slot].token_ids);
    std::free(ctx->cache[slot].embeds_f32);
    ctx->cache[slot].valid = true;
    ctx->cache[slot].hash = hash;
    ctx->cache[slot].last_use = ++ctx->lru_clock;
    ctx->cache[slot].num_tokens = num_tokens;
    ctx->cache[slot].token_ids = tids;
    ctx->cache[slot].embeds_f32 = embeds;

    std::memcpy(out_token_ids, tids, size_t(num_tokens) * sizeof(int32_t));
    *out_embeds_ptr = embeds;
    *out_num_tokens = num_tokens;
    pthread_mutex_unlock(&ctx->mu);
    return true;
}

} // namespace

extern "C" {

void* squid_vision_init(const char* model_dir) {
    if (!model_dir || !*model_dir) return nullptr;
    char st_path[1200];
    std::snprintf(st_path, sizeof(st_path), "%s/model-00001-of-00001.safetensors", model_dir);
    FILE* f = std::fopen(st_path, "rb");
    if (!f) return nullptr;

    VisIndex* idx = static_cast<VisIndex*>(std::calloc(1, sizeof(VisIndex)));
    if (!idx) { std::fclose(f); return nullptr; }
    if (!parse_safetensors_visual_index(f, idx)) {
        std::free(idx);
        std::fclose(f);
        return nullptr;
    }

    VisionContext* ctx = static_cast<VisionContext*>(std::calloc(1, sizeof(VisionContext)));
    if (!ctx) { std::free(idx); std::fclose(f); return nullptr; }
    pthread_mutex_init(&ctx->mu, nullptr);

    ctx->patch_w = load_bf16(f, idx, "model.visual.backbone.embeddings.patch_embedding.weight", size_t(V_HIDDEN) * V_PATCH_DIM);
    ctx->pos = load_bf16_as_f32(f, idx, "model.visual.backbone.embeddings.position_embedding", size_t(V_POS_GRID) * V_POS_GRID * V_HIDDEN);
    ctx->cls = load_bf16_as_f32(f, idx, "model.visual.backbone.embeddings.class_embedding", V_HIDDEN);
    float* cls_pos = load_bf16_as_f32(f, idx, "model.visual.backbone.embeddings.cls_position_embedding", V_HIDDEN);
    bool ok = ctx->patch_w && ctx->pos && ctx->cls && cls_pos;
    if (ok) {
        for (int c = 0; c < V_HIDDEN; ++c) ctx->cls[c] += cls_pos[c];
    }
    std::free(cls_pos);

    ctx->ln_pre_w = load_bf16_as_f32(f, idx, "model.visual.backbone.ln_pre.weight", V_HIDDEN);
    ctx->ln_pre_b = load_bf16_as_f32(f, idx, "model.visual.backbone.ln_pre.bias", V_HIDDEN);
    ok = ok && ctx->ln_pre_w && ctx->ln_pre_b;

    char name[128];
    const char* qkv_names[3] = {"q_proj", "k_proj", "v_proj"};
    for (int i = 0; i < V_LAYERS && ok; ++i) {
        HostVisionLayer& L = ctx->layers[i];
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.ln_1.weight", i); L.ln1_w = load_bf16_as_f32(f, idx, name, V_HIDDEN);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.ln_1.bias", i);   L.ln1_b = load_bf16_as_f32(f, idx, name, V_HIDDEN);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.ln_2.weight", i); L.ln2_w = load_bf16_as_f32(f, idx, name, V_HIDDEN);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.ln_2.bias", i);   L.ln2_b = load_bf16_as_f32(f, idx, name, V_HIDDEN);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.ls_1.gamma", i);  L.ls1 = load_bf16_as_f32(f, idx, name, V_HIDDEN);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.ls_2.gamma", i);  L.ls2 = load_bf16_as_f32(f, idx, name, V_HIDDEN);

        L.wqkv = static_cast<uint16_t*>(std::malloc(size_t(3) * V_HIDDEN * V_HIDDEN * 2));
        L.bqkv = static_cast<float*>(std::malloc(size_t(3) * V_HIDDEN * sizeof(float)));
        ok = ok && L.ln1_w && L.ln1_b && L.ln2_w && L.ln2_b && L.ls1 && L.ls2 && L.wqkv && L.bqkv;
        for (int p = 0; p < 3 && ok; ++p) {
            std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.self_attn.%s.weight", i, qkv_names[p]);
            ok = ok && load_bf16_into(f, idx, name, size_t(V_HIDDEN) * V_HIDDEN, L.wqkv + size_t(p) * V_HIDDEN * V_HIDDEN);
            std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.self_attn.%s.bias", i, qkv_names[p]);
            float* bp = load_bf16_as_f32(f, idx, name, V_HIDDEN);
            if (!bp) { ok = false; break; }
            std::memcpy(L.bqkv + size_t(p) * V_HIDDEN, bp, V_HIDDEN * sizeof(float));
            std::free(bp);
        }
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.self_attn.out_proj.weight", i); L.wo = load_bf16(f, idx, name, size_t(V_HIDDEN) * V_HIDDEN);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.self_attn.out_proj.bias", i);   L.bo = load_bf16_as_f32(f, idx, name, V_HIDDEN);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.mlp.c_fc.weight", i);           L.fc = load_bf16(f, idx, name, size_t(V_MLP_DIM) * V_HIDDEN);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.mlp.c_fc.bias", i);             L.bfc = load_bf16_as_f32(f, idx, name, V_MLP_DIM);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.mlp.c_proj.weight", i);         L.proj = load_bf16(f, idx, name, size_t(V_HIDDEN) * V_MLP_DIM);
        std::snprintf(name, sizeof(name), "model.visual.backbone.layers.%d.mlp.c_proj.bias", i);           L.bproj = load_bf16_as_f32(f, idx, name, V_HIDDEN);
        ok = ok && L.wo && L.bo && L.fc && L.bfc && L.proj && L.bproj;
    }

    ctx->proj_in = load_bf16(f, idx, "model.visual.proj_in.weight", size_t(V_MERGED) * V_MERGED);
    ctx->proj_out = load_bf16(f, idx, "model.visual.proj_out.weight", size_t(V_OUT_DIM) * V_MERGED);
    ok = ok && ctx->proj_in && ctx->proj_out;

    std::free(idx);
    std::fclose(f);
    if (!ok) {
        std::fprintf(stderr, "[vision] Failed loading vision weights from %s\n", st_path);
        free_vision_ctx(ctx);
        return nullptr;
    }

    for (int i = 0; i < V_ROPE_FREQS; ++i) {
        ctx->inv_freq[i] = 1.0f / std::pow(10000.0f, float(2 * i) / float(2 * V_ROPE_FREQS));
    }

    std::fprintf(stdout, "[vision] Loaded Inkling-2 vision tower in host RAM (607 MiB RAM, 0 MiB permanent VRAM, 23 layers, LRU=%d)\n", V_CACHE_MAX);
    std::fflush(stdout);
    return ctx;
}

void squid_vision_free(void* handle) {
    free_vision_ctx(static_cast<VisionContext*>(handle));
}

int squid_vision_encode_data_uri(void* handle, const char* uri, size_t uri_len, int max_tokens,
                                 void* stream, int32_t* out_token_ids, int max_out_ids,
                                 const float** out_embeds_ptr, int* out_num_tokens) {
    if (!handle || !uri || uri_len == 0 || !out_token_ids || !out_embeds_ptr || !out_num_tokens) return -1;
    if (!starts_with_nocase(uri, uri_len, "data:")) return -2;
    const char* comma = static_cast<const char*>(std::memchr(uri, ',', uri_len));
    if (!comma) return -3;
    const size_t header_len = size_t(comma - (uri + 5));
    if (header_len < 7 || !starts_with_nocase(comma - 7, 7, ";base64")) return -4;

    const char* b64 = comma + 1;
    const size_t b64_len = uri_len - size_t(b64 - uri);
    size_t raw_len = 0;
    uint8_t* raw = decode_base64(b64, b64_len, &raw_len);
    if (!raw) return -5;

    bool ok = decode_and_encode_bytes(static_cast<VisionContext*>(handle), raw, raw_len, max_tokens,
                                      static_cast<cudaStream_t>(stream), out_token_ids, max_out_ids,
                                      out_embeds_ptr, out_num_tokens);
    std::free(raw);
    return ok ? 0 : -6;
}

int squid_vision_encode_file(void* handle, const char* path, int max_tokens,
                             void* stream, int32_t* out_token_ids, int max_out_ids,
                             const float** out_embeds_ptr, int* out_num_tokens) {
    if (!handle || !path || !*path || !out_token_ids || !out_embeds_ptr || !out_num_tokens) return -1;
    FILE* f = std::fopen(path, "rb");
    if (!f) return -2;
    if (fseeko(f, 0, SEEK_END) != 0) { std::fclose(f); return -3; }
    off_t sz = ftello(f);
    if (sz <= 0 || sz > (64LL << 20)) { std::fclose(f); return -3; }
    fseeko(f, 0, SEEK_SET);
    uint8_t* raw = static_cast<uint8_t*>(std::malloc(size_t(sz)));
    if (!raw) { std::fclose(f); return -4; }
    if (std::fread(raw, 1, size_t(sz), f) != size_t(sz)) { std::free(raw); std::fclose(f); return -4; }
    std::fclose(f);

    bool ok = decode_and_encode_bytes(static_cast<VisionContext*>(handle), raw, size_t(sz), max_tokens,
                                      static_cast<cudaStream_t>(stream), out_token_ids, max_out_ids,
                                      out_embeds_ptr, out_num_tokens);
    std::free(raw);
    return ok ? 0 : -5;
}

} // extern "C"

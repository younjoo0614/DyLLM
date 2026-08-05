#include "common.h"

#include <cuda_bf16.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>
#include <tuple>

struct BlockInfo {
  __device__ BlockInfo(const int* cu_seqlens_q, const int* cu_seqlens_k, const int* cu_salientlens, int batch_id)
      : cu_seqlens_q_curr(cu_seqlens_q[batch_id]), cu_seqlens_q_next(cu_seqlens_q[batch_id + 1]),
        cu_seqlens_k_curr(cu_seqlens_k[batch_id]), cu_seqlens_k_next(cu_seqlens_k[batch_id + 1]),
        salientlen(cu_salientlens[batch_id + 1] - cu_salientlens[batch_id]),
        seqlen_q(cu_seqlens_q_next - cu_seqlens_q_curr), seqlen_kv(cu_seqlens_k_next - cu_seqlens_k_curr) {}
  const int cu_seqlens_q_curr;
  const int cu_seqlens_q_next;
  const int cu_seqlens_k_curr;
  const int cu_seqlens_k_next;
  const int salientlen;
  const int seqlen_q;
  const int seqlen_kv;
};

__global__ void cosine_similarity_reduce_kernel(const float* __restrict__ stats, // [total_tokens, 3]
                                                bool* __restrict__ out,          // [total_tokens]
                                                const int total_tokens, const int* __restrict__ cu_seqlens_q,
                                                const float threshold) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_tokens)
    return;

  // Use __ldg to hint read-only cache (though compiler usually does this)
  float dot = stats[idx * 3 + 0];
  float norm_a = stats[idx * 3 + 1];
  float norm_b = stats[idx * 3 + 2];

  // Use fast math intrinsics
  float rsqrt_norm_a = rsqrtf(fmaxf(norm_a, 1e-8f));
  float rsqrt_norm_b = rsqrtf(fmaxf(norm_b, 1e-8f));

  float cos_sim = dot * rsqrt_norm_a * rsqrt_norm_b;
  out[idx] = (cos_sim < threshold);
}

void compute_cosine_similarity(const float* stats, bool* out, const int total_tokens, const int* cu_seqlens_q,
                               const float threshold) {
  int threads = 256;
  int blocks = (total_tokens + threads - 1) / threads;
  cosine_similarity_reduce_kernel<<<blocks, threads, 0>>>(stats, out, total_tokens, cu_seqlens_q, threshold);
}

__global__ void overwrite_salient_kernel(const nv_bfloat16* __restrict__ c,
                                          const nv_bfloat16* __restrict__ o_sal,
                                          nv_bfloat16* __restrict__ o,
                                          const int* __restrict__ idx_salient_row,
                                          float* __restrict__ cosine_stats, const int vector_size) {
  const int salient_idx = blockIdx.x;
  const int token_idx = idx_salient_row[salient_idx];
  const int pair_count = vector_size / 2;
  const nv_bfloat162* old_row = reinterpret_cast<const nv_bfloat162*>(c + token_idx * vector_size);
  const nv_bfloat162* new_row = reinterpret_cast<const nv_bfloat162*>(o_sal + salient_idx * vector_size);
  nv_bfloat162* out_row = reinterpret_cast<nv_bfloat162*>(o + token_idx * vector_size);

  float3 local = {0.0f, 0.0f, 0.0f};
  for (int pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
    const nv_bfloat162 old_v = old_row[pair];
    const nv_bfloat162 new_v = new_row[pair];
    out_row[pair] = new_v;
    const float2 old_f = __bfloat1622float2(old_v);
    const float2 new_f = __bfloat1622float2(new_v);
    local.x += new_f.x * old_f.x + new_f.y * old_f.y;
    local.y += new_f.x * new_f.x + new_f.y * new_f.y;
    local.z += old_f.x * old_f.x + old_f.y * old_f.y;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local.x += __shfl_down_sync(0xffffffff, local.x, offset);
    local.y += __shfl_down_sync(0xffffffff, local.y, offset);
    local.z += __shfl_down_sync(0xffffffff, local.z, offset);
  }

  __shared__ float3 warp_stats[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_stats[warp] = local;
  __syncthreads();

  if (warp == 0) {
    local = lane < blockDim.x / 32 ? warp_stats[lane] : make_float3(0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local.x += __shfl_down_sync(0xffffffff, local.x, offset);
      local.y += __shfl_down_sync(0xffffffff, local.y, offset);
      local.z += __shfl_down_sync(0xffffffff, local.z, offset);
    }
    if (lane == 0) {
      cosine_stats[token_idx * 3 + 0] = local.x;
      cosine_stats[token_idx * 3 + 1] = local.y;
      cosine_stats[token_idx * 3 + 2] = local.z;
    }
  }
}

__global__ void scatter_idx_map_kernel(const int* idx_salient_row, const int* cu_salientlens,
                                       const int* cu_seqlen_to_use, int* idx_map, const int num_salient,
                                       const int batch_size, const int max_seqlen_padded) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_salient)
    return;

  int l = 0;
  int r = batch_size;
  int b = 0;

  while (l < r) {
    int mid = (l + r) >> 1;
    if (cu_salientlens[mid + 1] <= idx) {
      l = mid + 1;
    } else {
      r = mid;
    }
  }
  b = l;

  int global_idx = idx_salient_row[idx];
  int seq_start = cu_seqlen_to_use[b];
  int local_idx = global_idx - seq_start;
  int sal_start = cu_salientlens[b];
  int rank = idx - sal_start;

  if (local_idx >= 0 && local_idx < max_seqlen_padded) {
    idx_map[b * max_seqlen_padded + local_idx] = rank;
  }
}

__global__ void compute_block_mask_kernel(const int* idx_map, bool* block_mask, int total_blocks, int block_k) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_blocks)
    return;

  int start_offset = idx * block_k;
  bool exists = false;

  // Check if any element in this block is valid (!= -1)
  for (int i = 0; i < block_k; ++i) {
    if (idx_map[start_offset + i] != -1) {
      exists = true;
      break;
    }
  }
  block_mask[idx] = exists;
}

__global__ void compute_k_block_mask_kernel(const int* __restrict__ idx_salient_row,
                                            const int* __restrict__ cu_salientlens,
                                            const int* __restrict__ cu_seqlens_k, uint64_t* __restrict__ row_masks,
                                            const int total_blocks, const int num_blk_k, const int block_k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_blocks)
    return;

  const int batch_id = idx / num_blk_k;
  const int block_id = idx - batch_id * num_blk_k;
  const int seq_start = cu_seqlens_k[batch_id];
  const int seq_end = cu_seqlens_k[batch_id + 1];
  const int block_start = seq_start + block_id * block_k;
  if (block_start >= seq_end) {
    row_masks[idx] = 0;
    return;
  }
  const int block_end = min(block_start + block_k, seq_end);

  int left = cu_salientlens[batch_id];
  int right = cu_salientlens[batch_id + 1];
  while (left < right) {
    const int mid = (left + right) >> 1;
    if (idx_salient_row[mid] < block_start)
      left = mid + 1;
    else
      right = mid;
  }
  uint64_t mask = 0;
  const int salient_end = cu_salientlens[batch_id + 1];
  while (left < salient_end && idx_salient_row[left] < block_end) {
    mask |= 1ULL << (idx_salient_row[left] - block_start);
    ++left;
  }
  row_masks[idx] = mask;
}

void compute_k_masks(const int batch_size, const int* cu_seqlens_k, const int* cu_salientlens_k,
                     const int* idx_salient_row_k, const int num_salient_k, const int BLOCK_K, const int max_seqlen_k,
                     uint64_t* row_masks, int* idx_map_k) {

  int num_blk_k = (max_seqlen_k + BLOCK_K - 1) / BLOCK_K;
  const int total_blocks = batch_size * num_blk_k;
  constexpr int threads = 256;
  const int blocks = cdiv(total_blocks, threads);
  compute_k_block_mask_kernel<<<blocks, threads>>>(idx_salient_row_k, cu_salientlens_k, cu_seqlens_k, row_masks,
                                                   total_blocks, num_blk_k, BLOCK_K);
}

void compute_q_idx_map(const int batch_size, const int* cu_seqlens_q, const int* cu_salientlens_q,
                       const int* idx_salient_row_q, const int num_salient_q, const int BLOCK_Q, const int max_seqlen_q,
                       int* idx_map_q) {

  int num_blk_q = (max_seqlen_q + BLOCK_Q - 1) / BLOCK_Q;
  int max_seqlen_q_padded = num_blk_q * BLOCK_Q;

  if (num_salient_q > 0) {
    int threads = 256;
    int blocks = (num_salient_q + threads - 1) / threads;

    scatter_idx_map_kernel<<<blocks, threads>>>(idx_salient_row_q, cu_salientlens_q, cu_seqlens_q, idx_map_q,
                                                num_salient_q, batch_size, max_seqlen_q_padded);
  }
}

template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, bool MASK_V_ROWS>
__launch_bounds__(NUM_WARPS* WARP_SIZE) __global__
    void attention_sparse_varlen_kernel(const nv_bfloat16* Q,     // [sum(len_q), H_q, D]
                                        const nv_bfloat16* K,     // [sum(len_kv), H_kv, D]
                                        const nv_bfloat16* V,     // [sum(len_kv), H_kv, D]
                                        const nv_bfloat16* C,     // [sum(len_kv), H_q, D]
                                        const nv_bfloat16* O_sal, // [sum(salientlen), H_q, D]
                                        nv_bfloat16* O,           // [sum(len_q), H_q, D]
                                        const int B, const int H, const int H_kv, const int* cu_seqlens_q,
                                        const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k,
                                        const int* cu_salientlens, const uint64_t* row_masks, const int* idx_map_k,
                                        const int* idx_map_q, float* cosine_stats) {

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int num_blk_q = cdiv(max_seqlen_q, BLOCK_Q);
  const int num_blk_kv = cdiv(max_seqlen_k, BLOCK_KV);
  const int batch_id = (bid / (H * num_blk_q));
  const int head_id = (bid % (H * num_blk_q)) / num_blk_q;
  const int blk_q_id = bid % num_blk_q;

  // GQA: map query head to kv head
  const int num_queries_per_kv = H / H_kv;
  const int kv_head_id = head_id / num_queries_per_kv;

  BlockInfo binfo(cu_seqlens_q, cu_seqlens_k, cu_salientlens, batch_id);

  const int num_kv_iter = cdiv(binfo.seqlen_kv, BLOCK_KV);

  const uint64_t* row_masks_batch = row_masks + batch_id * num_blk_kv;

  const int ldg_q = H * DIM;     // Q, C, O use num_q_heads
  const int ldg_kv = H_kv * DIM; // K, V use num_kv_heads

  // QKVO offset
  const int batch_offset_q = binfo.cu_seqlens_q_curr * ldg_q;
  const int block_offset_q = blk_q_id * BLOCK_Q * ldg_q;
  const int head_offset_q = head_id * DIM;
  const int head_offset_kv = kv_head_id * DIM;
  const int batch_offset_kv = binfo.cu_seqlens_k_curr * ldg_kv;

  if (batch_offset_q + block_offset_q >= binfo.cu_seqlens_q_next * ldg_q)
    return;

  // each threadblock handles 1 BLOCK_Q

  Q += batch_offset_q + block_offset_q + head_offset_q;
  K += batch_offset_kv + head_offset_kv;
  V += batch_offset_kv + head_offset_kv;
  C += batch_offset_q + block_offset_q + head_offset_q;
  O_sal += cu_salientlens[batch_id] * ldg_q + head_offset_q;
  O += batch_offset_q + block_offset_q + head_offset_q;

  // exit if out of bound

  // we overlap Q_smem with (K_smem + V_smem), since we only need to load Q_smem once
  extern __shared__ nv_bfloat16 smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem; // double buffer for K
  const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);

  // FA2: shard BLOCK_Q among all warps
  // replicate K and V on all warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // mma.m16n8k16
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  // set up registers
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

  // let compiler decide register reuse?
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

  // rescale O_rmem once we obtain new rowmax, then accumulate to O_rmem for P @ V
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

  // pre-compute address and swizzling for ldmatrix
  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    // A tile
    const int row_off = warp_id * WARP_Q + (lane_id % 16);
    const int col_off = lane_id / 16 * 8;
    Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // B tile
    const int row_off = lane_id % 8;
    const int col_off = lane_id / 8 * 8;
    K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // B tile trans
    const int row_off = lane_id % 16;
    const int col_off = lane_id / 16 * 8;
    V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }

  const float softmax_scale = rsqrtf(static_cast<float>(DIM));

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // load Q [BLOCK_Q, DIM]
  global_to_shared_swizzle_zero_pad<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, ldg_q, tid, blk_q_id, 0, binfo.seqlen_q);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // shared -> registers
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      uint32_t addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16); // row
      addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);       // col
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  // we need a syncthreads() here so that we don't load K global->shared
  // before finishing loading Q shared->reg
  __syncthreads();

  auto load_KV = [&](int kv_id, uint64_t row_mask) {
    if (kv_id < num_kv_iter) {
      // Load K
      const uint32_t dst_k = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
      const nv_bfloat16* src_k = K + kv_id * BLOCK_KV * ldg_kv;
      global_to_shared_swizzle_zero_pad<BLOCK_KV, DIM, TB_SIZE>(dst_k, src_k, ldg_kv, tid, kv_id, 0, binfo.seqlen_kv);

      // Load V
      if (row_mask != 0) {
        const uint32_t dst_v = V_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        const nv_bfloat16* src_v = V + kv_id * BLOCK_KV * ldg_kv;
        if constexpr (MASK_V_ROWS) {
          global_to_shared_swizzle_zero_pad_with_row_mask<BLOCK_KV, DIM, TB_SIZE>(
              dst_v, src_v, ldg_kv, tid, kv_id, 0, binfo.seqlen_kv, row_mask);
        } else {
          // V is already a sparse delta tensor, so a direct tile copy is exact.
          global_to_shared_swizzle_zero_pad<BLOCK_KV, DIM, TB_SIZE>(dst_v, src_v, ldg_kv, tid, kv_id, 0,
                                                                    binfo.seqlen_kv);
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  };

  // prefetch K and V
  uint64_t current_row_mask = num_kv_iter > 0 ? row_masks_batch[0] : 0;
  load_KV(0, current_row_mask);

  // sparse attention
  // full attention
  float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    // Issue next block
    const uint64_t next_row_mask = kv_id + 1 < num_kv_iter ? row_masks_batch[kv_id + 1] : 0;
    load_KV(kv_id + 1, next_row_mask);

    // Wait for current block
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    const bool sparse_block = current_row_mask == 0;
    current_row_mask = next_row_mask;
    // bool sparse_block = false; // for debugging

    // K shared -> registers
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16); // row
        addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);        // col
        ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d], S_rmem[mma_id_q][mma_id_kv]);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // apply softmax scale
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        for (int reg_id = 0; reg_id < 4; reg_id++)
          S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

        // K is zero-padded in shared memory, but padded logits must not
        // participate in the softmax denominator. Each lane owns two adjacent
        // columns for each of its two MMA output rows.
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        const int key_col = kv_id * BLOCK_KV + mma_id_kv * MMA_N + (lane_id % 4) * 2;
        if (key_col >= binfo.seqlen_kv) {
          regs[0] = -FLT_MAX;
          regs[2] = -FLT_MAX;
        }
        if (key_col + 1 >= binfo.seqlen_kv) {
          regs[1] = -FLT_MAX;
          regs[3] = -FLT_MAX;
        }
      }

      // rowmax
      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        if (mma_id_kv == 0) {
          this_rowmax[0] = max(regs[0], regs[1]); // c0 and c1
          this_rowmax[1] = max(regs[2], regs[3]); // c2 and c3
        } else {
          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1])); // c0 and c1
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3])); // c2 and c3
        }
      }

      // butterfly reduction within 4 threads
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      // new rowmax
      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      // rescale for previous O
      float rescale[2];
      rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      // save new rowmax
      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      // rowsumexp
      float this_rowsumexp[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]); // c0
        regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]); // c1
        regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]); // c2
        regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]); // c3

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = regs[0] + regs[1];
          this_rowsumexp[1] = regs[2] + regs[3];
        } else {
          this_rowsumexp[0] += regs[0] + regs[1];
          this_rowsumexp[1] += regs[2] + regs[3];
        }

        // pack to P registers for next MMA
        // we need to change from m16n8 to m16k16
        nv_bfloat162* this_P_rmem = reinterpret_cast<nv_bfloat162*>(P_rmem[mma_id_q][mma_id_kv / 2]);
        if (sparse_block)
          continue;
        this_P_rmem[(mma_id_kv % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
        this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
      }

      // butterfly reduction within 4 threads
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // accumulate to total rowsumexp
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }

    // V shared -> registers
    if (sparse_block)
      continue;

    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
        uint32_t addr = V_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16); // row
        addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);        // col
        ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA O += P @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d], O_rmem[mma_id_q][mma_id_d]);
  }

  // write to O
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
    const int global_row_0 = binfo.cu_seqlens_q_curr + blk_q_id * BLOCK_Q + row;
    if (global_row_0 >= binfo.cu_seqlens_q_next)
      continue;

    const bool valid_1 = global_row_0 + 8 < binfo.cu_seqlens_q_next;
    const float inv_rowsum_0 = __frcp_rn(rowsumexp[mma_id_q][0]);
    const float inv_rowsum_1 = valid_1 ? __frcp_rn(rowsumexp[mma_id_q][1]) : 0.0f;

    float3 stats_0 = {0.0f, 0.0f, 0.0f};
    float3 stats_1 = {0.0f, 0.0f, 0.0f};
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      nv_bfloat162 old_c_0, old_c_1, new_c_0, new_c_1;
      float2 old_c_0_f2, old_c_1_f2, new_c_0_f2, new_c_1_f2;
      old_c_0 = reinterpret_cast<const nv_bfloat162*>(C + (row + 0) * ldg_q + col)[0];
      old_c_0_f2 = __bfloat1622float2(old_c_0);

      if (valid_1) {
        old_c_1 = reinterpret_cast<const nv_bfloat162*>(C + (row + 8) * ldg_q + col)[0];
        old_c_1_f2 = __bfloat1622float2(old_c_1);
      }

      float* regs = O_rmem[mma_id_q][mma_id_d];

      // Compute the sparse delta for every row. Salient rows are overwritten
      // in one compact post-kernel after all head CTAs complete.
      regs[0] *= inv_rowsum_0;
      regs[1] *= inv_rowsum_0;
      new_c_0_f2 = {old_c_0_f2.x + regs[0], old_c_0_f2.y + regs[1]};
      new_c_0 = __float22bfloat162_rn(new_c_0_f2);

      // Handle second vector (row + 8)
      if (valid_1) {
        regs[2] *= inv_rowsum_1;
        regs[3] *= inv_rowsum_1;
        new_c_1_f2 = {old_c_1_f2.x + regs[2], old_c_1_f2.y + regs[3]};
        new_c_1 = __float22bfloat162_rn(new_c_1_f2);
      }

      stats_0.x += new_c_0_f2.x * old_c_0_f2.x + new_c_0_f2.y * old_c_0_f2.y;
      stats_0.y += new_c_0_f2.x * new_c_0_f2.x + new_c_0_f2.y * new_c_0_f2.y;
      stats_0.z += old_c_0_f2.x * old_c_0_f2.x + old_c_0_f2.y * old_c_0_f2.y;

      if (valid_1) {
        stats_1.x += new_c_1_f2.x * old_c_1_f2.x + new_c_1_f2.y * old_c_1_f2.y;
        stats_1.y += new_c_1_f2.x * new_c_1_f2.x + new_c_1_f2.y * new_c_1_f2.y;
        stats_1.z += old_c_1_f2.x * old_c_1_f2.x + old_c_1_f2.y * old_c_1_f2.y;
      }

      // store O
      reinterpret_cast<nv_bfloat162*>(O + (row + 0) * ldg_q + col)[0] = new_c_0;
      if (valid_1) {
        reinterpret_cast<nv_bfloat162*>(O + (row + 8) * ldg_q + col)[0] = new_c_1;
      }
    }

// Butterfly reduction for 4 threads (lane_id % 4)
#pragma unroll
    for (int i = 1; i <= 2; i *= 2) {
      stats_0.x += __shfl_xor_sync(0xffffffff, stats_0.x, i);
      stats_0.y += __shfl_xor_sync(0xffffffff, stats_0.y, i);
      stats_0.z += __shfl_xor_sync(0xffffffff, stats_0.z, i);

      stats_1.x += __shfl_xor_sync(0xffffffff, stats_1.x, i);
      stats_1.y += __shfl_xor_sync(0xffffffff, stats_1.y, i);
      stats_1.z += __shfl_xor_sync(0xffffffff, stats_1.z, i);
    }

    // reduce among thread blocks and store to global memory
    if (lane_id % 4 == 0) {
      int global_token_idx_0 = global_row_0;

      if (global_token_idx_0 < binfo.cu_seqlens_q_next) {
        atomicAdd(&cosine_stats[global_token_idx_0 * 3 + 0], stats_0.x);
        atomicAdd(&cosine_stats[global_token_idx_0 * 3 + 1], stats_0.y);
        atomicAdd(&cosine_stats[global_token_idx_0 * 3 + 2], stats_0.z);
      }

      int global_token_idx_1 = global_token_idx_0 + 8;
      if (global_token_idx_1 < binfo.cu_seqlens_q_next) {
        atomicAdd(&cosine_stats[global_token_idx_1 * 3 + 0], stats_1.x);
        atomicAdd(&cosine_stats[global_token_idx_1 * 3 + 1], stats_1.y);
        atomicAdd(&cosine_stats[global_token_idx_1 * 3 + 2], stats_1.z);
      }
    }
  }
}

template <typename scalar_t>
void attention_sparse_varlen(const scalar_t* q,     // [sum(len_q), H_q, D]
                             const scalar_t* k,     // [sum(len_kv), H_kv, D]
                             const scalar_t* v,     // [sum(len_kv), H_kv, D]
                             const scalar_t* c,     // [sum(len_kv), H_q, D]
                             const scalar_t* o_sal, // [sum(salientlen), H_q, D]
                             scalar_t* o,           // [sum(len_q), H_q, D]
                             const int B, const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                             const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                             const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                             const int* idx_salient_row_q, uint64_t* row_masks, int* idx_map_k, int* idx_map_q,
                             float* cosine_stats, // [sum(len_q), 3] - intermediate buffer, must be zero-initialized
                             bool* cosine_out,    // [sum(len_q)] - output bool mask (true if cos_sim < threshold)
                             float threshold, int dim) {

  // Only support BFloat16 for now
  if constexpr (!std::is_same_v<scalar_t, at::BFloat16>) {
    std::cerr << "Only BFloat16 is supported" << std::endl;
    exit(1);
  }

  const nv_bfloat16* q_ptr = reinterpret_cast<const nv_bfloat16*>(q);
  const nv_bfloat16* k_ptr = reinterpret_cast<const nv_bfloat16*>(k);
  const nv_bfloat16* v_ptr = reinterpret_cast<const nv_bfloat16*>(v);
  const nv_bfloat16* c_ptr = reinterpret_cast<const nv_bfloat16*>(c);
  const nv_bfloat16* o_sal_ptr = reinterpret_cast<const nv_bfloat16*>(o_sal);
  nv_bfloat16* o_ptr = reinterpret_cast<nv_bfloat16*>(o);

  const int BLOCK_Q = 64;
  const int BLOCK_KV = 64;
  const int NUM_WARPS = 4;

  const int num_blocks = B * H * cdiv(max_seqlen_q, BLOCK_Q);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;

  compute_k_masks(B, cu_seqlens_k, cu_salientlens, idx_salient_row_k, num_salient, BLOCK_KV, max_seqlen_k, row_masks,
                  idx_map_k);

  auto launch_kernel_fn = [&](auto kernel, int dim_val) {
    const int smem_size = max(BLOCK_Q, BLOCK_KV * 4) * dim_val * sizeof(nv_bfloat16);
    launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, q_ptr, k_ptr, v_ptr, c_ptr, o_sal_ptr, o_ptr, B, H, H_kv,
                  cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, cu_salientlens, row_masks, idx_map_k,
                  idx_map_q, cosine_stats);
  };

  // On H100, masked zero-fill wins for short grids where K/V bandwidth and
  // launch underfill dominate; direct copies have lower instruction overhead
  // once there are enough CTAs to saturate the GPU.
  const bool mask_v_rows = num_blocks <= 1024;
  if (dim == 64) {
    if (mask_v_rows)
      launch_kernel_fn(attention_sparse_varlen_kernel<BLOCK_Q, BLOCK_KV, 64, NUM_WARPS, true>, 64);
    else
      launch_kernel_fn(attention_sparse_varlen_kernel<BLOCK_Q, BLOCK_KV, 64, NUM_WARPS, false>, 64);
  } else if (dim == 128) {
    if (mask_v_rows)
      launch_kernel_fn(attention_sparse_varlen_kernel<BLOCK_Q, BLOCK_KV, 128, NUM_WARPS, true>, 128);
    else
      launch_kernel_fn(attention_sparse_varlen_kernel<BLOCK_Q, BLOCK_KV, 128, NUM_WARPS, false>, 128);
  } else {
    std::cerr << "Unsupported dim=" << dim << std::endl;
    exit(1);
  }

  if (num_salient > 0) {
    overwrite_salient_kernel<<<num_salient, 256>>>(c_ptr, o_sal_ptr, o_ptr, idx_salient_row_q, cosine_stats, H * dim);
  }

  compute_cosine_similarity(cosine_stats, cosine_out, total_seqlen_q, cu_seqlens_q, threshold);
}

// Explicit instantiation
template void attention_sparse_varlen<at::BFloat16>(
    const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v, const at::BFloat16* c,
    const at::BFloat16* o_sal, at::BFloat16* o, const int B, const int H, const int H_kv, const int* cu_seqlens_q,
    const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
    const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
    uint64_t* row_masks, int* idx_map_k, int* idx_map_q, float* cosine_stats, bool* cosine_out, float threshold,
    int dim);

__global__ void fused_offset_kernel(const int* idxs, const int* num_idxs_ptr, const int* cu_promptlens,
                                    const int* cu_salientlens, const int num_groups, long* out_tensor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int limit = *num_idxs_ptr;
  if (tid >= limit)
    return;

  int left = 0;
  int right = num_groups;
  int group_idx = 0;

  while (left < right) {
    int mid = left + (right - left) / 2;
    if (cu_salientlens[mid + 1] <= tid) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  group_idx = left;

  long offset_val = (long)cu_promptlens[group_idx + 1];

  out_tensor[tid] = (long)idxs[tid] + offset_val;
}

void fused_offset_launch(torch::Tensor idxs, torch::Tensor num_idxs, torch::Tensor cu_promptlens,
                         torch::Tensor cu_salientlens, torch::Tensor out_tensor, int max_threads) {
  const int threads = 256;
  const int blocks = (max_threads + threads - 1) / threads;

  fused_offset_kernel<<<blocks, threads>>>(idxs.data_ptr<int>(), num_idxs.data_ptr<int>(),
                                           cu_promptlens.data_ptr<int>(), cu_salientlens.data_ptr<int>(),
                                           cu_promptlens.size(0) - 1, out_tensor.data_ptr<long>());
}

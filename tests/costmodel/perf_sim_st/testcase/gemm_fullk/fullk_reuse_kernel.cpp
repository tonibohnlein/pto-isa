/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Full-K operand-reuse GEMM kernel: validates the full-K traffic model and the
// bandwidth-weighted stationary-operand choice (L1->L0A 441 GB/s vs L1->L0B 220.5).
//
//   full-K  : k == K, so the whole reduction lives in one L0 block (kLoop == 1).
//   reuse   : one operand panel is held stationary in L0 and reused across the
//             orthogonal output dimension; only the moving operand is re-extracted.
//
//   A-stationary: outer i (rows), inner j (cols). Extract A[m,K] once per row,
//                 stream B[K,n] per (i,j).   MTE1_A = mLoop,  MTE1_B = mLoop*nLoop.
//   B-stationary: outer j (cols), inner i (rows). Extract B[K,n] once per col,
//                 stream A[m,K] per (i,j).   MTE1_B = nLoop,  MTE1_A = mLoop*nLoop.
//
// Single-buffered (no ping-pong): we read per-pipe BUSY cycles, which accumulate
// per op independent of overlap, so single buffering does not bias the MTE1 metric.
// The split-K / no-reuse baseline is RunGemmE2E with baseK == K (re-extracts both).

#include "gemm_performance_kernel.cpp"

using namespace pto;

template <typename T, typename U, typename S, uint32_t M, uint32_t K, uint32_t N, uint32_t baseM, uint32_t baseN,
          bool AStationary>
AICORE inline void RunGemmFullKReuse(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr uint32_t baseK = K; // full-K: the entire reduction dim fits one L0 block
    constexpr uint32_t mLoop = M / baseM;
    constexpr uint32_t nLoop = N / baseN;

    using TileMatA = Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor>;
    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + baseM * baseK * sizeof(U));
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using NDValidShapeA = TileShape2D<U, baseM, baseK, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, M, K, Layout::ND>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;

    using NDValidShapeB = TileShape2D<U, baseK, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, K, N, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    using NDValidShapeC = TileShape2D<T, baseM, baseN, Layout::ND>;
    using NDWholeShapeC = BaseShape2D<T, M, N, Layout::ND>;
    using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;

    // Prime the WAR flags consumed by the first iteration (no prior MAD / store yet).
    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_FIX, PIPE_M>(0);

    constexpr uint32_t outerLoop = AStationary ? mLoop : nLoop;
    constexpr uint32_t innerLoop = AStationary ? nLoop : mLoop;

    for (uint32_t o = 0; o < outerLoop; o++) {
        for (uint32_t in = 0; in < innerLoop; in++) {
            const uint32_t i = AStationary ? o : in;
            const uint32_t j = AStationary ? in : o;
            const bool refreshA = AStationary ? (in == 0) : true; // A held across cols when A-stationary
            const bool refreshB = AStationary ? true : (in == 0); // B held across rows when B-stationary

            // Wait until the previous MAD released L0A/L0B before overwriting them.
            WaitFlag<PIPE_M, PIPE_MTE1>(0);
            // Operands L1-resident (autotiler scope is L1->L0): only L1->L0 extracts.
            if (refreshA) {
                TEXTRACT(aTile, aMatTile, 0, 0);
            }
            if (refreshB) {
                TEXTRACT(bTile, bMatTile, 0, 0);
            }
            SetFlag<PIPE_MTE1, PIPE_M>(0);
            WaitFlag<PIPE_MTE1, PIPE_M>(0);

            // Wait until the previous store released L0C before accumulating into it.
            WaitFlag<PIPE_FIX, PIPE_M>(0);
            MatmulAcc(cTile, aTile, bTile, 0);
            SetFlag<PIPE_M, PIPE_MTE1>(0); // L0A/L0B free for next extract
            SetFlag<PIPE_M, PIPE_FIX>(0);
            WaitFlag<PIPE_M, PIPE_FIX>(0);

            GlobalDataOut dst(out + i * baseM * N + j * baseN);
            TSTORE(dst, cTile);
            SetFlag<PIPE_FIX, PIPE_M>(0); // L0C free for next MAD
        }
    }

    // Drain the primed flags.
    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_FIX, PIPE_M>(0);
}

template <uint32_t M, uint32_t K, uint32_t N, uint32_t baseM, uint32_t baseN, bool AStationary>
__global__ AICORE void GemmFullKReuse(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    RunGemmFullKReuse<float, half, half, M, K, N, baseM, baseN, AStationary>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

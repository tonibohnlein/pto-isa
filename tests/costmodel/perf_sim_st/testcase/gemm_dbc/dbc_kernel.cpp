/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// L0C double-buffering experiment: split-K GEMM with NUM_C in-flight L0C
// accumulators, ping-ponged across output tiles.
//
//   NUM_C == 1 : single L0C. The next tile's MAD must wait for this tile's
//                FIXPIPE drain (same buffer) -> the drain is EXPOSED on the cube.
//   NUM_C == 2 : double L0C. store(tile t) overlaps MAD(tile t+1) (other buffer)
//                -> the drain is HIDDEN, at the cost of halving the usable L0C
//                (each accumulator <= L0C/2).
//
// We read total_cycles (wall): the NUM_C=1 vs NUM_C=2 delta at a fixed tile
// isolates the exposed-FIXPIPE term; comparing the best DB tile against the
// best single-buffer (larger) tile answers whether enabling L0C-DB pays off.
// Operands are single-buffered identically in both, so only L0C buffering varies.

#include "gemm_performance_kernel.cpp"

using namespace pto;

template <typename T, typename U, typename S, uint32_t M, uint32_t K, uint32_t N, uint32_t baseM, uint32_t baseK,
          uint32_t baseN, uint32_t NUM_C>
AICORE inline void RunGemmSplitKDBC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr uint32_t mLoop = M / baseM;
    constexpr uint32_t nLoop = N / baseN;
    constexpr uint32_t kLoop = K / baseK;

    using TileMatA = Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor>;
    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    LeftTile aTile;
    RightTile bTile;
    ResTile cTile[NUM_C];

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + baseM * baseK * sizeof(U));
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    for (uint32_t b = 0; b < NUM_C; b++) {
        TASSIGN(cTile[b], b * (baseM * baseN * sizeof(T))); // distinct L0C slots
    }

    using NDValidShapeA = TileShape2D<U, baseM, baseK, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, M, K, Layout::ND>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;
    using NDValidShapeB = TileShape2D<U, baseK, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, K, N, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;
    using NDValidShapeC = TileShape2D<T, baseM, baseN, Layout::ND>;
    using NDWholeShapeC = BaseShape2D<T, M, N, Layout::ND>;
    using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;

    SetFlag<PIPE_M, PIPE_MTE1>(0);
    for (uint32_t b = 0; b < NUM_C; b++) {
        SetFlag<PIPE_FIX, PIPE_M>(b); // all L0C buffers start free
    }

    uint32_t t = 0;
    for (uint32_t i = 0; i < mLoop; i++) {
        for (uint32_t j = 0; j < nLoop; j++) {
            const uint32_t b = t % NUM_C;
            // accumulate into cTile[b] only once its previous store completed.
            WaitFlag<PIPE_FIX, PIPE_M>(b);
            for (uint32_t kk = 0; kk < kLoop; kk++) {
                WaitFlag<PIPE_M, PIPE_MTE1>(0); // operand slot free
                // Operands are L1-resident (the autotiler's scope is L1->L0): no GM->L1
                // TLOAD, only the L1->L0 TEXTRACT. This removes MTE2 from the wall-clock.
                TEXTRACT(aTile, aMatTile, 0, 0);
                TEXTRACT(bTile, bMatTile, 0, 0);
                SetFlag<PIPE_MTE1, PIPE_M>(0);
                WaitFlag<PIPE_MTE1, PIPE_M>(0);
                MatmulAcc(cTile[b], aTile, bTile, kk); // kk==0 overwrite, else accumulate
                SetFlag<PIPE_M, PIPE_MTE1>(0);
            }
            // non-blocking store: do NOT wait FIX->M here, so it overlaps the next tile.
            SetFlag<PIPE_M, PIPE_FIX>(b);
            WaitFlag<PIPE_M, PIPE_FIX>(b);
            GlobalDataOut dst(out + i * baseM * N + j * baseN);
            TSTORE(dst, cTile[b]);
            SetFlag<PIPE_FIX, PIPE_M>(b); // store done; same buffer reused NUM_C tiles later
            t++;
        }
    }

    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    for (uint32_t b = 0; b < NUM_C; b++) {
        WaitFlag<PIPE_FIX, PIPE_M>(b);
    }
}

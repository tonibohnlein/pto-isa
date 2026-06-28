/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Variant 4 -- asymmetric-buffered full-K (A-stationary). The stationary operand
// (A[m,K]) is single-buffered: loaded once per row and held in L0A, so it can use
// the FULL L0A buffer (no /2). The moving operand (B[K,n]) is MOVDB-buffered:
//   MOVDB == 1 : single -> B load serializes with the MAD (== plain full-K reuse)
//   MOVDB == 2 : double -> B(j+1) load overlaps MAD(j) (load/compute overlap)
//
// Traffic is identical for both (A once/row, B per tile); only the wall-clock
// differs. We read total_cycles: MOVDB=2 should beat MOVDB=1 by the hidden
// B-extract time, while keeping MTE1 busy unchanged.

#include "gemm_performance_kernel.cpp"

using namespace pto;

template <typename T, typename U, typename S, uint32_t M, uint32_t K, uint32_t N, uint32_t baseM, uint32_t baseN,
          uint32_t MOVDB>
AICORE inline void RunGemmFullKAsymBuf(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr uint32_t baseK = K; // full-K
    constexpr uint32_t mLoop = M / baseM;
    constexpr uint32_t nLoop = N / baseN;

    using TileMatA = Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor>;
    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    TileMatA aMatTile;
    TileMatB bMatTile[MOVDB];
    LeftTile aTile;       // stationary, single-buffered (full L0A)
    RightTile bTile[MOVDB]; // moving, MOVDB-buffered
    ResTile cTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(aTile, 0x0);
    TASSIGN(cTile, 0x0);
    for (uint32_t b = 0; b < MOVDB; b++) {
        TASSIGN(bMatTile[b], baseM * baseK * sizeof(U) + b * baseK * baseN * sizeof(S));
        TASSIGN(bTile[b], b * (baseK * baseN * sizeof(S)));
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

    SetFlag<PIPE_M, PIPE_MTE1>(2); // A slot free (event 2)
    for (uint32_t b = 0; b < MOVDB; b++) {
        SetFlag<PIPE_M, PIPE_MTE1>(b); // B slot b free (events 0..MOVDB-1)
    }
    SetFlag<PIPE_FIX, PIPE_M>(0); // L0C free

    for (uint32_t i = 0; i < mLoop; i++) {
        // Extract the stationary A[i] row panel once; held across all columns.
        WaitFlag<PIPE_M, PIPE_MTE1>(2);
        GlobalDataSrcA gmA(src0 + i * baseM * K);
        TLOAD(aMatTile, gmA);
        TEXTRACT(aTile, aMatTile, 0, 0);
        SetFlag<PIPE_MTE1, PIPE_M>(2);
        WaitFlag<PIPE_MTE1, PIPE_M>(2); // A ready (waited once per row)

        for (uint32_t j = 0; j < nLoop; j++) {
            const uint32_t b = j % MOVDB;
            WaitFlag<PIPE_M, PIPE_MTE1>(b); // B slot b free (MAD that used it is done)
            GlobalDataSrcB gmB(src1 + j * baseN * K);
            TLOAD(bMatTile[b], gmB);
            TEXTRACT(bTile[b], bMatTile[b], 0, 0);
            SetFlag<PIPE_MTE1, PIPE_M>(b);
            WaitFlag<PIPE_MTE1, PIPE_M>(b); // B[b] ready

            WaitFlag<PIPE_FIX, PIPE_M>(0); // L0C free (prev store)
            MatmulAcc(cTile, aTile, bTile[b], 0);
            SetFlag<PIPE_M, PIPE_MTE1>(b); // B slot b free

            SetFlag<PIPE_M, PIPE_FIX>(0);
            WaitFlag<PIPE_M, PIPE_FIX>(0);
            GlobalDataOut dst(out + i * baseM * N + j * baseN);
            TSTORE(dst, cTile);
            SetFlag<PIPE_FIX, PIPE_M>(0);
        }
        SetFlag<PIPE_M, PIPE_MTE1>(2); // A slot free for next row
    }

    WaitFlag<PIPE_M, PIPE_MTE1>(2);
    for (uint32_t b = 0; b < MOVDB; b++) {
        WaitFlag<PIPE_M, PIPE_MTE1>(b);
    }
    WaitFlag<PIPE_FIX, PIPE_M>(0);
}

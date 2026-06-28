/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Variant 3 -- accumulator / C-blocking: split-K GEMM with NACC in-flight L0C
// accumulators covering NACC adjacent N-tiles. Each A panel [m,k] is extracted
// once and MAC'd against B[k, n_0 .. n_{NACC-1}] into the NACC accumulators, so A
// is reused across NACC output columns WITHOUT requiring k == K (unlike full-K).
//
//   A extracts: mLoop * (nLoop/NACC) * kLoop   (1/NACC of plain split-K)
//   B extracts: mLoop *  nLoop       * kLoop   (unchanged)
//   constraint: NACC * m * n * bytes_c <= L0C   (the accumulators share L0C)
//
// NACC == 1 reduces to plain split-K (the no-reuse baseline). We read MTE1 busy:
// it should fall as NACC grows, by exactly the shrinking A-extract count.

#include "gemm_performance_kernel.cpp"

using namespace pto;

template <typename T, typename U, typename S, uint32_t M, uint32_t K, uint32_t N, uint32_t baseM, uint32_t baseK,
          uint32_t baseN, uint32_t NACC>
AICORE inline void RunGemmAccBlock(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr uint32_t mLoop = M / baseM;
    constexpr uint32_t njLoop = N / (baseN * NACC); // N-blocks of NACC tiles
    constexpr uint32_t kLoop = K / baseK;

    using TileMatA = Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor>;
    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    LeftTile aTile; // single A panel, reused across the NACC accumulators
    RightTile bTile;
    ResTile cTile[NACC];

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + baseM * baseK * sizeof(U));
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    for (uint32_t a = 0; a < NACC; a++) {
        TASSIGN(cTile[a], a * (baseM * baseN * sizeof(T)));
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

    SetFlag<PIPE_M, PIPE_MTE1>(0); // A slot free
    SetFlag<PIPE_M, PIPE_MTE1>(1); // B slot free
    for (uint32_t a = 0; a < NACC; a++) {
        SetFlag<PIPE_FIX, PIPE_M>(a); // all accumulators free
    }

    for (uint32_t i = 0; i < mLoop; i++) {
        for (uint32_t jb = 0; jb < njLoop; jb++) {
            for (uint32_t a = 0; a < NACC; a++) {
                WaitFlag<PIPE_FIX, PIPE_M>(a); // accumulator a free (prev store done)
            }
            for (uint32_t kk = 0; kk < kLoop; kk++) {
                // Extract A[i,kk] once; it feeds all NACC MACs this kk.
                WaitFlag<PIPE_M, PIPE_MTE1>(0);
                GlobalDataSrcA gmA(src0 + i * baseM * K + kk * baseK);
                TLOAD(aMatTile, gmA);
                TEXTRACT(aTile, aMatTile, 0, 0);
                SetFlag<PIPE_MTE1, PIPE_M>(0);
                WaitFlag<PIPE_MTE1, PIPE_M>(0); // A ready (cube waits once)
                for (uint32_t a = 0; a < NACC; a++) {
                    const uint32_t jcol = jb * NACC + a;
                    WaitFlag<PIPE_M, PIPE_MTE1>(1);
                    GlobalDataSrcB gmB(src1 + jcol * baseN * K + kk * baseK);
                    TLOAD(bMatTile, gmB);
                    TEXTRACT(bTile, bMatTile, 0, 0);
                    SetFlag<PIPE_MTE1, PIPE_M>(1);
                    WaitFlag<PIPE_MTE1, PIPE_M>(1); // B ready
                    MatmulAcc(cTile[a], aTile, bTile, kk);
                    SetFlag<PIPE_M, PIPE_MTE1>(1); // B slot free
                }
                SetFlag<PIPE_M, PIPE_MTE1>(0); // A slot free (all NACC MACs consumed it)
            }
            for (uint32_t a = 0; a < NACC; a++) {
                const uint32_t jcol = jb * NACC + a;
                SetFlag<PIPE_M, PIPE_FIX>(a);
                WaitFlag<PIPE_M, PIPE_FIX>(a);
                GlobalDataOut dst(out + i * baseM * N + jcol * baseN);
                TSTORE(dst, cTile[a]);
                SetFlag<PIPE_FIX, PIPE_M>(a);
            }
        }
    }

    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_M, PIPE_MTE1>(1);
    for (uint32_t a = 0; a < NACC; a++) {
        WaitFlag<PIPE_FIX, PIPE_M>(a);
    }
}

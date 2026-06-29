/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Truly fused single-core chained matmul for the gm_l1_tile_study:
//   C[M,Ki] = A[M,K1] * B[K1,Ki]      (MM1)
//   E[M,N2] = C[M,Ki] * D[Ki,N2]      (MM2)
// The intermediate C is drained L0C -> L1 (TMOV / copy_matrix_cc_to_cbuf, fp32->bf16)
// and consumed by MM2 from L1 -- it NEVER round-trips GM. So the GM->L1 reload (MTE2)
// covers only the boundary operands A, B, D, and the GM store (FixPipe) covers only E.
// This directly confirms cube_operand_reload()'s `produced`-operand exclusion that the
// gml1_chain decomposition validates indirectly.
//
// Scope (single core, one M row-band, Ki one tile): M==bm, the C tile [bm,Ki] stays in
// L1 across the MM1->MM2 boundary. Includes gemm_performance_kernel.cpp for MatmulAcc /
// SetFlag / WaitFlag.

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "gemm_performance_kernel.cpp"

namespace gm_l1_chain {
using namespace pto;

// MM1: accumulate C[bm, Ki] = A[bm, K1] * B[K1, Ki] in L0C, one output tile.
template <typename U, typename S, typename T, int M, int K1, int Ki, uint32_t bm, uint32_t bk,
          typename TileMatA, typename TileMatB, typename LeftTile, typename RightTile, typename AccTile>
AICORE inline void Mm1AccumulateC(__gm__ U *A, __gm__ S *B, TileMatA &aMat, TileMatB &bMat, LeftTile &aL0,
                                  RightTile &bL0, AccTile &cAcc)
{
    using NDShapeA = TileShape2D<U, bm, bk, Layout::ND>;
    using WholeA = BaseShape2D<U, M, K1, Layout::ND>;
    using GmA = GlobalTensor<U, NDShapeA, WholeA, Layout::ND>;
    using NDShapeB = TileShape2D<S, bk, Ki, Layout::DN>;
    using WholeB = BaseShape2D<S, K1, Ki, Layout::DN>;
    using GmB = GlobalTensor<S, NDShapeB, WholeB, Layout::DN>;

    constexpr uint32_t kLoop = K1 / bk;
    for (uint32_t k = 0; k < kLoop; k++) {
        GmA gmA(A + k * bk);             // A[0:bm, k*bk : k*bk+bk]
        GmB gmB(B + k * bk * Ki);        // B[k*bk : k*bk+bk, 0:Ki]
        TLOAD(aMat, gmA);                // GM -> L1 (MTE2): boundary operand A
        TLOAD(bMat, gmB);                // GM -> L1 (MTE2): boundary operand B
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TEXTRACT(aL0, aMat, 0, 0);       // L1 -> L0A (MTE1)
        TEXTRACT(bL0, bMat, 0, 0);       // L1 -> L0B (MTE1)
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);
        MatmulAcc(cAcc, aL0, bL0, k);    // Cube
        SetFlag<PIPE_M, PIPE_MTE1>(0);
        WaitFlag<PIPE_M, PIPE_MTE1>(0);
    }
}

// MM2: E[bm, N2] = C[bm, Ki] * D[Ki, N2], with C resident in L1 (cMat). One K-slice
// (Ki == bk2 == the cMat width), tiled over N2 in bnE columns; each E tile drains to GM.
template <typename U, typename S, typename T, int M, int Ki, int N2, uint32_t bm, uint32_t bnE,
          typename CMatTile, typename DMatTile, typename CLeftTile, typename DRightTile, typename AccTile>
AICORE inline void Mm2ConsumeC(__gm__ S *D, __gm__ T *E, CMatTile &cMat, DMatTile &dMat, CLeftTile &cL0,
                               DRightTile &dL0, AccTile &eAcc)
{
    using NDShapeD = TileShape2D<S, Ki, bnE, Layout::DN>;
    using WholeD = BaseShape2D<S, Ki, N2, Layout::DN>;
    using GmD = GlobalTensor<S, NDShapeD, WholeD, Layout::DN>;
    using NDShapeE = TileShape2D<T, bm, bnE, Layout::ND>;
    using WholeE = BaseShape2D<T, M, N2, Layout::ND>;
    using GmE = GlobalTensor<T, NDShapeE, WholeE, Layout::ND>;

    constexpr uint32_t nLoop = N2 / bnE;
    for (uint32_t j = 0; j < nLoop; j++) {
        GmD gmD(D + j * bnE);            // D[0:Ki, j*bnE : j*bnE+bnE]
        TLOAD(dMat, gmD);               // GM -> L1 (MTE2): boundary operand D
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
        TEXTRACT(cL0, cMat, 0, 0);      // L1 -> L0A (MTE1): C from L1, NOT GM
        TEXTRACT(dL0, dMat, 0, 0);      // L1 -> L0B (MTE1)
        SetFlag<PIPE_MTE1, PIPE_M>(1);
        WaitFlag<PIPE_MTE1, PIPE_M>(1);
        TMATMUL(eAcc, cL0, dL0);        // Cube (single K-slice -> plain matmul)
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);
        GmE gmE(E + j * bnE);
        TSTORE(gmE, eAcc);              // L0C -> GM (FixPipe): only E is stored
        SetFlag<PIPE_FIX, PIPE_M>(0);
        WaitFlag<PIPE_FIX, PIPE_M>(0);
    }
}

// Ki == bk2 == bnC: the whole intermediate C[bm, Ki] is one L0C tile drained to one L1
// (Mat) tile and consumed as MM2's single left K-slice. Operands bf16, accumulate fp32.
template <typename T, typename U, typename S, int M, int K1, int Ki, int N2, uint32_t bm, uint32_t bk, uint32_t bnE>
AICORE inline void RunGemmChainFused(__gm__ T *E, __gm__ U *A, __gm__ S *B, __gm__ S *D)
{
    using TileMatA = Tile<TileType::Mat, U, bm, bk, BLayout::ColMajor, bm, bk, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, S, bk, Ki, BLayout::RowMajor, bk, Ki, SLayout::ColMajor>;
    using CMatTile = Tile<TileType::Mat, S, bm, Ki, BLayout::ColMajor, bm, Ki, SLayout::RowMajor>;
    using DMatTile = Tile<TileType::Mat, S, Ki, bnE, BLayout::RowMajor, Ki, bnE, SLayout::ColMajor>;
    using LeftTile = TileLeft<U, bm, bk, bm, bk>;
    using RightTile = TileRight<S, bk, Ki, bk, Ki>;
    using CLeftTile = TileLeft<S, bm, Ki, bm, Ki>;
    using DRightTile = TileRight<S, Ki, bnE, Ki, bnE>;
    using AccTile = TileAcc<T, bm, bnE, bm, bnE>;
    using CAccTile = TileAcc<T, bm, Ki, bm, Ki>;

    TileMatA aMat;
    TileMatB bMat;
    CMatTile cMat;     // C resident in L1 (never to GM)
    DMatTile dMat;
    LeftTile aL0;
    RightTile bL0;
    CLeftTile cL0;
    DRightTile dL0;
    AccTile eAcc;
    CAccTile cAcc;

    // L1 layout: A panel, B panel, C tile, D panel at distinct offsets.
    TASSIGN(aMat, 0x0);
    TASSIGN(bMat, 0x0 + bm * bk * sizeof(U));
    TASSIGN(cMat, 0x0 + bm * bk * sizeof(U) + bk * Ki * sizeof(S));
    TASSIGN(dMat, 0x0 + bm * bk * sizeof(U) + bk * Ki * sizeof(S) + bm * Ki * sizeof(S));
    TASSIGN(aL0, 0x0);
    TASSIGN(bL0, 0x0);
    TASSIGN(cL0, 0x0);
    TASSIGN(dL0, 0x0);
    TASSIGN(eAcc, 0x0);
    TASSIGN(cAcc, 0x0 + bm * bnE * sizeof(T));

    // MM1: C = A * B, accumulate in L0C.
    Mm1AccumulateC<U, S, T, M, K1, Ki, bm, bk, TileMatA, TileMatB, LeftTile, RightTile, CAccTile>(
        A, B, aMat, bMat, aL0, bL0, cAcc);

    // Drain C: L0C -> L1 (TMOV / copy_matrix_cc_to_cbuf, fp32 acc -> bf16). NO GM store.
    SetFlag<PIPE_M, PIPE_FIX>(1);
    WaitFlag<PIPE_M, PIPE_FIX>(1);
    TMOV(cMat, cAcc);
    SetFlag<PIPE_FIX, PIPE_MTE1>(0);
    WaitFlag<PIPE_FIX, PIPE_MTE1>(0);

    // MM2: E = C * D, C from L1, store E to GM.
    Mm2ConsumeC<U, S, T, M, Ki, N2, bm, bnE, CMatTile, DMatTile, CLeftTile, DRightTile, AccTile>(
        D, E, cMat, dMat, cL0, dL0, eAcc);
}

} // namespace gm_l1_chain

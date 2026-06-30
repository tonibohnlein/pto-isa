/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// ── Minimal mixed cube+vector probe ──────────────────────────────────────────
// Feasibility probe for the cost-model study: a single output tile computed by
//   1) CUBE:    C = A @ B          (fp16 in, fp32 acc)   -> stored to GM
//   2) VECTOR:  C = C + C          (reads C back from GM) -> stored to GM
//
// The CUBE half reuses the proven gemm_performance_kernel (single 128x128x128
// tile, identical to gemm_perf_sim's Small case). The VECTOR half is appended
// inline. Both run in ONE kernel function; the perf-sim recorder routes each op
// to its pipe (CUBE/MTE2_AIC/MTE1/FIXP on AIC, MTE2_AIV/VEC/MTE3 on AIV).
//
// Two variants isolate the overlap question:
//   mixed_serial : vector reads the SAME GM buffer the cube wrote  -> data dep
//                  (FIXP store -> MTE2_AIV load) should SERIALIZE the two pipes.
//   mixed_indep  : vector reads a DIFFERENT GM buffer (no dep)      -> the AIC
//                  and AIV timelines should OVERLAP (run concurrently).
//
// No explicit FFTS / set_flag is used for the cube->vector handoff: the perf-sim
// TileDepTracker derives the cross-pipe wait purely from the matching GM data
// address. This probe checks whether that implicit handoff is sufficient.
// -----------------------------------------------------------------------------

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>
#include <gtest/gtest.h>

#include "gemm_performance_kernel.cpp" // RunGemmE2E (cube)

using namespace pto;

// Distinct, non-null GM bases so A/B/C never alias in the dep tracker.
// C lives at GM_C; the cube stores there and the vector reads it back.
static __gm__ half *const GM_A = reinterpret_cast<__gm__ half *>(0x10000000);
static __gm__ half *const GM_B = reinterpret_cast<__gm__ half *>(0x20000000);
static __gm__ float *const GM_C = reinterpret_cast<__gm__ float *>(0x30000000);
// A second, independent C-shaped GM buffer the cube never touches.
static __gm__ float *const GM_C_INDEP = reinterpret_cast<__gm__ float *>(0x40000000);

constexpr int MDIM = 128;
constexpr int KDIM = 128;
constexpr int NDIM = 128;

// ── Cube stage: C = A @ B, single 128x128x128 tile (no K-loop, no DB) ──
AICORE inline void CubeMatmul()
{
    RunGemmE2E<float, half, half, float, /*blockDim=*/1, MDIM, KDIM, NDIM, // m,k,n
               MDIM, KDIM, NDIM,                                           // validM,K,N
               MDIM, KDIM, NDIM,                                           // singleCoreM,K,N
               MDIM, KDIM, NDIM,                                           // baseM,K,N
               1, 1, 1, 1>(GM_C, GM_A, GM_B);                             // stepM,Ka,Kb,N
}

// ── Vector stage: V = C + C, reading C back from GM `cbuf` ──
// `ub_off` keeps the UB tile address clear of the cube buffers in the tracker.
AICORE inline void VectorAddFromGM(__gm__ float *cbuf, std::size_t ub_off)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using VT = pto::Tile<pto::TileType::Vec, float, MDIM, NDIM, pto::BLayout::RowMajor, MDIM, NDIM>;

    VT v;
    TASSIGN(v, ub_off);
    ShapeDyn shape(1, 1, 1, MDIM, NDIM);
    StrideDyn stride(MDIM * NDIM, MDIM * NDIM, MDIM * NDIM, NDIM, 1);
    Global g(cbuf, shape, stride);
    TLOAD(v, g);     // GM -> UB (MTE2_AIV); on `cbuf == GM_C` this waits on the cube store
    TADD(v, v, v);   // VEC compute
    TSTORE(g, v);    // UB -> GM (MTE3)
}

// Serial: vector consumes the cube's C buffer -> cross-pipe data dependency.
void mixed_serial()
{
    CubeMatmul();
    VectorAddFromGM(GM_C, 0x100000);
}

// Independent: vector touches a separate GM buffer -> no dep, pipes overlap.
void mixed_indep()
{
    CubeMatmul();
    VectorAddFromGM(GM_C_INDEP, 0x100000);
}

TEST(MixedProbe, Serial)
{
    LAUNCH_KERNEL(mixed_serial, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    bool has_cube = false, has_vec = false;
    for (auto &rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Matrix)
            has_cube = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::Vector)
            has_vec = true;
    }
    EXPECT_TRUE(has_cube) << "Expected CUBE (TMATMUL) instructions";
    EXPECT_TRUE(has_vec) << "Expected VEC (TADD) instructions";
}

TEST(MixedProbe, Independent)
{
    LAUNCH_KERNEL(mixed_indep, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    bool has_cube = false, has_vec = false;
    for (auto &rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Matrix)
            has_cube = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::Vector)
            has_vec = true;
    }
    EXPECT_TRUE(has_cube) << "Expected CUBE (TMATMUL) instructions";
    EXPECT_TRUE(has_vec) << "Expected VEC (TADD) instructions";
}

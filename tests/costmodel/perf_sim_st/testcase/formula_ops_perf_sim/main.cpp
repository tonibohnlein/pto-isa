/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This software is licensed under the CANN Open Software License Agreement Version 2.0.
See LICENSE in the root of the software repository for details.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <string_view>
#include <vector>

using namespace pto;

namespace {

template <typename T, int Rows, int Cols>
using VecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;

void runFormulaOps()
{
    VecTile<float, 64, 64> src0(64, 64);
    VecTile<float, 64, 64> src1(64, 64);
    VecTile<float, 64, 64> dst(64, 64);
    VecTile<float, 64, 64> tmp(64, 64);
    VecTile<float, 1, 64> col_dst(1, 64);
    TASSIGN(src0, 0x00000);
    TASSIGN(src1, 0x04000);
    TASSIGN(dst, 0x08000);
    TASSIGN(tmp, 0x0c000);
    TASSIGN(col_dst, 0x10000);

    TADDS(dst, src0, 1.5f);
    TMUL(dst, src0, src1);
    TROWSUM(dst, src0, tmp);
    TCOLSUM(col_dst, src0, tmp, true);
    TEXP(dst, src0);
    TSQRT(dst, src0);
}

void runTopkInactiveF32()
{
    VecTile<float, 1, 32> dst(1, 32);
    TASSIGN(dst, 0x00000);
    TEXPANDS(dst, -3.402823e38F);
}

void runTopkInactiveI32()
{
    VecTile<int32_t, 1, 32> dst(1, 32);
    TASSIGN(dst, 0x00000);
    TEXPANDS(dst, 0);
}

void runSixteenBitScalarSignatures()
{
    VecTile<half, 1, 32> fp16_dst(1, 32);
    VecTile<bfloat16_t, 1, 32> bf16_dst(1, 32);
    TASSIGN(fp16_dst, 0x00000);
    TASSIGN(bf16_dst, 0x01000);
    TEXPANDS(fp16_dst, static_cast<half>(1.0));
    TEXPANDS(bf16_dst, static_cast<bfloat16_t>(2.0));
}

void runRmsNormSignatures()
{
    VecTile<bfloat16_t, 8, 128> bf16_full(8, 128);
    VecTile<bfloat16_t, 1, 128> bf16_row(1, 128);
    VecTile<float, 8, 128> fp32_full(8, 128);
    VecTile<float, 8, 128> fp32_out(8, 128);
    VecTile<float, 8, 128> fp32_tmp(8, 128);
    VecTile<float, 1, 128> fp32_row(1, 128);
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1> row_scale(8, 1);
    VecTile<float, 1, 128> col_scale(1, 128);
    VecTile<float, 1, 8> small0(1, 8);
    VecTile<float, 1, 8> small1(1, 8);
    VecTile<float, 1, 8> small_tmp(1, 8);
    TASSIGN(bf16_full, 0x00000);
    TASSIGN(bf16_row, 0x01000);
    TASSIGN(fp32_full, 0x02000);
    TASSIGN(fp32_out, 0x06000);
    TASSIGN(fp32_tmp, 0x0a000);
    TASSIGN(fp32_row, 0x0e000);
    TASSIGN(row_scale, 0x0f000);
    TASSIGN(col_scale, 0x10000);
    TASSIGN(small0, 0x11000);
    TASSIGN(small1, 0x11100);
    TASSIGN(small_tmp, 0x11200);

    TEXPANDS(small0, 0.0F);
    TCVT(fp32_full, bf16_full, RoundMode::CAST_ROUND);
    TCVT(fp32_row, bf16_row, RoundMode::CAST_ROUND);
    TCVT(bf16_full, fp32_full, RoundMode::CAST_RINT);
    TMUL(fp32_out, fp32_full, fp32_full);
    TROWSUM(row_scale, fp32_out, fp32_tmp);
    TADD(small1, small0, small0);
    TMULS(small1, small0, 2.44140625e-4F);
    TADDS(small1, small0, 1.0e-6F);
    TRSQRT(small1, small0, small_tmp);
    TROWEXPANDMUL(fp32_out, fp32_full, row_scale);
    TCOLEXPANDMUL(fp32_out, fp32_full, col_scale);
}

void runTopkSelectSignatures()
{
    static thread_local std::array<std::byte, 2 * 1024 * 1024> storage{};
    VecTile<float, 1, 4096> scores(1, 4096);
    VecTile<int32_t, 1, 4096> indices(1, 4096);
    VecTile<uint32_t, 1, 2048> ci2048(1, 2048);
    VecTile<uint32_t, 1, 512> ci512(1, 512);
    VecTile<uint32_t, 1, 4096> ci4096(1, 4096);
    using Padded512 =
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min>;
    VecTile<float, 1, 512> unpadded512(1, 512);
    Padded512 padded512(1, 512);

    using SortSrc4096 = VecTile<float, 1, 4096>;
    using SortDst8192 = VecTile<float, 1, 8192>;
    using SortIdx4096 = VecTile<uint32_t, 1, 4096>;
    SortSrc4096 sort_src(1, 4096);
    SortDst8192 sort_dst(1, 8192);
    SortIdx4096 sort_idx(1, 4096);
    SortSrc4096 sort_tmp(1, 4096);

    VecTile<float, 1, 128> merge_dst(1, 128);
    VecTile<float, 1, 128> merge_tmp(1, 128);
    VecTile<float, 1, 64> merge_src0(1, 64);
    VecTile<float, 1, 64> merge_src1(1, 64);
    VecTile<float, 1, 32> gather_f32(1, 32);
    VecTile<int32_t, 1, 32> gather_i32(1, 32);
    using Padded64 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min>;
    using PaddedF32 =
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min>;
    using PaddedI32 =
        Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min>;
    Padded64 padded_gather_src(1, 64);
    PaddedF32 padded_gather_f32(1, 32);
    PaddedI32 padded_gather_i32(1, 32);

    uint64_t address = reinterpret_cast<uint64_t>(storage.data());
    auto assign = [&address](auto& tile) {
        TASSIGN(tile, address);
        address += 0x10000;
    };
    assign(scores);
    assign(indices);
    assign(ci2048);
    assign(ci512);
    assign(ci4096);
    assign(unpadded512);
    assign(padded512);
    assign(sort_src);
    assign(sort_dst);
    assign(sort_idx);
    assign(sort_tmp);
    assign(merge_dst);
    assign(merge_tmp);
    assign(merge_src0);
    assign(merge_src1);
    assign(gather_f32);
    assign(gather_i32);
    assign(padded_gather_src);
    assign(padded_gather_f32);
    assign(padded_gather_i32);

    TEXPANDS(scores, -3.402823e38F);
    TEXPANDS(indices, 0);
    TCI<decltype(ci2048), uint32_t, false>(ci2048, 0U);
    TCI<decltype(ci512), uint32_t, false>(ci512, 0U);
    TCI<decltype(ci4096), uint32_t, false>(ci4096, 0U);
    TFILLPAD(padded512, unpadded512);
    TSORT32(sort_dst, sort_src, sort_idx, sort_tmp);
    TMRGSORT(sort_dst, sort_dst, 64U);
    TMRGSORT(sort_dst, sort_dst, 256U);
    TMRGSORT(sort_dst, sort_dst, 1024U);

    MrgSortExecutedNumList executed{};
    TMRGSORT<decltype(merge_dst), decltype(merge_tmp), decltype(merge_src0), decltype(merge_src1), false>(
        merge_dst, executed, merge_tmp, merge_src0, merge_src1);
    TGATHER<decltype(gather_f32), decltype(merge_src0), MaskPattern::P0101>(gather_f32, merge_src0);
    TGATHER<decltype(gather_i32), decltype(merge_src0), MaskPattern::P1010>(gather_i32, merge_src0);
    TGATHER<decltype(padded_gather_f32), decltype(padded_gather_src), MaskPattern::P0101>(
        padded_gather_f32, padded_gather_src);
    TGATHER<decltype(padded_gather_i32), decltype(padded_gather_src), MaskPattern::P1010>(
        padded_gather_i32, padded_gather_src);

    const float score = gather_f32.GetValue(0);
    scores.SetValue(0, score);
    const int32_t index = gather_i32.GetValue(0);
    indices.SetValue(0, index);
    const float padded_score = padded_gather_f32.GetValue(0);
    scores.SetValue(0, padded_score);
    const int32_t padded_index = padded_gather_i32.GetValue(0);
    indices.SetValue(0, padded_index);
    const int32_t candidate_index = indices.GetValue(0);
    gather_i32.SetValue(0, candidate_index);
}

bool HasOpcode(const std::vector<::pto::perf_sim::InstrRecord>& records, std::string_view opcode)
{
    return std::ranges::any_of(records, [opcode](const auto& record) { return record.opcode == opcode; });
}

} // namespace

TEST(FormulaOpsPerfSim, RecordsFormulaSupportedOperations)
{
    LAUNCH_KERNEL(runFormulaOps, , (1, nullptr, nullptr));
    const auto& records = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    for (std::string_view opcode : {"TADDS", "TMUL", "TROWSUM", "TCOLSUM", "TEXP", "TSQRT"}) {
        EXPECT_TRUE(HasOpcode(records, opcode)) << opcode;
    }
}

TEST(FormulaOpsPerfSim, RecordsTopkInactiveTExpandsSignatures)
{
    LAUNCH_KERNEL(runTopkInactiveF32, , (1, nullptr, nullptr));
    const auto& f32_records = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    const auto f32 = std::ranges::find_if(f32_records, [](const auto& record) { return record.opcode == "TEXPANDS"; });
    ASSERT_NE(f32, f32_records.end());
    EXPECT_EQ(f32->dtype, "fp32");
    EXPECT_EQ(f32->rows, 1);
    EXPECT_EQ(f32->cols, 32);

    LAUNCH_KERNEL(runTopkInactiveI32, , (1, nullptr, nullptr));
    const auto& i32_records = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    const auto i32 = std::ranges::find_if(i32_records, [](const auto& record) { return record.opcode == "TEXPANDS"; });
    ASSERT_NE(i32, i32_records.end());
    EXPECT_EQ(i32->dtype, "int32");
    EXPECT_EQ(i32->rows, 1);
    EXPECT_EQ(i32->cols, 32);
}

TEST(FormulaOpsPerfSim, RecordsRmsNormSignatures)
{
    LAUNCH_KERNEL(runRmsNormSignatures, , (1, nullptr, nullptr));
    const auto& records = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    for (std::string_view opcode :
         {"TEXPANDS", "TCVT", "TMUL", "TROWSUM", "TADD", "TMULS", "TADDS", "TRSQRT", "TROWEXPANDMUL",
          "TCOLEXPANDMUL"}) {
        EXPECT_TRUE(HasOpcode(records, opcode)) << opcode;
    }
}

TEST(FormulaOpsPerfSim, SeparatesSixteenBitDtypesAndConstantsWhenToolchainCanRepresentThem)
{
    LAUNCH_KERNEL(runSixteenBitScalarSignatures, , (1, nullptr, nullptr));
    const auto& records = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    std::vector<::pto::perf_sim::InstrRecord> expands;
    std::ranges::copy_if(
        records, std::back_inserter(expands), [](const auto& record) { return record.opcode == "TEXPANDS"; });
    ASSERT_EQ(expands.size(), 2U);
    EXPECT_NE(expands[0].scalar_args, expands[1].scalar_args);
    if constexpr (std::is_same_v<half, bfloat16_t>) {
        EXPECT_EQ(expands[0].dtype, "fp16_or_bf16");
        EXPECT_EQ(expands[1].dtype, "fp16_or_bf16");
    } else {
        EXPECT_EQ(expands[0].dtype, "fp16");
        EXPECT_EQ(expands[1].dtype, "bf16");
    }
}

TEST(FormulaOpsPerfSim, RecordsTopkSelectSignatures)
{
    LAUNCH_KERNEL(runTopkSelectSignatures, , (1, nullptr, nullptr));
    const auto& records = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    for (std::string_view opcode :
         {"TEXPANDS", "TCI", "TFILLPAD", "TSORT32", "TMRGSORT", "TGATHER", "TGETVAL", "TSETVAL"}) {
        EXPECT_TRUE(HasOpcode(records, opcode)) << opcode;
    }
}

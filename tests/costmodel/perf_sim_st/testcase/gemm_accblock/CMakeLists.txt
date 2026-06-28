pto_costmodel_sim_st(gemm_accblock)
target_include_directories(gemm_accblock PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/gemm_performance
)
target_compile_options(gemm_accblock PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)

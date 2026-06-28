pto_costmodel_sim_st(gemm_sweep)
target_include_directories(gemm_sweep PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/gemm_performance
)
target_compile_options(gemm_sweep PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)

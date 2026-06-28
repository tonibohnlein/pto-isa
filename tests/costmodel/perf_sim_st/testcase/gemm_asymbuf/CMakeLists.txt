pto_costmodel_sim_st(gemm_asymbuf)
target_include_directories(gemm_asymbuf PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/gemm_performance
)
target_compile_options(gemm_asymbuf PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)

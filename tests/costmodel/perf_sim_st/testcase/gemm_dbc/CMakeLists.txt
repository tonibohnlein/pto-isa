pto_costmodel_sim_st(gemm_dbc)
target_include_directories(gemm_dbc PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/gemm_performance
)
target_compile_options(gemm_dbc PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)

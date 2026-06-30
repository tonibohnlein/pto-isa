pto_costmodel_sim_st(mixed_ddr_bound)
target_include_directories(mixed_ddr_bound PRIVATE ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/gemm_performance)
target_compile_options(mixed_ddr_bound PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)

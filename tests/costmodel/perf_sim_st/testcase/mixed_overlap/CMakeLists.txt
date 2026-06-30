pto_costmodel_sim_st(mixed_overlap)
target_include_directories(mixed_overlap PRIVATE ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/gemm_performance)
target_compile_options(mixed_overlap PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)

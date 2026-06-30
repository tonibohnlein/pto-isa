pto_costmodel_sim_st(mixed_serial)
target_include_directories(mixed_serial PRIVATE ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/gemm_performance)
target_compile_options(mixed_serial PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)

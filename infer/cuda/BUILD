load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "llm_mat_mul",
    srcs = ["llm_mat_mul.cu"],
    hdrs = ["llm_mat_mul.h"],
    includes = ["."],
    visibility = ["//transformer/mlp:__pkg__"],
)

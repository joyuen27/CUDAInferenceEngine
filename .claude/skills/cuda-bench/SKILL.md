---
name: bench-setup
description: Skill for benchmarking setup custom CUDA kernels
---

## Scripts

### [setup.sh](setup.sh)
<!-- when to use: At the start of a benchmarking session, before any timed runs. Triggers: "set up for benchmarking", "stabilize CPU", "prep for a kernel sweep". Snapshots the current Windows power plan, switches to High Performance, pins min/max processor state to 100%, disables turbo boost. Requires WSL launched from an elevated Windows terminal. -->

### [check-binary.sh](check-binary.sh)
<!-- when to use: Before timed runs, after building the benchmark binary and after setup.sh. Triggers: "check my binary", "will this JIT", "verify build flags". Detects the running GPU's compute capability and confirms the binary contains a matching cubin so no PTX JIT fires during timing. Pass the binary path as the first argument. On failure, reports the exact -gencode flag to rebuild with. -->

### [teardown.sh](teardown.sh)
<!-- when to use: After the benchmarking session completes. Triggers: "tear down", "restore", "clean up", "done benchmarking". Reads the saved power-plan GUID and restores the original Windows power plan, then removes the state file. Always run this after setup.sh, even if the benchmark failed. -->


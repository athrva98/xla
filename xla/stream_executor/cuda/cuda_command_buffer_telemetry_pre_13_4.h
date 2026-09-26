/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// CUDA 13.2-13.3 telemetry. CUPTI records there carry only the executable
// graph's `graphId` and a synthesized `graphNodeId = (exec_id << 32) |
// ordinal`. The ordinal numbers each graph's direct nodes first, then recurses
// depth-first into child and conditional body graphs. Child-graph and cuDNN
// updates don't re-project, so those nodes keep their instantiation-time
// annotations.
//
// TODO(b/483450622): Delete once CUDA 13.4 is the minimum for telemetry.

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_PRE_13_4_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_PRE_13_4_H_

#include "third_party/gpus/cuda/include/cuda.h"

#if CUDA_VERSION >= 13020
namespace stream_executor::gpu::command_buffer_telemetry {

// Projects template graph node annotations from `root_graph` (and its nested
// child / conditional graphs) into `CudaGraphAnnotationRegistry` keyed by
// `exec`'s `cuGraphExecGetId` and synthesized CUPTI `graphNodeId` in
// `KeySpace::kPre134ExecGraph`. Makes CUDA calls, so callers check
// `CudaCommandBuffer::ShouldRecordAnnotations()` first.
void RegisterPre134ExecGraphNodes(CUgraph root_graph, CUgraphExec exec);

}  // namespace stream_executor::gpu::command_buffer_telemetry
#endif  // CUDA_VERSION >= 13020

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_PRE_13_4_H_

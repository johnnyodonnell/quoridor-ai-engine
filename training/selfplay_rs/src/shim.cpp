// Minimal C FFI over libtorch's AOTInductor package loader and CUDA-graph +
// stream API (neither is exposed by tch-rs). Symbols resolve against the
// torch_cuda / c10_cuda libs that tch links. Port of chess-ai-engine's shim.cpp.
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/Context.h>
#include <ATen/Functions.h> // at::zeros (factory) for the batch-guard probe
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {

// --- CUDA events: gate the scatter thread's readback on the forward's
// completion so it overlaps the inference thread's next forward. Recorded on the
// thread's current CUDA stream.
void* cg_event_new() {
    cudaEvent_t e;
    // BlockingSync: cudaEventSynchronize parks the waiting thread on an OS wait
    // (woken by the completion interrupt) instead of spin-polling. The scatter
    // thread waits this event once per forward, so the few-us wake latency is
    // negligible at batch granularity, and it stops scatter from burning a full
    // core spinning, which matters whenever the workers are short of CPU.
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync);
    return reinterpret_cast<void*>(e);
}
void cg_event_record(void* e) {
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(e),
                    c10::cuda::getCurrentCUDAStream().stream());
}
void cg_event_sync(void* e) {
    cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(e));
}
void cg_event_free(void* e) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e));
}

// --- AOTInductor package loader (the fused, Python-free forward) -------------
// tch tensor handles (C_tensor*) ARE at::Tensor* under the hood, so we
// reinterpret them directly. Outputs are copy_'d into Rust-owned tensors so all
// lifetime stays on the tch side.
void* aoti_load(const char* path) {
    // run_single_threaded=true runs the model inline on the calling thread instead
    // of via AOTI's thread pool. The pooled path issues an op that's illegal during
    // CUDA-graph capture ("operation not permitted when stream is capturing", pytorch
    // #158834); the single-threaded path is capture-safe, letting us graph the AOTI
    // forward (and it has no downside for our single-stream inference).
    return reinterpret_cast<void*>(new torch::inductor::AOTIModelPackageLoader(
        std::string(path), "model", /*run_single_threaded=*/true));
}

void aoti_run(void* loader_, const void* in_, void* out_logits_, void* out_values_) {
    auto* loader = reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
    const at::Tensor& in = *reinterpret_cast<const at::Tensor*>(in_);
    std::vector<at::Tensor> outs = loader->run({in});
    reinterpret_cast<at::Tensor*>(const_cast<void*>(out_logits_))->copy_(outs[0]);
    reinterpret_cast<at::Tensor*>(const_cast<void*>(out_values_))->copy_(outs[1]);
}

void aoti_free(void* loader_) {
    delete reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
}

// Startup guard: verify the loaded package was compiled for `batch`. The .pt2 is
// exported with a STATIC batch (no dynamic_shapes), so feeding a wrong batch trips
// an AOTI input-shape check (which throws). We catch it here because throwing
// across the extern "C" boundary into Rust is UB. Returns 0 = ok, 1 = the forward
// threw (batch mismatch / other error), 2 = output batch dim mismatch. Doubles as
// a warmup forward. (c, h, w) is the per-position input shape.
int aoti_check_batch(void* loader_, long batch, long c, long h, long w) {
    try {
        auto* loader = reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
        auto opts = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
        at::Tensor x = at::zeros({static_cast<int64_t>(batch), static_cast<int64_t>(c),
                                  static_cast<int64_t>(h), static_cast<int64_t>(w)}, opts);
        std::vector<at::Tensor> outs = loader->run({x});
        return (!outs.empty() && outs[0].size(0) == batch) ? 0 : 2;
    } catch (...) {
        return 1;
    }
}

// In-place weight refresh (no recompile; chess-ai-engine b058439, with two stream
// races fixed). names[] are the MANGLED constant names (dotted FQN with '.'->'_');
// tensors[] are tch C_tensor handles (at::Tensor*) on CUDA, bf16. Push them into
// the INACTIVE constant buffer, re-derive the _FOLDED_CONST_* tensors (requires
// the .pt2 compiled with use_runtime_constant_folding=True), and swap it live.
//
// AOTI orders none of this against a non-default stream, and the inference
// thread runs on a non-blocking c10 pool stream:
//   1. update_constant_buffer copies the primaries with plain cudaMemcpy (legacy
//      default stream). Folding on the caller's stream raced those copies and
//      produced NaN/garbage (chess's original sequence; reproduced here).
//   2. run_const_fold hands its outputs over by ownership (no copy) and returns
//      without waiting, so forwards on the caller's stream after the swap could
//      read folded constants the fold kernels had not finished writing (~50% of
//      swaps corrupt on real positions: value mean |Δ| ~0.8 vs fp32).
// Hence: finish the caller's stream (uploaded tensors complete, no forward in
// flight), update + fold on the legacy default stream (ordered with AOTI's
// copies), swap, then synchronize the device so every later forward on any
// stream sees the finished constants. Measured exact (bit-identical to a
// natively compiled package) 10/10 under pipeline conditions; on the legacy
// stream alone it was exact 12/12. The cost is one GPU drain per swap.
void aoti_swap_weights(void* loader_, const char** names, const void** tensors, int n) {
    auto* loader = reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(loader_);
    auto* runner = loader->get_runner();
    std::unordered_map<std::string, at::Tensor> map;
    map.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        map[std::string(names[i])] = *reinterpret_cast<const at::Tensor*>(tensors[i]);
    }
    c10::cuda::getCurrentCUDAStream().synchronize();
    {
        c10::cuda::CUDAStreamGuard on_default(c10::cuda::getDefaultCUDAStream());
        runner->update_constant_buffer(map, /*use_inactive=*/true, /*validate_full_updates=*/false);
        auto stream = reinterpret_cast<AOTInductorStreamHandle>(
            c10::cuda::getDefaultCUDAStream().stream());
        runner->run_const_fold(/*use_inactive=*/true, stream);
        runner->swap_constant_buffer();
    }
    c10::cuda::device_synchronize();
}

// Enable cuDNN benchmark (algorithm autotuning): picks fast conv algorithms on
// the first calls (during warmup). tch does not expose this.
void cg_set_cudnn_benchmark(bool on) {
    at::globalContext().setBenchmarkCuDNN(on);
}

void* cg_new() {
    return reinterpret_cast<void*>(new at::cuda::CUDAGraph());
}

void cg_capture_begin(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->capture_begin();
}

void cg_capture_end(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->capture_end();
}

void cg_replay(void* g) {
    reinterpret_cast<at::cuda::CUDAGraph*>(g)->replay();
}

void cg_free(void* g) {
    delete reinterpret_cast<at::cuda::CUDAGraph*>(g);
}

// Switch the current CUDA stream to a fresh non-default stream from the pool.
// CUDAGraph::capture_begin requires a non-default capture stream.
void cg_set_side_stream() {
    auto s = c10::cuda::getStreamFromPool(false);
    c10::cuda::setCurrentCUDAStream(s);
}

void cg_device_synchronize() {
    c10::cuda::device_synchronize();
}

}  // extern "C"

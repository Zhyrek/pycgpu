// Minimal C ABI over Metal for the pycalphad Metal backend.
//
// Compiled on first use with the system clang++ (Xcode command line tools)
// and loaded through ctypes, so the backend needs no Python dependencies
// beyond numpy. Buffers use shared storage: on Apple silicon the CPU and
// GPU see the same memory, so numpy views of buffer contents are zero-copy.
//
// Every entry point returns 0 on success and a nonzero code on failure, with
// a message written to `err`.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <cstdint>

namespace {

void set_err(char* err, int errlen, NSString* msg) {
    if (err == nullptr || errlen <= 0) return;
    const char* s = msg ? msg.UTF8String : "unknown Metal error";
    std::strncpy(err, s ? s : "unknown Metal error", (size_t)errlen - 1);
    err[errlen - 1] = '\0';
}

struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

}  // namespace

extern "C" {

int pyc_mtl_init(void** ctx_out, char* name, int namelen, char* err, int errlen) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            set_err(err, errlen, @"no Metal device available");
            return 1;
        }
        Context* ctx = new Context();
        ctx->device = dev;
        ctx->queue = [dev newCommandQueue];
        if (name && namelen > 0) {
            std::strncpy(name, dev.name.UTF8String, (size_t)namelen - 1);
            name[namelen - 1] = '\0';
        }
        *ctx_out = ctx;
        return 0;
    }
}

uint64_t pyc_mtl_recommended_working_set(void* ctx_) {
    return ((Context*)ctx_)->device.recommendedMaxWorkingSetSize;
}

int pyc_mtl_compile(void* ctx_, const char* source, int fast_math, void** lib_out,
                    char* err, int errlen) {
    @autoreleasepool {
        Context* ctx = (Context*)ctx_;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        // Double-float arithmetic relies on exact IEEE rounding and on the
        // compiler not reassociating error-compensation terms.
        if (@available(macOS 15.0, *)) {
            opts.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = fast_math ? YES : NO;
#pragma clang diagnostic pop
        }
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src options:opts error:&error];
        if (lib == nil) {
            set_err(err, errlen, error ? error.localizedDescription : @"Metal compilation failed");
            return 1;
        }
        *lib_out = (__bridge_retained void*)lib;
        return 0;
    }
}

int pyc_mtl_pipeline(void* ctx_, void* lib_, const char* fn_name, void** pso_out,
                     int* max_threads, int* simd_width, char* err, int errlen) {
    @autoreleasepool {
        Context* ctx = (Context*)ctx_;
        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)lib_;
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
        if (fn == nil) {
            set_err(err, errlen, [NSString stringWithFormat:@"kernel '%s' not found", fn_name]);
            return 1;
        }
        NSError* error = nil;
        id<MTLComputePipelineState> pso =
            [ctx->device newComputePipelineStateWithFunction:fn error:&error];
        if (pso == nil) {
            set_err(err, errlen, error ? error.localizedDescription : @"pipeline creation failed");
            return 1;
        }
        *max_threads = (int)pso.maxTotalThreadsPerThreadgroup;
        *simd_width = (int)pso.threadExecutionWidth;
        *pso_out = (__bridge_retained void*)pso;
        return 0;
    }
}

void* pyc_mtl_buffer(void* ctx_, uint64_t nbytes, void** contents) {
    @autoreleasepool {
        Context* ctx = (Context*)ctx_;
        id<MTLBuffer> buf = [ctx->device newBufferWithLength:(NSUInteger)(nbytes ? nbytes : 16)
                                                     options:MTLResourceStorageModeShared];
        if (buf == nil) return nullptr;
        *contents = buf.contents;
        return (__bridge_retained void*)buf;
    }
}

void pyc_mtl_release(void* obj) {
    if (obj) CFRelease(obj);
}

void pyc_mtl_destroy(void* ctx_) {
    delete (Context*)ctx_;
}

// Dispatch `nthreads` threads of `pso` in threadgroups of `tg` threads.
// Buffers bind to indices 0..nbuf-1; the `params` bytes (if any) bind to
// index nbuf. Blocks until the GPU finishes.
int pyc_mtl_dispatch(void* ctx_, void* pso_, int nbuf, void** bufs, const void* params,
                     uint64_t params_len, uint32_t nthreads, uint32_t tg, double* gpu_seconds,
                     char* err, int errlen) {
    @autoreleasepool {
        Context* ctx = (Context*)ctx_;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_;
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (int i = 0; i < nbuf; ++i) {
            [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:0 atIndex:(NSUInteger)i];
        }
        if (params && params_len > 0) {
            [enc setBytes:params length:(NSUInteger)params_len atIndex:(NSUInteger)nbuf];
        }
        [enc dispatchThreads:MTLSizeMake(nthreads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            set_err(err, errlen, cb.error ? cb.error.localizedDescription : @"command buffer failed");
            return 1;
        }
        if (gpu_seconds) *gpu_seconds = cb.GPUEndTime - cb.GPUStartTime;
        return 0;
    }
}

}  // extern "C"

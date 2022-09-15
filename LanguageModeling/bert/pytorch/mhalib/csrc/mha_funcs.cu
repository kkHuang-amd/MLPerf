#include <vector>
#include <iostream>
//#ifdef __gfx908__  
//Uncomment ifdef and endif only if you need to undef the HIP_HALF ops below just for gfx908 and not for others
//below lines enable hip float to half conversion which are disabled by default in hip_fp16.h
#undef __HIP_NO_HALF_OPERATORS__   
#undef __HIP_NO_HALF_CONVERSIONS__
//#endif

#include <ATen/ATen.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
//#include <cuda_profiler_api.h>
//#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
//#include "/opt/pytorch/apex/apex/contrib/csrc/multihead_attn/softmax.h"
#include "./softmax.cuh"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

#define nstreams 16

// global variables.
cudaStream_t stream[nstreams];
cublasHandle_t handle;

rocblas_datatype a_type       = rocblas_datatype_f16_r;
rocblas_datatype b_type       = rocblas_datatype_f16_r;
rocblas_datatype c_type       = rocblas_datatype_f16_r;
rocblas_datatype d_type       = rocblas_datatype_f16_r;
rocblas_datatype compute_type       = rocblas_datatype_f32_r;

rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
int32_t           solution_index = 0;

int get_flag(){
  int flag;
#ifdef __HIP_PLATFORM_HCC__
	#define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 + ROCBLAS_VERSION_MINOR)
	#define USE_GEMM_FLAGS_FP16_ALT_IMPL (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242)
	#if USE_GEMM_FLAGS_FP16_ALT_IMPL
	  #ifdef BACKWARD_PASS_GUARD
		flag = at::BACKWARD_PASS_GUARD_CLASS::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
	  #endif
	#endif
#endif
	return flag;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Fprop_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
			 torch::Tensor &D,
	    	         int batch,
  	  	         torch::Tensor &seq_len,
                         int heads,
		         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{

    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()) + (strided ? embed : 0)); 	// key
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr())); 				// query
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr())); 	        		// output
    void *ptrD = static_cast<void*>(static_cast<half*>(D.data_ptr()));                          // output

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams]: at::cuda::getCurrentCUDAStream());
//        cublasGemmStridedBatchedEx(handle,
//                                   CUBLAS_OP_T,
//                                   CUBLAS_OP_N,
//                                   seqlen[i],
//                                   seqlen[i],
//                                   embed,
//                                   static_cast<const void*>(scale ? &alpha : &one),
//                                   ptrA,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   ptrB,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   static_cast<const void*>(&zero),
//                                   ptrC,
//                                   CUDA_R_16F,
//                                   seqlen[i],
//                                   seqlen[i]*seqlen[i],
//                                   enable_stream ? heads : batch*heads,
//                                   CUDA_R_32F,
//                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	int flag = get_flag();
        rocblas_gemm_strided_batched_ex(handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   seqlen[i],
                                   seqlen[i],
                                   embed,
                                   static_cast<const void*>(scale ? &alpha : &one),
                                   ptrA,
                                   a_type, 
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   b_type,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   c_type,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
				   ptrD,
				   d_type,
				   seqlen[i],
				   seqlen[i]*seqlen[i],
                                   enable_stream ? heads : batch*heads,
                                   compute_type,
                                   algo,
				   solution_index,
				   flag);
	ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
	ptrB = static_cast<void*>(static_cast<half*>(ptrB) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
	ptrC = static_cast<void*>(static_cast<half*>(ptrC) + heads*seqlen[i]*seqlen[i]);
	ptrD = static_cast<void*>(static_cast<half*>(ptrD) + heads*seqlen[i]*seqlen[i]); // is this correct, also using same values for ldd and stride_d as ldc and stride_c
    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Fprop_(torch::Tensor &A,
                    torch::Tensor &B,
                    torch::Tensor &C,
		    torch::Tensor &D,
                    int batch,
                    torch::Tensor &seq_len,
                    int heads,
                    int embed,
		    bool scale,
		    bool strided,
		    bool enable_stream,
		    bool sync)
{

    float one = 1.0, zero = 0.0;

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()) + (strided ? 2*embed : 0));  // value 
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));            		// query*key
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()));           		 // output
    void *ptrD = static_cast<void*>(static_cast<half*>(D.data_ptr()));                          // output

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams]: at::cuda::getCurrentCUDAStream());
//        cublasGemmStridedBatchedEx(handle,
//                                   CUBLAS_OP_N,
//                                   CUBLAS_OP_N,
//                                   embed,
//                                   seqlen[i],
//                                   seqlen[i],
//                                   static_cast<const void*>(&one),
//                                   ptrA,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   ptrB,
//                                   CUDA_R_16F,
//                                   seqlen[i],
//                                   seqlen[i]*seqlen[i],
//                                   static_cast<const void*>(&zero),
//                                   ptrC,
//                                   CUDA_R_16F,
//                                   enable_stream ? heads*embed : batch*heads*embed,
//                                   embed,
//                                   enable_stream ? heads : batch*heads,
//                                   CUDA_R_32F,
//                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	int flag = get_flag();
        rocblas_gemm_strided_batched_ex(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   embed,
                                   seqlen[i],
                                   seqlen[i],
				   static_cast<const void*>(&one),
                                   ptrA,
                                   a_type,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   b_type,
				   seqlen[i],
				   seqlen[i]*seqlen[i],
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   c_type,
                                   enable_stream ? heads*embed : batch*heads*embed,
				   embed,
				   ptrD,
				   d_type,
				   enable_stream ? heads*embed : batch*heads*embed,
				   embed,
                                   enable_stream ? heads : batch*heads,
                                   compute_type,
                                   algo,
				   solution_index,
				   flag);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + heads*seqlen[i]*seqlen[i]);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + seqlen[i]*heads*embed);
        ptrD = static_cast<void*>(static_cast<half*>(ptrD) + seqlen[i]*heads*embed);

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Dgrad1_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
			 torch::Tensor &D,
                         int batch,
                         torch::Tensor &seq_len,
                         int heads,
                         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{

    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()));           		// query
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()) + (strided ? embed : 0)); 	// grad_key
    void *ptrD = static_cast<void*>(static_cast<half*>(D.data_ptr()) + (strided ? embed : 0)); 	// grad_key

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
//        cublasGemmStridedBatchedEx(handle,
//                                   CUBLAS_OP_N,
//                                   CUBLAS_OP_T,
//                                   embed,
//                                   seqlen[i],
//                                   seqlen[i],
//                                   static_cast<const void*>(scale ? &alpha : &one),
//                                   ptrA,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   ptrB,
//                                   CUDA_R_16F,
//                                   seqlen[i],
//                                   seqlen[i]*seqlen[i],
//                                   static_cast<const void*>(&zero),
//                                   ptrC,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   enable_stream ? heads : heads*batch,
//                                   CUDA_R_32F,
//                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	int flag = get_flag();
        rocblas_gemm_strided_batched_ex(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_T,
                                   embed,
                                   seqlen[i],
                                   seqlen[i],
                                   static_cast<const void*>(scale ? &alpha : &one),
                                   ptrA,
                                   a_type,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   b_type,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   c_type,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
				   ptrD,
				   d_type,
				   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
				   strided ? 3*embed : embed,
                                   enable_stream ? heads : heads*batch,
                                   compute_type,
                                   algo,
				   solution_index,
				   flag);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + heads*seqlen[i]*seqlen[i]);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrD = static_cast<void*>(static_cast<half*>(ptrD) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Dgrad1_(torch::Tensor &A,
                     torch::Tensor &B,
                     torch::Tensor &C,
		     torch::Tensor &D,
                     int batch,
                     torch::Tensor &seq_len,
                     int heads,
                     int embed,
		     bool scale,
		     bool strided,
		     bool enable_stream,
		     bool sync)
{

    float one = 1.0, zero = 0.0;

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()) + (strided ? 2*embed : 0));  // value
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()));
    void *ptrD = static_cast<void*>(static_cast<half*>(D.data_ptr()));

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
//        cublasGemmStridedBatchedEx(handle,
//                                   CUBLAS_OP_T,
//                                   CUBLAS_OP_N,
//                                   seqlen[i],
//                                   seqlen[i],
//                                   embed,
//                                   static_cast<const void*>(&one),
//                                   ptrA,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   ptrB,
//                                   CUDA_R_16F,
//				   enable_stream ? heads*embed : batch*heads*embed,
//                                   embed,
//                                   static_cast<const void*>(&zero),
//                                   ptrC,
//                                   CUDA_R_16F,
//                                   seqlen[i],
//                                   seqlen[i]*seqlen[i],
//                                   enable_stream ? heads : batch*heads,
//                                   CUDA_R_32F,
//                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	int flag = get_flag();
        rocblas_gemm_strided_batched_ex(handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   seqlen[i],
                                   seqlen[i],
                                   embed,
                                   static_cast<const void*>(&one),
                                   ptrA,
                                   a_type,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   b_type,
				   enable_stream ? heads*embed : batch*heads*embed,
                                   embed,
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   c_type,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
				   ptrD,
				   d_type,
				   seqlen[i],
				   seqlen[i]*seqlen[i],
                                   enable_stream ? heads : batch*heads,
                                   compute_type,
                                   algo,
				   solution_index,
				   flag);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + seqlen[i]*heads*embed);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + heads*seqlen[i]*seqlen[i]);
        ptrD = static_cast<void*>(static_cast<half*>(ptrD) + heads*seqlen[i]*seqlen[i]);

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Dgrad2_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
			 torch::Tensor &D,
                         int batch,
                         torch::Tensor &seq_len,
                         int heads,
                         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{

    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()) + (strided ? embed : 0));  	// key
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()));          		// grad query
    void *ptrD = static_cast<void*>(static_cast<half*>(D.data_ptr()));          		// grad query

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
//        cublasGemmStridedBatchedEx(handle,
//                                   CUBLAS_OP_N,
//                                   CUBLAS_OP_N,
//                                   embed,
//                                   seqlen[i],
//                                   seqlen[i],
//                                   static_cast<const void*>(scale ? &alpha : &one),
//                                   ptrA,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   ptrB,
//                                   CUDA_R_16F,
//                                   seqlen[i],
//                                   seqlen[i]*seqlen[i],
//                                   static_cast<const void*>(&zero),
//                                   ptrC,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   enable_stream ? heads : batch*heads,
//                                   CUDA_R_32F,
//                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	int flag = get_flag();
        rocblas_gemm_strided_batched_ex(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   embed,
                                   seqlen[i],
                                   seqlen[i],
                                   static_cast<const void*>(scale ? &alpha : &one),
                                   ptrA,
                                   a_type,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   b_type,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   c_type,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
				   ptrD,
				   d_type,
				   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
				   strided ? 3*embed : embed,
                                   enable_stream ? heads : batch*heads,
                                   compute_type,
                                   algo,
				   solution_index,
				   flag);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + heads*seqlen[i]*seqlen[i]);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrD = static_cast<void*>(static_cast<half*>(ptrD) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Dgrad2_(torch::Tensor &A,
                     torch::Tensor &B,
                     torch::Tensor &C,
		     torch::Tensor &D,
                     int batch,
                     torch::Tensor &seq_len,
                     int heads,
                     int embed,
		     bool scale,
		     bool strided,
		     bool enable_stream,
		     bool sync)
{

    float one = 1.0, zero = 0.0;

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()));
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()) + (strided ? 2*embed : 0));  // grad-value
    void *ptrD = static_cast<void*>(static_cast<half*>(D.data_ptr()) + (strided ? 2*embed : 0));  // grad-value

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
//        cublasGemmStridedBatchedEx(handle,
//                                   CUBLAS_OP_N,
//                                   CUBLAS_OP_T,
//                                   embed,
//                                   seqlen[i],
//                                   seqlen[i],
//                                   static_cast<const void*>(&one),
//                                   ptrA,
//                                   CUDA_R_16F,
//				   enable_stream ? heads*embed : batch*heads*embed,
//                                   embed,
//                                   ptrB,
//                                   CUDA_R_16F,
//                                   seqlen[i],
//                                   seqlen[i]*seqlen[i],
//                                   static_cast<const void*>(&zero),
//                                   ptrC,
//                                   CUDA_R_16F,
//                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
//                                   strided ? 3*embed : embed,
//                                   enable_stream ? heads : batch*heads,
//                                   CUDA_R_32F,
//                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	int flag = get_flag();
        rocblas_gemm_strided_batched_ex(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_T,
                                   embed,
                                   seqlen[i],
                                   seqlen[i],
                                   static_cast<const void*>(&one),
                                   ptrA,
                                   a_type,
				   enable_stream ? heads*embed : batch*heads*embed,
                                   embed,
                                   ptrB,
                                   b_type,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   c_type,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
				   ptrD,
				   d_type,
				   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
				   strided ? 3*embed : embed,
                                   enable_stream ? heads : batch*heads,
                                   compute_type,
                                   algo,
				   solution_index,
				   flag);

        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + seqlen[i]*heads*embed);
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + heads*seqlen[i]*seqlen[i]);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrD = static_cast<void*>(static_cast<half*>(ptrD) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastSoftmaxFprop_(torch::Tensor &input,
		  int batch,
                  torch::Tensor &seq_len,
		  int heads,
		  bool enable_stream,
		  bool sync)
{
    int *seqlen = static_cast<int*>(seq_len.data_ptr());
    void *ptrIn = static_cast<void*>(input.data_ptr());

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        dispatch_softmax<half, half, float>(
                                 reinterpret_cast<half*>(ptrIn),
                                 reinterpret_cast<const half*>(ptrIn),
                                 seqlen[i],
                                 seqlen[i],
                                 enable_stream ? heads*seqlen[i] : batch*heads*seqlen[i]);
        ptrIn = static_cast<void*>(static_cast<half*>(ptrIn) + heads*seqlen[i]*seqlen[i]);
    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastSoftmaxBprop_(torch::Tensor &input,
		       torch::Tensor &output,
                       int batch,
                       torch::Tensor &seq_len,
                       int heads,
		       bool enable_stream,
		       bool sync)
{
    int *seqlen = static_cast<int*>(seq_len.data_ptr());
    void *ptrIn = static_cast<void*>(input.data_ptr());
    void *ptrOut = static_cast<void*>(output.data_ptr());

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        dispatch_softmax_backward_stream<half, half, float>(
                                 static_cast<half*>(ptrOut),
                                 static_cast<half*>(ptrOut),
                                 reinterpret_cast<half const*>(ptrIn),
                                 seqlen[i],
                                 seqlen[i],
                                 enable_stream ? heads*seqlen[i] : batch*heads*seqlen[i], 
				 enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
        ptrIn = static_cast<void*>(static_cast<half*>(ptrIn) + heads*seqlen[i]*seqlen[i]);
        ptrOut = static_cast<void*>(static_cast<half*>(ptrOut) + heads*seqlen[i]*seqlen[i]);	
    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }	
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastMaskSoftmaxFprop_(torch::Tensor &input,
                           torch::Tensor &mask,
                           int batch,
                           torch::Tensor &seq_len,
                           int heads,
			   bool enable_stream,
			   bool sync)
{
    int *seqlen = static_cast<int*>(seq_len.data_ptr());
    void *ptrIn = static_cast<void*>(input.data_ptr());
    void *ptrMask = static_cast<void*>(mask.data_ptr());

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        dispatch_additive_masked_softmax_stream<half, half, float>(
                                 reinterpret_cast<half*>(ptrIn),
                                 reinterpret_cast<const half*>(ptrIn),
                                 reinterpret_cast<const half*>(ptrMask),				 
                                 seqlen[i],
                                 seqlen[i],
                                 enable_stream ? heads*seqlen[i] : batch*heads*seqlen[i],
				 enable_stream ? heads*seqlen[i] : heads*seqlen[i], 
				 enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
        ptrIn = static_cast<void*>(static_cast<half*>(ptrIn) + heads*seqlen[i]*seqlen[i]);
        ptrMask = static_cast<void*>(static_cast<half*>(ptrMask) + seqlen[i]);	
    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<torch::Tensor> FastMaskSoftmaxDropoutFprop_(torch::Tensor &input,
                                  torch::Tensor &mask,
                                  int batch,
                                  torch::Tensor &seq_len,
                                  int heads,
                                  float dropout_prob,
                                  bool enable_stream,
                                  bool sync,
                                  bool is_training)
{
    int *seqlen = static_cast<int*>(seq_len.data_ptr());
    void *ptrIn = static_cast<void*>(input.data_ptr());
    void *ptrMask = static_cast<void*>(mask.data_ptr());

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        dispatch_additive_masked_softmax_stream<half, half, float>(
                                 reinterpret_cast<half*>(ptrIn),
                                 reinterpret_cast<const half*>(ptrIn),
                                 reinterpret_cast<const half*>(ptrMask),
                                 seqlen[i],
                                 seqlen[i],
                                 enable_stream ? heads*seqlen[i] : batch*heads*seqlen[i],
                                 enable_stream ? heads*seqlen[i] : heads*seqlen[i],
                                 enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
        ptrIn = static_cast<void*>(static_cast<half*>(ptrIn) + heads*seqlen[i]*seqlen[i]);
        ptrMask = static_cast<void*>(static_cast<half*>(ptrMask) + seqlen[i]);
    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }

    int ntokens = seqlen[0];
    for(int i = 1; i < (enable_stream ? batch : 2); i++) {
        ntokens += seqlen[i];
    }
    auto act_options  = input.options().requires_grad(false);
    auto mask_options = act_options.dtype(torch::kUInt8);
    torch::Tensor dropout_results   = torch::empty({batch*heads, ntokens},   act_options);
    torch::Tensor dropout_mask      = torch::empty({batch*heads, ntokens},   mask_options);
    //torch::Tensor dropout_results   = torch::empty({batch*heads, seqlen[0], seqlen[0]},   act_options);
    //torch::Tensor dropout_mask      = torch::empty({batch*heads, seqlen[0], seqlen[0]},   mask_options);
    if (is_training) {
        //use at:: function so that C++ version generates the same random mask as python version
	//auto dropout_tuple = at::native_dropout(input, 1.0f-dropout_prob, 1./(1.0f-dropout_prob), true);
	auto dropout_tuple = at::_fused_dropout(input, 1.0f-dropout_prob);
        dropout_results = std::get<0>(dropout_tuple);
        dropout_mask = std::get<1>(dropout_tuple);
    }
    return {dropout_results, dropout_mask};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastMaskSoftmaxDropoutBprop_(torch::Tensor &input,
                              torch::Tensor &output,
                              torch::Tensor &dropout_mask,
                              int batch,
                              torch::Tensor &seq_len,
                              int heads,
                              float dropout_prob,
                              bool enable_stream,
                              bool sync)
{
    int *seqlen = static_cast<int*>(seq_len.data_ptr());
    void *ptrIn = static_cast<void*>(input.data_ptr());
    void *ptrOut = static_cast<void*>(output.data_ptr());
    void *ptrDropoutMask = static_cast<void*>(dropout_mask.data_ptr());

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        dispatch_masked_scale_softmax_backward_stream<half, half, float, false>(
                                 static_cast<half*>(ptrOut),
                                 static_cast<half*>(ptrOut),
                                 reinterpret_cast<half const*>(ptrIn),
                                 reinterpret_cast<uint8_t const*>(ptrDropoutMask),
                                 1.0/(1.0-dropout_prob),
                                 seqlen[i],
                                 seqlen[i],
                                 enable_stream ? heads*seqlen[i] : batch*heads*seqlen[i],
                                 enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
        ptrIn = static_cast<void*>(static_cast<half*>(ptrIn) + heads*seqlen[i]*seqlen[i]);
        ptrOut = static_cast<void*>(static_cast<half*>(ptrOut) + heads*seqlen[i]*seqlen[i]);
    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mha_cuda_extension()
{
    // CUDA Stream.
    for(int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // CuBlas Handle.
    cublasCreate(&handle);
    //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("InitMHACUDAExtension", &init_mha_cuda_extension, "InitMHACUDAExtension");
  m.def("FastBmm1Fprop", &FastBmm1Fprop_, "FastBmm1Fprop");
  m.def("FastBmm1Dgrad1", &FastBmm1Dgrad1_, "FastBmm1Dgrad1"); 
  m.def("FastBmm1Dgrad2", &FastBmm1Dgrad2_, "FastBmm1Dgrad2"); 
  m.def("FastBmm2Fprop", &FastBmm2Fprop_, "FastBmm2Fprop");
  m.def("FastBmm2Dgrad1", &FastBmm2Dgrad1_, "FastBmm2Dgrad1");
  m.def("FastBmm2Dgrad2", &FastBmm2Dgrad2_, "FastBmm2Dgrad2");
  m.def("FastSoftmaxFprop", &FastSoftmaxFprop_, "FastSoftmaxFprop");
  m.def("FastSoftmaxBprop", &FastSoftmaxBprop_, "FastSoftmaxBprop");
  m.def("FastMaskSoftmaxFprop", &FastMaskSoftmaxFprop_, "FastMaskSoftmaxFprop");
  m.def("FastMaskSoftmaxDropoutFprop", &FastMaskSoftmaxDropoutFprop_, "FastMaskSoftmaxDropoutFprop");  
  m.def("FastMaskSoftmaxDropoutBprop", &FastMaskSoftmaxDropoutBprop_, "FastMaskSoftmaxDropoutBprop");
}

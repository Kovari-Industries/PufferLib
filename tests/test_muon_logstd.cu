// Reuse the headless learner fixture. Build with the alignment test flags.
#define main alignment_fixture_main
#include "test_muon_alignment.cu"
#undef main
#include <vector>

int main() {
    cublas_init_handle();
    Allocator params{}, scratch{}, acts{}, grads{};
    DecoderWeights dw{}; dw.hidden_dim=4; dw.output_dim=3; dw.continuous=true;
    DecoderActivations da{};
    decoder_reg_params(&dw,&params);
    decoder_reg_train(&dw,&da,&acts,&grads,4);
    assert(ndim(dw.logstd.shape)==1 && ndim(da.logstd_scratch.shape)==1);
    alloc_create(&params);
    Muon m{}; muon_init(&m,&params,.95f,&scratch); alloc_create(&scratch);
    long n=params.total_bytes/sizeof(precision_t);
    cudaMemset(params.mem,0,params.total_bytes);
    Prec param{.data=static_cast<precision_t*>(params.mem),.shape={n}};
    Float weights{}; master_weights_setup(&weights,&param,true,0);
    std::vector<precision_t> g(n,from_float(0.0f));
    long offset=dw.logstd.data-param.data;
    // Exactly representable in both precisions.
    float raw[3]={.125f,-.25f,.5f};
    for(int i=0;i<3;++i) g[offset+i]=from_float(raw[i]);
    Prec grad{.shape={n}}; cudaMalloc(&grad.data,n*sizeof(precision_t));
    cudaMemcpy(grad.data,g.data(),n*sizeof(precision_t),cudaMemcpyHostToDevice);
    float lr=.015f; cudaMemcpy(m.lr,&lr,sizeof(lr),cudaMemcpyHostToDevice);
    muon_step(&m,weights,grad,100);
    assert(cudaDeviceSynchronize()==cudaSuccess);
    float got[3]; cudaMemcpy(got,weights.data+offset,sizeof(got),cudaMemcpyDeviceToHost);
    for(int i=0;i<3;++i) {
        // Nesterov updates are stored in the selected precision before application.
        float expected=-lr*to_float(from_float(1.95f*raw[i]));
        assert(isfinite(got[i]) && fabsf(got[i]-expected)<1e-7f);
    }
    puts("continuous logstd vector update PASS");
}

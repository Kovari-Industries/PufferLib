// Focused contracts for GP objective/gradients, score endpoints, and direction.
// Build: nvcc -O2 -arch=sm_120 -UNDEBUG -Isrc tests/test_protein_contracts.cu \
//        -lcublas -lcusolver -lcurand -o build/test_protein_contracts
// The numerical fixture supplies the production reduction helper.
#define main numerical_fixture_main
#include "test_protein_numerics.cu"
#undef main

static void near_value(const char* name, double got, double expected, double tol) {
    if (!isfinite(got) || !isfinite(expected) || fabs(got - expected) > tol) {
        fprintf(stderr, "%s: got %.9g expected %.9g tolerance %.3g\n", name, got, expected, tol);
        exit(1);
    }
}

// Independent log-density evaluation from the computed Cholesky factor and
// linear solve. Accumulate in double; do not call the training loss code.
static double objective(GaussianProcess* gp, const float* y) {
    assert(gp_recompute(gp, 0) == 0);
    int n = gp->n;
    float L[64], alpha[8];
    assert(n <= 8);
    cudaMemcpy(L, gp->d_L, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(alpha, gp->d_alpha, n*sizeof(float), cudaMemcpyDeviceToHost);
    double quad = 0, logdet = 0;
    for (int i=0; i<n; ++i) {
        quad += (double(y[i]) - gp->mean) * alpha[i];
        logdet += log(double(L[i*n+i]));
    }
    double noise = log1p(exp(double(gp->raw_noise))) + SP_LB;
    double z = (log(noise) - double(NOISE_PRIOR_MU)) / double(NOISE_PRIOR_SIGMA);
    double log_prior = -0.5*z*z - log(noise) - log(double(NOISE_PRIOR_SIGMA));
    return (0.5*quad + logdet + 0.5*n*log(2.0*M_PI) - log_prior) / n;
}

static void gp_contract(int n) {
    GaussianProcess gp{};
    gp_init(&gp, 1, 8, 8, 0.05f);
    gp.mean = 0.3f;
    float X[8], y[8];
    for (int i=0; i<n; ++i) { X[i] = -0.8f + 0.23f*i; y[i] = 0.7f + 0.2f*X[i]; }
    float m[16]{}, v[16]{}, vmax[16]{};
    int t=0;
    float loss = gp_train(&gp, m, v, vmax, &t, 0, X, y, n, 1, 0);
    near_value("normalized GP loss", loss, objective(&gp, y), 2e-5);
    for (int param=0; param<2; ++param) {
        float* value = param ? &gp.mean : &gp.raw_noise;
        float original=*value, h=param ? 0.002f : 0.01f;
        *value=original+h; double plus=objective(&gp,y);
        *value=original-h; double minus=objective(&gp,y);
        *value=original;
        double numerical=(plus-minus)/(2*h);
        double analytic=m[gp.kernel->n_params+param]/(1.0f-0.9f);
        near_value(param ? "mean gradient" : "noise gradient", analytic, numerical, 2e-3);
        printf("n=%d %s gradient analytic=%.9g finite_difference=%.9g\n",
            n, param ? "mean" : "noise", analytic, numerical);
    }
    assert(gp_recompute(&gp, 0) == 0);
    float* prediction;
    cudaMalloc(&prediction, n*sizeof(float));
    gp_predict(&gp, gp.d_X, prediction, n, 0);
    float got[8], alpha[8], K[64];
    cudaMemcpy(got,prediction,n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(alpha,gp.d_alpha,n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(K,gp.d_Ks,n*n*sizeof(float),cudaMemcpyDeviceToHost);
    for (int j=0;j<n;++j) {
        double expected=gp.mean;
        for (int i=0;i<n;++i) expected += double(K[j*n+i])*alpha[i];
        near_value("prediction includes learned mean",got[j],expected,2e-5);
    }
    cudaFree(prediction);
}

static ProteinSweep* sweep(SweepSpace* space, int logit=0) {
    ProteinSweep init{};
    init.space=space; init.num_random_samples=2; init.suggestions_per_pareto=8;
    init.gp_training_iter=3; init.gp_learning_rate=.01f; init.gp_max_obs=32;
    init.infer_batch_size=32; init.use_success_prob=1; init.prune_pareto=1;
    init.use_logit=logit; init.global_search_scale=1; init.max_suggestion_cost=1e9f;
    init.expansion_rate=.05f; init.early_stop_quantile=.3f;
    init.success_cap=32; init.failure_cap=32; init.top_k=3; init.rng_seed=4242;
    return protein_sweep_create(init);
}

static void endpoint_contract() {
    Space dim{.type=SPACE_LINEAR,.min=0,.max=1,.scale=.5f};
    SweepSpace space{&dim,1,-1,1};
    ProteinSweep* sw=sweep(&space,1);
    float values[]={0,nextafterf(0,1),nextafterf(1,0),1,NAN,INFINITY,-INFINITY};
    float previous=-INFINITY;
    for(int i=0;i<7;++i) {
        float x=-.9f+.25f*i;
        protein_sweep_observe(sw,&x,values[i],1,0);
        if(i<4) {
            float transformed=logit_transform(values[i]);
            assert(isfinite(transformed) && transformed>=previous);
            previous=transformed;
        }
    }
    assert(sw->succ_n==4 && sw->fail_n==3);
    puts("logit endpoints accepted; non-finite scores rejected");
}

static void direction_contract() {
    Space dim{.type=SPACE_LINEAR,.min=-1,.max=1,.scale=.5f};
    SweepSpace max_space{&dim,1,-1,1}, min_space{&dim,1,-1,-1};
    ProteinSweep *a=sweep(&max_space), *b=sweep(&min_space);
    // maximize -loss must equal minimize loss, including Pareto/search centers.
    for(int i=0;i<14;++i) {
        float x=-.9f+.13f*i, loss=2+x*x, cost=20+i;
        protein_sweep_observe(a,&x,-loss,cost,i==5);
        protein_sweep_observe(b,&x, loss,cost,i==5);
        assert(a->n_top==b->n_top);
        for(int j=0;j<a->n_top;++j) assert(a->top_idx[j]==b->top_idx[j]);
        float sa,sb;
        srand(100+i); auto ia=protein_sweep_suggest(a,&sa,NAN);
        srand(100+i); auto ib=protein_sweep_suggest(b,&sb,NAN);
        near_value("mirrored suggestion",sa,sb,1e-6);
        near_value("mirrored prediction",ia.predicted_score,-ib.predicted_score,1e-6);
        near_value("mirrored rating",ia.rating,ib.rating,1e-6);
        assert(ia.n_pareto==ib.n_pareto && ia.n_candidates==ib.n_candidates);
    }
    // Force the fitted early-stop path, independent of the minimum fit count.
    a->cm_fitted=b->cm_fitted=1; a->cm_upper=b->cm_upper=100;
    a->cm_A=b->cm_A=-2; a->cm_B=b->cm_B=0;
    assert(protein_sweep_should_stop(a,-3,50)==1);
    assert(protein_sweep_should_stop(b, 3,50)==1);
    assert(protein_sweep_should_stop(a,-1,50)==0);
    assert(protein_sweep_should_stop(b, 1,50)==0);
    b->max_suggestion_cost=0; float out;
    auto rejected=protein_sweep_suggest(b,&out,NAN);
    assert(rejected.is_random); // No invalid GP candidate may win with rating zero.
    puts("minimize/maximize mirrored search and stopping pass");
}

int main() {
    gp_contract(4); gp_contract(7);
    endpoint_contract(); direction_contract();
    assert(cudaDeviceSynchronize()==cudaSuccess);
    puts("PROTEIN contracts PASS");
}

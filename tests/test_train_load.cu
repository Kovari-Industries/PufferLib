// Exercise the real training entry point with zero learning rate and sentinel weights.
// Build with the same flags as test_muon_alignment. Each run is a new process.
#define main alignment_fixture_main
#include "test_muon_alignment.cu"
#undef main
#include <vector>

int main(int argc,char** argv) {
    assert(argc==4 && "usage: test_train_load output_dir async missing");
    Ini ini{}; puf_ini_load_file(&ini,"config/default.ini");
    puf_ini_put(&ini,"base.env_name","muon_alignment");
    puf_ini_put(&ini,"base.run_id","load");
    puf_ini_put(&ini,"base.checkpoint_dir",argv[1]);
    puf_ini_put(&ini,"base.log_dir",argv[1]);
    puf_ini_put(&ini,"base.async",argv[2]);
    puf_ini_put(&ini,"base.cudagraphs","-1");
    puf_ini_put(&ini,"base.eval_episodes","0");
    puf_ini_put(&ini,"vec.total_agents","4");
    puf_ini_put(&ini,"vec.num_buffers","1");
    puf_ini_put(&ini,"vec.num_threads","1");
    puf_ini_put(&ini,"policy.hidden_size","32");
    puf_ini_put(&ini,"policy.num_layers","1");
    puf_ini_put(&ini,"train.horizon","4");
    puf_ini_put(&ini,"train.minibatch_size","16");
    puf_ini_put(&ini,"train.total_timesteps","16");
    puf_ini_put(&ini,"train.learning_rate","0");
    puf_ini_put(&ini,"train.anneal_lr","0");
    TrainContext ctx{.rank=0,.world_size=1,.gpu_id=0,.artifact_owner=1};
    PuffeRL* probe=create_pufferl(&ini,&ctx);
    size_t n=numel(probe->policies[0].master_weights.shape);
    close_pufferl(probe);
    mkdir_p(argv[1]);
    char path[4096]; snprintf(path,sizeof(path),"%s/sentinel.bin",argv[1]);
    std::vector<float> expected(n,.125f), got(n);
    FILE* f=fopen(path,"wb"); assert(f);
    assert(fwrite(expected.data(),sizeof(float),n,f)==n); fclose(f);
    puf_ini_put(&ini,"base.load_model_path",atoi(argv[3]) ? "/missing/puffer-policy.bin" : path);
    run_train(&ini,&ctx);
    assert(!atoi(argv[3]) && "missing policy must not silently start fresh");
    snprintf(path,sizeof(path),"%s/muon_alignment/load/0000000000000016.bin",argv[1]);
    f=fopen(path,"rb"); assert(f);
    assert(fread(got.data(),sizeof(float),n,f)==n); fclose(f);
    assert(memcmp(expected.data(),got.data(),n*sizeof(float))==0);
    puts("train entry point preserves loaded sentinel PASS");
}

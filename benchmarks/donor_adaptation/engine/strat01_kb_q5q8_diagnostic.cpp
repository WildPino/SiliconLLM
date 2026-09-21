#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t kTokens=8,kHeads=32,kQHead=192,kNope=128,kOutput=512,kBlocks=kNope/QK5_0;
constexpr size_t kRows=kHeads*kOutput,kInputCount=kTokens*kHeads*kQHead,kOutputCount=kTokens*kHeads*kOutput;
constexpr uint64_t kTensorOffset=278524800;
constexpr size_t kTensorBytes=1441792;

struct options { std::filesystem::path model,input,output;bool selftest=false; };
[[noreturn]] void fail(const std::string &message){throw std::runtime_error(message);}

options parse(int argc,char **argv){options o;for(int i=1;i<argc;++i){std::string a=argv[i];auto value=[&](const char *name){if(i+1>=argc)fail(std::string("missing value for ")+name);return std::filesystem::path(argv[++i]);};if(a=="--model")o.model=value("--model");else if(a=="--input")o.input=value("--input");else if(a=="--output-dir")o.output=value("--output-dir");else if(a=="--selftest")o.selftest=true;else fail("unknown argument: "+a);}return o;}

std::vector<float> read_f32(const std::filesystem::path &path){if(!std::filesystem::is_regular_file(path)||std::filesystem::file_size(path)!=kInputCount*sizeof(float))fail("input byte count mismatch");std::vector<float> v(kInputCount);std::ifstream f(path,std::ios::binary);if(!f.read(reinterpret_cast<char *>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(float))))fail("cannot read input");for(float x:v)if(!std::isfinite(x))fail("non-finite input");return v;}

std::vector<uint8_t> read_weights(const std::filesystem::path &path){std::vector<uint8_t> v(kTensorBytes);std::ifstream f(path,std::ios::binary);f.seekg(static_cast<std::streamoff>(kTensorOffset));if(!f||!f.read(reinterpret_cast<char *>(v.data()),static_cast<std::streamsize>(v.size())))fail("cannot read frozen K-B span");return v;}

template<class T> void write_payload(const std::filesystem::path &path,const std::vector<T> &v){std::ofstream f(path,std::ios::binary|std::ios::trunc);if(!f.write(reinterpret_cast<const char *>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T))))fail("cannot write "+path.string());}

int selftest(){float input[kNope];for(size_t i=0;i<kNope;++i)input[i]=std::sin(static_cast<float>(i)*0.071f);block_q8_0 q8[kBlocks];quantize_row_q8_0(input,q8,kNope);if(sizeof(block_q5_0)!=22||sizeof(block_q8_0)!=34)fail("pinned block layout mismatch");for(const auto &block:q8)if(block.d==0)fail("unexpected zero Q8 scale");std::cout<<"K-B pinned helper selftest passed\n";return 0;}

int run(const options &o){if(o.model.empty()||o.input.empty()||o.output.empty())fail("model, input, and output-dir are required");std::filesystem::create_directories(o.output);auto input=read_f32(o.input);auto weights=read_weights(o.model);std::vector<block_q8_0> q8(kTokens*kHeads*kBlocks);std::vector<float> output(kOutputCount);
    for(size_t token=0;token<kTokens;++token)for(size_t head=0;head<kHeads;++head){const float *x=input.data()+(token*kHeads+head)*kQHead;block_q8_0 *qx=q8.data()+(token*kHeads+head)*kBlocks;quantize_row_q8_0(x,qx,kNope);for(size_t feature=0;feature<kOutput;++feature){const size_t row=head*kOutput+feature;const auto *qw=reinterpret_cast<const block_q5_0 *>(weights.data()+row*kBlocks*sizeof(block_q5_0));float value=0.0f;ggml_vec_dot_q5_0_q8_0(kNope,&value,sizeof(float),qw,kBlocks*sizeof(block_q5_0),qx,kBlocks*sizeof(block_q8_0),1);if(!std::isfinite(value))fail("non-finite pinned output");output[(token*kHeads+head)*kOutput+feature]=value;}}
    write_payload(o.output/"q8.bin",q8);write_payload(o.output/"q5q8.f32le",output);std::cout<<"wrote pinned K-B Q5_0/Q8_0 payloads\n";return 0;}
}

int main(int argc,char **argv){try{ggml_cpu_init();options o=parse(argc,argv);return o.selftest?selftest():run(o);}catch(const std::exception &e){std::cerr<<"K-B pinned helper: "<<e.what()<<"\n";return 2;}}

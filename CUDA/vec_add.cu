#include <iostream>
#include <cassert>
#include<time.h>
#include <cuda_runtime.h>

void add_host(long N, long *c, long *a, long *b) {
    for(long i=0; i<N; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__
void add(long N, long *c, long *a, long *b) {
    const int start = (blockIdx.x*blockDim.x + threadIdx.x)*N, end = start+N;
    //printf("%d %d\n", start, end);
    for(long i=start; i<end; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__
void add1(long N, long *c, long *a, long *b) {
    const int start = (blockIdx.x*blockDim.x + threadIdx.x)*N, end = start+N;
    //printf("%d %d\n", start, end);
    for(long i=start; i<end; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    long *h_a, *h_b, *h_c, *h_c_copy, size=128;
    long *d_a, *d_b, *d_c;
    if(argc>=2)
        size = std::atol(argv[1]);
    std::cout<<"Size " << size << std::endl;

    h_a = (long*)malloc(size * sizeof(long));
    h_b = (long*)malloc(size * sizeof(long));
    h_c = (long*)malloc(size * sizeof(long));
    h_c_copy = (long*)malloc(size * sizeof(long));

    cudaMalloc((void**)&d_a, size * sizeof(long));
    cudaMalloc((void**)&d_b, size * sizeof(long));
    cudaMalloc((void**)&d_c, size * sizeof(long));

    for(long i=0; i<size; i++) {
        h_a[i] = std::rand();
        h_b[i] = std::rand();
    }

    cudaMemcpy(d_a, h_a, size * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(long), cudaMemcpyHostToDevice);
    clock_t start,end;
    double time;
    start = clock();
    add_host(size, h_c, h_a, h_b);
    end = clock();
    time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("time %lf[ms]\n", time);

    assert(size >= 2*32);
    const long num_blocks = 2;
    const long num_threds_per_block = 32;
    const long block_size = size/num_blocks;
    start = clock();
    add<<<num_blocks, block_size/num_threds_per_block>>>(num_threds_per_block, d_c, d_a, d_b);
    add1<<<num_blocks, block_size/num_threds_per_block>>>(num_threds_per_block, d_c, d_a, d_b);
    cudaMemcpy(h_c_copy, d_c, size * sizeof(long), cudaMemcpyDeviceToHost);
    end = clock();
    time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("time %lf[ms]\n", time);

    bool ret = std::equal(h_c, h_c+size, h_c_copy);
    if (ret){
        printf("Equal test succeed.\n");
    }else{
    std::cout<<"Equals test "<<ret<<std::endl;
}

//    print("Output Array1\n", h_c, size);
//    print("Output Array2\n", h_c_copy, size);

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
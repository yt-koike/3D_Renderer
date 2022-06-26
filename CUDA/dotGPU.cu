#include<iostream>
#include"VectorGPU.h"
#define N (1024*1024*32)
#define THREADS_PER_BLOCK 512

typedef struct {
double x,y,z;
}Vec3Simple;

__global__ void dot(Vec3Simple* a,Vec3Simple* b,double *dotResult)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Vec3Simple v1 = a[i];
    Vec3Simple v2 = b[i];
//    printf("%d %f\n",i,v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
    dotResult[i] = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Vec3Simple simplize(Vec3 v){
    Vec3Simple res;
    res.x = v.getX();
    res.y = v.getY();
    res.z = v.getZ();
    return res;
}

Vec3 vectorize(Vec3Simple v_simple){
    Vec3 v(v_simple.x,v_simple.y,v_simple.z);
    return v;
}
inline int ceil(double x,double deno){int div = x/deno;return div+(x>div*deno);}

int main()
{
    Vec3Simple *a, *b;
    double *c,*dev_c;
    Vec3Simple *dev_a, *dev_b;

    int size = N * sizeof(Vec3Simple);

   //allocate space for the variables on the device
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);
   //allocate space for the variables on the host
   a = (Vec3Simple *)malloc(size);
   b = (Vec3Simple *)malloc(size);
   c = (double *)malloc(sizeof(double)*N);

   //generate numbers
   for (int i = 0; i < N; i++)
   {
    Vec3 tmpV;
       tmpV.set(rand() % 10,rand() % 10,rand() % 10);
       a[i] = simplize(tmpV);
       tmpV.set(rand() % 10,rand() % 10,rand() % 10);
       b[i] = simplize(tmpV);
   }
clock_t st,ed;
st = clock();
   for (int i = 0; i < N; i++)
   {
    Vec3 v1 = vectorize(a[i]);
    Vec3 v2 = vectorize(b[i]);
    Vec3 dotResult = v1.dot(v2);
   }
   ed = clock();
   double CPU_time = (double)(ed-st)/CLOCKS_PER_SEC;
   cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dim3 block(3,1,1);
    dim3 grid(ceil(N,block.x),1,1);

st = clock();
   dot<<< grid, block >>>(dev_a, dev_b, dev_c);
   //triangleIntersection<<<grid,block>>>(dev_a,dev_b,nullptr);
   ed = clock();
   double GPU_time = (double)(ed-st)/CLOCKS_PER_SEC;

   cudaMemcpy(c, dev_c, sizeof(double)*N, cudaMemcpyDeviceToHost);

   for (int i = 0; i < 5; i++)
   {
    Vec3 v1 = vectorize(a[i]);
    Vec3 v2 = vectorize(b[i]);
    v1.print();
    v2.print();
       printf("CPU:%f, GPU:%f \n",v1.dot(v2),c[i]);
   }
   printf("CPU: %f s\n",CPU_time);
   printf("GPU: %f s\n",GPU_time);
   free(a);
   free(b);
   free(c);

   cudaFree(a);
   cudaFree(b);
   cudaFree(c);

   //system("pause");

   return 0;

 }
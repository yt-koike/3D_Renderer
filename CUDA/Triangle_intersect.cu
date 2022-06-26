#ifndef TRI_INTERSECT_H
#define TRI_INTERSECT_H
#include"../headers/shapes/ShapeSuite.h"

__global__ void gpuCalc(double *triVertex,double *distance,Ray *r){
    int i = blockIdx.x;
    if(threadIdx.x!=0)return;
    distance[0] = 0;
}

void getTriVertex(int size,Triangle** tris,double* triVertex){
for(int i=0;i<size;i++){
    Triangle* tri = tris[i];
    int offset = i * 9;
    Vec3* vertexPs[3];
    vertexPs[0] = tri->getV1p();
    vertexPs[1] = tri->getV2p();
    vertexPs[2] = tri->getV3p();
    for(int j=0;j<3;j++){
    triVertex[offset+j*3+0] = vertexPs[j]->getX();
    triVertex[offset+j*3+1] = vertexPs[j]->getY();
    triVertex[offset+j*3+2] = vertexPs[j]->getZ();
    }
}
}
IntersectionPoint testIntersectionGPU(int size,double* triVertex,Triangle** tris, Ray r){
    IntersectionPoint result;    
    size_t vertex_byte = size*sizeof(double)*3*3; // three coords per three vertexes
double *d_triVs,*d_distance;
Ray *d_ray;
cudaMalloc((void**)&d_triVs,vertex_byte);
cudaMalloc((void**)&d_distance,size*sizeof(double));
cudaMalloc((void**)&d_ray,sizeof(Ray));
cudaMemcpy(d_triVs, triVertex, vertex_byte, cudaMemcpyHostToDevice);
cudaMemcpy(d_ray,&r, sizeof(Ray), cudaMemcpyHostToDevice);
const int blocksize = 9;
dim3 block(blocksize,1,1);
dim3 grid(vertex_byte/block.x,1,1);
gpuCalc<<<grid,block>>>(d_triVs,d_distance,d_ray);
    double *distance = new double(size);
cudaMemcpy(distance, d_distance, size, cudaMemcpyDeviceToHost);
cudaFree(d_triVs);
cudaFree(d_distance);
cudaFree(d_ray);

for(int i=0;i<size;i++){
    printf("%f\n",distance[i]);
}

int triIdx = -1;
double minDistance = -1;
for(int i=0;i<size;i++){
    if(distance[i]<0)continue;
    if(minDistance<distance[i] || minDistance==-1){
        triIdx = i;
        minDistance = distance[i];
    }
}
delete distance;
if(triIdx<0){return result;}
result.exists=1;
result.distance = minDistance;
result.normal = tris[triIdx]->getNormalV().copy();
result.position = r.getDir().mult(minDistance).add(r.getPoint());

return result;
}
#endif
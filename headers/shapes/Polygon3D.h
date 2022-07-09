#ifndef POLYGON3D_H
#define POLYGON3D_H
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"
#include "ShapeSuite.h"
#include "Triangle.h"
#include "../kdTree.h"

class Polygon3D : public Shape
{
private:
  unsigned int size = 0;
  unsigned int maxSize = 0;
  Triangle **tris;
  BoundaryBox *boundary = nullptr;
  std::vector<Triangle*> kdTree,nearestToCamera;
public:
  Polygon3D(int maxSize)
  {
    this->maxSize = maxSize;
    tris = new Triangle *[this->maxSize];
    boundary = new BoundaryBox();
  }
  Polygon3D rotate(Vec3 origin, Vec3 axis, double rad)
  {
    Polygon3D poly(maxSize);
    for (int i = 0; i < size; i++)
    {
      poly.addTriangle(new Triangle(tris[i]->rotate(origin, axis, rad)));
    }
    return poly;
  }
  Polygon3D move(Vec3 dV)
  {
    Polygon3D poly(maxSize);
    for (int i = 0; i < size; i++)
    {
      poly.addTriangle(new Triangle(tris[i]->move(dV)));
    }
    Vec3 startV = boundary->getStartV().add(dV);
    Vec3 endV = boundary->getEndV().add(dV);
    poly.setBoundary(new BoundaryBox(startV, endV));
    return poly;
  }
  void print()
  {
    printf("Polygon: %d triangles\n", size);
    for (int i = 0; i < size; i++)
      tris[i]->print();
  }
  void generateBoundary();
  BoundaryBox *getBoundary() { if(boundary==nullptr)generateBoundary(); return boundary; }
  Polygon3D *copy()
  {
    Polygon3D *newPoly = new Polygon3D(maxSize);
    for (int i = 0; i < size; i++)
      newPoly->addTriangle(tris[i]->copy());
    newPoly->setBoundary(boundary->copy());
    return newPoly;
  }
  void setBoundary(BoundaryBox *boundary) { this->boundary = boundary; }
  void addTriangle(Triangle *tri);
  Triangle **getTriangles(int *n)
  {
    *n = size;
    return tris;
  }
  IntersectionPoint testIntersection(Ray r);
  int find(std::vector<Vec3 *> vs, Vec3 v)
  {
    for (int i = 0; i < vs.size(); i++)
    {
      if (vs[i]->equals(v))
        return i;
    }
    return -1;
  }
  void buildKdTree(){
    kdTree = makeKdTree(size,tris);
  }
  std::vector<Triangle *> searchNearest(Vec3 p,int queryN){
    if(kdTree.size()!=size)buildKdTree();
    return searchKdTree(&kdTree,p,queryN);
  }
#ifdef GPU_MODE
  void testIntersections(int rayN, Ray *rs, IntersectionPoint *result);
#endif
};

void Polygon3D::addTriangle(Triangle *tri)
{
  if (size < maxSize)
  {
    tris[size] = tri;
    size++;
  }
}

void Polygon3D::generateBoundary()
{
  boundary = new BoundaryBox();
  for (int i = 0; i < size; i++)
  {
    Triangle *tri = tris[i];
    boundary->includeV(tri->getV1());
    boundary->includeV(tri->getV2());
    boundary->includeV(tri->getV3());
  }
}


IntersectionPoint Polygon3D::testIntersection(Ray r)
{
  IntersectionPoint boundaryCross = boundary->testIntersection(r);
  IntersectionPoint cross;
  IntersectionPoint noCross;
  if (!boundaryCross.exists)return cross;
  if(nearestToCamera.size()!=size)
    nearestToCamera = searchNearest(r.getPoint(),size);
  Triangle* hitTri;
  int closestId = -1;
  double closestDistance = -1;
  for (int i = 0; i < nearestToCamera.size(); i++)
  {
    cross = nearestToCamera[i]->testIntersection(r);
    if (cross.exists)
    {
      if (closestId == -1)
      {
        closestId = i;
        closestDistance = cross.distance;
        hitTri = nearestToCamera[i];
        break;
      }
    }
  }
  if(closestId==-1)
    return noCross;
  const unsigned int searchLimitN = 100;
  for(int i=closestId;i<closestId+searchLimitN;i++){
        cross = nearestToCamera[i]->testIntersection(r);
    if (cross.exists)
    {
      if (cross.distance<closestDistance)
      {
        closestId = i;
        closestDistance = cross.distance;
        hitTri = nearestToCamera[i];
      }
    }
  }
  /*
  std::vector<Triangle*> nearestTriangles = searchNearest(cross.position,searchLimitN);
  for (int i = 0; i < searchLimitN; i++)
  {
    cross = nearestTriangles[i]->testIntersection(r);
    if (cross.exists)
    {
      if (cross.distance<closestDistance)
      {
        closestId = i;
        closestDistance = cross.distance;
        hitTri = nearestTriangles[i];
      }
    }
  } 
  */
  return hitTri->testIntersection(r);
}


#ifdef GPU_MODE
typedef struct
{
  double x, y, z;
} Vec3Simple;

Vec3Simple simplize(Vec3 v)
{
  Vec3Simple res;
  res.x = v.getX();
  res.y = v.getY();
  res.z = v.getZ();
  return res;
}

Vec3 vectorize(Vec3Simple v_simple)
{
  Vec3 v(v_simple.x, v_simple.y, v_simple.z);
  return v;
}

__device__ inline double simpleDot(Vec3Simple a, Vec3Simple b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ void simpleCross(Vec3Simple a, Vec3Simple b, Vec3Simple *result)
{
  result->x = a.y * b.z - b.y * a.z;
  result->y = a.z * b.x - b.z * a.x;
  result->z = a.x * b.y - b.x * a.y;
}

__device__ void simpleAdd(Vec3Simple a, Vec3Simple b, Vec3Simple *result)
{
  result->x = a.x + b.x;
  result->y = a.y + b.y;
  result->z = a.z + b.z;
}

__device__ void simpleSub(Vec3Simple a, Vec3Simple b, Vec3Simple *result)
{
  result->x = a.x - b.x;
  result->y = a.y - b.y;
  result->z = a.z - b.z;
}

__device__ void simpleMult(Vec3Simple a, double n, Vec3Simple *result)
{
  result->x = a.x * n;
  result->y = a.y * n;
  result->z = a.z * n;
}

__constant__ double ray_dir[3];
__constant__ double ray_pos[3];
__global__ void triIntersection_GPU(Vec3Simple *vs1, Vec3Simple *vs2, Vec3Simple *vs3, double *distance, unsigned int *d_hitIdxN)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  distance[i] = -1;
  Vec3Simple v1, v2, v3;
  v1 = vs1[i];
  v2 = vs2[i];
  v3 = vs3[i];
  Vec3Simple E0, E1, D, T, P, Q;
  simpleSub(v2, v1, &E0);
  simpleSub(v3, v1, &E1);
  D.x = ray_dir[0];
  D.y = ray_dir[1];
  D.z = ray_dir[2];
  T.x = ray_pos[0] - v1.x;
  T.y = ray_pos[1] - v1.y;
  T.z = ray_pos[2] - v1.z;
  simpleCross(D, E1, &P);
  simpleCross(T, E0, &Q);
  double deno = simpleDot(P, E0);
  double u = simpleDot(P, T) / deno;
  double v = simpleDot(Q, D) / deno;
  if (!(u >= 0 && v >= 0 && u + v <= 1))
    return;
  distance[i] = simpleDot(Q, E1) / deno;
  atomicAdd(d_hitIdxN, 1);
}

inline int ceil(double x, double deno)
{
  int div = x / deno;
  return div + (x > div * deno);
}
/*
__global__ void min_GPU(double *ary,int *idx){
unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;
idx[globalId]=globalId;
for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
  if(threadIdx.x<s){
    if(ary[globalId+s]<ary[globalId]){
      ary[globalId]=ary[globalId+s];
      idx[globalId]=idx[globalId+s];
    }
  }
  __syncthreads();
}
}
*/
__host__ void PolyIntersection_GPU(int triN, Triangle **tris, int rayN, Ray *rs, BoundaryBox *boundary, IntersectionPoint *result)
{
  Vec3Simple *v1, *v2, *v3;
  Vec3Simple *d_v1, *d_v2, *d_v3;
  double *distance = (double *)malloc(triN * sizeof(double));
  // unsigned int *hitIdx = (unsigned int *)malloc(triN * sizeof(unsigned int));

  int size = triN * sizeof(Vec3Simple);

  // allocate space for the variables on the device
  cudaMalloc((void **)&d_v1, size);
  cudaMalloc((void **)&d_v2, size);
  cudaMalloc((void **)&d_v3, size);
  double *d_distance;
  cudaMalloc((void **)&d_distance, triN * sizeof(double));
  unsigned int *d_hitIdxN;
  cudaMalloc((void **)&d_hitIdxN, sizeof(unsigned int));
  // allocate space for the variables on the host
  v1 = (Vec3Simple *)malloc(size);
  v2 = (Vec3Simple *)malloc(size);
  v3 = (Vec3Simple *)malloc(size);
  // generate numbers
  for (int i = 0; i < triN; i++)
  {
    v1[i] = simplize(tris[i]->getV1());
    v2[i] = simplize(tris[i]->getV2());
    v3[i] = simplize(tris[i]->getV3());
  }

  cudaMemcpy(d_v1, v1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, v2, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3, v3, size, cudaMemcpyHostToDevice);

  for (int rayIdx = 0; rayIdx < rayN; rayIdx++)
  {
    result[rayIdx].exists = 0;
    Ray r = rs[rayIdx];
    if (!boundary->testIntersection(r).exists)
      continue;

    unsigned int hitIdxN = 0;
    Vec3Simple simple_dir = simplize(r.getDir());
    Vec3Simple simple_pos = simplize(r.getPoint());
    cudaMemcpyToSymbol(ray_dir, &simple_dir, sizeof(Vec3Simple));
    cudaMemcpyToSymbol(ray_pos, &simple_pos, sizeof(Vec3Simple));
    cudaMemcpy(d_hitIdxN, &hitIdxN, sizeof(unsigned int), cudaMemcpyHostToDevice);
    dim3 block(512, 1, 1);
    dim3 grid(ceil(triN, block.x), 1, 1);
    triIntersection_GPU<<<grid, block>>>(d_v1, d_v2, d_v3, d_distance, d_hitIdxN);
    cudaMemcpy(&hitIdxN, d_hitIdxN, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (hitIdxN == 0)
      continue;
    cudaMemcpy(distance, d_distance, triN * sizeof(double), cudaMemcpyDeviceToHost);
    int triIdx = -1;
    int foundFlag = 0;
    double minDistance;
    for (int i = 0; i < triN; i++)
    {
      double d = distance[i];
      if (d < 0)
        continue;
      if (!foundFlag || d < minDistance)
      {
        triIdx = i;
        minDistance = d;
        foundFlag = 1;
      }
    }
    if (triIdx >= 0)
    {
      result[rayIdx].exists = 1;
      result[rayIdx].distance = minDistance;
      result[rayIdx].normal = tris[triIdx]->getNormalV();
      result[rayIdx].position = r.getDir().mult(minDistance).add(r.getPoint());
    }
  }
  free(v1);
  free(v2);
  free(v3);
  free(distance);
  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_v3);
  cudaFree(d_distance);
  cudaFree(d_hitIdxN);
  return;
}

void Polygon3D::testIntersections(int rayN, Ray *rays, IntersectionPoint *result)
{
  PolyIntersection_GPU(size,tris,rayN,rays,boundary,result);return;
  if(nearestToCamera.size()!=size)
    nearestToCamera = searchNearest(rays[0].getPoint(),size);
  const int checkTriN = size/2;
  Triangle** nearestTris = new Triangle*[checkTriN];
  for(int i=0;i<checkTriN;i++){
    nearestTris[i] = nearestToCamera[i];
  }
  PolyIntersection_GPU(checkTriN,nearestTris,rayN,rays,boundary,result);
  delete nearestTris;
  return;
}
#endif


#endif
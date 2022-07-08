#ifndef GPU_MAN_H
#define GPU_MAN_H
#include "shapes/ShapeSuite.h"
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
__global__ void triIntersection_GPU(Vec3Simple *vs1, Vec3Simple *vs2, Vec3Simple *vs3, double *distance)
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
}

inline int ceil(double x, double deno)
{
    int div = x / deno;
    return div + (x > div * deno);
}

__host__ void PolyIntersection_GPU(int triN, Triangle **tris, int rayN, Ray *rs, IntersectionPoint *result)
{
    Vec3Simple *v1, *v2, *v3;
    Vec3Simple *d_v1, *d_v2, *d_v3;

    int size = triN * sizeof(Vec3Simple);

    // allocate space for the variables on the device
    cudaMalloc((void **)&d_v1, size);
    cudaMalloc((void **)&d_v2, size);
    cudaMalloc((void **)&d_v3, size);
    double *d_distance;
    cudaMalloc((void **)&d_distance, triN * sizeof(double));
    // allocate space for the variables on the host
    v1 = (Vec3Simple *)malloc(size);
    v2 = (Vec3Simple *)malloc(size);
    v3 = (Vec3Simple *)malloc(size);
    double *distance = (double *)malloc(triN * sizeof(double));

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
        Ray r = rs[rayIdx];
        Vec3Simple simple_dir = simplize(r.getDir());
        Vec3Simple simple_pos = simplize(r.getPoint());
        cudaMemcpyToSymbol(ray_dir, &simple_dir, sizeof(Vec3Simple));
        cudaMemcpyToSymbol(ray_pos, &simple_pos, sizeof(Vec3Simple));
        dim3 block(3, 1, 1);
        dim3 grid(ceil(triN, block.x), 1, 1);
        triIntersection_GPU<<<grid, block>>>(d_v1, d_v2, d_v3, d_distance);
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
        IntersectionPoint cross;
        if (triIdx >= 0)
        {
            cross.exists = 1;
            cross.distance = minDistance;
            cross.normal = tris[triIdx]->getNormalV();
            cross.position = r.getDir().mult(minDistance).add(r.getPoint());
        }
        result[rayIdx] = cross;
    }
    free(v1);
    free(v2);
    free(v3);
    free(distance);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_v3);
    cudaFree(d_distance);
    return;
}

class GPU_Manager
{
private:
    const int maxSize = 100000;
    int rayN;
    Ray *rs;
    IntersectionPoint *ps;

public:
    GPU_Manager()
    {
        rayN = 0;
        rs = new Ray[maxSize];
        ps = new IntersectionPoint[maxSize];
    }

    ~GPU_Manager(){
        delete rs;
        delete ps;
    }

    void calc(int triN, Triangle **tris)
    {
        PolyIntersection_GPU(triN, tris, rayN, rs, ps);
        rayN = 0;
    }

    int addRay(Ray r)
    {
        rs[rayN] = r;
        rayN ++;
        return rayN - 1; // return ray ID
    }

    IntersectionPoint getResult(int rayId)
    {
        return ps[rayId];
    }
};
#endif

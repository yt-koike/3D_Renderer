#include "headers/kdTree.h"
#include "headers/STL.h"

Triangle* normalSearch(int triN,Triangle** tris,Vec3 p){
    int idx = 0;
    double minDistanceSq = triG(tris[idx]).sub(p).magSq();
    for(int i=1;i<triN;i++){
        Vec3 G = triG(tris[i]);
        double distanceSq = G.sub(p).magSq();
        if(distanceSq<minDistanceSq){
            minDistanceSq = distanceSq;
            idx = i;
        }
    }
    return tris[idx];
}

int main()
{
    Polygon3D sphere = STLBinLoad("ICO_Sphere.stl");
    int n;
    Triangle ** tris = sphere.getTriangles(&n);
    std::vector<Triangle*> kdTree = makeKdTree(n,tris);
/*
    for(int i=0;i<kdTree.size();i++){
      //  kdTree[i]->print();
    }
    */
    Vec3 p(10);
    std::vector<Triangle*> a = sphere.searchNearest(p,80);
    Triangle *b = normalSearch(n,tris,p);
    for(int i=0;i<a.size();i++){
    printf("%f\n",triG(a[i]).sub(p).mag());
    }
    printf("%f",triG(b).sub(p).mag());
    return 0;
}
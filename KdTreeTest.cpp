#include<iostream>
#include"headers/kdTree.h"
#include"headers/STL.h"
    Vec3 triG(Triangle *tri)
    {
        return (tri->getV1()).add(tri->getV2()).add(tri->getV3());
    }

int main(){
    Polygon3D cone = STLBinLoad("STL/Cone.stl").move(Vec3(1));
    int size;
    Triangle** tris;
    tris = cone.getTriangles(&size);
    Vec3 p(0.5f);
    const int queryN = cone.getKdTree()->getNodeN();
    Triangle* res[queryN];
    cone.getKdTree()->searchNearest(p,queryN,res);
    for(int i=0;i<queryN;i++){printf("%d:\n",i);res[i]->print();}
    printf("%d\n",size);
    unsigned int minIdx=0;
    double minDis=triG(tris[0]).sub(p).magSq();
    for(int i=0;i<size;i++){
        double distance = triG(tris[i]).sub(p).magSq();
        if(distance<minDis){
            minDis=distance;
            minIdx = i;
        }
    }
    tris[minIdx]->print();
    return 0;
}
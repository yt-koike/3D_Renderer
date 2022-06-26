#include <stdio.h>
#include<stdlib.h>
#include<iostream>
#include"../headers/STL.h"
#include"../headers/shapes/ShapeSuite.h"

#include"Triangle_intersect.cu"
using namespace std;

int main(){
    Polygon3D cone = STLBinLoad("../STL/Cone.stl");
    Triangle **tris;
    int size;
    tris = cone.getTriangles(&size);
/*    for(int i=0;i<size;i++){
        tris[i]->print();
    }
    cout<<size<<endl;
    cout<<sizeof(Triangle)<<endl;
    */
   
    Ray r(Vec3(0,0,-10),Vec3(0,0,1));
    double* triVertex = new double[size*3*3];
    getTriVertex(size,tris,triVertex);
    IntersectionPoint res = testIntersectionGPU(size,triVertex,tris,r);
    if(res.exists){
        res.position.print();
    }else{
        cout<<"Not found";
    }
    return 0;
}
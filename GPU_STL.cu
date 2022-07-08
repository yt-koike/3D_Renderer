#include<stdio.h>
#include <string.h>
#include <unistd.h>
#include "headers/RenderSuite.h"
#include "headers/PPM.h"
#include "headers/STL.h"
#define POLY_INTERSECTION_GPU_H
int main(int argn,char** argv)
{
    Ray camera(Vec3(0, 0,-5), Vec3(0, 0,1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237));
    // load cube from file
    Material coneMt(Color(Vec3(0.1)), Color(Vec3(0.69,0,0)), Color(Vec3(0.3)), Vec3(8));
    Polygon3D cone = STLBinLoad("STL/Cone.stl").move(Vec3(-1,0.1,5));
    cone.setMaterial(coneMt);
    scene.add(&cone);
    Polygon3D ICO = STLBinLoad("STL/ICO_Sphere.stl").move(Vec3(1,0.1,3));
    scene.add(&ICO);
    Polygon3D ICO2 = *ICO.copy();
    ICO2.move(Vec3(0,0,10));
    scene.add(&ICO2);
    Material mirrorMt(Color(Vec3(0.01)), Color(Vec3(0.1)), Color(Vec3(0.1)), Vec3(8));
    mirrorMt.setUsePerfectReflectance(1);
    mirrorMt.setCatadioptricFactor(Color(Vec3(0.7)));
    Sphere* sp = new Sphere(Vec3(0,1,10),0.5,mirrorMt);
    //scene.add(sp);
    printf("Load Complete.\n");

    scene.add(new Plane(Vec3(0,-1,0),Vec3(0,1,0)));

    // add lights
    scene.addLight(new PointLightSource(Vec3(0, 0, -5), Color(Vec3(1))));
    scene.addLight(new PointLightSource(Vec3(0, 20, 0), Color(Vec3(1))));

    int width,height;
    switch(argn){
        case 2:
        sscanf(argv[1],"%d",&width);
        sscanf(argv[1],"%d",&height);
        break;
        case 3:
        sscanf(argv[1],"%d",&width);
        sscanf(argv[2],"%d",&height);
        break;
        default:
        width = height = 256;
        break;
    }
    char filename[100];
    PPM ppmwriter(width, height, 255);
    clock_t st,ed;
    printf("Render Start. (%d x %d)\n",width, height);
    st = clock();
    ColorImage img = scene.draw(width, height);
    ed = clock();
    printf("Render End.\n");
    sprintf(filename, "GPU_STL.ppm");
    ppmwriter.import(img);
    ppmwriter.writePPM(filename);
    //irfanview(filename);
    return 0;
}
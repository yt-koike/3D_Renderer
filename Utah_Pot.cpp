#include<stdio.h>
#include <string.h>
#include <unistd.h>
#include "headers/RenderSuite.h"
#include "headers/PPM.h"
#include "headers/STL.h"

void irfanview(const char *filename)
{
    const char *irfanPath = "C:\\Software\\IrfanView\\i_view64.exe";
    char directory[128] = "D:\\GoogleDrive\\3年春\\プロジェクト(小池)\\C++\\Renderer\\";
    execl(irfanPath, irfanPath, strcat(directory, filename));
}

int main(int argn,char** argv)
{
    Ray camera(Vec3(0, 0, -50), Vec3(0, 0,1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237));
    // load cube from file
    Material coneMt(Color(Vec3(0.1)), Color(Vec3(0.69,0,0)), Color(Vec3(0.3)), Vec3(8));
    Polygon3D pot = STLBinLoad("Utah_teapot.stl").move(Vec3(0,0,5));
    pot.setMaterial(coneMt);
    scene.add(&pot);
    pot.getBoundary()->print();
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
    printf("time: %f s\n",(double)(ed-st)/CLOCKS_PER_SEC);
    sprintf(filename, "Utah_Pot.ppm");
    ppmwriter.import(img);
    ppmwriter.writePPM(filename);
   // irfanview(filename);
    return 0;
}
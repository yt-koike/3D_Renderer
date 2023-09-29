#define BOUNDARY_BOX_MODE
// #define KD_TREE_MODE
// #define GPU_MODE
#define SHADOW_ENABLE
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "headers/RenderSuite.h"
#include "headers/PPM.h"
#include "headers/STL.h"

int main(int argn, char **argv)
{
    Ray camera(Vec3(0, 0, -5), Vec3(0, 0, 1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237), 100, 100);
    // load cube from file
    Material whiteMt(Color(Vec3(0.1)), Color(Vec3(0.69)), Color(Vec3(0.3)), Vec3(8));
    Material redMt(Color(Vec3(0.1)), Color(Vec3(0.69, 0, 0)), Color(Vec3(0.3, 0, 0)), Vec3(8));
    Material greenMt(Color(Vec3(0.1)), Color(Vec3(0, 0.69, 0)), Color(Vec3(0, 0.3, 0)), Vec3(8));
    Material blueMt(Color(Vec3(0.1)), Color(Vec3(0, 0, 0.69)), Color(Vec3(0, 0, 0.3)), Vec3(8));

    Material coneMt(Color(Vec3(0.1)), Color(Vec3(0.69, 0, 0)), Color(Vec3(0.3)), Vec3(8));
    Polygon3D cone = STLBinLoad("STL/Cone.stl").move(Vec3(-1, 0.1, 5));
    cone.setMaterial(coneMt);
    /*
        int size;
        Triangle **tris = cone.getTriangles(&size);
        for(int i=0;i<size;i++){
            Triangle *t = tris[i];
            Vec3 triG =t->getV1().add(t->getV2()).add(t->getV3()).mult(0.33f);
            scene.add(new Sphere(triG,0.05));
        }
        */
    scene.add(&cone);
    Polygon3D ICO = STLBinLoad("STL/ICO_Sphere.stl").move(Vec3(1, 1, 3));
    scene.add(&ICO);
    Polygon3D ICO2 = *ICO.copy();
    ICO2.move(Vec3(0, 0, 10));
    // scene.add(&ICO2);
    printf("Load Complete.\n");

    scene.add(new Plane(Vec3(0, -1, 0), Vec3(0, 1, 0)));

    // add lights
    scene.addLight(new PointLightSource(Vec3(0, 0, -5), Color(Vec3(1))));
    scene.addLight(new PointLightSource(Vec3(0, 20, 0), Color(Vec3(1))));

    int width, height;
    switch (argn)
    {
    case 2:
        sscanf(argv[1], "%d", &width);
        sscanf(argv[1], "%d", &height);
        break;
    case 3:
        sscanf(argv[1], "%d", &width);
        sscanf(argv[2], "%d", &height);
        break;
    default:
        width = height = 256;
        break;
    }
    char filename[100];
    PPM ppmwriter(width, height, 255);
    printf("Render Start. (%d x %d)\n", width, height);
    clock_t st, ed;
    st = clock();
    ColorImage img = scene.draw(width, height);
    ed = clock();
    printf("time %f\n", (double)(ed - st) / CLOCKS_PER_SEC);
    printf("Render End.\n");
    sprintf(filename, "STL_Render.ppm");
    ppmwriter.import(img);
    ppmwriter.writePPM(filename);

    return 0;
}

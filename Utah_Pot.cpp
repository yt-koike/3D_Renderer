#define BOUNDARY_BOX_MODE
//#define KD_TREE_MODE
//#define GPU_MODE
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
    int boundary_enabled, kd_tree_enabled, gpu_enabled;
    boundary_enabled = 0;
    kd_tree_enabled = 0;
    gpu_enabled = 0;
    #ifdef BOUNDARY_BOX_MODE
    boundary_enabled=1;
    #endif
    #ifdef KD_TREE_MODE
    kd_tree_enabled = 1;
    #endif
    #ifdef GPU_MODE
    gpu_enabled = 1;
    #endif
    printf("Optimization: \"%c%c%c\"\n",(boundary_enabled)?'B':'-',(kd_tree_enabled)?'K':'-',(gpu_enabled)?'G':'-');
    Ray camera(Vec3(0, 5, -35), Vec3(0, 0, 1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237),100,100);
    // load cube from file
    Material whiteMt(Color(Vec3(0.1)), Color(Vec3(0.69)), Color(Vec3(0.3)), Vec3(8));
    Material redMt(Color(Vec3(0.1)), Color(Vec3(0.69, 0, 0)), Color(Vec3(0.3, 0, 0)), Vec3(8));
    Material greenMt(Color(Vec3(0.1)), Color(Vec3(0, 0.69, 0)), Color(Vec3(0, 0.3, 0)), Vec3(8));
    Material blueMt(Color(Vec3(0.1)), Color(Vec3(0, 0, 0.69)), Color(Vec3(0, 0, 0.3)), Vec3(8));

    Polygon3D pot1 = STLBinLoad("STL/Utah_teapot.stl").move(Vec3(0, 0, 5));
    pot1.setMaterial(whiteMt);
    scene.add(&pot1);
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
    printf("Render time: %f s\n", (double)(ed - st) / CLOCKS_PER_SEC);
    printf("Render End.\n");
    sprintf(filename, "Utah_Pot.ppm");
    ppmwriter.import(img);
    ppmwriter.writePPM(filename);

    return 0;
}
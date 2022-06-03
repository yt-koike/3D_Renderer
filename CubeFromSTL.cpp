#include <string.h>
#include <unistd.h>
#include "headers/Render.h"
#include "headers/PPM.h"
#include "headers/STL.h"

void irfanview(const char *filename)
{
    const char *irfanPath = "C:\\Software\\IrfanView\\i_view64.exe";
    char directory[128] = "D:\\GoogleDrive\\3年春\\プロジェクト(小池)\\C++\\Renderer\\";
    execl(irfanPath, irfanPath, strcat(directory, filename));
}

int main()
{
    Ray camera(Vec3(0, 0, -10), Vec3(0, 0, 1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237));

    // load cube from file
    Material mt1(Color(Vec3(0.1)), Color(Vec3(0.2)), Color(Vec3(0)), Vec3(8));
    mt1.setUsePerfectReflectance(1);
    mt1.setCatadioptricFactor(Color(Vec3(0.5)));
    Polygon3D poly = STLBinLoad("TiltedCube_bin.stl");
    printf("Load Complete.\n");
    poly.setMaterial(mt1);
    scene.add(&poly);

    scene.add(new Plane(Vec3(0,-2,0),Vec3(0,1,0)));
    Material mt0(Color(Vec3(0.01)), Color(Vec3(0.69)), Color(Vec3(0.3)), Vec3(8));
    scene.add(new Sphere(Vec3(-3,0,0),1,mt0));

    // add lights
    scene.addLight(new PointLightSource(Vec3(0, 0, -5), Color(Vec3(1))));

    const int width = 4096;
    const int height = 4096;
    char filename[100];
    PPM ppmwriter(width, height, 255);
    printf("Render Start.\n");
    ColorImage img = scene.draw(width, height);
    printf("Render End.\n");
    sprintf(filename, "CubeFromSTL.ppm");
    ppmwriter.import(img);
    ppmwriter.writePPM(filename);
    //irfanview(filename);
    return 0;
}
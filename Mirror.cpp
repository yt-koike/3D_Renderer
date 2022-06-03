#include <string.h>
#include <unistd.h>
#include "headers/Render.h"
#include "headers/PPM.h"

void irfanview(const char *filename)
{
    const char *irfanPath = "C:\\Software\\IrfanView\\i_view64.exe";
    char directory[128] = "D:\\GoogleDrive\\3年春\\プロジェクト(小池)\\C++\\Renderer\\";
    execl(irfanPath, irfanPath, strcat(directory, filename));
}

int main()
{
    Ray camera(Vec3(0, 0, -1), Vec3(0, 0, 1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237));

    // add spheres
    Material mt1(Color(Vec3(0.01)), Color(Vec3(0)), Color(Vec3(0)), Vec3(0));
    mt1.setUsePerfectReflectance(1);
    mt1.setCatadioptricFactor(Color(Vec3(1)));
    scene.add(new Sphere(Vec3(-0.25,-0.5,3), 0.5, mt1));

    // add floor
    Material redWall(Color(Vec3(0.01)), Color(255,0,0), Color(Vec3(0)), Vec3(8));
    Material whiteWall(Color(Vec3(0.01)), Color(255,255,255), Color(Vec3(0)), Vec3(8));
    Material greenWall(Color(Vec3(0.01)), Color(0,255,0), Color(Vec3(0)), Vec3(8));
    scene.add(new Plane(Vec3(-1,0,0), Vec3(1, 0, 0), redWall));
    scene.add(new Plane(Vec3(1,0,0), Vec3(-1, 0, 0), greenWall));
    scene.add(new Plane(Vec3(0,1,0), Vec3(0, -1, 0), whiteWall));
    scene.add(new Plane(Vec3(0,-1,0), Vec3(0, 1, 0), whiteWall));
    scene.add(new Plane(Vec3(0,0,5), Vec3(0, 0, -1), whiteWall));

    // add lights
    scene.addLight(new PointLightSource(Vec3(0,0.9,2.5), Color(Vec3(1.0))));

    const int width = 4096;
    const int height = 4096;
    ColorImage img = scene.draw(width, height);
    PPM ppmwriter(img);
    const char *filename = "Mirror.ppm";
    ppmwriter.writePPM(filename);
    irfanview(filename);
    return 0;
}
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
    Ray camera(Vec3(0, 0, -5), Vec3(0, 0, 1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237));

    // add spheres
    Material mt1(Color(Vec3(0.01)), Color(Vec3(0.69, 0, 0)), Color(Vec3(0.3)), Vec3(8));
    Material mt2(Color(Vec3(0.01)), Color(Vec3(0, 0.69, 0)), Color(Vec3(0.3)), Vec3(8));
    Material mt3(Color(Vec3(0.01)), Color(Vec3(0, 0, 0.69)), Color(Vec3(0.3)), Vec3(8));
    Material mt4(Color(Vec3(0.01)), Color(Vec3(0, 0.69, 0.69)), Color(Vec3(0.3)), Vec3(8));
    Material mt5(Color(Vec3(0.01)), Color(Vec3(0.69, 0, 0.69)), Color(Vec3(0.3)), Vec3(8));
    scene.add(new Sphere(Vec3(3, 0, 25), 1, mt1));
    scene.add(new Sphere(Vec3(2, 0, 20), 1, mt2));
    scene.add(new Sphere(Vec3(1, 0, 15), 1, mt3));
    scene.add(new Sphere(Vec3(0, 0, 10), 1, mt4));
    scene.add(new Sphere(Vec3(-1, 0, 5), 1, mt5));
    // scene.add(new Triangle(Vec3(1,3,0),Vec3(1,1,0),Vec3(-1,3,0),mt1));

    // add floor
    Material floorPlaneMt(Color(Vec3(0.01)), Color(Vec3(0.69)), Color(Vec3(0.3)), Vec3(8));
    scene.add(new Plane(Vec3(3, -1, 0), Vec3(0, 1, 0), floorPlaneMt));

    // add lights
    scene.addLight(new PointLightSource(Vec3(-5, 5, -5), Color(Vec3(0.5))));
    scene.addLight(new PointLightSource(Vec3(5, 0, -5), Color(Vec3(0.5))));
    scene.addLight(new PointLightSource(Vec3(5,20,-5), Color(Vec3(0.5))));

    const int width = 512;
    const int height = 512;
    ColorImage img = scene.draw(width, height);
    PPM ppmwriter(img);
    const char *filename = "SceneTest.ppm";
    ppmwriter.writePPM(filename);
    irfanview(filename);
    return 0;
}
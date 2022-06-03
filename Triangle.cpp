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
    Material mt1(Color(Vec3(0.01)), Color(Vec3(0.69)), Color(Vec3(0.3)), Vec3(8));
    Vec3 v1(0, 0, 0), v2(-1, 1, 0), v3(1, 1, 0), v4(0, -1, 1);


    Polygon3D poly(10);
    poly.addTriangle(new Triangle(v1, v2, v3));
    poly.addTriangle(new Triangle(v1, v3, v4));
    poly.addTriangle(new Triangle(v2, v3, v4));
    scene.add(&poly);

    // add lights
    scene.addLight(new PointLightSource(Vec3(0, 0, -5), Color(Vec3(1.0))));

    const int width = 512;
    const int height = 512;
    char filename[100];
    PPM ppmwriter(width, height, 255);
    for (int i = 0; i < 1; i++)    {
        ColorImage img = scene.draw(width, height);
        sprintf(filename, "Triangle%d.ppm", i);
        ppmwriter.import(img);
        ppmwriter.writePPM(filename);
    }
    //irfanview(filename);
    return 0;
}
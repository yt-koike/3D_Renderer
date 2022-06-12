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
    Ray camera(Vec3(0, 0,-5), Vec3(0, 0,1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237));
    // load cube from file
//    BoundaryBox b(Vec3(0),Vec3(1));
//    scene.add(&b);
    Polygon3D ICO = STLBinLoad("ICO_sphere.stl");
    ICO.getBoundary()->print();
    scene.add(&ICO);
    
    scene.addLight(new PointLightSource(Vec3(0, 20, -30), Color(Vec3(1))));

    int size;
    printf("Size?:");
    scanf("%d",&size);
    char filename[100];
    PPM ppmwriter(size, size, 255);
    printf("Render Start.\n");
    ColorImage img = scene.draw(size, size);
    printf("Render End.\n");
    sprintf(filename, "STL_Render.ppm");
    ppmwriter.import(img);
    ppmwriter.writePPM(filename);
    irfanview(filename);
    return 0;
}
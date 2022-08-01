#include <string.h>
#include <unistd.h>
#include "headers/RenderSuite.h"
#include "headers/PPM.h"

int main(int argc,char **argv)
{
    Ray camera(Vec3(0,0,-5), Vec3(0, 0, 1));
    Scene scene(camera, Color(255, 255, 255), Color(100, 149, 237));

    // add spheres
    Material mt1(Color(Vec3(0.01)), Color(Vec3(0)), Color(Vec3(0)), Vec3(0));
    mt1.setUsePerfectReflectance(1);
    mt1.setCatadioptricFactor(Color(Vec3(1)));
    scene.add(new Sphere(Vec3(-0.4,-0.65,3), 0.35,mt1));

    Material mt2(Color(Vec3(0.01)), Color(Vec3(0)), Color(Vec3(0)), Vec3(0));
    mt2.setUsePerfectReflectance(1);
    mt2.setCatadioptricFactor(Color(Vec3(1)));
    mt2.setUseRefraction(1);
    mt2.setRefractionIndex(1.51);
    scene.add(new Sphere(Vec3(0.5,-0.65,2), 0.35,mt2));

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

    // set width and height of the result image
    int width,height;
    switch(argc){
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

    // render
    printf("Render Start. (%d x %d)\n",width, height);
    clock_t st,ed;
    st = clock();
    ColorImage img = scene.draw(width, height);
    ed = clock();
    printf("time %f\n",(double)(ed-st)/CLOCKS_PER_SEC);
    printf("Render End.\n");

    // export image to file
    PPM ppmwriter(img);
    const char *filename = "Refraction.ppm";
    ppmwriter.writePPM(filename);
    return 0;
}
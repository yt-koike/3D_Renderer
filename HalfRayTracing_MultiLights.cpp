#include <list>
#include "headers/Suite.h"
#include "headers/PPM.h"
using namespace std;
int main()
{
  Material floorPlaneMt(Color(Vec3(0.1)), Color(Vec3(0.5)), Color(Vec3(0.3)), Vec3(4));
  Material mt1(Color(Vec3(0.01)), Color(Vec3(0.69,0,0)), Color(Vec3(0.3)), Vec3(8));
  Material mt2(Color(Vec3(0.01)), Color(Vec3(0,0.69,0)), Color(Vec3(0.3)), Vec3(8));
  Material mt3(Color(Vec3(0.01)), Color(Vec3(0,0,0.69)), Color(Vec3(0.3)), Vec3(8));
  Material mt4(Color(Vec3(0.01)), Color(Vec3(0,0.69,0.69)), Color(Vec3(0.3)), Vec3(8));
  Material mt5(Color(Vec3(0.01)), Color(Vec3(0.69,0,0.69)), Color(Vec3(0.3)), Vec3(8));
  Shape* shapes[6];
  shapes[0] = new Sphere(Vec3(3, 0, 25), 1, mt1);
  shapes[1] = new Sphere(Vec3(2, 0, 20), 1, mt2);
  shapes[2] = new Sphere(Vec3(1, 0, 15), 1, mt3);
  shapes[3] = new Sphere(Vec3(0, 0, 10), 1, mt4);
  shapes[4] = new Sphere(Vec3(-1, 0, 5), 1, mt5);
  shapes[5] = new Plane(Vec3(0,5,0),Vec3(0,-1,0),floorPlaneMt);
  Color envLight(255,255,255);
  PointLightSource* lights[3];
  lights[0] = new PointLightSource(Vec3(-5), Color(Vec3(0.5)));
  lights[1] = new PointLightSource(Vec3(5, 0, -5), Color(Vec3(0.5)));
  lights[2] = new PointLightSource(Vec3(5, -20, -5), Color(Vec3(0.5)));
  int W = 512;
  int H = 512;
  PPM image(W, H, 255);
  Material mt();
  for (double y = 0; y < H; y += 1)
  {
    for (double x = 0; x < W; x += 1)
    {
      Ray camera(0, 0, -5, x / W - 0.5, y / H - 0.5, 1);
      Color finalColor(100, 149, 237);
      double distance = -1;
      for (int i = 0;i<sizeof(shapes)/sizeof(shapes[0]);i++)
      {
        Shape* shape = shapes[i];
        IntersectionPoint cross = shape -> testIntersection(camera);
        if (cross.exists)
        {
          if (distance < 0)
            distance = cross.distance;
          if (cross.distance <= distance)
          {
            finalColor = shape -> totalLightness(cross, camera.getDir(), sizeof(lights)/sizeof(lights[0]),lights, envLight);
            distance = cross.distance;
          }
        }
      }
      image.setPixel(x, y, finalColor.getR(), finalColor.getG(), finalColor.getB());
    }
  }
  image.writePPM("HalfRayTrace_MultiLights.ppm");
}
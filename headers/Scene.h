#ifndef SCENE_H
#define SCENE_H
#include "Vector.h"
#include "Color.h"
#include "shapes/Shape.h"
#include "Array2D.h"
#include "Ray.h"
#include <float.h>
#include <vector>

#define MAX_RECURSION 8

class Scene
{
private:
  std::vector<Shape *> shapes;
  std::vector<PointLightSource *> lights;
  Ray camera;
  Color envLight;
  Color background;
  float globalRefractionIndex = 1.0;

public:
  Scene(Ray camera, Color envLight, Color background)
  {
    this->camera = camera;
    this->envLight = envLight;
    this->background = background;
  }
  void add(Shape *s) { shapes.push_back(s); }
  void addLight(PointLightSource *light) { lights.push_back(light); }
  ColorImage draw(int width, int height);
  void testIntersectionPointWithAll(int rayN, Ray *rs, IntersectionPoint *crosses, unsigned int *shapeIds);
  void testIntersectionPointWithAll(Ray r, IntersectionPoint *cross, unsigned int *shapeId) { testIntersectionPointWithAll(1, &r, cross, shapeId); }
  void rayTraceRecursive(int rayN, Ray *rs, int recursionLevel, Color *result);
};

ColorImage Scene::draw(int width, int height)
{
  ColorImage img(width, height);
  int rayN = width * height;
  Ray *rs = new Ray[rayN];
  for (double y = 0; y < height; y++)
  {
    for (double x = 0; x < width; x++)
    {
      Ray r(camera.getPoint(), camera.getDir().add(Vec3(x / width - 0.5, -y / height + 0.5, 0).mult(0.5)));
      rs[(int)(y * width + x)] = r;
    }
  }

  Color *c = new Color[rayN];
  rayTraceRecursive(rayN, rs, 3, c);
  for (double y = 0; y < height; y++)
  {
    for (double x = 0; x < width; x++)
    {
      int rayIdx = y * width + x;
      img.set(x, y, c[rayIdx].getR(), c[rayIdx].getG(), c[rayIdx].getB());
    }
  }
  return img;
}

void Scene::testIntersectionPointWithAll(int rayN, Ray *rs, IntersectionPoint *crosses, unsigned int *shapeIds)
{
  IntersectionPoint *crosses_tmp = new IntersectionPoint[rayN];
  for (int i = 0; i < shapes.size(); i++)
  {
    if (!shapes[i]->isVisible())
      continue;
    clock_t st, ed;
    st = clock();
    shapes[i]->testIntersections(rayN, rs, crosses_tmp);
    ed = clock();
    //printf("Object %d: %f s\n", i, (double)(ed - st) / CLOCKS_PER_SEC);
    for (int rayIdx = 0; rayIdx < rayN; rayIdx++)
    {
      IntersectionPoint cross = crosses_tmp[rayIdx];
      if (cross.exists)
      {
        if (!crosses[rayIdx].exists || cross.distance <= crosses[rayIdx].distance)
        {
          crosses[rayIdx] = cross;
          if(shapeIds != nullptr)
          shapeIds[rayIdx] = i;
        }
      }
    }
  }
  delete crosses_tmp;
}

void Scene::rayTraceRecursive(int rayN, Ray *rs, int recursionLevel, Color *result)
{
  IntersectionPoint *crosses = new IntersectionPoint[rayN];
  unsigned int *shapeIds = new unsigned int[rayN];
  testIntersectionPointWithAll(rayN, rs, crosses, shapeIds);
  for (int rayIdx = 0; rayIdx < rayN; rayIdx++)
  {
    Ray r = rs[rayIdx];
    IntersectionPoint cross = crosses[rayIdx];
    Color c;
    if (cross.exists)
    {
      Shape *s = shapes[shapeIds[rayIdx]];
      c = s->envLightness(envLight);
      for (int i = 0; i < lights.size(); i++)
      {
        PointLightSource light = *lights[i];
        Vec3 shadowCheckerPos = cross.position.add(cross.normal.mult(0.01));
        Vec3 shadowCheckerDir = light.position.sub(shadowCheckerPos);
        Ray shadowChecker(shadowCheckerPos, shadowCheckerDir);
        IntersectionPoint shadowCross;
        testIntersectionPointWithAll(shadowChecker, &shadowCross, nullptr);
        if (shadowCross.exists)
          continue;
        c = c.add(s->lightness(cross, r.getDir(), light));
      }
    }
    else
    {
      c = background;
    }
    /*
    Ray *shadowCheckRay = new Ray[rayN];
    for (int rayIdx = 0; rayIdx < rayN; rayIdx++)
    {
      IntersectionPoint cross = crosses[rayIdx];
      Vec3 sPos = cross.position.add(cross.normal.mult(0.01));
      for (int i = 0; i < lights.size(); i++)
      {
        Vec3 sDir = lights[i]->position.sub(sPos);
        shadowCheckRay[rayIdx] = Ray(sPos, sDir);
        IntersectionPoint *shadow_crosses = new IntersectionPoint[rayN];
        testIntersectionsPointWithAll(rayN, shadowCheckRay, shadow_crosses, nullptr);
            for (int sRayIdx = 0; sRayIdx < rayN; sRayIdx++){
              if(shadow_crosses[sRayIdx].exists){
                c[sRayIdx] = Vec3(0);
                break;
              }
            }

      }

    }
    */
    result[rayIdx] = c;
  }
  delete crosses;
  delete shapeIds;
}
#endif
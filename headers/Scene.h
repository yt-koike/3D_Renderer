#ifndef SCENE_H
#define SCENE_H
#include "Vector.h"
#include "Color.h"
#include "shapes/Shape.h"
#include "ColorImage.h"
#include "Ray.h"
#include <float.h>
#include <vector>

#define MAX_RECURSION 8

class Scene
{
private:
  unsigned int shapesN;
  unsigned int shapesCapacity;
  Shape **shapes;

  unsigned int lightsN;
  unsigned int lightsCapacity;
  PointLightSource **lights;

  Ray camera;
  Color envLight;
  Color background;
  float globalRefractionIndex = 1.0;

public:
  Scene(Ray camera, Color envLight, Color background, unsigned int shapesCapacity, unsigned int lightsCapacity)
  {
    this->camera = camera;
    this->envLight = envLight;
    this->background = background;
    this->shapesCapacity = shapesCapacity;
    this->shapesN = 0;
    this->shapes = new Shape *[shapesCapacity];
    this->lightsCapacity = lightsCapacity;
    this->lightsN = 0;
    this->lights = new PointLightSource *[lightsCapacity];
  }
  void add(Shape *s)
  {
    if (shapesN >= shapesCapacity)
    {
      printf("Error: shapes capacity overflow\n");
      return;
    }
    shapes[shapesN] = s;
    shapesN++;
  }
  void addLight(PointLightSource *light)
  {
    if (lightsN >= lightsCapacity)
    {
      printf("Error: lights capacity overflow\n");
      return;
    }

    lights[lightsN] = light;
    lightsN++;
  }
  ColorImage draw(int width, int height);
  void testIntersectionPointWithAll(int rayN, Ray *rs, IntersectionPoint *crosses, unsigned int *shapeIds);
  Color rayCalc(Ray ray, IntersectionPoint cross, Shape *shape, int recursionLevel);
  void rayTrace(int rayN, Ray *rs, Color *result);
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
  rayTrace(rayN, rs, c);
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

// test rays whether it hits any objects
void Scene::testIntersectionPointWithAll(int rayN, Ray *rs, IntersectionPoint *crosses, unsigned int *shapeIds)
{
  IntersectionPoint *crosses_tmp = new IntersectionPoint[rayN];
  for (unsigned int i = 0; i < shapesN; i++)
  {
    if (!shapes[i]->isVisible())
      continue;
    shapes[i]->testIntersections(rayN, rs, crosses_tmp);
    for (int rayIdx = 0; rayIdx < rayN; rayIdx++)
    {
      IntersectionPoint cross = crosses_tmp[rayIdx];
      if (cross.exists)
      {
        if (!crosses[rayIdx].exists || cross.distance <= crosses[rayIdx].distance)
        {
          crosses[rayIdx] = cross;
          shapeIds[rayIdx] = i;
        }
      }
    }
  }
  delete crosses_tmp;
}

// calculate color out of ray, shape and the cross where it hits the shape
Color Scene::rayCalc(Ray ray, IntersectionPoint cross, Shape *shape, int recursionLevel)
{
  // rayTrace START
  if (!cross.exists)
    return background;
  if (recursionLevel > MAX_RECURSION)
    return Vec3(0);
  Color color = shape->envLightness(envLight);
  for (int i = 0; i < lightsN; i++)
  {
    PointLightSource light = *lights[i];
    #ifdef SHADOW_ENABLE
    // add shadows
    Vec3 shadowCheckerPos = cross.position.add(cross.normal.mult(0.001));
    Vec3 shadowCheckerDir = light.position.sub(shadowCheckerPos);
    Ray shadowChecker(shadowCheckerPos, shadowCheckerDir);
    IntersectionPoint shadowCross;
    unsigned int shadowShapeId;
    testIntersectionPointWithAll(1, &shadowChecker, &shadowCross, &shadowShapeId);
    if (shadowCross.exists && shadowCross.distance < shadowCheckerDir.mag())
      continue;
    #endif
    color = color.add(shape->lightness(cross, ray.getDir(), light));
  }
  Material mt = shape->getMaterial();
  if (mt.getUsePerfectReflectance())
  {
    Vec3 n = cross.normal;
    Vec3 v = ray.getDir().mult(-1);
    Vec3 mirrorPos = cross.position.add(cross.normal.mult(0.1));
    Vec3 mirrorDir = n.mult(2 * v.dot(n)).sub(v);
    Ray reRay(mirrorPos, mirrorDir);
    IntersectionPoint reRayCross;
    unsigned int reRayShapeId;
    testIntersectionPointWithAll(1, &reRay, &reRayCross, &reRayShapeId);
    Color reflect = rayCalc(reRay, reRayCross, shapes[reRayShapeId], recursionLevel + 1).clamp();
    Color catadioptricColor;
    if (mt.getUseRefraction())
    {
      double mu1, mu2;
      if (n.dot(v) > 0)
      {
        mu1 = globalRefractionIndex;
        mu2 = mt.getRefractionIndex();
      }
      else
      {
        mu1 = mt.getRefractionIndex();
        mu2 = globalRefractionIndex;
        n = n.mult(-1);
      }
      double muR = mu2 / mu1;
      double cos1 = v.dot(n);
      double cos2 = mu1 / mu2 * sqrt((long double)(muR * muR - (1 - cos1 * cos1)));
      double omega = muR * cos2 - cos1;
      double rouParallel = (muR * cos1 - cos2) / (muR * cos1 + cos2);
      double rouVertical = omega / (muR * cos2 + cos1);
      double Cr = (rouParallel * rouParallel + rouVertical * rouVertical) / 2;
      Vec3 refractionDir = (ray.getDir().sub(n.mult(omega))).mult(mu1 / mu2);
      Vec3 refractionPos = cross.position.add(refractionDir.mult(0.01));
      Ray refractionRay(refractionPos, refractionDir);
      IntersectionPoint refractionCross;
      unsigned int refractionShapeId;
      testIntersectionPointWithAll(1, &refractionRay, &refractionCross, &refractionShapeId);
      Color refraction = rayCalc(refractionRay, refractionCross, shapes[refractionShapeId], recursionLevel + 1);
      catadioptricColor = reflect.mult(Cr).add(refraction.mult(1 - Cr));
    }
    else
    {
      catadioptricColor = reflect;
    }
    Color catadioptricFactor = mt.getCatadioptricFactor();
    color = color.add(catadioptricColor.mask(catadioptricFactor));
  }
  return color.clamp();
  // rayTrace END
}

void Scene::rayTrace(int rayN, Ray *rays, Color *result)
{
  IntersectionPoint *crosses = new IntersectionPoint[rayN];
  unsigned int *shapeIds = new unsigned int[rayN];
  testIntersectionPointWithAll(rayN, rays, crosses, shapeIds);
  for (int i = 0; i < rayN; i++)
  {
    result[i] = rayCalc(rays[i], crosses[i], shapes[shapeIds[i]], 0);
  }
  delete crosses;
  delete shapeIds;
  return;
}

#endif
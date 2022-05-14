#ifndef SCENE_H
#define SCENE_H
#include "Vector.h"
#include "Color.h"
#include "Shapes.h"
#include "Array2D.h"
#include "Ray.h"
#include <float.h>
#include <vector>

#define MAX_RECURSION 8

class IntersectionTestResult
{
public:
  Shape *shape;
  IntersectionPoint intersectionPoint;
};

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
  IntersectionTestResult testIntersectionPointWithAll(Ray ray) { return testIntersectionPointWithAll(ray, __DBL_MAX__, false); }
  IntersectionTestResult testIntersectionPointWithAll(Ray ray, double maxDist, bool exitOnceFound);
  Color rayTrace(Ray camera);
  Color rayTraceRecursive(Ray ray, int recusionLevel);
};

ColorImage Scene::draw(int width, int height)
{
  ColorImage img(width, height);
  for (double y = 0; y < height; y++)
  {
    for (double x = 0; x < width; x++)
    {
      Ray ray(camera.getPoint(), camera.getDir().add(Vec3(x / width - 0.5, -y / height + 0.5, 0).mult(0.5)));
      Color c = rayTrace(ray);
      img.set(x, y, c.getR(), c.getG(), c.getB());
    }
  }
  return img;
}

IntersectionTestResult Scene::testIntersectionPointWithAll(Ray ray, double maxDist, bool exitOnceFound)
{
  IntersectionTestResult res;
  res.intersectionPoint.distance = -1;
  for (int i = 0; i < shapes.size(); i++)
  {
    IntersectionPoint cross = shapes[i]->testIntersection(ray);
    if (cross.exists && cross.distance <= maxDist)
    {
      if (cross.distance <= res.intersectionPoint.distance || res.intersectionPoint.distance < 0)
      {
        res.shape = shapes[i];
        res.intersectionPoint = cross;
      }
      if (exitOnceFound)
        return res;
    }
  }
  return res;
}

Color Scene::rayTraceRecursive(Ray ray, int recursionLevel)
{
  if (recursionLevel > MAX_RECURSION)
  {
    return background;
  }
  else
  {
    // rayTrace START
    IntersectionTestResult intersection = testIntersectionPointWithAll(ray);
    if (intersection.intersectionPoint.exists)
    {
      IntersectionPoint cross = intersection.intersectionPoint;
      Shape *shape = intersection.shape;
      Color color = shape->envLightness(envLight);
      for (int i = 0; i < lights.size(); i++)
      {
        PointLightSource light = *lights[i];

        Vec3 shadowCheckerPos = cross.position.add(cross.normal.mult(0.001));
        Vec3 shadowCheckerDir = light.position.sub(shadowCheckerPos);
        Ray shadowChecker(shadowCheckerPos, shadowCheckerDir);
        IntersectionTestResult shadow = testIntersectionPointWithAll(shadowChecker, shadowCheckerDir.mag(), true);
        if (shadow.intersectionPoint.exists)
          continue;

        color = color.add(shape->lightness(cross, camera.getDir(), light));
      }
      Material mt = shape->getMaterial();
      if (mt.getUsePerfectReflectance())
      {
        Vec3 n = cross.normal;
        Vec3 v = ray.getDir().mult(-1);
        Vec3 mirrorPos = cross.position.add(cross.normal.mult(0.001));
        Vec3 mirrorDir = n.mult(2 * v.dot(n)).sub(v);
        Ray reRay(mirrorPos, mirrorDir);
        Color catadioptricFactor = mt.getCatadioptricFactor();
        Color reflect = rayTraceRecursive(reRay, recursionLevel + 1);
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
            n=n.mult(-1);
          }
          double muR = mu2 / mu1;
          double cos1 = v.dot(n);
          double cos2 = mu1 / mu2 * sqrt((long double)(muR * muR - (1 - cos1 * cos1)));
          double omega = muR * cos2 - cos1;
          double rouParallel = (muR * cos1 - cos2) / (muR * cos1 + cos2);
          double rouVertical = omega / (muR * cos2 + cos1);
          double Cr = (rouParallel * rouParallel + rouVertical * rouVertical)/2;
          Vec3 refractionDir = (ray.getDir().sub(n.mult(omega))).mult(mu1/mu2);
          Vec3 refractionPos = cross.position.add(refractionDir.mult(0.01));
          Ray refractionRay(refractionPos,refractionDir);
          Color refraction = rayTraceRecursive(refractionRay,recursionLevel+1);
          catadioptricColor = reflect.mult(Cr).add(refraction.mult(1 - Cr));
        }
        else
        {
          catadioptricColor = reflect;
        }
          color = color.add(catadioptricColor.mask(catadioptricFactor));
      }

      return color.clamp();
    }
    else
    {
      return background;
    }
    // rayTrace END
  }
}

Color Scene::rayTrace(Ray camera)
{
  return rayTraceRecursive(camera, 0);
}


/*
Color Scene::rayTraceSimple(Ray camera)
{
  IntersectionTestResult intersection = testIntersectionPointWithAll(camera);
  if (intersection.intersectionPoint.exists)
  {
    IntersectionPoint cross = intersection.intersectionPoint;
    Shape *shape = intersection.shape;
    Color color = shape->envLightness(envLight);
    for (int i = 0; i < lights.size(); i++)
    {
      PointLightSource light = *lights[i];

      Vec3 shadowCheckerPos = cross.position.add(cross.normal.mult(0.001));
      Vec3 shadowCheckerDir = light.position.sub(shadowCheckerPos);
      Ray shadowChecker(shadowCheckerPos, shadowCheckerDir);

      IntersectionTestResult shadow = testIntersectionPointWithAll(shadowChecker, shadowCheckerDir.mag(), true);
      if (shadow.intersectionPoint.exists)
        continue;
      color = color.add(shape->lightness(cross, camera.getDir(), light));
    }

    return color.clamp();
  }else{
    return background;
  }
}
*/
#endif
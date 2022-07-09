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
  Color rayCalc(Ray ray, IntersectionPoint cross, Shape *shape,int recursionLevel);
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

void Scene::testIntersectionPointWithAll(int rayN, Ray *rs, IntersectionPoint *crosses, unsigned int *shapeIds)
{
  IntersectionPoint *crosses_tmp = new IntersectionPoint[rayN];
  for (int i = 0; i < shapes.size(); i++)
  {
    if (!shapes[i]->isVisible())
      continue;
    clock_t st,ed;st = clock();
    shapes[i]->testIntersections(rayN, rs, crosses_tmp);
    ed = clock();
    printf("object %d: %f s\n",i,(double)(ed-st)/CLOCKS_PER_SEC);
    for (int rayIdx = 0; rayIdx < rayN; rayIdx++)
    {
      IntersectionPoint cross = crosses_tmp[rayIdx];
      if (cross.exists)
      {
        if (!crosses[rayIdx].exists || cross.distance <= crosses[rayIdx].distance)
        {
          crosses[rayIdx] = cross;
          if (shapeIds != nullptr)
            shapeIds[rayIdx] = i;
        }
      }
    }
  }
  delete crosses_tmp;
}

Color Scene::rayCalc(Ray ray, IntersectionPoint cross, Shape *shape,int recursionLevel)
{
  // rayTrace START
  if(recursionLevel>MAX_RECURSION)return background;
    Color color = shape->envLightness(envLight);
    for (int i = 0; i < lights.size(); i++)
    {
      PointLightSource light = *lights[i];
/*
      Vec3 shadowCheckerPos = cross.position.add(cross.normal.mult(0.01));
      Vec3 shadowCheckerDir = light.position.sub(shadowCheckerPos);
      Ray shadowChecker(shadowCheckerPos, shadowCheckerDir);
      IntersectionPoint shadowCross;
      testIntersectionPointWithAll(shadowChecker, &shadowCross,nullptr);
      if (shadowCross.exists)
        continue;
*/
      color = color.add(shape->lightness(cross, ray.getDir(), light));
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
      IntersectionPoint reRayCross;
      unsigned int reRayShapeId;
      testIntersectionPointWithAll(reRay,&reRayCross,&reRayShapeId);
      Color reflect = rayCalc(reRay,reRayCross,&shape[reRayShapeId],recursionLevel+1);
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
      testIntersectionPointWithAll(reRay,&refractionCross,&refractionShapeId);
        Color refraction = rayCalc(refractionRay,refractionCross,&shape[refractionShapeId],recursionLevel+1);
        catadioptricColor = reflect.mult(Cr).add(refraction.mult(1 - Cr));
      }
      else
      {
        catadioptricColor = reflect;
      }
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
    if(crosses[i].exists){
      result[i] = rayCalc(rays[i], crosses[i], shapes[shapeIds[i]],0);
    }else{
      result[i] = background;
    }
  }
}

#endif
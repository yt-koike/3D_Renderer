#ifndef SHAPE_H
#define SHAPE_H
#include <vector>
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"

class Shape
{
protected:
  int visible = 1;
  Material mt = Material(Color(Vec3(0.01)), Color(Vec3(0.69)), Color(Vec3(0.3)), Vec3(8));

public:
  Material getMaterial() { return mt; }
  void setMaterial(Material mt) { this->mt = mt; }
  int isVisible(){return visible;}
  void setVisible(int flag){visible=flag;}
  virtual IntersectionPoint testIntersection(Ray r) {}
  virtual int doesHit(Ray r) {}
  virtual void print() { printf("Null shape"); }
  Color envLightness(Color envRayIntensity);
  Color lightness(IntersectionPoint cross, Vec3 cameraDir, PointLightSource light);
};

Color Shape::envLightness(Color envRayIntensity)
{
  Color Ra = Color(mt.getKa().mask(envRayIntensity)).clamp();
  return Ra;
}

Color Shape::lightness(IntersectionPoint cross, Vec3 cameraDir, PointLightSource light)
{
  Vec3 cross1 = cross.position;
  Vec3 n = cross.normal;
  Vec3 l = light.position.sub(cross1).normalize();
  Color Rd = Color(light.intensity.mult(n.dot(l)).mask(mt.getKd())).clamp();
  Vec3 r = n.mult(2 * n.dot(l)).sub(l);
  Vec3 antiViewVec = cameraDir.mult(-1).normalize();
  Color Rs = Color(light.intensity.mask(Vec3(1, 1, 1).mult(r.dot(antiViewVec)).vecPow(mt.getAlpha())).mask(mt.getKs())).clamp();
  return Color(Rd.add(Rs)).clamp();
}

#endif
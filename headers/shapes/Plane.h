#ifndef PLANE_H
#define PLANE_H
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"
#include "Shape.h"
class Plane : public Shape
{
private:
  Vec3 pointV;
  Vec3 normalV;

public:
  Plane(Vec3 point, Vec3 normal)
  {
    setPointV(point);
    setNormalV(normal);
  }
  Plane(Vec3 point, Vec3 normal, Material mt)
  {
    setPointV(point);
    setNormalV(normal);
    setMaterial(mt);
  }
  Vec3 getPointV() { return pointV; }
  Vec3 getNormalV() { return normalV; }
  void setPointV(Vec3 v) { this->pointV = v; }
  void setNormalV(Vec3 v) { this->normalV = v.normalize(); }
  int isCross(Vec3 s, Vec3 d);
  Vec3 firstCross(Vec3 s, Vec3 d);
  IntersectionPoint testIntersection(Ray r);
  virtual void print()
  {
    printf("Plane:\n point:");
    pointV.print();
    printf(" normalV:");
    normalV.print();
  }
};


IntersectionPoint Plane::testIntersection(Ray r)
{
  IntersectionPoint res;
  Vec3 s = r.getPoint().sub(pointV);
  Vec3 d = r.getDir().normalize();
  Vec3 N = normalV.copy();
  if (N.dot(d) == 0)
    return res;
  if (N.cos(d) > 0)
  {
    N = N.mult(-1);
  }
  double t = -N.dot(s) / N.dot(d);
  if (t < 0)
    return res;
  res.exists = 1;
  res.position = s.add(d.mult(t)).add(pointV);
  res.normal = N;
  res.distance = res.position.sub(r.getPoint()).mag();
  return res;
}

#endif
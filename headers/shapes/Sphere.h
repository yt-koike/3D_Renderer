#ifndef SPHERE_H
#define SPHERE_H
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"
#include "Shape.h"
class Sphere : public Shape
{
private:
  Vec3 center;
  double radius;
  BoundaryBox* boundary;

public:
  Sphere(){};
  Sphere(Vec3 c, double r);
  Vec3 getCenter() { return center; }
  IntersectionPoint testIntersection(Ray r);
  void print()
  {
    printf("Sphere:\n Pos:");
    center.print();
    printf(" r:%f\n", radius);
  }
  void generateBoundary(){
    this->setBoundary(new BoundaryBox(center.sub(Vec3(radius)),center.add(Vec3(radius))));
  }
  void setBoundary(BoundaryBox* b){boundary = b;}
  BoundaryBox* getBoundary(){return boundary;}

};

Sphere::Sphere(Vec3 c, double r)
{
  this->center = c;
  this->radius = r;
  generateBoundary();
}

IntersectionPoint Sphere::testIntersection(Ray r)
{
  IntersectionPoint res;
//  if(!getBoundary()->doesHit(r))return res; // faster to remove
  Vec3 s = r.getPoint().sub(center);
  Vec3 d = r.getDir();
  double A, B, C;
  A = d.magSq();
  B = 2 * s.dot(d);
  C = s.magSq() - radius * radius;
  if (B * B - 4 * A * C < 0)
    return res;
  double t = (-B - sqrt((long double)B * B - 4 * A * C)) / (2 * A);
  if (t < 0)
  {
    t = (-B + sqrt((long double)B * B - 4 * A * C)) / (2 * A);
    if (t < 0)
      return res;
  }
  res.exists = 1;
  res.position = r.getPoint().add(d.mult(t));
  res.normal = res.position.sub(getCenter()).normalize();
  res.distance = res.position.sub(r.getPoint()).mag();
  return res;
}


#endif
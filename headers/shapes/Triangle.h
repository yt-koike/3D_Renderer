#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"
#include "Shape.h"
#include "Plane.h"
#include "BoundaryBox.h"

class Triangle : public Shape
{
private:
  Vec3 v1, v2, v3;
  Vec3 E0,E1;
  Vec3 normalV;
  BoundaryBox *boundary;

public:
  Triangle(Vec3 v1, Vec3 v2, Vec3 v3)
  {
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
    E0 = v2.sub(v1);
    E1 = v3.sub(v1);
    normalV = E0.cross(E1).normalize();
    generateBoundary();
  }
  Triangle(Vec3 v1, Vec3 v2, Vec3 v3, Material mt)
  {
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
    E0 = v2.sub(v1);
    E1 = v3.sub(v1);
    normalV = E0.cross(E1).normalize();
    generateBoundary();
    setMaterial(mt);
  }
  Triangle *copy() { return new Triangle(v1.copy(), v2.copy(), v3.copy()); }
  Triangle rotate(Vec3 origin, Vec3 axis, double rad) { return Triangle(v1.rotate(origin, axis, rad), v2.rotate(origin, axis, rad), v3.rotate(origin, axis, rad)); }
  void generateBoundary()
  {
    boundary = new BoundaryBox();
    boundary->includeV(v1);
    boundary->includeV(v2);
    boundary->includeV(v3);
  }
  virtual IntersectionPoint testIntersection(Ray r);
  virtual void print()
  {
    printf("Triangle:\n");
    v1.print();
    v2.print();
    v3.print();
  }
  Triangle move(Vec3 dV)
  {
    return Triangle(v1.add(dV), v2.add(dV), v3.add(dV));
  }
  Vec3 getV1() { return v1; }
  Vec3 getV2() { return v2; }
  Vec3 getV3() { return v3; }
  Vec3 *getV1p() { return &v1; }
  Vec3 *getV2p() { return &v2; }
  Vec3 *getV3p() { return &v3; }
};

IntersectionPoint Triangle::testIntersection(Ray r)
{
//  if(!boundary->doesHit(r)){return cross;} // Faster to comment out
/*
  IntersectionPoint cross;
  Plane plane(v1, normalV);
  cross = plane.testIntersection(r);
  if (!cross.exists)
    return cross;
  cross.exists = 0;
  Vec3 crossPos = cross.position;
  // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
  if (v2.sub(v1).cross(crossPos.sub(v1)).dot(normalV) >= 0 && v3.sub(v2).cross(crossPos.sub(v2)).dot(normalV) >= 0 && v1.sub(v3).cross(crossPos.sub(v3)).dot(normalV) >= 0)
  {
    cross.exists = 1;
  }
  return cross;
*/
  // https://dc.ewu.edu/cgi/viewcontent.cgi?article=1093&context=theses
  Vec3 D = r.getDir();
  Vec3 T = r.getPoint().sub(v1);
  Vec3 P = D.cross(E1);
  Vec3 Q = T.cross(E0);
  double deno = P.dot(E0);
  double distance = Q.dot(E1) / deno;
  double u = P.dot(T)/deno;
  double v = Q.dot(D)/deno;
  IntersectionPoint cross;
  if(!(u>=0&&v>=0&&u+v<=1))return cross;
  if (distance < 0)
    return cross;
  cross.exists = 1;
  cross.distance = distance;
  cross.normal = normalV;
  cross.position = D.mult(distance).add(r.getPoint());
  return cross;
}
#endif
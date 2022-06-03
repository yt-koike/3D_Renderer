#ifndef SHAPE_H
#define SHAPE_H
#include <vector>
#include "Vector.h"
#include "Color.h"
#include "Ray.h"
#include "Material.h"
class Shape
{
protected:
  Material mt = Material(Color(Vec3(0.01)), Color(Vec3(0.69)), Color(Vec3(0.3)), Vec3(8));

public:
  Material getMaterial() { return mt; }
  void setMaterial(Material mt) { this->mt = mt; }
  virtual IntersectionPoint testIntersection(Ray r) {}
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

class Sphere : public Shape
{
private:
  Vec3 center;
  double radius;

public:
  Sphere(){};
  Sphere(Vec3 c, double r, Material mt);
  Vec3 getCenter() { return center; }
  virtual IntersectionPoint testIntersection(Ray r);
  virtual void print()
  {
    printf("Sphere:\n Pos:");
    center.print();
    printf(" r:%f\n", radius);
  }
};

Sphere::Sphere(Vec3 c, double r, Material mt)
{
  this->center = c;
  this->radius = r;
  setMaterial(mt);
}

IntersectionPoint Sphere::testIntersection(Ray r)
{
  Vec3 s = r.getPoint().sub(center);
  Vec3 d = r.getDir();
  double A, B, C;
  A = d.magSq();
  B = 2 * s.dot(d);
  C = s.magSq() - radius * radius;
  IntersectionPoint res;
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
  Vec3 setPointV(Vec3 v) { this->pointV = v; }
  Vec3 setNormalV(Vec3 v) { this->normalV = v.normalize(); }
  int isCross(Vec3 s, Vec3 d);
  Vec3 firstCross(Vec3 s, Vec3 d);
  virtual IntersectionPoint testIntersection(Ray r);
  virtual void print()
  {
    printf("Sphere:\n point:");
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

class Triangle : public Shape
{
private:
  Vec3 v1, v2, v3;

public:
  Triangle(Vec3 v1, Vec3 v2, Vec3 v3)
  {
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
  }
  Triangle(Vec3 v1, Vec3 v2, Vec3 v3, Material mt)
  {
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
    setMaterial(mt);
  }
  Triangle copy(){return Triangle(v1.copy(),v2.copy(),v3.copy());}
  Triangle rotate(Vec3 origin, Vec3 axis, double rad) { return Triangle(v1.rotate(origin, axis, rad), v2.rotate(origin, axis, rad), v3.rotate(origin, axis, rad)); }
  virtual IntersectionPoint testIntersection(Ray r);
  virtual void print()
  {
    printf("Triangle:\n");
    v1.print();
    v2.print();
    v3.print();
  }
  Triangle move(Vec3 dV){
    return Triangle(v1.add(dV),v2.add(dV),v3.add(dV));
  }
  Vec3 getV1(){return v1;}
  Vec3 getV2(){return v2;}
  Vec3 getV3(){return v3;}
};

IntersectionPoint Triangle::testIntersection(Ray r)
{
  Vec3 a = v2.sub(v1);
  Vec3 b = v3.sub(v1);
  Vec3 normalV = a.cross(b).normalize();
  Plane plane(v1, normalV);
  IntersectionPoint cross = plane.testIntersection(r);
  if (!cross.exists)
    return cross;
  cross.exists = 0;
  Vec3 crossPos = cross.position;
  // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
  if (v2.sub(v1).cross(crossPos.sub(v1)).dot(normalV) > -0.001 && v3.sub(v2).cross(crossPos.sub(v2)).dot(normalV) > -0.001 && v1.sub(v3).cross(crossPos.sub(v3)).dot(normalV) > -0.001)
  {
    cross.exists = 1;
  }
  return cross;
}

class Polygon3D : public Shape
{
private:
  int size = 0;
  int maxSize = 0;
  Triangle **tris;

public:
  Polygon3D(int size)
  {
    this->maxSize = size;
    tris = new Triangle *[this->maxSize];
  }
  Polygon3D rotate(Vec3 origin, Vec3 axis, double rad)
  {
    Polygon3D poly(maxSize);
    for (int i = 0; i < size; i++)
    {
      poly.addTriangle(new Triangle(tris[i]->rotate(origin, axis, rad))); 
    }
    return poly;
  }
  Polygon3D move(Vec3 dV){
        Polygon3D poly(maxSize);
    for (int i = 0; i < size; i++)
    {
      poly.addTriangle(new Triangle(tris[i]->move(dV))); 
    }
    return poly;
  }
  void print(){
    printf("Polygon: %d triangles\n",size);
    for (int i = 0; i < size; i++) tris[i]->print();
  }
  void addTriangle(Triangle *tri);
  virtual IntersectionPoint testIntersection(Ray r);
};

void Polygon3D::addTriangle(Triangle *tri)
{
  if (size < maxSize)
  {
    tris[size] = tri;
    size++;
  }
}

IntersectionPoint Polygon3D::testIntersection(Ray r)
{
  IntersectionPoint cross;
  int closestId = -1;  
  double closestDistance = -1;
  for (int i = 0; i < size; i++)
  {
    cross = tris[i]->testIntersection(r);
    if (cross.exists){
      if(closestId == -1){
        closestId = i;
        closestDistance = cross.distance;
      }else if (cross.distance < closestDistance){
        closestId = i;
        closestDistance = cross.distance;
      }
    }
  }
  if(closestId != -1){
    return tris[closestId]->testIntersection(r);
  }
  IntersectionPoint noCross;
  return noCross;
}

#endif
#ifndef POLYGON3D_H
#define POLYGON3D_H
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"
#include "Shape.h"
#include "Triangle.h"

class Polygon3D : public Shape
{
private:
  int size = 0;
  int maxSize = 0;
  Triangle **tris;
  BoundaryBox* boundary;

public:
  Polygon3D(int maxSize)
  {
    this->maxSize = maxSize;
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
    Vec3 startV = boundary->getStartV().add(dV);
    Vec3 endV = boundary->getEndV().add(dV);
    poly.setBoundary(new BoundaryBox(startV,endV));
    return poly;
  }
  void print(){
    printf("Polygon: %d triangles\n",size);
    for (int i = 0; i < size; i++) tris[i]->print();
  }
  void generateBoundary();
  BoundaryBox* getBoundary(){return boundary;}
  void setBoundary(BoundaryBox* boundary){this->boundary = boundary;}
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

double min4(double a,double b,double c,double d){
  double res = (a<b)?a:b;
  res = (c<res)?c:res;
  res = (d<res)?d:res;
  return res;
}
double max4(double a,double b,double c,double d){
  double res = (a>b)?a:b;
  res = (c>res)?c:res;
  res = (d>res)?d:res;
  return res;  
}

void Polygon3D::generateBoundary(){
  double minX,minY,minZ,maxX,maxY,maxZ;
  minX = maxX = tris[0]->getV1().getX();
  minY = maxY = tris[0]->getV1().getY();
  minZ = maxZ = tris[0]->getV1().getZ();
  for (int i = 0; i < size; i++)
  {
    Triangle* tri = tris[i];
    Vec3 v1 = tri->getV1();
    Vec3 v2 = tri->getV2();
    Vec3 v3 = tri->getV3();
    minX = min4(minX,v1.getX(),v2.getX(),v3.getX());
    maxX = max4(maxX,v1.getX(),v2.getX(),v3.getX());
    minY = min4(minY,v1.getY(),v2.getY(),v3.getY());
    maxY = max4(maxY,v1.getY(),v2.getY(),v3.getY());
    minZ = min4(minZ,v1.getZ(),v2.getZ(),v3.getZ());
    maxZ = max4(maxZ,v1.getZ(),v2.getZ(),v3.getZ());
  }
  boundary = new BoundaryBox(Vec3(minX,minY,minZ),Vec3(maxX,maxY,maxZ));
}

IntersectionPoint Polygon3D::testIntersection(Ray r)
{
  IntersectionPoint cross;
  if(!boundary->doesHit(r)) return cross;
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
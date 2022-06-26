#ifndef POLYGON3D_H
#define POLYGON3D_H
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"
#include "ShapeSuite.h"
#include "Triangle.h"
#include "../kdTree.h"

class Polygon3D : public Shape
{
private:
  int size = 0;
  int maxSize = 0;
  Triangle **tris;
  BoundaryBox *boundary;
  TreeNode *kdTree;

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
  Polygon3D move(Vec3 dV)
  {
    Polygon3D poly(maxSize);
    for (int i = 0; i < size; i++)
    {
      poly.addTriangle(new Triangle(tris[i]->move(dV)));
    }
    Vec3 startV = boundary->getStartV().add(dV);
    Vec3 endV = boundary->getEndV().add(dV);
    poly.setBoundary(new BoundaryBox(startV, endV));
    return poly;
  }
  void print()
  {
    printf("Polygon: %d triangles\n", size);
    for (int i = 0; i < size; i++)
      tris[i]->print();
  }
  void generateBoundary();
  BoundaryBox *getBoundary() { return boundary; }
  Polygon3D *copy()
  {
    Polygon3D *newPoly = new Polygon3D(maxSize);
    for (int i = 0; i < size; i++)
      newPoly->addTriangle(tris[i]->copy());
    newPoly->setBoundary(boundary->copy());
    return newPoly;
  }
  void setBoundary(BoundaryBox *boundary) { this->boundary = boundary; }
  void addTriangle(Triangle *tri);
  Triangle **getTriangles(int *n)
  {
    *n = size;
    return tris;
  }
  IntersectionPoint testIntersection(Ray r);
  int find(std::vector<Vec3 *> vs, Vec3 v)
  {
    for (int i = 0; i < vs.size(); i++)
    {
      if (vs[i]->equals(v))
        return i;
    }
    return -1;
  }
  TreeNode *getKdTree() { return kdTree; }
  void setKdTree(TreeNode *root) { kdTree = root; }
  void buildKdTree()
  {
    printf("Building KdTree\n");
    std::vector<Vec3 *> v;
    for (int i = 0; i < size; i++)
    {
      Triangle *tri = tris[i];
      if (find(v, tri->getV1()) == -1)
        v.push_back(tri->getV1p());
      if (find(v, tri->getV2()) == -1)
        v.push_back(tri->getV2p());
      if (find(v, tri->getV3()) == -1)
        v.push_back(tri->getV3p());
    }
    Voxel vox(boundary->getStartV().sub(Vec3(1)), boundary->getEndV().add(Vec3(1)));
    TreeNode *root = recBuild(0, v, vox);
    setKdTree(root);
  }
};

void Polygon3D::addTriangle(Triangle *tri)
{
  if (size < maxSize)
  {
    tris[size] = tri;
    size++;
  }
}

void Polygon3D::generateBoundary()
{
  boundary = new BoundaryBox();
  for (int i = 0; i < size; i++)
  {
    Triangle *tri = tris[i];
    boundary->includeV(tri->getV1());
    boundary->includeV(tri->getV2());
    boundary->includeV(tri->getV3());
  }
}
#ifdef GPU_MODE
#include "../GPU_manager.h"

IntersectionPoint Polygon3D::testIntersection(Ray r)
{
  IntersectionPoint boundaryCross = boundary->testIntersection(r);
  IntersectionPoint cross;
  if (!boundaryCross.exists)
    return cross;
  //  clock_t st, ed;  st = clock();
  GPU_Manager m;
  m.addRay(r);
  m.calc(size,tris);
  cross = m.getResult(0);
//  PolyIntersection_GPU(size, tris, 1, &r,&cross);
  //  ed = clock();  printf("%f\n", (double)(ed - st) / CLOCKS_PER_SEC);
  return cross;
}
#endif

#ifndef GPU_MODE
IntersectionPoint Polygon3D::testIntersection(Ray r)
{
  IntersectionPoint boundaryCross = boundary->testIntersection(r);
  IntersectionPoint cross;
  if (!boundaryCross.exists)
    return cross;
  //  clock_t st, ed;st = clock();
  int closestId = -1;
  double closestDistance = -1;
  for (int i = 0; i < size; i++)
  {
    cross = tris[i]->testIntersection(r);
    if (cross.exists)
    {
      if (closestId == -1)
      {
        closestId = i;
        closestDistance = cross.distance;
      }
      else if (cross.distance < closestDistance)
      {
        closestId = i;
        closestDistance = cross.distance;
      }
    }
  }
  //  ed = clock();printf("%f\n", (double)(ed - st) / CLOCKS_PER_SEC);
  if (closestId != -1)
  {
    return tris[closestId]->testIntersection(r);
  }
  IntersectionPoint noCross;
  return noCross;
}
#endif
#endif
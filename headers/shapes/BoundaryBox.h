#ifndef BOUNDARY_BOX_H
#define BOUNDARY_BOX_H
#include "Shape.h"
#include "RectanglePlane.h"
class BoundaryBox : public Shape
{
private:
  Vec3 startV;
  Vec3 endV;
  RectanglePlane *walls[6];

public:
Vec3 getStartV(){return startV;}
Vec3 getEndV(){return endV;}
void setV(Vec3 V1, Vec3 V2){
    startV = V1;
    endV = V2;
    double x1, y1, z1, x2, y2, z2;
    x1 = V1.getX();
    y1 = V1.getY();
    z1 = V1.getZ();
    x2 = V2.getX();
    y2 = V2.getY();
    z2 = V2.getZ();
    walls[0] = new RectanglePlane(V1,Vec3(x1, y2, z2) );
    walls[1] = new RectanglePlane(V1,Vec3(x2, y1, z2) );
    walls[2] = new RectanglePlane(V1,Vec3(x2, y2, z1) );
    walls[3] = new RectanglePlane(V2,Vec3(x2, y1, z1) );
    walls[4] = new RectanglePlane(V2,Vec3(x1, y2, z1) );
    walls[5] = new RectanglePlane(V2,Vec3(x1, y1, z2) );
}
BoundaryBox(){}
  BoundaryBox(Vec3 V1, Vec3 V2)
  {
setV(V1,V2);
  }
  virtual int doesHit(Ray r) {
    for(int i=0;i<6;i++){
      if(walls[i]->doesHit(r))return 1;
    }
    return 0;
  }
  virtual IntersectionPoint testIntersection(Ray r);
  virtual void print(){printf("Boundary box\n");startV.print();endV.print();};
};



IntersectionPoint BoundaryBox::testIntersection(Ray r)
{
  IntersectionPoint res;
  for (int i = 0; i < 6; i++)
  {
    RectanglePlane *wall = walls[i];
    IntersectionPoint cross = wall->testIntersection(r);

    if (cross.exists)
    {
      if (!res.exists || cross.distance < res.distance)
      {
        res = cross;
      }
    }
  }
  return res;
}
#endif
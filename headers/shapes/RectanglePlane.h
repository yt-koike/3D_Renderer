#ifndef RECTANGLE_PLANE_H
#define RECTANGLE_PLANE_H
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"
#include "Shape.h"


class RectanglePlane : public Shape{
private:
  Vec3 startP;
  Vec3 endP;
  Vec3 normalV;
void orderVec(Vec3* a,Vec3* b){
    if(a->getX()>b->getX()){
        double x = a->getX();
        a->setX(b->getX());
        b->setX(x);
    }
    if(a->getY()>b->getY()){
        double y = a->getY();
        a->setY(b->getY());
        b->setY(y);
    }
    if(a->getZ()>b->getZ()){
        double z = a->getZ();
        a->setZ(b->getZ());
        b->setZ(z);
    }
}
int isBetween(double a,double x,double b){
  return (a<=x)&&(x<=b);
}

public:
RectanglePlane(Vec3 startP,Vec3 endP){
    this->startP = startP;
    this->endP = endP;
    orderVec(&(this->startP),&(this->endP));
    Vec3 dv = startP.sub(endP);
    if(dv.getX()==0){
      normalV = Vec3(1,0,0);
    }else if (dv.getY()==0){
      normalV = Vec3(0,1,0);
    }else if (dv.getZ()==0){
      normalV = Vec3(0,0,1);
    }else{
      printf("Error: invalid RectanglePlane\n");
    }
    // if it has one zero or less
    if(abs(normalV.getX()*normalV.getY())+abs(normalV.getY()*normalV.getZ())+abs(normalV.getZ()*normalV.getX())>0.001){
        printf("Error: invalid RectanglePlane\n");
    }
}
IntersectionPoint testIntersection(Ray r);
int doesHit(Ray r);
void print() { printf("Rectangle"); }
};

int RectanglePlane::doesHit(Ray r){
  if(normalV.getX()==1){
    if(r.getDir().getX()==0)return 0;
    double t = startP.sub(r.getPoint()).getX()/r.getDir().getX();
    if(t<0)return 0;
    double y = r.getDir().getY()*t + r.getPoint().getY();
    if(!isBetween(startP.getY(),y,endP.getY()))return 0;
    double z = r.getDir().getZ()*t + r.getPoint().getZ();
    if(!isBetween(startP.getZ(),z,endP.getZ()))return 0; 
  }else if (normalV.getY()==1){
    if(r.getDir().getY()==0)return 0;
    double t = startP.sub(r.getPoint()).getY()/r.getDir().getY();
    if(t<0)return 0;
    double x = r.getDir().getX()*t + r.getPoint().getX();
    if(!isBetween(startP.getX(),x,endP.getX()))return 0;
    double z = r.getDir().getZ()*t + r.getPoint().getZ();
    if(!isBetween(startP.getZ(),z,endP.getZ()))return 0; 
  }else if (normalV.getZ()==1){
    if(r.getDir().getZ()==0)return 0;
    double t = startP.sub(r.getPoint()).getZ()/r.getDir().getZ();
    if(t<0)return 0;
    double x = r.getDir().getX()*t + r.getPoint().getX();
    if(!isBetween(startP.getX(),x,endP.getX()))return 0;
    double y = r.getDir().getY()*t + r.getPoint().getY();
    if(!isBetween(startP.getY(),y,endP.getY()))return 0;
  }
  return 1; 
}

IntersectionPoint RectanglePlane::testIntersection(Ray r)
{
  IntersectionPoint res;
  double t,x,y,z;
  if(normalV.getX()==1){
    if(r.getDir().getX()==0)return res;
    x = startP.getX();
    t = startP.sub(r.getPoint()).getX()/r.getDir().getX();
    if(t<0)return res;
    y = r.getDir().getY()*t + r.getPoint().getY();
    if(!isBetween(startP.getY(),y,endP.getY()))return res;
    z = r.getDir().getZ()*t + r.getPoint().getZ();
    if(!isBetween(startP.getZ(),z,endP.getZ()))return res;
  }else if (normalV.getY()==1){
    if(r.getDir().getY()==0)return res;
    t = startP.sub(r.getPoint()).getY()/r.getDir().getY();
    if(t<0)return res;
    x = r.getDir().getX()*t + r.getPoint().getX();
    if(!isBetween(startP.getX(),x,endP.getX()))return res;
    y = startP.getY();
    z = r.getDir().getZ()*t + r.getPoint().getZ();
    if(!isBetween(startP.getZ(),z,endP.getZ()))return res;
  }else if (normalV.getZ()==1){
    if(r.getDir().getZ()==0)return res;
    t = startP.sub(r.getPoint()).getZ()/r.getDir().getZ();
    if(t<0)return res;
    x = r.getDir().getX()*t + r.getPoint().getX();
    if(!isBetween(startP.getX(),x,endP.getX()))return res;
    y = r.getDir().getY()*t + r.getPoint().getY();
    if(!isBetween(startP.getY(),y,endP.getY()))return res;
    z = startP.getZ();
  }
  res.exists = 1;
  res.position = Vec3(x,y,z);
  if(normalV.cos(r.getDir())<0){
    res.normal = normalV.mult(-1);
  }else{
    res.normal = normalV.copy();
  }
  res.distance = t;
  return res;
}
#endif
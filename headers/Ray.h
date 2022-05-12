#ifndef RAY_H
#define RAY_H
#include "Color.h"
class Ray{
    private:
        Vec3 start;
        Vec3 direction;
    public:
        Ray(){};
        Ray(Vec3 pos,Vec3 dir){start=pos;direction=dir.normalize();}
        Ray(double px,double py,double pz,double dx,double dy,double dz);
        Vec3 getPoint(){return start;}
        Vec3 getPoint(double t){return start.add(direction.mult(t));}
        Vec3 getDir(){return direction;}
};

Ray::Ray(double px,double py,double pz,double dx,double dy,double dz){
    start.setX(px);
    start.setY(py);
    start.setZ(pz);
    direction.setX(dx);
    direction.setY(dy);
    direction.setZ(dz);
}

class IntersectionPoint{
    public:
    int exists;
    double distance;
    Vec3 position;
    Vec3 normal;
    IntersectionPoint(){exists=0;}
};

class LightSource{
  public:
  Vec3 lightingAt;
};

class PointLightSource: public LightSource{
  public:
  Vec3 position;
  Color intensity;
  PointLightSource(Vec3 pos,Color intensity){this->position=pos;this->intensity=intensity;}
};

class DirectionalLightSource: public LightSource{
  public:
  Vec3 direction;
  Color intensity;
};

#endif
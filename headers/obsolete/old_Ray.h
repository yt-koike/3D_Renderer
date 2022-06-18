#ifndef RAY_H
#define RAY_H
#include "Vector.h"
#include "Color.h"
class Ray{
    private:
        Vec3 position;
        Vec3 direction;
        Color color;
    public:
        Ray(double px,double py,double pz,double dx,double dy,double dz);
        Vec3 getPos(){return position;}
        Vec3 getDir(){return direction;}
        void setColor(int r,int g,int b){color.setRGB(r,g,b);}
        Color getColor(){return color;}
};

Ray::Ray(double px,double py,double pz,double dx,double dy,double dz){
    position.setX(px);
    position.setY(py);
    position.setZ(pz);
    direction.setX(dx);
    direction.setY(dy);
    direction.setZ(dz);
    setColor(255,255,255);
}

#endif
#ifndef COLOR_H
#define COLOR_H
#include "Vector.h"
class Color : public Vec3{
    private:
    double alpha;
    public:
    Color(){setRGB(0,0,0);alpha=0;}
    Color(Vec3 colorRatio){set(colorRatio.getX(),colorRatio.getY(),colorRatio.getZ());alpha=1;}
    Color(int r,int g,int b){setRGBA(r,g,b,1);}
    Color(int r,int g,int b,double alpha){setRGBA(r,g,b,alpha);}
    void setRGB(double r,double g,double b){set(r/255,g/255,b/255);}
    void setAlpha(double alpha){this->alpha=alpha;}
    void setRGBA(double r,double g,double b,double alpha){setRGB(r,g,b);setAlpha(alpha);}
    int getR(){return getX()*255;}
    int getG(){return getY()*255;}
    int getB(){return getZ()*255;}
    Color clamp();
    double getAlpha(){return alpha;}
    void print(){printf("Color: %d,%d,%d\n",getR(),getG(),getB());};
};

Color Color::clamp(){
    Color c = this->copy();
    if(c.getX()<0)c.setX(0);
    if(c.getY()<0)c.setY(0);
    if(c.getZ()<0)c.setZ(0);
    if(c.getX()>1)c.setX(1);
    if(c.getY()>1)c.setY(1);
    if(c.getZ()>1)c.setZ(1);
    return c;
}
#endif
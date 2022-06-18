#ifndef OBJECTS_H
#define OBJECTS_H
#include "Vector.h"
#include "Color.h"
class Object{
    protected:
    double ka,kd,ks,alpha;
    Color surfaceColor;
    public:
    Object(){ka=kd=ks=alpha=1;}
    double getKa(){return ka;}
    double getKd(){return kd;}
    double getKs(){return ks;}
    void setKa(double ka){this->ka=ka;}
    void setKd(double kd){this->kd=kd;}
    void setKs(double ks){this->ks=ks;}
    double getAlpha(){return alpha;}
    void setSurfaceColor(Color c){surfaceColor = c;}
    Color getSurfaceColor(){return surfaceColor;}
};

class Sphere : public Object{
    private:
    Vec3 c;
    double r;
    public:
    void init(Vec3 c,double r);
    void init(Vec3 c,double r,double ka,double kd,double ks,double alpha);
    Vec3 getCenter(){return c;}
    int isCross(Vec3 s,Vec3 d);
    Vec3 firstCross(Vec3 s,Vec3 d);
    double lightness(Vec3 cameraPos,Vec3 cameraDir,Vec3 pointRay,double envRayIntensity,double rayIntensity);
};

void Sphere::init(Vec3 c,double r){
    this -> c = c;
    this -> r = r;
    this -> ka = 0;
    this -> kd = 1;
    this -> ks = 0;
    this -> alpha = 0;
}

void Sphere::init(Vec3 c,double r,double ka,double kd,double ks,double alpha){
    this -> c = c;
    this -> r = r;
    this -> ka = ka;
    this -> kd = kd;
    this -> ks = ks;
    this -> alpha = alpha;
}

int Sphere::isCross(Vec3 s,Vec3 d){
  s = s.sub(c);
  double A,B,C;
  A = d.magSq();
  B = 2*s.dot(d);
  C = s.magSq() - r*r;
  return B*B-4*A*C >= 0;
}

Vec3 Sphere::firstCross(Vec3 s,Vec3 d){
  double A,B,C;
  A = d.magSq();
  B = 2*s.sub(c).dot(d);
  C = s.sub(c).magSq() - r*r;
  double t = (-B - sqrt(B*B-4*A*C))/(2*A);
  return s.add(d.mult(t));
}


double Sphere::lightness(Vec3 cameraPos,Vec3 cameraDir,Vec3 pointRay,double envRayIntensity,double rayIntensity){
        double Ra =  envRayIntensity * getKa(); 
        Vec3 cross1 = firstCross(cameraPos,cameraDir);
        Vec3 n = cross1.sub(getCenter()).normalize();
        Vec3 l = pointRay.sub(cross1).normalize();
        double Rd = getKd() * rayIntensity * n.dot(l);
        if(Rd<0)Rd=0;
        Vec3 r = n.mult(2*n.dot(l)).sub(l);
        double Rs = getKs() * rayIntensity * pow(r.dot(cameraDir.mult(-1).normalize()),getAlpha());
        return Ra + Rd + Rs;
}

class Plane : public Object{
  private:
  Vec3 pointV;
  Vec3 normalV;
  public:
  void init(Vec3 point,Vec3 normal){setPointV(point);setNormalV(normal);}
  Vec3 getPointV(){return pointV;}
  Vec3 getNormalV(){return normalV;}
  Vec3 setPointV(Vec3 v){this->pointV = v;}
  Vec3 setNormalV(Vec3 v){this->normalV = v;}
  int isCross(Vec3 s,Vec3 d);
  Vec3 firstCross(Vec3 s,Vec3 d);
  double lightness(Vec3 cameraPos,Vec3 cameraDir,Vec3 pointRay,double envRayIntensity,double rayIntensity);
};

int Plane::isCross(Vec3 s,Vec3 d){
  return normalV.dot(d) != 0;
}

Vec3 Plane::firstCross(Vec3 s,Vec3 d){
  double t = - normalV.dot(s) / normalV.dot(d);
  return s.add(d.mult(t));
}

double Plane::lightness(Vec3 cameraPos,Vec3 cameraDir,Vec3 pointRay,double envRayIntensity,double rayIntensity){
        double Ra =  envRayIntensity * getKa(); 
        Vec3 cross1 = firstCross(cameraPos,cameraDir);
        Vec3 n = normalV.normalize();
        Vec3 l = pointRay.sub(cross1).normalize();
        double Rd = getKd() * rayIntensity * n.dot(l);
        if(Rd<0)Rd=0;
        Vec3 r = n.mult(2*n.dot(l)).sub(l);
        double Rs = getKs() * rayIntensity * pow(r.dot(cameraDir.mult(-1).normalize()),getAlpha());
        return Ra ;//+ Rd + Rs;
}

#endif
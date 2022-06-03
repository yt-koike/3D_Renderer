#include<iostream>
#include <fstream>
#include "PPM.h"
#include "vec.h"
#include "Ray.h"
#include "Objects.h"
using namespace std;

int main(){
  Vec3 s(0,0,-5);
  int W = 512;
  int H = 512;
  Vec3 pointRay(-5,-5,-5);
  double envRayIntensity = 0.1;
  double rayIntensity = 1;
  Vec3 spCenter(0,0,5);
  double r = 1;
  Sphere sp;
  sp.init(spCenter,r,0.01,0.69,0.3,8);
  PPM ppmSphere(W,H,255);
  for(double y = 0;y<H;y+=1){
    for(double x = 0;x<W;x+=1){
      Vec3 d(x/W-0.5,y/H-0.5,1);
      if (sp.isCross(s,d)){
        double Ra =  envRayIntensity * sp.getKa(); 
        Vec3 firstCross = sp.firstCross(s,d);
        Vec3 n = firstCross.sub(sp.getCenter()).normalize();
        Vec3 l = pointRay.sub(firstCross).normalize();
        double Rd = sp.getKd() * rayIntensity * n.dot(l);
        if(Rd<0)Rd=0;
        Vec3 r = n.mult(2*n.dot(l)).sub(l);
        double Rs = sp.getKs() * rayIntensity * pow(r.dot(d.mult(-1).normalize()),sp.getAlpha());
        double lightness = Ra + Rd + Rs;
        ppmSphere.setPixel(x,y,255*lightness,255*lightness,255*lightness);
      }else{
        ppmSphere.setPixel(x,y,100,149,237);
      }
    }
  }
  printf("Rendering Complete");
  ppmSphere.writePPM("SimpleShading.ppm");
  return 0;
}
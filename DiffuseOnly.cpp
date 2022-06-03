#include<iostream>
#include <fstream>
#include "PPM.h"
#include "headers/Vector.h"
#include "headers/Ray.h"
#include "headers/Objects.h"
using namespace std;

int main(){
  Vec3 s(0,0,-5);
  int W = 512;
  int H = 512;
  Vec3 pointRay(-5,-5,-5);
  Vec3 spCenter(0,0,5);
  double r = 1;
  Sphere sp;
  sp.init(spCenter,r);
  PPM ppmSphere(W,H,255);
  for(double y = 0;y<H;y+=1){
    for(double x = 0;x<W;x+=1){
      Vec3 d(x/W-0.5,y/H-0.5,1);
     //printf("%f,%f : %d\n",x,y,sp.isCross(s,d));
      if (sp.isCross(s,d)){
        Vec3 firstCross = sp.firstCross(s,d);
        Vec3 n = firstCross.sub(sp.getCenter());
        double lightness = n.cos(pointRay.sub(firstCross));
        if(lightness<0)lightness=0;
        ppmSphere.setPixel(x,y,255*lightness,255*lightness,255*lightness);
      }else{
        ppmSphere.setPixel(x,y,100,149,237);
      }
    }
  }
  printf("Rendering Complete");
  ppmSphere.writePPM("DiffuseOnly.ppm");
  return 0;
}
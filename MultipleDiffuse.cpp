#include<iostream>
#include <fstream>
#include "headers/PPM.h"
#include "headers/Vector.h"
#include "headers/Ray.h"
#include "headers/Objects.h"
using namespace std;

int main(){
  Vec3 s(0,0,-5);
  int W = 512;
  int H = 512;
  Vec3 pointRay(-5,-5,-5);
  Sphere spheres[2];
  spheres[0].init(Vec3(0,0,10),1);
  spheres[1].init(Vec3(2,0,5),1);
  PPM ppmSphere(W,H,255);
  for(double y = 0;y<H;y+=1){
    for(double x = 0;x<W;x+=1){
      Vec3 d(x/W-0.5,y/H-0.5,1);
     //printf("%f,%f : %d\n",x,y,sp.isCross(s,d));
     Sphere sp;
     double lightness = -1;
     for(int i=0;i<2;i++){
      sp = spheres[i];
      if (sp.isCross(s,d)){
        Vec3 firstCross = sp.firstCross(s,d);
        Vec3 n = firstCross.sub(sp.getCenter());
        lightness = n.cos(pointRay.sub(firstCross));
        if(lightness<0)lightness=0;
      }
     }
     if(lightness<0){
        ppmSphere.setPixel(x,y,100,149,237);
     }else{
        ppmSphere.setPixel(x,y,lightness*255,lightness*255,lightness*255);
     }
    }
  }
  printf("Rendering Complete");
  ppmSphere.writePPM("DiffuseOnly.ppm");
  return 0;
}
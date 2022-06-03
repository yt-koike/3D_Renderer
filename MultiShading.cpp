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
  double envRayIntensity = 0.1;
  double rayIntensity = 1;
  Vec3 spCenter(0,0,5);
  double r = 1;
  Sphere spheres[3];
  spheres[0].init(Vec3(0,0,10),1,0.01,0.7,0.1,5);
  spheres[1].init(Vec3(2,0,5),1,0.8,0.5,0.2,10);
  spheres[2].init(Vec3(-3,0,5),2);
  Plane pla;
  pla.init(Vec3(0,-20,0),Vec3(0,1,0));
  pla.setKa(10);
  PPM ppmSphere(W,H,255);
  for(double y = 0;y<H;y+=1){
    for(double x = 0;x<W;x+=1){
      Vec3 d(x/W-0.5,y/H-0.5,1);
      double finalLightness = -1;
      if(pla.isCross(s,d)){
        double lightness = pla.lightness(s,d,pointRay,envRayIntensity,rayIntensity);
        if (lightness>=0) finalLightness = lightness;
      }
      Sphere sp;
      for(int i=0;i<3;i++){
        sp = spheres[i];
        if (sp.isCross(s,d)){
          double lightness = sp.lightness(s,d,pointRay,envRayIntensity,rayIntensity);
          if (lightness<0) lightness = 0;
          finalLightness = lightness;
        }
      }
      if(finalLightness>=0){
        ppmSphere.setPixel(x,y,255*finalLightness,255*finalLightness,255*finalLightness);
      }else{
        ppmSphere.setPixel(x,y,100,149,237);
      }
    }
  }
  printf("Rendering Complete");
  ppmSphere.writePPM("MultiShading.ppm");
  return 0;
}
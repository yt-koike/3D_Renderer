#include<iostream>
#include <fstream>
#include "headers/PPM.h"
#include "headers/Vector.h"
#include "headers/Objects.h"
using namespace std;

int main(){
  Vec3 c(0,0,0);
  double r = 1;
  Circle circ(&c,r);

  Vec3 s(0,0,-5);
  int W = 512;
  int H = 512;
  Vec3 lightV(1,1,1);
  PPM ppmCirc(W,H,255);
  for(double y = 0;y<H;y+=1){
    for(double x=0;x<W;x+=1){
      Vec3 d(x/W-0.5,y/H-0.5,1);
//      printf("%f,%f : %d\n",x,y,circ.isCross(s,d));
      if (circ.isCross(s,d)){
        ppmCirc.setPixel(x,y,255,0,0);
      }else{
        ppmCirc.setPixel(x,y,0,0,255);
      }
    }
  }
  ppmCirc.writePPM("Circle.ppm");
  return 0;
}
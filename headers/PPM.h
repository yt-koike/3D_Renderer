#ifndef PPM_H
#define PPM_H
#include<iostream>
#include<fstream>
#include"Array2D.h"

class Pixel{
  public:
    int r,g,b;
};

class PPM{
private:
  int width,height,maxV;
  ColorImage img;
public:
  PPM(int w,int h,int maxV);
  PPM(ColorImage image){import(image);}
  void init(int w,int h,int maxV);
  Pixel getPixel(int x,int y);
  void setPixel(int x,int y,int r,int g,int b);
  int readPPM(const char* filename);
  int writePPM(const char* filename);
  void import(ColorImage image){width=image.getWidth();height=image.getHeight();maxV=255;this->img = image;}
  int getWidth(){return width;};
  int getHeight(){return height;};
};

PPM::PPM(int w,int h,int maxV){
  init(w,h,maxV);
}

void PPM::init(int w,int h,int maxV){
  this->width = w;
  this->height = h;
  this->maxV = maxV;
  img.init(w,h);
}

Pixel PPM::getPixel(int x,int y){
  Pixel result;
  result.r = img.getR(x,y);
  result.g = img.getG(x,y);
  result.b = img.getB(x,y);
  return result;
}

void PPM::setPixel(int x,int y,int r,int g,int b){
  img.set(x,y,r,g,b);
}

int PPM::readPPM(const char* filename){
  std::ifstream ifs(filename);
  char buf[100];
  ifs >> buf;
  ifs >> width >> height;
  ifs >> maxV;
  init(width,height,maxV);
  int r,g,b;
  int x = 0;
  int y = 0;
  while(!ifs.eof()){
    ifs >> r >> g >> b;
    setPixel(x,y,r,g,b);
    x++;
    if(x>=width){
      x = 0;
      y++;
    }
  }
  ifs.close();
  return 0;
}

int PPM::writePPM(const char* filename){
  std::ofstream ofs(filename);
  ofs << "P3" << std::endl;
  ofs << width << " " << height << std::endl;
  ofs << maxV << std::endl;
  for(int i =0;i<height;i++){
    for(int j=0;j<width;j++){
      ofs << img.getR(j,i) << " " << img.getG(j,i) <<" "<< img.getB(j,i) << std::endl;
    }
  }
  ofs.close();
  return 0;
}
#endif
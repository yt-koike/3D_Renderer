#ifndef SHAPE_H
#define SHAPE_H
#include <vector>
#include "../Vector.h"
#include "../Color.h"
#include "../Ray.h"
#include "../Material.h"

class Shape
{
protected:
  int visible = 1;
  Material mt;

public:
  Shape(){
    Material defaultMt(Color(Vec3(0.01)), Color(Vec3(0.69)), Color(Vec3(0.3)), Vec3(8)); 
    setMaterial(defaultMt);
  }
  Material getMaterial() { return mt; }
  void setMaterial(Material mt) { this->mt=mt;}
  int isVisible(){return visible;}
  void setVisible(int flag){visible=flag;}
  virtual IntersectionPoint testIntersection(Ray r){IntersectionPoint cross;return cross;}
  virtual void testIntersections(int rayN,Ray* rs,IntersectionPoint* result){for(int i=0;i<rayN;i++){result[i]=testIntersection(rs[i]);}}
  virtual void print() { printf("Null shape"); }
  Color envLightness(Color envRayIntensity);
  Color lightness(IntersectionPoint cross, Vec3 cameraDir, PointLightSource light);
};

Color Shape::envLightness(Color envRayIntensity)
{
  Color Ra = Color(mt.getKa().mask(envRayIntensity)).clamp();
  return Ra;
}

Color Shape::lightness(IntersectionPoint cross, Vec3 cameraDir, PointLightSource light)
{
  Vec3 cross1 = cross.position;
  Vec3 n = cross.normal;
  Vec3 l = light.position.sub(cross1).normalize();
  Color Rd = Color(light.intensity.mult(n.dot(l)).mask(mt.getKd()));
  Vec3 r = n.mult(2 * n.dot(l)).sub(l);
  Vec3 antiViewVec = cameraDir.mult(-1).normalize();
  Color Rs = Color(light.intensity.mask(Vec3(1, 1, 1).mult(r.dot(antiViewVec)).vecPow(mt.getAlpha())).mask(mt.getKs()));
  return Color(Rd.add(Rs));
}

#endif
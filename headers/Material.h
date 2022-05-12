#ifndef MATERIAL_H
#define MATERIAL_H
#include"Vector.h"
#include"Color.h"
class Material{
    private:
    Color ka,kd,ks;
    Vec3 alpha;
    int usePerfectReflectance = 0;
    Color catadioptricFactor = Color(Vec3(0));
    int useRefraction = 0; // 屈折を使用するかどうか
    double refractionIndex = 1.0; // 絶対屈折率
    public:
    Material(){};
    Material(Color ka,Color kd,Color ks,Vec3 alpha){setKa(ka);setKd(kd);setKs(ks);setAlpha(alpha);};
    Color getKa(){return ka;}
    Color getKd(){return kd;}
    Color getKs(){return ks;}
    Vec3 getAlpha(){return alpha;}
    void setKa(Color ka){this->ka=ka;}
    void setKd(Color kd){this->kd=kd;}
    void setKs(Color ks){this->ks=ks;}
    void setAlpha(Vec3 alpha){this->alpha=alpha;}
    int getUsePerfectReflectance(){return usePerfectReflectance;}
    Color getCatadioptricFactor(){return catadioptricFactor;}
    void setUsePerfectReflectance(int flag){usePerfectReflectance=flag;}
    void setCatadioptricFactor(Color c){catadioptricFactor=c;}
    int getUseRefraction(){return useRefraction;}
    double getRefractionIndex(){return refractionIndex;}
    void setUseRefraction(int flag){useRefraction=flag;}
    void setRefractionIndex(double x){refractionIndex=x;}
};
#endif
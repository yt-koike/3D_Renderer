#ifndef VEC_H
#define VEC_H
#include <stdio.h>
#include <math.h>

class Vec3
{
private:
    double x, y, z;

public:
    Vec3() { set(0, 0, 0); }
    Vec3(double a) { set(a, a, a); }
    Vec3(double x, double y, double z) { set(x, y, z); }
    
    double getX() { return x; }
    double getY() { return y; }
    double getZ() { return z; }
    void set(double x, double y, double z)
    {
        setX(x);
        setY(y);
        setZ(z);
    }
    void setX(double x) { this->x = x; }
    void setY(double y) { this->y = y; }
    void setZ(double z) { this->z = z; }
    Vec3 copy() { return Vec3(this->getX(), this->getY(), this->getZ()); }
    Vec3 add(double x, double y, double z) { return Vec3(this->getX() + x, this->getY() + y, this->getZ() + z); }
    Vec3 sub(double x, double y, double z) { return Vec3(this->getX() - x, this->getY() - y, this->getZ() - z); }
    Vec3 mult(double k) { return Vec3(this->getX() * k, this->getY() * k, this->getZ() * k); }
    Vec3 add(Vec3 v) { return this->add(v.getX(), v.getY(), v.getZ()); }
    Vec3 sub(Vec3 v) { return this->sub(v.getX(), v.getY(), v.getZ()); }
    Vec3 mask(Vec3 v) { return Vec3(getX() * v.getX(), getY() * v.getY(), getZ() * v.getZ()); }
    Vec3 vecPow(Vec3 v) { return Vec3(pow(getX(), v.getX()), pow(getY(), v.getY()), pow(getZ(), v.getZ())); }
    double dot(Vec3 v) { return this->getX() * v.getX() + this->getY() * v.getY() + this->getZ() * v.getZ(); }
    Vec3 cross(Vec3 v);
    double mag() { return sqrt((long double)getX() * getX() + getY() * getY() + getZ() * getZ()); }
    double magSq() { return getX() * getX() + getY() * getY() + getZ() * getZ(); }
    double cos(Vec3 v) { return dot(v) / mag() / v.mag(); }
    Vec3 normalize() { return mult(1 / mag()); }
    Vec3 rotate(Vec3 origin, Vec3 axis, double rad);
    Vec3 max(Vec3 v){
        double newX = (x>v.getX())?x:v.getX();
        double newY = (y>v.getY())?y:v.getY();
        double newZ = (z>v.getZ())?z:v.getZ();
        return Vec3(newX,newY,newZ);
    }
    Vec3 min(Vec3 v){
        double newX = (x<v.getX())?x:v.getX();
        double newY = (y<v.getY())?y:v.getY();
        double newZ = (z<v.getZ())?z:v.getZ();
        return Vec3(newX,newY,newZ);
    }
    void print() { printf("%f,%f,%f\n", x, y, z); };
};

Vec3 Vec3::cross(Vec3 v)
{
    double a1 = getX();
    double a2 = getY();
    double a3 = getZ();
    double b1 = v.getX();
    double b2 = v.getY();
    double b3 = v.getZ();
    return Vec3(a2 * b3 - b2 * a3, a3 * b1 - b3 * a1, a1 * b2 - b1 * a2);
}

Vec3 Vec3::rotate(Vec3 origin, Vec3 axis, double rad)
{
    //ロドリゲスの回転公式 https://www.mynote-jp.com/entry/2016/04/30/201249
    Vec3 v = this->sub(origin);
    if(v.mag()==0)return *this;
    double s = sin((long double)rad);
    double c = cos((long double)rad);
    double Nx, Ny, Nz;
    Nx = axis.getX();
    Ny = axis.getY();
    Nz = axis.getZ();
    double new_x, new_y, new_z;
    new_x = v.dot(Vec3(c + Nx * Nx * (1 - c), Nx * Ny * (1 - c) - Nz * s, Nz * Nx * (1 - c) + Ny * s));
    new_y = v.dot(Vec3(Nx * Ny * (1 - c) + Nz * s, c + Ny * Ny * (1 - c), Ny * Nz * (1 - c) - Nx * s));
    new_z = v.dot(Vec3(Nz * Nx * (1 - c) - Ny * s, Ny * Nz * (1 - c) + Nx * s, c + Nz * Nz * (1 - c)));
    return Vec3(new_x, new_y, new_z).add(origin);
}

class Vec3m // Vector 3 mutable
{
private:
    double x, y, z;

public:
    Vec3m() { set(0, 0, 0); }
    Vec3m(double a) { set(a, a, a); }
    Vec3m(double x, double y, double z) { set(x, y, z); }
    Vec3m(Vec3 v) { set(v.getX(), v.getY(), v.getZ()); }
    double getX() { return x; }
    double getY() { return y; }
    double getZ() { return z; }
    void set(double x, double y, double z)
    {
        setX(x);
        setY(y);
        setZ(z);
    }
    void setX(double x) { this->x = x; }
    void setY(double y) { this->y = y; }
    void setZ(double z) { this->z = z; }
    Vec3m copy() { return Vec3m(x, y, z);}
    Vec3 copyImmutable(){return Vec3(x, y, z);}
    void add(double x, double y, double z) {this->x += x; this->y += y; this->z += z;}
    void sub(double x, double y, double z) {this->x -= x; this->y -= y; this->z -= z;}
    void mult(double k) {this->x *= x; this->y *= y; this->z *= z;}
    void add(Vec3 v) { add(v.getX(),v.getY(),v.getZ()); }
    void sub(Vec3 v) { sub(v.getX(),v.getY(),v.getZ()); }
    void mask(Vec3 v) { this->x *= v.getX(); this->y *= y; this->z *= z; }
    void vecPow(Vec3 v) {this->x=pow(x, v.getX()); this->y=pow(y, v.getY());this->z=pow(z, v.getZ());}
    double dot(Vec3 v) { return x * v.getX() + y * v.getY() + z * v.getZ(); }
    void cross(Vec3 v);
    double mag() { return sqrt((long double)x*x + y*y + z*z); }
    double magSq() { return x*x+y*y+z*z; }
    double cos(Vec3 v) { return dot(v) / mag() / v.mag(); }
    void normalize() { mult(1 / mag()); }
    Vec3m max(Vec3 v){
        x = (x>v.getX())?x:v.getX();
        y = (y>v.getY())?y:v.getY();
        z = (z>v.getZ())?z:v.getZ();
    }
    Vec3m min(Vec3 v){
        x = (x<v.getX())?x:v.getX();
        y = (y<v.getY())?y:v.getY();
        z = (z<v.getZ())?z:v.getZ();
    }
    void print() { printf("%f,%f,%f\n", x, y, z); };
};

void Vec3m::cross(Vec3 v)
{
    double a1 = getX();
    double a2 = getY();
    double a3 = getZ();
    double b1 = v.getX();
    double b2 = v.getY();
    double b3 = v.getZ();
    x = a2 * b3 - b2 * a3;
    y = a3 * b1 - b3 * a1;
    z = a1 * b2 - b1 * a2;
}

#endif
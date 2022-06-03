#include "headers/kdTree.h"

int main(){
    std::vector<Triangle *> t;
    Vec3 v1,v2,v3;
    v1 = Vec3(0,0,0);
    v2 = Vec3(1,0,0);
    v3 = Vec3(0,1,0);
    t.push_back(new Triangle(v1,v2,v3));
    v1 = Vec3(0,0,10);
    v2 = Vec3(1,0,10);
    v3 = Vec3(0,1,10);
    t.push_back(new Triangle(v1,v2,v3));
    Voxel v(Vec3(-100),Vec3(100));
    TreeNode* root = (TreeNode*)recBuild(t,v);
    seekPrint(root);
    return 0;
}
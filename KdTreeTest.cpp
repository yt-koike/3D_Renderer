#include "headers/kdTree.h"
#include "headers/STL.h"

int main()
{
    Polygon3D sphere = STLBinLoad("ICO_Sphere.stl");
    TreeNode* root = sphere.buildKdTree();
    Vec3* a = search(root,Vec3(0,0,-1.0f));
    a->print();
    return 0;
}
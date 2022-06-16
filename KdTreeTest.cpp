#include "headers/kdTree.h"
#include "headers/STL.h"



int main()
{
    Polygon3D sphere = STLBinLoad("ICO_Sphere.stl");
    TreeNode* root = sphere.buildKdTree();
    //seekPrint(root);
    //sphere.getBoundary()->print();

    search(root,Vec3(0,0,-1.0f))->print();    
    return 0;
}
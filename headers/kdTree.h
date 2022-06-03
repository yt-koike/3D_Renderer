#ifndef TD_TREE_H
#define TD_TREE_H
#include "Vector.h"
#include"Shapes.h"
#include <vector>

class Voxel{
  private:
  Vec3 start;
  Vec3 end;
  public:
  Voxel(Vec3 start,Vec3 end){this->start=start;this->end=end;}
Vec3 getStart(){return start;}
Vec3 getEnd(){return end;}
};

class TreeNode{
    private:
    Vec3* pos;
    Vec3* normal;
    TreeNode* left = nullptr;
    TreeNode* right = nullptr;
    int leafFlag = 0;
    Triangle* leaf = nullptr;
    public:
    TreeNode(Vec3* pos,Vec3* normal,TreeNode* left,TreeNode* right){this->pos = pos;this->normal=normal;this-> left = left;this-> right = right;}
    TreeNode(Triangle* t){leafFlag=1;leaf=t;}
    Vec3* getPos(){return pos;}
    Vec3* getNormal(){return normal;}
    TreeNode* getLeft(){return left;}
    TreeNode* getRight(){return right;}
    int isLeaf(){return leafFlag;}
    Triangle* getTriangle(){return leaf;}
};

int inVoxel(Vec3 p,Voxel v){
    Vec3 start= v.getStart();
    Vec3 end = v.getEnd();
    return (start.getX() <= p.getX() && p.getX() <=end.getX()) &&
    (start.getY() <= p.getY() && p.getY() <=end.getY()) &&
    (start.getZ() <= p.getZ() && p.getZ() <=end.getZ());
}

int inVoxel(Triangle t,Voxel v){
     return inVoxel(t.getV1(),v) || inVoxel(t.getV2(),v) || inVoxel(t.getV3(),v);
}

Plane findPlane(std::vector<Triangle *> ts,Voxel v){
    return Plane(Vec3(0,0,5),Vec3(0,0,1));
}

TreeNode* recBuild(std::vector<Triangle *> ts,Voxel v){
    if (ts.size() == 1){
        return new TreeNode(ts[0]);
    }
    Plane p = findPlane(ts,v);
    Vec3 planeP = p.getPointV();
    Vec3 planeN = p.getNormalV();

    Vec3 vL_start = v.getStart();
    Vec3 vR_end = v.getEnd();
    Vec3 vR_start  = vL_start.copy();
    Vec3 vL_end  = vR_end.copy();
    if(abs(planeN.getX()-1)<0.001f){ // YZ
        double pX = planeP.getX();
        vR_start.setX(pX);
        vL_end.setX(pX);
    }else if(abs(planeN.getY()-1)<0.001f){ // XZ
        double pY = planeP.getY();
        vR_start.setY(pY);
        vL_end.setY(pY);
    }else if(abs(planeN.getZ()-1)<0.001f){ // XY
        double pZ = planeP.getZ();
        vR_start.setZ(pZ);
        vL_end.setZ(pZ);
    }else{
        perror("invalid plane angle!");
    }

    Voxel vL(vL_start,vL_end);
    Voxel vR(vR_start,vR_end);

    std::vector<Triangle *> tL,tR;
    for(int i=0;i<ts.size();i++){
        Triangle* t = ts[i];
        if(inVoxel(*t,vL)){
            tL.push_back(t);
        }
        if(inVoxel(*t,vR)){
            tR.push_back(t);
        }
    }
    return new TreeNode(&planeP,&planeN,recBuild(tL,vL),recBuild(tR,vR));
}


void seekPrint(TreeNode* root){
    if(root->isLeaf()){
        Triangle tri = root->getTriangle()->copy();
        tri.print();
    }else{
        Vec3 p=root->getPos()->copy();
        Vec3 n=root->getNormal()->copy();
        p.print();
        n.print();
        seekPrint(root->getLeft());
        seekPrint(root->getRight());
    }
}
#endif
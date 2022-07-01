#ifndef TD_TREE_H
#define TD_TREE_H
#include <stdlib.h>
#include "Vector.h"
#include <vector>
#include "shapes/Plane.h"
#include "shapes/Triangle.h"
class Voxel
{
private:
    Vec3 start;
    Vec3 end;

public:
    Voxel(Vec3 start, Vec3 end)
    {
        this->start = start;
        this->end = end;
    }
    Vec3 getStart() { return start; }
    Vec3 getEnd() { return end; }
};

class TreeNode
{
private:
    Vec3 *pos;
    Vec3 *normal;
    std::vector<Vec3 *> onPlane;
    TreeNode *left = nullptr;
    TreeNode *right = nullptr;
    int leafFlag = 0;
    Vec3 *leaf;

public:
    TreeNode(Vec3 *pos, Vec3 *normal, TreeNode *left, TreeNode *right)
    {
        this->pos = pos;
        this->normal = normal;
        this->left = left;
        this->right = right;
    }
    TreeNode(Vec3 *v)
    {
        this->leafFlag = 1;
        this->leaf = v;
    }
    Vec3 *getPos() { return pos; }
    Vec3 *getNormal() { return normal; }
    TreeNode *getLeft() { return left; }
    TreeNode *getRight() { return right; }
    int isLeaf() { return leafFlag; }
    Vec3 *getLeaf() {return leaf; }
    void setOnPlane(std::vector<Vec3 *> vs) { onPlane = vs; }
    std::vector<Vec3 *> getOnPlane() { return onPlane; }
    void move(Vec3 dV){
        if(isLeaf()){
            *leaf = leaf->add(dV);
        }else{
            *pos = pos->add(dV);
            getLeft()->move(dV);
            getRight()->move(dV);
        }
        
    }
    TreeNode *copy()
    {
        if (isLeaf())
        {
            return new TreeNode(leaf);
        }
        else
        {
            return new TreeNode(pos, normal, left, right);
        }
    }
    Vec3* searchNearest(Vec3 v);
    void search(Vec3 v,int n,std::vector<Vec3*> *li);
};

int isBetween(double a, double x, double b)
{
    return a <= x && x <= b;
}

int inVoxel(Vec3 p, Voxel v)
{
    Vec3 start = v.getStart();
    Vec3 end = v.getEnd();
    return isBetween(start.getX(), p.getX(), end.getX()) &&
           isBetween(start.getY(), p.getY(), end.getY()) &&
           isBetween(start.getZ(), p.getZ(), end.getZ());
}

int inVoxel(Triangle t, Voxel v)
{
    return inVoxel(t.getV1(), v) || inVoxel(t.getV2(), v) || inVoxel(t.getV3(), v);
}

int compare_double(const void *a, const void *b)
{
    return *(double *)a - *(double *)b;
}

Plane findPlane(const int depth, std::vector<Vec3 *> vs, Voxel v)
{
    Vec3 N;
    switch (depth % 3)
    {
    case 0:
        N.set(1, 0, 0);
        break;
    case 1:
        N.set(0, 1, 0);
        break;
    case 2:
        N.set(0, 0, 1);
        break;
    }
    Vec3 pos;
    double *ary = new double[vs.size()];
    for (int i = 0; i < vs.size(); i++)
    {
        ary[i] = vs[i]->dot(N);
    }
    qsort(ary, vs.size(), sizeof(double), compare_double);
    double center = ary[vs.size() / 2];
    pos = N.mult(center);
    delete ary;
    return Plane(pos, N);
}


TreeNode *recBuild(const int depth, std::vector<Vec3 *> vs, Voxel v)
{
    if (vs.size() == 0)
    {
        return nullptr;
    }
    if (vs.size() == 1)
    {
        return new TreeNode(vs[0]);
    }
    Plane p = findPlane(depth, vs, v);
    Vec3 planeP = p.getPointV();
    Vec3 planeN = p.getNormalV();

    Vec3 vL_start = v.getStart();
    Vec3 vR_end = v.getEnd();
    Vec3 vR_start = vL_start.copy();
    Vec3 vL_end = vR_end.copy();
    if (planeN.getX() == 1)
    { // YZ
        double pX = planeP.getX();
        vR_start.setX(pX);
        vL_end.setX(pX);
    }
    else if (planeN.getY() == 1)
    { // XZ
        double pY = planeP.getY();
        vR_start.setY(pY);
        vL_end.setY(pY);
    }
    else if (planeN.getZ() == 1)
    { // XY
        double pZ = planeP.getZ();
        vR_start.setZ(pZ);
        vL_end.setZ(pZ);
    }
    else
    {
        perror("invalid plane angle!");
    }

    Voxel voxL(vL_start, vL_end);
    Voxel voxR(vR_start, vR_end);

    std::vector<Vec3 *> vL, vR;
    std::vector<Vec3 *> onPlane;
    double center = planeP.dot(planeN);

    for (int i = 0; i < vs.size(); i++)
    {
        Vec3 *v = vs[i];
        if (v->dot(planeN) == center)
        {
            onPlane.push_back(v);
            continue;
        }
        if (inVoxel(*v, voxL))
        {
            vL.push_back(v);
        }
        else if (inVoxel(*v, voxR))
        {
            vR.push_back(v);
        }
    }
    TreeNode *result = new TreeNode(&planeP, &planeN, recBuild(depth + 1, vL, voxL), recBuild(depth + 1, vR, voxR));
    result->setOnPlane(onPlane);
    return result;
}
TreeNode *recBuild(std::vector<Vec3 *> vs, Voxel v){return recBuild(0,vs,v);}

void seekPrint(TreeNode *root)
{
    if(!root)return;
    if (root->isLeaf())
    {
        Vec3 *leaf = root->getLeaf();
        leaf->print();
    }
    else
    {
        std::vector<Vec3 *> ps = root->getOnPlane();
        for (int i = 0; i < ps.size(); i++)
        {
            ps[i]->print();
        }
        seekPrint(root->getLeft());
        seekPrint(root->getRight());
    }
}

int doesHave(TreeNode *root, Vec3 v)
{
    if (root->isLeaf())
    {
        if (root->getLeaf())
        {
            return root->getLeaf()->equals(v);
        }
        else
        {
            return 0;
        }
    }
    std::vector<Vec3 *> onPlane = root->getOnPlane();
    for (int i = 0; i < onPlane.size(); i++)
    {
        if (onPlane[i]->equals(v))
            return 1;
    }
    return doesHave(root->getLeft(), v) || doesHave(root->getRight(), v);
}

Vec3* TreeNode::searchNearest(Vec3 v)
{
    if (isLeaf())
    {
        return getLeaf();
    }
    Vec3 *P = getPos();
    Vec3 *N = getNormal();
    std::vector<Vec3 *> vsOnPlane = getOnPlane();
    Vec3 *res = vsOnPlane[0];
    double distance = res->sub(v).magSq();
    for (int i = 1; i < vsOnPlane.size(); i++)
    {
        if (vsOnPlane[i]->sub(v).magSq() < distance)
        {
            res = vsOnPlane[i];
            distance = res->sub(v).magSq();
        }
    }
    double x = v.sub(*P).dot(*N);
    Vec3 *vInNextVox = nullptr;
    if (x < 0)
    {
        vInNextVox = getLeft()->searchNearest(v);
    }
    else if (x > 0)
    {
        vInNextVox = getRight()->searchNearest(v);
    }
    if (vInNextVox != nullptr && vInNextVox->sub(v).magSq() < distance)
        res = vInNextVox;
    return res;
}

void TreeNode::search(Vec3 v,int n,std::vector<Vec3*> *li){
    if (isLeaf())
    {
        li->push_back(getLeaf());
        return;
    }
    Vec3 *P = getPos();
    Vec3 *N = getNormal();
    double x = v.sub(*P).dot(*N);
    if(x<0){
        getLeft()->search(v,n,li);
        getRight()->search(v,n,li);
    }else if(x>0){
        getRight()->search(v,n,li);
        getLeft()->search(v,n,li);
    }
    std::vector<Vec3 *> vsOnPlane = getOnPlane();
    for (int i = 0; i < vsOnPlane.size(); i++)
    {
        if(li->size()>=n)return;
        li->push_back(vsOnPlane[i]);
    }
    return;
}


#endif
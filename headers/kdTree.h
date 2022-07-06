#ifndef TD_TREE_H
#define TD_TREE_H
#include <stdlib.h>
#include "Vector.h"
#include <vector>
#include <algorithm>
#include "shapes/Plane.h"
#include "shapes/Triangle.h"

typedef struct KdTreeNode
{
    KdTreeNode *c1, *c2;
    Vec3 *split;
    int isLeaf;
    Vec3 *leaf;
};

class KdTree
{
private:
    KdTreeNode *root;

public:
    void build(int N, Triangle **tris, Vec3 start, Vec3 end);
    void build(std::vector<Triangle *> tris_vector, Vec3 start, Vec3 end, KdTreeNode *node, char d);
    void print()
    {
        printRec(root);
    }
    void printRec(KdTreeNode *node)
    {
        printf("1");
        if(node==nullptr)return;
        if (node->isLeaf==1)
        {
            (node->leaf)->print();
        }
        else if(node->isLeaf<0){
            printf("Error\n");
        }else if(node->isLeaf==0)
        {
            printRec(node->c1);
            printRec(node->c2);
        }
    }
};

Vec3 triG(Triangle *tri)
{
    return (tri->getV1()).add(tri->getV2()).add(tri->getV3());
}

std::vector<Triangle *> sortByXYZ(std::vector<Triangle *> tris, char d)
{
    switch (d)
    {
    case 'x':
        std::sort(tris.begin(), tris.end(), [](Triangle *a, Triangle *b)
                  { return triG(a).getX() < triG(b).getX(); });
        break;
    case 'y':
        std::sort(tris.begin(), tris.end(), [](Triangle *a, Triangle *b)
                  { return triG(a).getY() < triG(b).getY(); });
        break;
    case 'z':
        std::sort(tris.begin(), tris.end(), [](Triangle *a, Triangle *b)
                  { return triG(a).getZ() < triG(b).getZ(); });
        break;
    default:
        break;
    }
    return tris;
}

int isBetween(double a, double x, double b)
{
    return a <= x && x < b;
}

int isIn(Vec3 p, Vec3 start, Vec3 end)
{
    return isBetween(start.getX(), p.getX(), end.getX()) &&
           isBetween(start.getY(), p.getY(), end.getY()) &&
           isBetween(start.getZ(), p.getZ(), end.getZ());
}

/*
int isIn(Triangle *t,Vec3 start,Vec3 end)
{
    return isIn(t->getV1(), start,end) || isIn(t->getV2(), start,end) || isIn(t->getV3(), start,end);
}
*/

std::vector<Triangle *> areaFilter(std::vector<Triangle *> tris, Vec3 start, Vec3 end)
{
    std::vector<Triangle *> result;
    for (int i = 0; i < tris.size(); i++)
    {
        if (isIn(triG(tris[i]), start, end))
            result.push_back(tris[i]);
    }
    return result;
}

void KdTree::build(int N, Triangle **tris, Vec3 start, Vec3 end)
{
    std::vector<Triangle *> tris_vector;
    for (int i = 0; i < N; i++)
    {
        Triangle *t = tris[i];
        tris_vector.push_back(tris[i]);
    }
    build(tris_vector, start, end, root, 'x');
}

void KdTree::build(std::vector<Triangle *> tris_vector, Vec3 start, Vec3 end, KdTreeNode *node, char d)
{
    node = new KdTreeNode();
    node->isLeaf = 0;
    if (tris_vector.size() == 0){
        node->isLeaf = -1;
        return;
    }
    if (tris_vector.size() == 1)
    {
        node->isLeaf = 1;
        node->leaf = new Vec3(triG(tris_vector[0]));
        return;
    }
    tris_vector = sortByXYZ(tris_vector, d);
    Vec3 med = triG(tris_vector[tris_vector.size() / 2]);
    node->split = new Vec3(med);
    if (tris_vector.size() == 2)
    {
        node->c1 = new KdTreeNode();
        node->c1->leaf = new Vec3(triG(tris_vector[0]));
        node->c2 = new KdTreeNode();
        node->c2->leaf = new Vec3(triG(tris_vector[1]));
        return;
    }
    if (d == 'z')
    {
        d = 'x';
    }
    else
    {
        d++;
    }
    build(areaFilter(tris_vector, start, med), start, med, node->c1, d);
    build(areaFilter(tris_vector, med, end), med, end, node->c2, d);
}

#endif
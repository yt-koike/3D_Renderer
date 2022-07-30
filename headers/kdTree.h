#ifndef TD_TREE_H
#define TD_TREE_H
#include <stdlib.h>
#include "Vector.h"
#include <vector>
#include <algorithm>
#include "shapes/Plane.h"
#include "shapes/Triangle.h"

const char firstDimension = 'z';

Vec3 triG(Triangle *tri)
{
    return (tri->getV1()).add(tri->getV2()).add(tri->getV3());
}

void sortTreeRange(unsigned int begin,
                   unsigned int end,
                   Triangle **tree, char d)
{
    for(unsigned int i=begin;i<end){
    for(unsigned int j=i;j<=end){
        int swapFlag;
        switch (d)
    {
    case 'x':
    swapFlag=triG(tree[i]).getX()>triG(tree[j]).getX();
        break;
    case 'y':
    swapFlag=triG(tree[i]).getY()>triG(tree[j]).getY();
        break;
    case 'z':
    swapFlag=triG(tree[i]).getZ()>triG(tree[j]).getZ();
        break;
    }
    if(swapFlag){
        Triangle* z = tree[i];
        tree[i] = tree[j];
        tree[j] = z;
    }
        }
            }
}

void sortTreeRec(unsigned int begin,
                 unsigned int end,
                 Triangle **tree, char d)
{
    unsigned int length = end - begin;
    if (length <= 1)
        return;
    sortTreeRange(begin,end,tree,d);
    d = (d=='z')?'x':d+1;

    sortTreeRec(begin, begin + length / 2, tree, d);
    sortTreeRec(begin + length / 2 + 1 + (length % 2), end, tree, d);
    return;
}

Triangle **makeKdTree(int triN, Triangle **tris)
{
    Triangle **tree = new Triangle *[triN];
    for(int i=0;i<triN;i++)tree[i]=tris[i];
    sortTreeRec(0,triN-1,tree,d);
    return tree;
}

void searchRec(unsigned int begin,
               unsigned int end,
               Triangle **tree, char d, int queryN, Triangle **result)
{
    return;
}

void searchKdTree(Triangle **tree, Vec3 p, int queryN)
{
    searchRec(tree->begin(), tree->end() - 1, tree, p, firstDimension, queryN, &result);
    return;
}

#endif
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

void sortTreeRec(std::vector<Triangle *, std::allocator<Triangle *>>::iterator begin,
                 std::vector<Triangle *, std::allocator<Triangle *>>::iterator end,
                 std::vector<Triangle *> *tree, char d)
{
    unsigned int length = std::distance(begin, end);
    if (length <= 1)
        return;
    switch (d)
    {
    case 'x':
        std::sort(begin, end, [](Triangle *a, Triangle *b)
                  { return triG(a).getX() < triG(b).getX(); });
        break;
    case 'y':
        std::sort(begin, end, [](Triangle *a, Triangle *b)
                  { return triG(a).getY() < triG(b).getY(); });
        break;
    case 'z':
        std::sort(begin, end, [](Triangle *a, Triangle *b)
                  { return triG(a).getZ() < triG(b).getZ(); });
        break;
    }
    if (d == 'z')
    {
        d = 'x';
    }
    else
    {
        d++;
    }

    sortTreeRec(begin, begin + length / 2, tree, d);
    sortTreeRec(begin + length / 2 + 1 + (length % 2), end, tree, d);
    return;
}

std::vector<Triangle *> makeKdTree(int triN, Triangle **tris)
{
    std::vector<Triangle *> tree;
    for (int i = 0; i < triN; i++)
    {
        tree.push_back(tris[i]);
    }
    sortTreeRec(tree.begin(), tree.end(), &tree, firstDimension);
    return tree;
}
void searchRec(std::vector<Triangle *, std::allocator<Triangle *>>::iterator begin,
                    std::vector<Triangle *, std::allocator<Triangle *>>::iterator end,
                    std::vector<Triangle *> *tree, Vec3 p, char d,int queryN, std::vector<Triangle*> *result)
{
    if(result->size()>=queryN)return;
    int length = std::distance(begin, end);
    int beginIdx = std::distance(tree->begin(), begin);
    if (length == 0)
    {
        result->push_back(tree->at(beginIdx));
        return;
    }
    int middleIdx = beginIdx + length / 2 + (length % 2);
    char way;
    switch (d)
    {
    case 'x':
        way = (p.getX() <= triG(tree->at(middleIdx)).getX()) ? 'l' : 'r';
        break;
    case 'y':
        way = (p.getY() <= triG(tree->at(middleIdx)).getY()) ? 'l' : 'r';
        break;
    case 'z':
        way = (p.getZ() <= triG(tree->at(middleIdx)).getZ()) ? 'l' : 'r';
        break;
    }
    if (d == 'z')
    {
        d = 'x';
    }
    else
    {
        d++;
    }
    if (way == 'l')
    {
        searchRec(begin, begin + length / 2, tree, p, d,queryN,result);
        searchRec(begin + length / 2 + 1, end, tree, p, d,queryN,result);
    }
    else if (way == 'r')
    {
        searchRec(begin + length / 2 + 1, end, tree, p, d,queryN,result);
        searchRec(begin, begin + length / 2, tree, p, d,queryN,result);
    }
    return;
}

std::vector<Triangle *> searchKdTree(std::vector<Triangle *> *tree, Vec3 p,int queryN)
{
    std::vector<Triangle *> result;
    searchRec(tree->begin(), tree->end()-1, tree, p, firstDimension,queryN,&result);
    return result;
}

#endif
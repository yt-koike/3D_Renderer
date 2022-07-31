#ifndef TD_TREE_H
#define TD_TREE_H
#include "Vector.h"
#include "shapes/Plane.h"
#include "shapes/Triangle.h"

class KdTree
{
private:
    const char firstDimension = 'z';
    unsigned int nodeN;
    Triangle **tree;
    Vec3 triG(Triangle *tri)
    {
        return (tri->getV1()).add(tri->getV2()).add(tri->getV3());
    }
    int searchNearestRec(unsigned int begin, unsigned int end, Vec3 p, char d, int queryN, Triangle **result)
    {
        if(end>=nodeN-1)end=nodeN-1;
        if (end < begin)
            return 0;
        if (end == begin)
        {
            result[0] = tree[begin];
            return 1;
        }
        char way;
        unsigned int middleIdx = begin + (end - begin + 1) / 2;
        switch (d)
        {
        case 'x':
            way = (p.getX() <= triG(tree[middleIdx]).getX()) ? 'l' : 'r';
            break;
        case 'y':
            way = (p.getY() <= triG(tree[middleIdx]).getY()) ? 'l' : 'r';
            break;
        case 'z':
            way = (p.getZ() <= triG(tree[middleIdx]).getZ()) ? 'l' : 'r';
            break;
        }
        d = (d == 'z') ? 'x' : d + 1;
        unsigned int offset;
        if (way == 'l')
        {
            offset = searchNearestRec(begin, middleIdx - 1, p, d, queryN, result);
            if (offset >= queryN)
                return offset;
            result[offset] = tree[middleIdx];
            offset++;
            if (offset >= queryN)
                return offset;
            offset += searchNearestRec(middleIdx + 1, end, p, d, queryN-offset-1, result + offset);
        }
        else if (way == 'r')
        {
            offset = searchNearestRec(middleIdx + 1, end, p, d, queryN, result);
            if (offset >= queryN)
                return offset;
            result[offset] = tree[middleIdx];
            offset++;
            if (offset >= queryN)
                return offset;
            offset += searchNearestRec(begin, middleIdx - 1, p, d, queryN-offset-1, result + offset);
        }
        return offset;
    }

    void sortTreeRange(unsigned int begin, unsigned int end, Triangle **tree, char d)
    {
        int swapFlag;
        for (unsigned int i = begin; i < end; i++)
        {
            for (unsigned int j = i + 1; j <= end; j++)
            {
                switch (d)
                {
                case 'x':
                    swapFlag = triG(tree[i]).getX() > triG(tree[j]).getX();
                    break;
                case 'y':
                    swapFlag = triG(tree[i]).getY() > triG(tree[j]).getY();
                    break;
                case 'z':
                    swapFlag = triG(tree[i]).getZ() > triG(tree[j]).getZ();
                    break;
                }
                if (swapFlag)
                {
                    Triangle *z = tree[i];
                    tree[i] = tree[j];
                    tree[j] = z;
                }
            }
        }
    }

    void sortTreeRec(unsigned int begin, unsigned int end, Triangle **tree, char d)
    {
        if (end <= begin)
            return;
        sortTreeRange(begin, end, tree, d);
        d = (d == 'z') ? 'x' : d + 1;
        unsigned int middleIdx = begin + (end - begin + 1) / 2;
        sortTreeRec(begin, middleIdx - 1, tree, d);
        sortTreeRec(middleIdx + 1, end, tree, d);
        return;
    }
    void printTreeRec(unsigned int begin, unsigned int end)
    {
        if (end <= begin)
        {
            triG(tree[begin]).print();
            return;
        }
        unsigned int middleIdx = begin + (end - begin + 1) / 2;
        if (middleIdx < 0)
            middleIdx = 0;
        if (middleIdx >= nodeN - 1)
            middleIdx = nodeN - 2;
        printTreeRec(begin, middleIdx - 1);
        triG(tree[middleIdx]).print();
        printTreeRec(middleIdx + 1, end);
    }

public:
    KdTree(int triN, Triangle **tris)
    {
        nodeN = triN;
        tree = new Triangle *[triN];
        for (int i = 0; i < triN; i++)
            tree[i] = tris[i];
        sortTreeRec(0, triN - 1, tree, firstDimension);
    }
    unsigned int getNodeN(){
        return nodeN;
    }
    void printTree()
    {
        printTreeRec(0, nodeN - 1);
    }
    void printTreeFlat()
    {
        for (unsigned int i = 0; i < nodeN; i++)
        {
            triG(tree[i]).print();
        }
    }
    void searchNearest(Vec3 p, unsigned int queryN, Triangle **result)
    {
        if(queryN>nodeN)queryN=nodeN;
        searchNearestRec(0, nodeN - 1, p, firstDimension, queryN, result);
    }
};

#endif
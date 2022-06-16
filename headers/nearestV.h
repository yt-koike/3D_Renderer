#ifndef NEAREST_H
#define NEAREST_H
#include <vector>
#include <unordered_map>
#include"shapes/Triangle.h"

void makeSortedList(Vec3 origin,int n,Triangle* ts){
    int compare(const void* a,const void* b){
        Triangle* t1 = (Triangle*) a;
        Triangle* t2 = (Triangle*) b;
        
    };
    qsort(ts,n,sizeof(Triangle),compare);
}

#endif
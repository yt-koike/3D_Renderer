#ifndef STL_H
#define STL_H
#include <iostream>
#include <fstream>
#include <string.h>
#include "shapes/ShapeSuite.h"

Polygon3D STLBinLoad(const char *filename)
{
    // Load binary STL file
    FILE *fp = fopen(filename, "r");
    if(!fp){
        perror("File not found");
    }
    float x, y, z;
    fseek(fp, 80, SEEK_SET);
    unsigned int n;
    size_t nil; // to avoid warnings when compiled with -O3
    nil = fread(&n, sizeof(unsigned int), 1, fp);
    printf("File %s has %d triangles.\n", filename,n);
    Polygon3D poly(n);
    Vec3* vertexes[3];
    while (n--)
    {
        nil = fread(&x, sizeof(float), 1, fp);
        nil = fread(&y, sizeof(float), 1, fp);
        nil = fread(&z, sizeof(float), 1, fp);
        Vec3 normalV(x,y,z);
        for (int i = 0; i < 3; i++)
        {
            nil = fread(&x, sizeof(float), 1, fp);
            nil = fread(&y, sizeof(float), 1, fp);
            nil = fread(&z, sizeof(float), 1, fp);
            vertexes[i] = new Vec3(x, y, z);
        }
        Triangle* tri = new Triangle(*vertexes[0],*vertexes[1],*vertexes[2]);
        poly.addTriangle(tri);
        fseek(fp, 2, SEEK_CUR);
    }
    fclose(fp);
    poly.generateBoundary();
    poly.buildKdTree();
    return poly;
}

Polygon3D STLload(const char *filename)
{
    // Load Ascii STL file
    Polygon3D poly(1000);
    std::ifstream ifs;
    ifs.open(filename, std::ios::in);
    if (ifs.fail())
    {
        std::cerr << "File read error!";
        return poly;
    }
    const int buf_size = 100;
    char str[buf_size];
    while (ifs.getline(str, buf_size))
    {
        if (!strcmp(str, "  outer loop"))
        {
            for (int i = 0; i < 3; i++)
            {
                ifs.getline(str, buf_size);
                double x, y, z;
                sscanf(str + 10, "%le%le%le", &x, &y, &z);
                std::cout << str + 10 << std::endl;
                printf("%f,%f,%f\n", x, y, z);
            }
        }
    }
    ifs.close();
    poly.generateBoundary();
    poly.buildKdTree();
    return poly;
}

#endif
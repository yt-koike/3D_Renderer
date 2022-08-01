#ifndef COLOR_IMAGE_H
#define COLOR_IMAGE_H

class Array2D
{
private:
    int width, height;
    int *ary;

public:
    Array2D(){};
    Array2D(int width, int height)
    {
        this->width = width;
        this->height = height;
    }
    void init(int w, int h);
    int get(int x, int y);
    void set(int x, int y, int n);
    int getWidth() { return width; }
    int getHeight() { return height; }
    void reset();
};

void Array2D::init(int w, int h)
{
    this->width = w;
    this->height = h;
    ary = new int[w * h];
    reset();
}

int Array2D::get(int x, int y)
{
    return ary[width * y + x];
}

void Array2D::set(int x, int y, int n)
{
    ary[width * y + x] = n;
}

void Array2D::reset()
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            set(x, y, 0);
        }
    }
}

class ColorImage
{
private:
    int width, height;
    Array2D r, g, b;

public:
    ColorImage(){};
    ColorImage(int width, int height){init(width,height);}
    void init(int width, int height);
    void set(int x, int y, int r, int g, int b)
    {
        this->r.set(x, y, r);
        this->g.set(x, y, g);
        this->b.set(x, y, b);
    }
    int getR(int x, int y) { return r.get(x, y); }
    int getG(int x, int y) { return g.get(x, y); }
    int getB(int x, int y) { return b.get(x, y); }
    int getWidth() { return width; }
    int getHeight() { return height; }
};

void ColorImage::init(int width, int height)
{
    this->width = width;
    this->height = height;
    this->r.init(width, height);
    this->g.init(width, height);
    this->b.init(width, height);
}

#endif
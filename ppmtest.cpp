
void PPMtest(){
  PPM ppmtest(256,256,255);
  ppmtest.readPPM("test.ppm");
  Pixel a = ppmtest.getPixel(0,0);
  printf("%d",a.r);
  /*
  for(int i=0;i<100;i++){
    for(int j=0;j<128;j++){
    ppmtest.setPixel(j,i,255,0,0);
    }
  }*/
  ppmtest.writePPM("test2.ppm");
}

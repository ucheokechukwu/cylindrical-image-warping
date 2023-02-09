#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

template<typename T, typename U>
U interpolate_pixel_4(const Mat& src, Point2f pt)
{
      // Binomial Interpolation of BGRA-channels
      const int x = (int)pt.x;
      const int y = (int)pt.y;
      
      const int x0 = borderInterpolate(x,     src.cols, BORDER_REFLECT_101);
      const int x1 = borderInterpolate(x + 1, src.cols, BORDER_REFLECT_101);
      const int y0 = borderInterpolate(y,     src.rows, BORDER_REFLECT_101);
      const int y1 = borderInterpolate(y + 1, src.rows, BORDER_REFLECT_101);
      
      const float d = pt.x - (float)x;
      const float c = pt.y - (float)y;
      
      const float one_minus_d = 1.f - d;
      const float one_minus_c = 1.f - c;
      
      const U y0_x0 = src.at<U>(y0, x0);
      const U y1_x0 = src.at<U>(y1, x0);
      const U y0_x1 = src.at<U>(y0, x1);
      const U y1_x1 = src.at<U>(y1, x1);
      
      const T r = (T)cvRound((one_minus_d * (float)(y0_x0[0]) + d * (float)(y0_x1[0])) * one_minus_c +
                             (one_minus_d * (float)(y1_x0[0]) + d * (float)(y1_x1[0])) * c);
      const T g = (T)cvRound((one_minus_d * (float)(y0_x0[1]) + d * (float)(y0_x1[1])) * one_minus_c +
                             (one_minus_d * (float)(y1_x0[1]) + d * (float)(y1_x1[1])) * c);
      const T b = (T)cvRound((one_minus_d * (float)(y0_x0[2]) + d * (float)(y0_x1[2])) * one_minus_c +
                             (one_minus_d * (float)(y1_x0[2]) + d * (float)(y1_x1[2])) * c);
      const T a = (T)cvRound((one_minus_d * (float)(y0_x0[3]) + d * (float)(y0_x1[3])) * one_minus_c +
                             (one_minus_d * (float)(y1_x0[3]) + d * (float)(y1_x1[3])) * c);
      return U(r,g,b,a);
}

template<typename T, typename U>
Mat* project_cylinder(Mat& src)
{
      //Pixel Remapping
      const int height = src.rows;
      const int width  = src.cols;
      const float r  = (float)width * 0.5f;
      const float xf = (float)width * 0.5f;
      const float yf = (float)height * 0.5f;
      const float zf = hypot((float)width, (float)height);
      
      const float zfr = cos(1);
      const float xfr = sin(1);
      const int dst_height = (int)ceil((float)height / xfr);
      
      Mat *dst = new Mat(Mat::zeros(dst_height, width, src.type()));
      
      for (int i = 0; i < width; i++)
      {
            const float zb = r * (cos((r - (float)i) / r) - zfr);
            const float z_ratio = (zf - zb) / zf;
            
            const float i_dst = r * asin(xfr * ((float)i - xf) * z_ratio / r) + r;
            for (int j = 0; j < dst_height; j++)
            {
                  const float j_dst = yf + (xfr * (float)j - yf) * z_ratio;
                  //keep within src image borders
                  if (i_dst >= 0.0f && i_dst <= (float)width && j_dst >= 0.0f && j_dst <= (float)height)
                  {
                        dst->at<U>(j, i) = interpolate_pixel_4<T, U>(src, Point2f(i_dst, j_dst));
                  }
            }
      }
      return dst;
}

int main(int argc, char *argv[])
{
      
      cout << "Warper program:" << endl;
      cout << "===============" << endl;
      cout << "Argument 1 - Source file" << endl << "Argument 2 - Output file." << endl;
      cout << "Will overwrite existing file with the same output filename." << endl;
      cout << "Only suitable for 4-channel (RGBA) and some 3-channel (RGB) images." << endl;
      Mat src = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
      Mat* (*project)(Mat&);
      switch (src.type()) {
                  //check for image type and 3-channel to 4-channel conversion
            case CV_8UC3:
            {
                  Mat newSrc = Mat(src.rows,src.cols,CV_8UC4);
                  int from_to[] = { 0,0, 1,1, 2,2 };
                  mixChannels(&src,1,&newSrc,1,from_to,3);
                  src=newSrc.clone();
                  newSrc.release();
                  project = project_cylinder<ushort, Vec4b>;
            }
                  break;
            case CV_16UC4: 
                  project = project_cylinder<ushort, Vec4w>;
                  break;
            default:
                  cerr << "Unexpected image type: " << src.type() << endl;
                  exit(1);
      }
      copyMakeBorder(src, src, 2, 2, 0, 0, BORDER_CONSTANT, Scalar::all(0)); //add border
      
      //store dst
      Mat *dst = project(src);
      imwrite(argv[2], *dst);
      
      //display images on windows
      namedWindow("Output Image", WINDOW_AUTOSIZE );
      imshow("Output Image", *dst);
      namedWindow("Input Image", WINDOW_AUTOSIZE );
      moveWindow("Input Image", 300,300);
      imshow("Input Image", src);
      waitKey(0);
      delete dst;
      return 0;
}

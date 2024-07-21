#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
int main()
{
    Mat img = imread("./girls.png");
    imshow("beauty", img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}

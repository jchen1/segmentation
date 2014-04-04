#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
  cv::Mat src = cv::imread("/Users/Jeff/pic.png");
    if(src.empty()){
        std::cerr<<"can't read the image"<<std::endl;
        return -1;
    }

    //step 1 : map the src to the samples
    cv::Mat samples(src.total(), 3, CV_32F);
    auto samples_ptr = samples.ptr<float>(0);
    for( int row = 0; row != src.rows; ++row){
        auto src_begin = src.ptr<uchar>(row);
        auto src_end = src_begin + src.cols * src.channels();
        //auto samples_ptr = samples.ptr<float>(row * src.cols);
        while(src_begin != src_end){
            samples_ptr[0] = src_begin[0];
            samples_ptr[1] = src_begin[1];
            samples_ptr[2] = src_begin[2];
            samples_ptr += 3; src_begin +=3;
        }        
    }

    //step 2 : apply kmeans to find labels and centers
    int clusterCount = 10;
    cv::Mat labels;
    int attempts = 5;
    cv::Mat centers;
    cv::kmeans(samples, clusterCount, labels,
               cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
                                10, 0.01),
               attempts, cv::KMEANS_PP_CENTERS, centers);

    //step 3 : map the centers to the output
    cv::Mat new_image(src.size(), src.type());
    for( int row = 0; row != src.rows; ++row){
        auto new_image_begin = new_image.ptr<uchar>(row);
        auto new_image_end = new_image_begin + new_image.cols * 3;
        auto labels_ptr = labels.ptr<int>(row * src.cols);

        while(new_image_begin != new_image_end){
            int const cluster_idx = *labels_ptr;
            auto centers_ptr = centers.ptr<float>(cluster_idx);
            new_image_begin[0] = centers_ptr[0];
            new_image_begin[1] = centers_ptr[1];
            new_image_begin[2] = centers_ptr[2];
            new_image_begin += 3; ++labels_ptr;
        }
    }

    cv::Mat binary, copy = src.clone();
    cv::GaussianBlur(copy, copy, cv::Size(5, 5), 0, 0);    
    cv::Canny(copy, binary, 30, 90);
    cv::vector<cv::Vec2f> lines;
    cv::HoughLines(binary, lines, 1, CV_PI/180, 100, 0, 0 );

    for( size_t i = 0; i < lines.size(); i++ )
    {
      float rho = lines[i][0], theta = lines[i][1];
      cv::Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 1000*(-b));
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b));
      pt2.y = cvRound(y0 - 1000*(a));
      cv::line( src, pt1, pt2, cv::Scalar(255,0,255), 1, CV_AA);
    }

    cv::imshow("original", src);
    cv::imshow("binary", binary);
    cv::imshow( "clustered image", new_image );

    cv::waitKey();

    return 0;
}
#ifndef FEATURE_EXACTION_LIME_H // 防止头文件被重复包含
#define FEATURE_EXACTION_LIME_H // 定义宏，表示已包含此头文件

#include <opencv2/core/core.hpp> // 包含 OpenCV 核心库
#include <opencv2/highgui/highgui.hpp> // 包含 OpenCV 高级 GUI 库
#include <iostream> // 包含标准输入输出流库
#include <opencv2/imgproc/imgproc.hpp> // 包含 OpenCV 图像处理库

namespace LIME // 定义一个名为 feature 的命名空间
{
    class lime 
    {
        public: 
        cv::Mat img_norm; //输入图像归一化
        cv::Mat R;
        cv::Mat out_lime; //增强后的图像
        cv::Mat dv;
        cv::Mat dh;
        cv::Mat T_hat;    //初始化的光照图
        cv::Mat W; 	      //权重矩阵
        cv::Mat veCDD;
        int channel; 	  //存储图像通道数
        int row; 		  //图像的行数
        int col; 		  //图像的列数
        float alpha=1;
        float rho =2;
        float gamma = 0.7;
        float epsilon;    //迭代系数
		float threshold;  //迭代收敛时间阈值

    	public: 
        lime(cv::Mat src); 
        cv::Mat Mat2Vec(cv::Mat mat);                         //矩阵向量化
        cv::Mat reshape1D(cv::Mat mat);						  //按行压缩矩阵到一维
        cv::Mat getReal(cv::Mat mat);						  //获取矩阵实部
        cv::Mat derivative(cv::Mat matrix);					  //求解矩阵导数
        cv::Mat solveT(cv::Mat G, cv::Mat Z, float u);		  //求解子问题T
		cv::Mat solveG(cv::Mat T,cv::Mat Z,float u,cv::Mat W);//求解子问题G
        cv::Mat solveZ(cv::Mat T,cv::Mat G,cv::Mat Z,float u);//求解子问题Z
        cv::Mat Dev(int n, int k);							  //求解矩阵一阶导数
        cv::Mat getMax(const cv::Mat &bgr);					  //获取色彩通道最大值
        cv::Mat optIllumMap();								  //获取优化后光照图
		cv::Mat enhance(cv::Mat &src);						  //图像增强					

        static inline float comp(float& a, float& b, float& c) // 声明一个静态内联函数，用于比较三个浮点数的大小，并返回最大值
        {
            return fmax(a, fmax(b, c));
        }

        void weightStrategy();
        void _init_IllumMap(cv::Mat src);
        void Illum_filter(cv::Mat& img_in, cv::Mat& img_out); //滤波、GAMMA矫正
        void Illumination(cv::Mat& src, cv::Mat& out); 		  //求解每个像素的光照强度
        void fft2(const cv::Mat& input, cv::Mat& output, int opt);
        void fft2_neon(const cv::Mat& input, cv::Mat& output, int opt);
        int ReverseBin(int a, int n);
        float solveU(float u);								  //求解子问题u
        float Frobenius(cv::Mat mat);
    };
} // 结束命名空间 feature

#endif //FEATURE_EXACTION_LIME_H // 结束头文件保护

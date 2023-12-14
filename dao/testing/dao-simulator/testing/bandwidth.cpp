#include <cuda_runtime.h>
#include <iostream>
#include <vector>

std::pair<double, double> lr(
    const std::vector<double>& x,
    const std::vector<double>& y) {
    double avex = 0.0;
    double avey = 0.0;
    double avexy = 0.0;
    double avexx = 0.0;
    for (int i = 0; i < x.size(); i++) {
        avex += x[i];
        avey += y[i];
    }
    avex /= x.size();
    avey /= y.size();
    for (int i = 0; i < x.size(); i++) {
        avexy += (x[i]-avex) * (y[i] - avey);
        avexx += (x[i]-avex) * (x[i]-avex);
    }
    double k = avexy / avexx;
    double b = avey - k * avex;
    double ssr = 0, sst = 0;
    for (int i = 0; i < x.size(); i++) {
        ssr += (k * x[i] + b - y[i]) * (k * x[i] + b - y[i]);
        sst += (y[i] - avey) * (y[i] - avey);
    }
    double r2 = 1 - ssr / sst;
    return {k, r2};
}

void test(int device_id) {
    cudaSetDevice(device_id);
    std::vector<double> sizes;
    std::vector<double> times_h2d; 
    std::vector<double> times_d2h;
    for (int i = 1024; i < 1024 * 1024 * 1024; i *= 2) {
        void* ptr, *ptr2;
        cudaMalloc(&ptr, i);
        cudaMalloc(&ptr2, i);
        {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventRecord(start);
            cudaMemcpy(ptr, ptr2, i, cudaMemcpyHostToDevice);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            times_h2d.push_back(ms);
        }

        {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventRecord(start);
            cudaMemcpy(ptr2, ptr, i, cudaMemcpyDeviceToHost);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            times_d2h.push_back(ms);
        }
        
        sizes.push_back(i);
        cudaFree(ptr);
        cudaFree(ptr2);
    }
    // do linear regression on the sizes and time 
    // to get the bandwidth
    auto [k, r2] = lr(times_h2d, sizes);
    auto [k2, r22] = lr(times_d2h, sizes);
    std::cout << "device: " << device_id << std::endl;
    std::cout << "h2d bandwidth: " << k << " r2: " << r2 << std::endl;
    std::cout << "d2h bandwidth: " << k2 << " r2: " << r22 << std::endl;
}

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; i++) {
        test(i);
    }
    return 0;
}
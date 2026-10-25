#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    // 假设你在 AutoDL 租了两张卡
    int num_gpus = 2;
    
    // 1. 准备句柄数组和设备列表
    // comms 数组用来存放 NCCL 为每张卡创建的通信域句柄
    std::vector<ncclComm_t> comms(num_gpus);
    // devs 数组告诉 NCCL：我要把哪几张卡编入这个通信组？
    std::vector<int> devs = {0, 1}; // 使用 GPU 0 和 GPU 1

    // ==========================================
    // 核心调用：一键初始化单进程多卡通信域
    // ==========================================
    // 参数：(句柄数组指针, 卡的总数, 设备ID数组)
    ncclResult_t res = ncclCommInitAll(comms.data(), num_gpus, devs.data());
    if (res != ncclSuccess) {
        std::cerr << "NCCL 初始化失败: " << ncclGetErrorString(res) << "\n";
        return -1;
    }
    std::cout << "NCCL 单进程双卡初始化成功！\n";

    // 2. 为每张卡准备资源 (Stream 和 显存)
    std::vector<cudaStream_t> streams(num_gpus);
    std::vector<float*> sendbuff(num_gpus);
    std::vector<float*> recvbuff(num_gpus);
    int data_size = 1024; // 每张卡 1024 个 float 数据

    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(devs[i]); // 切视⻆
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&sendbuff[i], data_size * sizeof(float));
        cudaMalloc(&recvbuff[i], data_size * sizeof(float));
        // (此处省略用 cudaMemcpy 或 Kernel 给 sendbuff 初始化的代码)
    }

    // 3. 启动通信操作 (必须用 Group 包裹)
    ncclGroupStart();
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(devs[i]);
        // 把双卡的 sendbuff 里的数据按位求和，结果放入 recvbuff
        ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], data_size, 
                      ncclFloat, ncclSum, comms[i], streams[i]);
    }
    ncclGroupEnd(); // 这里底层才真正开始调度传输

    // 4. 等待所有卡的数据搬运完成
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(devs[i]);
        cudaStreamSynchronize(streams[i]);
    }
    std::cout << "All-Reduce 通信完成！\n";

    // 5. 销毁资源
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(devs[i]);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
        cudaStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]); // 销毁 NCCL 句柄
    }

    return 0;
}
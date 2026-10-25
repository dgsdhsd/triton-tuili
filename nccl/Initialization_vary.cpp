#include <nccl.h>

void init_process(int my_rank, int total_ranks) {
    ncclUniqueId id;
    
    // ==========================================
    // 阶段 1：生成并广播 ID
    // ==========================================
    if (my_rank == 0) {
        ncclGetUniqueId(&id);
        // 主进程将 128 字节的 id 广播给其他进程 (伪代码)
        broadcast_to_others(&id, 128); 
    } else {
        // 其他进程阻塞等待接收主进程的 id (伪代码)
        receive_from_rank_0(&id, 128);
    }

    // ==========================================
    // 阶段 2：初始化通信域
    // ==========================================
    ncclComm_t comm;
    // 强制绑定当前进程对应的 GPU
    cudaSetDevice(my_rank); 
    
    // 所有进程在这里阻塞，直到底层拓扑建立完毕
    ncclCommInitRank(&comm, total_ranks, id, my_rank);
    
    printf("Rank %d 初始化成功，准备起飞！\n", my_rank);

    // ==========================================
    // 阶段 3：执行核心通信 (略)
    // ncclAllReduce(..., comm, ...);
    // ==========================================

    // ==========================================
    // 阶段 4：销毁
    // ==========================================
    ncclCommDestroy(comm);
}
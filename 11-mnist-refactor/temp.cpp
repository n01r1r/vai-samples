


// HxWxC calls: [H][W][C] -> [H][W][C][K*K], where K is odd
__global__ void im2col(
    const float* input, float* output,
    int H, int W, int C, int K)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;  // input row (0 <= h < H)
    int w = blockIdx.y * blockDim.y + threadIdx.y;  // input col (0 <= w < W)
    int c = blockIdx.z * blockDim.z + threadIdx.z;  // input col (0 <= c < C)

    int offset = ((h * W + w) * C + c) * K * K;

    int j = 0;
    for (int ki = 0; ki < K; ++ki)
    {
        int h_ = h + ki - K / 2;  
        for (int kj = 0; kj < K; ++kj, ++j)
        {
            int w_ = w + kj - K / 2;  
            float value = 0.0f;
            if (0 <= h_ && h_ < H && 0 <= w_ && w_ < W)
                value = input[(h_ * W + w_) * C + c];
            output[offset + j] = value; 
        }
    }
}

// (H*W*C)x(K*K) calls: [H][W][C] -> [H][W][C][K*K], where K is odd
__global__ void im2col_noForLoop(
    const float* input, float* output,
    int H, int W, int C, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // output row (0 <= i < H*W*C)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // output col (0 <= j < K*K)
    
    int h = i / (W * C);        // input row
    int w = (i % (W * C)) / C;  // input col
    int c = (i % (W * C)) % C;  // input channel

    int h_ = h + j / K - K / 2;  // kernel row offset
    int w_ = w + j % K - K / 2;  // kernel col offset

    float value = 0.0f;
    if (0 <= h_ && h_ < H && 0 <= w_ && w_ < W)
        value = input[(h_ * W + w_) * C + c]; 
    output[i * K * K + j] = value;  
}

// (H*W)x(C*K*K) calls: [H][W][C] -> [H][W][C][K*K], where K is odd
__global__ void im2col_noForLoop2(
    const float* input, float* output,
    int H, int W, int C, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // output row (0 <= i < H*W)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // output col (0 <= j < C*K*K)
    int KK = K * K;
    
    if (i >= H * W || j >= C * KK) return;  
    
    int h = i / W;        // image center row
    int w = i % W;        // image center col
    int c = j / KK;  // image channel

    int h_ = h + (j % KK) / K - K / 2;  // h_ = h + dh
    int w_ = w + (j % KK) % K - K / 2;  // w_ = w + dw

    float value = 0.0f;
    if (0 <= h_ && h_ < H && 0 <= w_ && w_ < W)
        value = input[(h_ * W + w_) * C + c]; 
    output[i * C * KK + j] = value; 
}

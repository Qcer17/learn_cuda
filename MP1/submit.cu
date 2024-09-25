#include <stdio.h>

float *load_floats(char *path, int *p_n_elements)
{
    FILE *file = fopen(path, "r");
    fscanf(file, "%d", p_n_elements);
    float *vector = (float *)malloc(sizeof(float) * (*p_n_elements));
    int i = 0;
    while (fscanf(file, "%f", vector + i) == 1)
    {
        i++;
    }
    return vector;
}

void print_floats(int n_elements, float *vector)
{
    for (int i = 0; i < n_elements; ++i)
    {
        printf("%f\n", *(vector + i));
    }
}

__global__ void add_vec_kernel(float *a_g, float *b_g, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        b_g[idx] += a_g[idx];
    }
}

void add_vec(float *a, float *b, float *c, int n)
{
    float *a_g, *b_g;
    int sz = sizeof(float) * n;
    cudaMalloc((void **)&a_g, sz);
    cudaMalloc((void **)&b_g, sz);

    cudaMemcpy(a_g, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(b_g, b, sz, cudaMemcpyHostToDevice);

    add_vec_kernel<<<ceil(n / 256.0), 256>>>(a_g, b_g, n);

    cudaMemcpy(c, b_g, sz, cudaMemcpyDeviceToHost);

    cudaFree(a_g);
    cudaFree(b_g);
}

float diff(float *a, float *b, int n)
{
    float sum = 0;
    for (int i = 0; i < n; ++i)
    {
        float t = b[i] - a[i];
        t = t < 0 ? -t : t;
        sum += t;
    }
    return sum;
}

int main(int argc, char **argv)
{
    int n_elements;
    float *vector1 = load_floats(argv[1], &n_elements);
    // print_floats(n_elements, vector1);
    float *vector2 = load_floats(argv[2], &n_elements);
    // print_floats(n_elements, vector2);
    float *res = (float *)malloc(sizeof(float) * n_elements);
    add_vec(vector1, vector2, res, n_elements);
    // print_floats(n_elements, res);

    float eps = 1e-5;
    float *ans = load_floats(argv[3], &n_elements);
    float d = diff(ans, res, n_elements);
    if (d < eps)
    {
        printf("correct\n");
    }
    else
    {
        printf("INCORRECT! %f\n", d);
    }

    return 0;
}
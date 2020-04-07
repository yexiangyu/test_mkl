
#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

#define ARRAY_SIZE 192

float randf()
{
    return rand() * 10.f / RAND_MAX - 5;
}

int main(int, char **)
{
    float *a = new float[ARRAY_SIZE];
    float *b = new float[ARRAY_SIZE];

    for (auto i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = randf();
        b[i] = randf();
    }

    auto print_array = [](float *ptr) {
        std::cout << "[";
        for (auto i = 0; i < ARRAY_SIZE; i++)
        {
            std::cout << ptr[i];
            if (i != ARRAY_SIZE - 1)
                std::cout << ",";
        }
        std::cout << "]" << std::endl;
    };

    auto get_norm = [](float *ptr) -> float {
        float norm = 0;
        for (auto i = 0; i < ARRAY_SIZE; i++)
        {
            norm += ptr[i] * ptr[i];
        }

        return sqrtf(norm);
    };

    // auto normalize = [&get_norm](float *ptr) {
    //     auto norm = get_norm(ptr);
    //     for (auto i = 0; i < ARRAY_SIZE; i++)
    //     {
    //         ptr[i] /= norm;
    //     }
    // };

    auto cos_naive = [&]() {
        float sum = 0;
        for (auto i = 0; i < ARRAY_SIZE; i++)
        {
            sum += (a[i] * b[i]);
        }
        std::cout << "cos_naive=" << sum / (get_norm(a) * get_norm(b)) << std::endl; // -0.01081336 from sklearn, -0.010813 from c++
        std::cout << "get_norm_a=" << get_norm(a) << std::endl;
        std::cout << "get_norm_b=" << get_norm(b) << std::endl;
    };
    auto cos_mkl = [&]() {
        auto mkl_norm_a = cblas_snrm2(ARRAY_SIZE, a, 1);
        cblas_sscal(ARRAY_SIZE, 1.f / mkl_norm_a, a, 1);
        auto mkl_norm_b = cblas_snrm2(ARRAY_SIZE, b, 1);
        cblas_sscal(ARRAY_SIZE, 1.f / mkl_norm_b, b, 1);
        std::cout << "cos_mkl=" << cblas_sdot(ARRAY_SIZE, a, 1, b, 1) << std::endl; // -0.01081336 from sklearn, -0.010813 from c++
    };

    // auto mkl_norm_a = cblas_snrm2(ARRAY_SIZE, a, 1);
    // auto mkl_norm_b = cblas_snrm2(ARRAY_SIZE, b, 1);

    // auto cos_naive_mkl = [&] {
    //     float sum = 0;
    //     for (auto i = 0; i < ARRAY_SIZE; i++)
    //     {
    //         sum += (a[i] * b[i]);
    //     }
    //     std::cout << "cos_mkl_naive=" << sum / (mkl_norm_a * mkl_norm_b) << std::endl; // -0.01081336 from sklearn, -0.010813 from c++
    // };

    // auto cos_mkl_blas_1 = [&] {
    //     std::cout << "cos_mkl_blas_1=" << cblas_sdot(ARRAY_SIZE, a, 1, b, 1) / (mkl_norm_a * mkl_norm_b) << std::endl; // -0.01081336 from sklearn, -0.010813 from c++
    // };

    // auto cos_mkl_blas_1_1 = [&] {
    //     std::cout << "cos_mkl_blas_1_1=" << cblas_sdot(ARRAY_SIZE, a, 1, b, 1) << std::endl; // -0.01081336 from sklearn, -0.010813 from c++
    // };

    auto cos_mkl_blas_3 = [&] {
        float ret;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, 1, ARRAY_SIZE, 1, a, ARRAY_SIZE, b, ARRAY_SIZE, 1, &ret, 1);
        std::cout << "cos_mkl_blas_3=" << ret << std::endl; // -0.01081336 from sklearn, -0.010813 from c++
    };

    // std::cout << "orgin:" << std::endl;
    // std::cout << "a=";
    // print_array(a);
    // std::cout << "norm_a=" << get_norm(a) << std::endl;
    // std::cout << "mkl_norm_a=" << mkl_norm_a << std::endl;
    // std::cout << "b=";
    // print_array(b);
    // std::cout << "norm_b=" << get_norm(b) << std::endl;
    // std::cout << "mkl_norm_b=" << mkl_norm_b << std::endl;

    cos_naive();
    cos_mkl();
    // cos_naive_mkl();
    // cos_mkl_blas_1();

    // normalize(a);
    // normalize(b);
    // std::cout << "normalized:" << std::endl;
    // std::cout << "a=";
    // print_array(a);
    // std::cout << "b=";
    // print_array(b);
    // cos_mkl_blas_1_1();
    // cos_mkl_blas_3();

    auto cos_mkl_blas_3_test = [&](size_t feat_size, size_t base_size, size_t query_size) {
        float *base = new float[feat_size * base_size];
        std::copy(a, a + ARRAY_SIZE, base);
        float *query = new float[feat_size * base_size];
        std::copy(b, b + ARRAY_SIZE, query);
        float *ret = new float[base_size * query_size];
        auto start = std::chrono::system_clock::now();
        //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, base_size, query_size, feat_size, 1, base, base_size, query, feat_size, 1, ret, query_size);
        //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, base_size, query_size, feat_size, 1, base, feat_size, query, query_size, 1, ret, query_size);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, base_size, query_size, feat_size, 1, base, feat_size, query, feat_size, 0, ret, query_size);
        auto delta = std::chrono::system_clock::now() - start;
        std::cout << "score=" << ret[0] << ",feat_size=" << feat_size << ",base_size=" << base_size << ",query_size=" << query_size << ",delta=" << std::chrono::duration_cast<std::chrono::microseconds>(delta).count() << std::endl;
        delete[] base;
        delete[] query;
        delete[] ret;
    };

    // cos_mkl_blas_3_test(192, 1, 1);
    // cos_mkl_blas_3_test(192, 10, 1);
    // cos_mkl_blas_3_test(192, 100, 1);
    // cos_mkl_blas_3_test(192, 1000, 1);
    // cos_mkl_blas_3_test(192, 10000, 1);
    // cos_mkl_blas_3_test(192, 100000, 1);
    // cos_mkl_blas_3_test(192, 1000000, 1);
    // cos_mkl_blas_3_test(192, 100000, 10);
    // cos_mkl_blas_3_test(192, 1000000, 10);
    // cos_mkl_blas_3_test(192, 100000, 100);
    static std::vector<int> bases{1, 10000, 1 << 14, 100000, 1 << 17, 1000000, 1 << 20};
    static std::vector<int> queries{1, 2, 4, 8, 10, 16, 32, 64, 100, 128, 256, 512, 1000, 1024};

    cos_mkl_blas_3_test(192, 64, 1);
    cos_mkl_blas_3_test(192, 64, 10);
    cos_mkl_blas_3_test(192, 64, 100);
    cos_mkl_blas_3_test(192, 64, 1000);
    cos_mkl_blas_3_test(192, 64, 1024);
    cos_mkl_blas_3_test(192, 64, 1 << 14);
    cos_mkl_blas_3_test(192, 64, 1 << 17);
    cos_mkl_blas_3_test(192, 64, 100000);

    // cos_mkl_blas_3_test(192, 32, 1);
    // cos_mkl_blas_3_test(192, 32, 10);
    // cos_mkl_blas_3_test(192, 32, 100);
    // cos_mkl_blas_3_test(192, 32, 1000);
    // cos_mkl_blas_3_test(192, 32, 1024);
    // cos_mkl_blas_3_test(192, 32, 1 << 14);
    // cos_mkl_blas_3_test(192, 32, 1 << 17);
    // cos_mkl_blas_3_test(192, 32, 100000);

    // cos_mkl_blas_3_test(192, 16, 1);
    // cos_mkl_blas_3_test(192, 16, 10);
    // cos_mkl_blas_3_test(192, 16, 100);
    // cos_mkl_blas_3_test(192, 16, 1000);
    // cos_mkl_blas_3_test(192, 16, 1024);
    // cos_mkl_blas_3_test(192, 16, 1 << 14);
    // cos_mkl_blas_3_test(192, 16, 1 << 17);
    // cos_mkl_blas_3_test(192, 16, 100000);

    // cos_mkl_blas_3_test(192, 1, 1);
    // cos_mkl_blas_3_test(192, 1, 10);
    // cos_mkl_blas_3_test(192, 1, 100);
    // cos_mkl_blas_3_test(192, 1, 1000);
    // cos_mkl_blas_3_test(192, 1, 1024);
    // cos_mkl_blas_3_test(192, 1, 1 << 14);
    // cos_mkl_blas_3_test(192, 1, 1 << 17);
    // cos_mkl_blas_3_test(192, 1, 100000);

    for (auto base : bases)
    {
        for (auto query : queries)
        {
            cos_mkl_blas_3_test(192, base, query);
        }
    }

    return 0;
}
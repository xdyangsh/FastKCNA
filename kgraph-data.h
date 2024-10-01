#ifndef WDONG_KGRAPH_DATA
#define WDONG_KGRAPH_DATA

#include <cmath>
#include <cstring>
#include <malloc.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <boost/assert.hpp>

#ifdef __GNUC__
#ifdef __AVX__
#define KGRAPH_MATRIX_ALIGN 32
#else
#ifdef __SSE2__
#define KGRAPH_MATRIX_ALIGN 16
#else
#define KGRAPH_MATRIX_ALIGN 4
#endif
#endif
#endif

namespace kgraph {

    /// L2 square distance with AVX instructions.
    /** AVX instructions have strong alignment requirement for t1 and t2.
     */
    extern float float_l2sqr_avx (float const *t1, float const *t2, unsigned dim);
    extern float float_dot_avx (float const *t1, float const *t2, unsigned dim);
    /// L2 square distance with SSE2 instructions.
    extern float float_l2sqr_sse2 (float const *t1, float const *t2, unsigned dim);
    extern float float_l2sqr_sse2 (float const *, unsigned dim);
    extern float float_dot_sse2 (float const *, float const *, unsigned dim);
    /// L2 square distance for uint8_t with SSE2 instructions (for SIFT).
    extern float uint8_l2sqr_sse2 (uint8_t const *t1, uint8_t const *t2, unsigned dim);

    extern float float_l2sqr (float const *, float const *, unsigned dim);
    extern float float_l2sqr (float const *, unsigned dim);
    extern float float_dot (float const *, float const *, unsigned dim);


    using std::vector;

    /// namespace for various distance metrics.
    namespace metric {
        /// L2 square distance.
        struct l2sqr {
            template <typename T>
            /// L2 square distance.
            static float apply (T const *t1, T const *t2, unsigned dim) {
                //printf("no SIMD\n");
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(t1[i]) - float(t2[i]);
                    v *= v;
                    r += v;
                }
                return r;
            }

            /// inner product.
            template <typename T>
            static float dot (T const *t1, T const *t2, unsigned dim) {
                //printf("Heihei5\n");
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    r += float(t1[i]) *float(t2[i]);
                }
                return r;
            }

            /// L2 norm.
            template <typename T>
            static float norm2 (T const *t1, unsigned dim) {
                //printf("Heihei6\n");
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(t1[i]);
                    v *= v;
                    r += v;
                }
                return r;
            }
        };

        struct l2 {
            template <typename T>
            static float apply (T const *t1, T const *t2, unsigned dim) {
                //printf("Heihei7\n");
                return sqrt(l2sqr::apply<T>(t1, t2, dim));
            }
        };
    }

    /// Matrix data.
    template <typename T, unsigned A = KGRAPH_MATRIX_ALIGN>
    class Matrix {
        unsigned col;
        unsigned row;
        size_t stride;
        char *data;

        void reset (unsigned r, unsigned c) {
            row = r;
            col = c;
            stride = (sizeof(T) * c + A - 1) / A * A;
            /*
            data.resize(row * stride);
            */
            if (data) free(data);
            data = (char *)memalign(A, row * stride); // SSE instruction needs data to be aligned
            if (!data) throw runtime_error("memalign");
        }
    public:
        Matrix (): col(0), row(0), stride(0), data(0) {}
        Matrix (unsigned r, unsigned c): data(0) {
            reset(r, c);
        }
        ~Matrix () {
            if (data) free(data);
        }
        unsigned size () const {
            return row;
        }
        unsigned dim () const {
            return col;
        }
        size_t step () const {
            return stride;
        }
        void resize (unsigned r, unsigned c) {
            reset(r, c);
        }
        char* getdata(unsigned i)
        {
            return data+stride*i;
        }
        T const *operator [] (unsigned i) const {
            return reinterpret_cast<T const *>(&data[stride * i]);
        }
        T *operator [] (unsigned i) {
            return reinterpret_cast<T *>(&data[stride * i]);
        }
        void zero () {
            memset(data, 0, row * stride);
        }

        void normalize2 () {
#pragma omp parallel for
            for (unsigned i = 0; i < row; ++i) {
                T *p = operator[](i);
                double sum = metric::l2sqr::norm2(p, col);
                sum = std::sqrt(sum);
                for (unsigned j = 0; j < col; ++j) {
                    p[j] /= sum;
                }
            }
        }
        
        void load (const std::string &path, unsigned dim, unsigned skip = 0, unsigned gap = 0) {
            std::ifstream is(path.c_str(), std::ios::binary);
            if (!is) throw io_error(path);
            is.seekg(0, std::ios::end);
            size_t size = is.tellg();
            size -= skip;
            unsigned line = sizeof(T) * dim + gap;
            unsigned N =  size / line;
            reset(N, dim);
            zero();
            is.seekg(skip, std::ios::beg);
            for (unsigned i = 0; i < N; ++i) {
                is.read(&data[stride * i], sizeof(T) * dim);
                is.seekg(gap, std::ios::cur);
            }
            if (!is) throw io_error(path);
        }

        int getRandomLevel(double reverse_size,std::default_random_engine &level_generator_) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }
        

        void getlevel(unsigned N,vector<int> &data2file,vector<int> &file2data,vector<int> &points_num,unsigned M_,unsigned random_seed)
        {
            std::default_random_engine level_generator_;
            double muti_=1 / log(1.0 * M_);
            level_generator_.seed(random_seed);
            data2file.reserve(N);
            data2file.clear();
            vector<int> levels;
            levels.resize(N);
            int max_level=0;
            for(int i=0;i<N;i++)
            {
                int level=getRandomLevel(muti_,level_generator_);
                levels[i]=level;
                if(max_level<level)
                {
                    max_level=level;
                }
            }
            points_num.resize(max_level+1,0);
            for(int i=max_level;i>=0;i--)
            {
                for(int j=0;j<N;j++)
                {
                    if(levels[j]==i)
                    {
                        data2file.push_back(j);
                        points_num[i]++;
                    }
                }
            }
            file2data.resize(N);
            for(int i=0;i<N;i++)
            {
                file2data[data2file[i]]=i;
            }
        }

        double hnsw_dataload (const std::string &path, unsigned dim, unsigned skip = 0, unsigned gap = 0, vector<int> &data2file=vector<int>(),vector<int> &points_num=vector<int>(),unsigned M_=50, unsigned random_seed=2024) {
            std::ifstream is(path.c_str(), std::ios::binary);
            if (!is) throw io_error(path);
            is.seekg(0, std::ios::end);
            size_t size = is.tellg();
            size -= skip;
            unsigned line = sizeof(T) * dim + gap;
            unsigned N =  size / line;
            vector<int> file2data;
            auto s = std::chrono::high_resolution_clock::now();
            getlevel(N,data2file,file2data,points_num,M_,random_seed);
            auto e = std::chrono::high_resolution_clock::now();
  	        std::chrono::duration<double> diff = e - s;
            double time = diff.count();
  	        std::cout << "get level time: " << time << "\n";
            reset(N, dim);
            zero();
            is.seekg(skip, std::ios::beg);
            int index;
            for (unsigned i = 0; i < N; ++i) {
                // auto it=find(labels.begin(),labels.end(),int(i));
                // if (it != labels.end()) 
                // {
                //     index = std::distance(labels.begin(), it);
                // }
                // else
                // {
                //     cout<<"error............"<<endl;
                // }
                index=file2data[i];
                is.read(&data[stride * index], sizeof(T) * dim);
                is.seekg(gap, std::ios::cur);
            }
            if (!is) throw io_error(path);
            cout<<"load over!"<<endl;
            return time;
        }

        void load_lshkit (std::string const &path) {
            static const unsigned LSHKIT_HEADER = 3;
            std::ifstream is(path.c_str(), std::ios::binary);
            unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
            is.read((char *)header, sizeof header);
            if (!is) throw io_error(path);
            if (header[0] != sizeof(T)) throw io_error(path);
            is.close();
            unsigned D = header[2];
            unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
            unsigned gap = 0;
            load(path, D, skip, gap);
        }

        void save_lshkit (std::string const &path) {
            std::ofstream os(path.c_str(), std::ios::binary);
            unsigned header[3];
            assert(sizeof header == 3*4);
            header[0] = sizeof(T);
            header[1] = row;
            header[2] = col;
            os.write((const char *)header, sizeof(header));
            for (unsigned i = 0; i < row; ++i) {
                os.write(&data[stride * i], sizeof(T) * col);
            }
        }
    };

    /// Matrix proxy to interface with 3rd party libraries (FLANN, OpenCV, NumPy).
    template <typename DATA_TYPE, unsigned A = KGRAPH_MATRIX_ALIGN>
    class MatrixProxy {
        unsigned rows;
        unsigned cols;      // # elements, not bytes, in a row, 
        size_t stride;    // # bytes in a row, >= cols * sizeof(element)
        uint8_t const *data;
    public:
        MatrixProxy (Matrix<DATA_TYPE> const &m)
            : rows(m.size()), cols(m.dim()), stride(m.step()), data(reinterpret_cast<uint8_t const *>(m[0])) {
        }

#ifndef __AVX__
#ifdef FLANN_DATASET_H_
        /// Construct from FLANN matrix.
        MatrixProxy (flann::Matrix<DATA_TYPE> const &m)
            : rows(m.rows), cols(m.cols), stride(m.stride), data(m.data) {
            if (stride % A) throw invalid_argument("bad alignment");
        }
#endif
#ifdef __OPENCV_CORE_HPP__
        /// Construct from OpenCV matrix.
        MatrixProxy (cv::Mat const &m)
            : rows(m.rows), cols(m.cols), stride(m.step), data(m.data) {
            if (stride % A) throw invalid_argument("bad alignment");
        }
#endif
#ifdef NPY_NDARRAYOBJECT_H
        /// Construct from NumPy matrix.
        MatrixProxy (PyArrayObject *obj) {
            if (!obj || (obj->nd != 2)) throw invalid_argument("bad array shape");
            rows = obj->dimensions[0];
            cols = obj->dimensions[1];
            stride = obj->strides[0];
            data = reinterpret_cast<uint8_t const *>(obj->data);
            if (obj->descr->elsize != sizeof(DATA_TYPE)) throw invalid_argument("bad data type size");
            if (stride % A) throw invalid_argument("bad alignment");
            if (!(stride >= cols * sizeof(DATA_TYPE))) throw invalid_argument("bad stride");
        }
#endif
#endif
        unsigned size () const {
            return rows;
        }
        unsigned dim () const {
            return cols;
        }
        DATA_TYPE const *operator [] (unsigned i) const {
            return reinterpret_cast<DATA_TYPE const *>(data + stride * i);
        }
        DATA_TYPE *operator [] (unsigned i) {
            return const_cast<DATA_TYPE *>(reinterpret_cast<DATA_TYPE const *>(data + stride * i));
        }
    };

    /// Oracle for Matrix or MatrixProxy.
    /** DATA_TYPE can be Matrix or MatrixProxy,
    * DIST_TYPE should be one class within the namespace kgraph.metric.
    */
    template <typename DATA_TYPE, typename DIST_TYPE>
    class MatrixOracle: public kgraph::IndexOracle {
        MatrixProxy<DATA_TYPE> proxy;
        
    public:
        float *selfdot;
        unsigned oracle_size;
        class SearchOracle: public kgraph::SearchOracle {
            MatrixProxy<DATA_TYPE> proxy;
            DATA_TYPE const *query;
        public:
            SearchOracle (MatrixProxy<DATA_TYPE> const &p, DATA_TYPE const *q): proxy(p), query(q) {
            }
            virtual unsigned size () const {
                return proxy.size();
            }
            virtual float operator () (unsigned i) const {
                return DIST_TYPE::apply(proxy[i], query, proxy.dim());
            }
        };
        template <typename MATRIX_TYPE>
        MatrixOracle (MATRIX_TYPE const &m): proxy(m) {
            oracle_size=proxy.size();
        }
        virtual unsigned size () const {
            return oracle_size;
        }
        virtual unsigned data_size () const {
            return proxy.size();
        }
        virtual void set_size (unsigned mysize){
            if(mysize>proxy.size())
            {
                oracle_size=proxy.size();
            }
            else
            {
                oracle_size=mysize;
            }
        }
        virtual unsigned dim() const {
            return proxy.dim();
        }
        virtual float operator () (unsigned i, unsigned j) const {
            return DIST_TYPE::apply(proxy[i], proxy[j], proxy.dim());
            //return selfdot[i]+selfdot[j]-2*DIST_TYPE::dot(proxy[i],proxy[j],proxy.dim());
        }
        virtual float operator () (unsigned i, unsigned j, unsigned offset, unsigned dim) const {
            return DIST_TYPE::apply(proxy[i]+offset, proxy[j]+offset, dim);
        }
        virtual float operator () (const float *query, unsigned i) const {
            return DIST_TYPE::apply(query, proxy[i], proxy.dim());
        }
        virtual float computeDot(unsigned i, unsigned j) const {
            return DIST_TYPE::dot(proxy[i],proxy[j],proxy.dim());
        }
        virtual void computeSelfDotPtr (){
            selfdot= new float[proxy.size()];
#pragma omp parallel for 
            for(unsigned i=0;i<proxy.size();i++)
            {
                selfdot[i]=DIST_TYPE::dot(proxy[i],proxy[i],proxy.dim());
            }
        }
        virtual float* Calcenter () const {
            float *center = new float[proxy.dim()];
            for (unsigned j = 0; j < proxy.dim(); j++) center[j] = 0;
                for (unsigned i = 0; i < proxy.size(); i++) {
                    //printf("i=%d\n", i);
                    for (unsigned j = 0; j < proxy.dim(); j++) {
                    //center[j] += data_[i * dimension_ + j];
                    center[j] += proxy[i][j];
                }
            }
        //printf("heihei0\n");
            for (unsigned j = 0; j < proxy.dim(); j++) {
                center[j] /= proxy.size();
            }//printf("heihei1\n");
            return center;
        }

        SearchOracle query (DATA_TYPE const *query) const {
            return SearchOracle(proxy, query);
        }
    };

    inline float AverageRecall (Matrix<float> const &gs, Matrix<float> const &result, unsigned K = 0) {
        if (K == 0) {
            K = result.dim();
        }
        if (!(gs.dim() >= K)) throw invalid_argument("gs.dim() >= K");
        if (!(result.dim() >= K)) throw invalid_argument("result.dim() >= K");
        if (!(gs.size() >= result.size())) throw invalid_argument("gs.size() > result.size()");
        float sum = 0;
        for (unsigned i = 0; i < result.size(); ++i) {
            float const *gs_row = gs[i];
            float const *re_row = result[i];
            // compare
            unsigned found = 0;
            unsigned gs_n = 0;
            unsigned re_n = 0;
            while ((gs_n < K) && (re_n < K)) {
                if (gs_row[gs_n] < re_row[re_n]) {
                    ++gs_n;
                }
                else if (gs_row[gs_n] == re_row[re_n]) {
                    ++found;
                    ++gs_n;
                    ++re_n;
                }
                else {
                    throw runtime_error("distance is unstable");
                }
            }
            sum += float(found) / K;
        }
        return sum / result.size();
    }


}

#ifndef KGRAPH_NO_VECTORIZE
#ifdef __GNUC__
#ifdef __AVX__
// #if 1
namespace kgraph { namespace metric {
        template <>
        inline float l2sqr::apply<float> (float const *t1, float const *t2, unsigned dim) {
            //printf("Heihei8\n");
            return float_l2sqr_avx(t1, t2, dim);
        }
        template <>
        inline float l2sqr::dot<float> (float const *t1, float const *t2, unsigned dim) {
            //printf("Heihei8\n");
            return float_dot_avx(t1, t2, dim);
        }
}}
// #endif
#else
#ifdef __SSE2__
namespace kgraph { namespace metric {
        template <>
        inline float l2sqr::apply<float> (float const *t1, float const *t2, unsigned dim) {
            //printf("Heihei0\n");
            return float_l2sqr_sse2(t1, t2, dim);
        }
        template <>
        inline float l2sqr::dot<float> (float const *t1, float const *t2, unsigned dim) {
            //printf("Heihei1\n");
            return float_dot_sse2(t1, t2, dim);
        }
        template <>
        inline float l2sqr::norm2<float> (float const *t1, unsigned dim) {
            //printf("Heihei2\n");
            return float_l2sqr_sse2(t1, dim);
        }
        template <>
        inline float l2sqr::apply<uint8_t> (uint8_t const *t1, uint8_t const *t2, unsigned dim) {
            //printf("Heihei3\n");
            return uint8_l2sqr_sse2(t1, t2, dim);
        }
}}
#endif
#endif
#endif
#endif



#endif


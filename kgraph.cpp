#ifndef KGRAPH_VERSION
#define KGRAPH_VERSION unknown
#endif
#ifndef KGRAPH_BUILD_NUMBER
#define KGRAPH_BUILD_NUMBER
#endif
#ifndef KGRAPH_BUILD_ID
#define KGRAPH_BUILD_ID
#endif
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#define STRINGIFY_HELPER(x) #x
static char const *kgraph_version = STRINGIFY(KGRAPH_VERSION) "-" STRINGIFY(KGRAPH_BUILD_NUMBER) "," STRINGIFY(KGRAPH_BUILD_ID);

// #ifdef DEBUG
// #define DEBUG
// #endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <unordered_set>
#include <mutex>
#include <stack>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <boost/timer/timer.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include <boost/dynamic_bitset.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include "boost/smart_ptr/detail/spinlock.hpp"
#include "kgraph.h"

#include <sys/resource.h>
#include <queue>

constexpr double kPi = 3.14159265358979323846264;
typedef std::lock_guard<std::mutex> LockGuard1;

namespace kgraph
{

    typedef std::vector<Neighbor> Neighbors;

    using namespace std;
    using namespace boost;
    using namespace boost::accumulators;

    unsigned verbosity = default_verbosity;

    typedef boost::detail::spinlock Lock;
    typedef std::lock_guard<Lock> LockGuard;

    
    template < typename T> 
    vector<unsigned>  sort_indexes(const vector< T>  & v) {

        // initialize original index locations
        vector<unsigned>  idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),
            [& v](unsigned i1, unsigned i2) {
                if(v[i1]==v[i2])
                {
                    return i1<i2;
                }
                return v[i1] <  v[i2];
            });

        return idx;
    }

    // generate size distinct random numbers < N
    template <typename RNG>
    static void GenRandom(RNG &rng, unsigned *addr, unsigned size, unsigned N)
    {
        if (N == size)
        {
            for (unsigned i = 0; i < size; ++i)
            {
                addr[i] = i;
            }
            return;
        }
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = rng() % (N - size);
        }
        sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i)
        {
            if (addr[i] <= addr[i - 1])
            {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = (addr[i] + off) % N;
        }
    }

    // both pool and knn should be sorted in ascending order
    static float EvaluateRecall(Neighbors const &pool, unsigned K0,Neighbors const &knn, unsigned K)
    {
        if (knn.empty())
            return 1.0;
        unsigned found = 0;
        // unsigned n_p = 0;
        // unsigned n_k = 0;
        // for (int i=0;i<K;i++)
        // {
        //     if (n_p >= pool.size()||n_p >=K)
        //         break;
        //     if (n_k >= knn.size()||n_p>=K)
        //         break;
        //     if (knn[n_k].dist < pool[n_p].dist)
        //     {
        //         ++n_k;
        //     }
        //     else if (knn[n_k].dist == pool[n_p].dist)
        //     {
        //         ++found;
        //         ++n_k;
        //         ++n_p;
        //     }
        //     else
        //     {
        //         cerr << knn[n_k].dist <<"       "<< pool[n_p].dist<<endl;
        //         cerr << "Distance is unstable." << endl;
        //         cerr << "Exact";
        //         for (auto const &p : knn)
        //         {
        //             cerr << ' ' << p.id << ':' << p.dist;
        //         }
        //         cerr << endl;
        //         cerr << "Approx";
        //         for (auto const &p : pool)
        //         {
        //             cerr << ' ' << p.id << ':' << p.dist;
        //         }
        //         cerr << endl;
        //         throw runtime_error("distance is unstable");
        //     }
        // }
        for(unsigned i=0;i<min(K0,K);i++)
        {
            unsigned id=pool[i].id;
            for(unsigned j=0;j<K;j++)
            {
                if(id==knn[j].id)
                {
                    found++;
                    break;
                }

            }
        }
        return float(found) / K;
    }

    static float EvaluateAccuracy(Neighbors const &pool, Neighbors const &knn)
    {
        unsigned m = std::min(pool.size(), knn.size());
        float sum = 0;
        unsigned cnt = 0;
        for (unsigned i = 0; i < m; ++i)
        {
            if (knn[i].dist > 0)
            {
                sum += abs(pool[i].dist - knn[i].dist) / knn[i].dist;
                ++cnt;
            }
        }
        return cnt > 0 ? sum / cnt : 0;
    }

    static float EvaluateOneRecall(Neighbors const &pool, Neighbors const &knn)
    {
        if (pool[0].dist == knn[0].dist)
            return 1.0;
        return 0;
    }

    static float EvaluateDelta(Neighbors const &pool, unsigned K)
    {
        unsigned c = 0;
        unsigned N = K;
        if (pool.size() < N)
            N = pool.size();
        for (unsigned i = 0; i < N; ++i)
        {
            if (pool[i].flag)
                ++c;
        }
        return float(c) / K;
    }

    struct Control
    {
        unsigned id;
        Neighbors neighbors;
    };

    // try insert nn into the list
    // the array addr must contain at least K+1 entries:
    //      addr[0..K-1] is a sorted list
    //      addr[K] is as output parameter
    // * if nn is already in addr[0..K-1], return K+1
    // * Otherwise, do the equivalent of the following
    //      put nn into addr[K]
    //      make addr[0..K] sorted
    //      return the offset of nn's index in addr (could be K)
    //
    // Special case:  K == 0
    //      addr[0] <- nn
    //      return 0
    template <typename NeighborT>
    unsigned UpdateKnnListHelper(NeighborT *addr, unsigned K, NeighborT nn)
    {
        // find the location to insert
        unsigned j;
        unsigned i = K;
        while (i > 0)
        {
            j = i - 1;
            if (addr[j].dist <= nn.dist)
                break;
            i = j;
        }
        // check for equal ID
        unsigned l = i;
        while (l > 0)
        {
            j = l - 1;
            if (addr[j].dist < nn.dist)
                break;
            if (addr[j].id == nn.id)
                return K + 1;
            l = j;
        }
        // i <= K-1
        j = K;
        while (j > i)
        {
            addr[j] = addr[j - 1];
            --j;
        }
        addr[i] = nn;
        return i;
    }
    static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
    {
        // find the location to insert
        int left = 0, right = K - 1;
        if (addr[left].dist > nn.dist)
        {
            memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].dist < nn.dist)
        {
            addr[K] = nn;
            return K;
        }
        while (left < right - 1)
        {
            int mid = (left + right) / 2;
            if (addr[mid].dist > nn.dist)
                right = mid;
            else
                left = mid;
        }

        while (left > 0)
        {
            if (addr[left].dist < nn.dist)
                break;
            if (addr[left].id == nn.id)
                return K + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)
            return K + 1;
        memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }
    static inline unsigned UpdateKnnList(Neighbor *addr, unsigned K, Neighbor nn)
    {
        return UpdateKnnListHelper<Neighbor>(addr, K, nn);
    }

    static inline unsigned UpdateKnnList(NeighborX *addr, unsigned K, NeighborX nn)
    {
        return UpdateKnnListHelper<NeighborX>(addr, K, nn);
    }

    void LinearSearch(IndexOracle const &oracle, unsigned i, unsigned K, vector<Neighbor> *pnns)
    {
        vector<Neighbor> nns(K + 1);
        unsigned N = oracle.size();
        Neighbor nn;
        nn.id = 0;
        nn.flag = true; // we don't really use this
        unsigned k = 0;
        while (nn.id < N)
        {
            if (nn.id != i)
            {
                nn.dist = oracle(i, nn.id);
                UpdateKnnList(&nns[0], k, nn);
                if (k < K)
                    ++k;
            }
            ++nn.id;
        }
        nns.resize(K);
        pnns->swap(nns);
    }

    unsigned SearchOracle::search(unsigned K, float epsilon, unsigned *ids, float *dists) const
    {
        vector<Neighbor> nns(K + 1);
        unsigned N = size();
        unsigned L = 0;
        for (unsigned k = 0; k < N; ++k)
        {
            float k_dist = operator()(k);
            if (k_dist > epsilon)
                continue;
            UpdateKnnList(&nns[0], L, Neighbor(k, k_dist));
            if (L < K)
                ++L;
        }
        if (ids)
        {
            for (unsigned k = 0; k < L; ++k)
            {
                ids[k] = nns[k].id;
            }
        }
        if (dists)
        {
            for (unsigned k = 0; k < L; ++k)
            {
                dists[k] = nns[k].dist;
            }
        }
        return L;
    }

    void GenerateControl(IndexOracle const &oracle, unsigned C, unsigned K, vector<Control> *pcontrols)
    {
        auto s_computegt = std::chrono::high_resolution_clock::now();
        vector<Control> controls(C);
        {
            vector<unsigned> index(oracle.size());
            int i = 0;
            for (unsigned &v : index)
            {
                v = i++;
            }
            random_shuffle(index.begin(), index.end());
#pragma omp parallel for
            for (unsigned i = 0; i < C; ++i)
            {
                controls[i].id = index[i];
                LinearSearch(oracle, index[i], K, &controls[i].neighbors);
            }
        }
        pcontrols->swap(controls);
        auto e_computegt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_computegt = e_computegt-s_computegt;
        double Time_computegt = diff_computegt.count();
        cout<<"compute groundtruth time: "<<Time_computegt<<endl;
    }

    static char const *KGRAPH_MAGIC = "KNNGRAPH";
    static unsigned constexpr KGRAPH_MAGIC_SIZE = 8;
    static uint32_t constexpr SIGNATURE_VERSION = 2;

    class KGraphImpl : public KGraph
    {
    protected:
        vector<unsigned> M;
        bool no_dist; // Distance & flag information in Neighbor is not valid.

        // actual M for a node that should be used in search time
        unsigned actual_M(unsigned pM, unsigned i) const
        {
            return std::min(std::max(M[i], pM), unsigned(graph[i].size()));
        }

    public:
        vector<vector<Neighbor>> graph;
        vector<vector<unsigned>> temp_graph;
        virtual ~KGraphImpl()
        {
        }
        virtual void load(char const *path)
        {
            static_assert(sizeof(unsigned) == sizeof(uint32_t), "unsigned must be 32-bit");
            ifstream is(path, ios::binary);
            char magic[KGRAPH_MAGIC_SIZE];
            uint32_t sig_version;
            uint32_t sig_cap;
            uint32_t N;
            is.read(magic, sizeof(magic));
            is.read(reinterpret_cast<char *>(&sig_version), sizeof(sig_version));
            is.read(reinterpret_cast<char *>(&sig_cap), sizeof(sig_cap));
            if (sig_version != SIGNATURE_VERSION)
                throw runtime_error("data version not supported.");
            is.read(reinterpret_cast<char *>(&N), sizeof(N));
            if (!is)
                runtime_error("error reading index file.");
            for (unsigned i = 0; i < KGRAPH_MAGIC_SIZE; ++i)
            {
                if (KGRAPH_MAGIC[i] != magic[i])
                    runtime_error("index corrupted.");
            }
            no_dist = sig_cap & FORMAT_NO_DIST;
            graph.resize(N);
            M.resize(N);
            vector<uint32_t> nids;
            for (unsigned i = 0; i < graph.size(); ++i)
            {
                auto &knn = graph[i];
                unsigned K;
                is.read(reinterpret_cast<char *>(&M[i]), sizeof(M[i]));
                is.read(reinterpret_cast<char *>(&K), sizeof(K));
                if (!is)
                    runtime_error("error reading index file.");
                knn.resize(K);
                if (no_dist)
                {
                    nids.resize(K);
                    is.read(reinterpret_cast<char *>(&nids[0]), K * sizeof(nids[0]));
                    for (unsigned k = 0; k < K; ++k)
                    {
                        knn[k].id = nids[k];
                        knn[k].dist = 0;
                        knn[k].flag = false;
                    }
                }
                else
                {
                    is.read(reinterpret_cast<char *>(&knn[0]), K * sizeof(knn[0]));
                }
            }
        }

        virtual void save(char const *path, int format) const
        {
            ofstream os(path, ios::binary);
            uint32_t N = graph.size();
            os.write(KGRAPH_MAGIC, KGRAPH_MAGIC_SIZE);
            os.write(reinterpret_cast<char const *>(&SIGNATURE_VERSION), sizeof(SIGNATURE_VERSION));
            uint32_t sig_cap = format;
            os.write(reinterpret_cast<char const *>(&sig_cap), sizeof(sig_cap));
            os.write(reinterpret_cast<char const *>(&N), sizeof(N));
            vector<unsigned> nids;
            for (unsigned i = 0; i < graph.size(); ++i)
            {
                auto const &knn = graph[i];
                uint32_t K = knn.size();
                os.write(reinterpret_cast<char const *>(&M[i]), sizeof(M[i]));
                os.write(reinterpret_cast<char const *>(&K), sizeof(K));
                if (format & FORMAT_NO_DIST)
                {
                    nids.resize(K);
                    for (unsigned k = 0; k < K; ++k)
                    {
                        nids[k] = knn[k].id;
                    }
                    os.write(reinterpret_cast<char const *>(&nids[0]), K * sizeof(nids[0]));
                }
                else
                {
                    os.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(knn[0]));
                }
            }
        }
        virtual void buildFastNSG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name ,const char *res_name);
        virtual void buildFastTauMNG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name ,const char *res_name);
        virtual void buildFastAlphaPG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name ,const char *res_name);
        virtual void buildFastHNSW(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name ,const char *res_name,char *&data_level0_memory_,char **linkLists_,unsigned level,unsigned* ep_ptr,size_t size_data_per_element_);

        void FatherSwapHNSWLayer0(IndexOracle const &oracle,IndexParams const &params,char *&data_level0_memory_)
        {
            size_t size_links_level0_=params.nsg_R*sizeof(unsigned)+sizeof(unsigned);
            size_t size_data_per_element_=size_links_level0_+sizeof(float)*oracle.dim()+sizeof(size_t);
            size_t offsetData_=size_links_level0_;
            size_t label_offset_ = size_links_level0_+sizeof(float)*oracle.dim();
            size_t offsetLevel0_ = 0;

            for(unsigned i=0;i<oracle.size();i++)
            {
                unsigned *cll= (unsigned *)(data_level0_memory_+i*size_data_per_element_+offsetLevel0_);
                int size=temp_graph[i].size();
                unsigned short linkCount = static_cast<unsigned short>(size);
                *((unsigned short int*)(cll))=*((unsigned short int *)&linkCount);
                unsigned *nei=(unsigned *)(cll+1);
                for(unsigned j=0;j<temp_graph[i].size();j++)
                {
                    nei[j]=temp_graph[i][j];
                }

            }
        }


        virtual unsigned search(SearchOracle const &oracle, SearchParams const &params, unsigned *ids, float *dists, SearchInfo *pinfo) const
        {
            if (graph.size() > oracle.size())
            {
                throw runtime_error("dataset larger than index");
            }
            if (params.P >= graph.size())
            {
                if (pinfo)
                {
                    pinfo->updates = 0;
                    pinfo->cost = 1.0;
                }
                return oracle.search(params.K, params.epsilon, ids, dists);
            }
            vector<NeighborX> knn(params.K + params.P + 1);
            vector<NeighborX> results;
            // flags access is totally random, so use small block to avoid
            // extra memory access
            boost::dynamic_bitset<> flags(graph.size(), false);

            if (params.init && params.T > 1)
            {
                throw runtime_error("when init > 0, T must be 1.");
            }

            unsigned seed = params.seed;
            unsigned updates = 0;
            if (seed == 0)
                seed = time(NULL);
            mt19937 rng(seed);
            unsigned n_comps = 0;
            for (unsigned trial = 0; trial < params.T; ++trial)
            {
                unsigned L = params.init;
                if (L == 0)
                { // generate random starting points
                    vector<unsigned> random(params.P);
                    GenRandom(rng, &random[0], random.size(), graph.size());
                    for (unsigned s : random)
                    {
                        if (!flags[s])
                        {
                            knn[L++].id = s;
                            // flags[s] = true;
                        }
                    }
                }
                else
                { // user-provided starting points.
                    if (!ids)
                        throw invalid_argument("no initial data provided via ids");
                    if (!(L < params.K))
                        throw invalid_argument("L < params.K");
                    for (unsigned l = 0; l < L; ++l)
                    {
                        knn[l].id = ids[l];
                    }
                }
                for (unsigned k = 0; k < L; ++k)
                {
                    auto &e = knn[k];
                    flags[e.id] = true;
                    e.flag = true;
                    e.dist = oracle(e.id);
                    e.m = 0;
                    e.M = actual_M(params.M, e.id);
                }
                sort(knn.begin(), knn.begin() + L);

                unsigned k = 0;
                while (k < L)
                {
                    auto &e = knn[k];
                    if (!e.flag)
                    { // all neighbors of this node checked
                        ++k;
                        continue;
                    }
                    unsigned beginM = e.m;
                    unsigned endM = beginM + params.S; // check this many entries
                    if (endM > e.M)
                    { // we are done with this node
                        e.flag = false;
                        endM = e.M;
                    }
                    e.m = endM;
                    // all modification to knn[k] must have been done now,
                    // as we might be relocating knn[k] in the loop below
                    auto const &neighbors = graph[e.id];
                    for (unsigned m = beginM; m < endM; ++m)
                    {
                        unsigned id = neighbors[m].id;
                        // BOOST_VERIFY(id < graph.size());
                        if (flags[id])
                            continue;
                        flags[id] = true;
                        ++n_comps;
                        float dist = oracle(id);
                        NeighborX nn(id, dist);
                        unsigned r = UpdateKnnList(&knn[0], L, nn);
                        BOOST_VERIFY(r <= L);
                        // if (r > L) continue;
                        if (L + 1 < knn.size())
                            ++L;
                        if (r < L)
                        {
                            knn[r].M = actual_M(params.M, id);
                            if (r < k)
                            {
                                k = r;
                            }
                        }
                    }
                }
                if (L > params.K)
                    L = params.K;
                if (results.empty())
                {
                    results.reserve(params.K + 1);
                    results.resize(L + 1);
                    copy(knn.begin(), knn.begin() + L, results.begin());
                }
                else
                {
                    // update results
                    for (unsigned l = 0; l < L; ++l)
                    {
                        unsigned r = UpdateKnnList(&results[0], results.size() - 1, knn[l]);
                        if (r < results.size() /* inserted */ && results.size() < (params.K + 1))
                        {
                            results.resize(results.size() + 1);
                        }
                    }
                }
            }
            results.pop_back();
            // check epsilon
            {
                for (unsigned l = 0; l < results.size(); ++l)
                {
                    if (results[l].dist > params.epsilon)
                    {
                        results.resize(l);
                        break;
                    }
                }
            }
            unsigned L = results.size();
            /*
            if (!(L <= params.K)) {
                cerr << L << ' ' << params.K << endl;
            }
            */
            if (!(L <= params.K))
                throw runtime_error("L <= params.K");
            // check epsilon
            if (ids)
            {
                for (unsigned k = 0; k < L; ++k)
                {
                    ids[k] = results[k].id;
                }
            }
            if (dists)
            {
                for (unsigned k = 0; k < L; ++k)
                {
                    dists[k] = results[k].dist;
                }
            }
            if (pinfo)
            {
                pinfo->updates = updates;
                pinfo->cost = float(n_comps) / graph.size();
            }
            return L;
        }

        virtual void get_nn(unsigned id, unsigned *nns, float *dist, unsigned *pM, unsigned *pL) const
        {
            if (!(id < graph.size()))
                throw invalid_argument("id too big");
            auto const &v = graph[id];
            *pM = M[id];
            *pL = v.size();
            if (nns)
            {
                for (unsigned i = 0; i < v.size(); ++i)
                {
                    nns[i] = v[i].id;
                }
            }
            if (dist)
            {
                if (no_dist)
                    throw runtime_error("distance information is not available");
                for (unsigned i = 0; i < v.size(); ++i)
                {
                    dist[i] = v[i].dist;
                }
            }
        }

        void prune1()
        {
            for (unsigned i = 0; i < graph.size(); ++i)
            {
                if (graph[i].size() > M[i])
                {
                    graph[i].resize(M[i]);
                }
            }
        }

        void prune2()
        {
            vector<vector<unsigned>> reverse(graph.size()); // reverse of new graph
            vector<unsigned> new_L(graph.size(), 0);
            unsigned L = 0;
            unsigned total = 0;
            for (unsigned i = 0; i < graph.size(); ++i)
            {
                if (M[i] > L)
                    L = M[i];
                total += M[i];
                for (auto &e : graph[i])
                {
                    e.flag = false; // haven't been visited yet
                }
            }
            progress_display progress(total, cerr);
            vector<unsigned> todo(graph.size());
            for (unsigned i = 0; i < todo.size(); ++i)
                todo[i] = i;
            vector<unsigned> new_todo(graph.size());
            for (unsigned l = 0; todo.size(); ++l)
            {
                BOOST_VERIFY(l <= L);
                new_todo.clear();
                for (unsigned i : todo)
                {
                    if (l >= M[i])
                        continue;
                    new_todo.push_back(i);
                    auto &v = graph[i];
                    BOOST_VERIFY(l < v.size());
                    if (v[l].flag)
                        continue;     // we have visited this one already
                    v[l].flag = true; // now we have seen this one
                    ++progress;
                    unsigned T;
                    {
                        auto &nl = new_L[i];
                        // shift the entry to add
                        T = v[nl].id = v[l].id;
                        v[nl].dist = v[l].dist;
                        ++nl;
                    }
                    reverse[T].push_back(i);
                    {
                        auto const &u = graph[T];
                        for (unsigned ll = l + 1; ll < M[i]; ++ll)
                        {
                            if (v[ll].flag)
                                continue;
                            for (unsigned j = 0; j < new_L[T]; ++j)
                            { // new graph
                                if (v[ll].id == u[j].id)
                                {
                                    v[ll].flag = true;
                                    ++progress;
                                    break;
                                }
                            }
                        }
                    }
                    {
                        for (auto r : reverse[i])
                        {
                            auto &u = graph[r];
                            for (unsigned ll = l; ll < M[r]; ++ll)
                            {
                                // must start from l: as item l might not have been checked
                                // for reverse
                                if (u[ll].id == T)
                                {
                                    if (!u[ll].flag)
                                        ++progress;
                                    u[ll].flag = true;
                                }
                            }
                        }
                    }
                }
                todo.swap(new_todo);
            }
            BOOST_VERIFY(progress.count() == total);
            M.swap(new_L);
            prune1();
        }

        virtual void prune(IndexOracle const &oracle, unsigned level)
        {
            if (level & PRUNE_LEVEL_1)
            {
                prune1();
            }
            if (level & PRUNE_LEVEL_2)
            {
                prune2();
            }
        }

        void reverse(int rev_k)
        {
            if (rev_k == 0)
                return;
            if (no_dist)
                throw runtime_error("Need distance information to reverse graph");
            {
                cerr << "Graph completion with reverse edges..." << endl;
                vector<vector<Neighbor>> ng(graph.size()); // new graph adds on original one
                // ng = graph;
                progress_display progress(graph.size(), cerr);
                for (unsigned i = 0; i < graph.size(); ++i)
                {
                    auto const &v = graph[i];
                    unsigned K = M[i];
                    if (rev_k > 0)
                    {
                        K = rev_k;
                        if (K > v.size())
                            K = v.size();
                    }
                    // if (v.size() < XX) XX = v.size();
                    for (unsigned j = 0; j < K; ++j)
                    {
                        auto const &e = v[j];
                        auto re = e;
                        re.id = i;
                        ng[i].push_back(e);
                        ng[e.id].push_back(re);
                    }
                    ++progress;
                }
                graph.swap(ng);
            }
            {
                cerr << "Reranking edges..." << endl;
                progress_display progress(graph.size(), cerr);
#pragma omp parallel for
                for (unsigned i = 0; i < graph.size(); ++i)
                {
                    auto &v = graph[i];
                    std::sort(v.begin(), v.end());
                    v.resize(std::unique(v.begin(), v.end()) - v.begin());
                    M[i] = v.size();
#pragma omp critical
                    ++progress;
                }
            }
        }

        virtual void swapResult(vector<vector<int>> &_knng)
        {
            // printf("HeiHei\n");
            unsigned N = graph.size();
            _knng.resize(N);
            for (unsigned n = 0; n < N; ++n)
            {
                auto &nns = graph[n];
                auto &knn = _knng[n];
                unsigned K = nns.size();
                knn.resize(K);
                for (unsigned k = 0; k < K; ++k)
                {
                    knn[k] = nns[k].id;
                }
            }
        }

        virtual void swapResult(vector<vector<unsigned>> &_knng)
        {
            // printf("HeiHei\n");
            unsigned N = graph.size();
            _knng.resize(N);
            for (unsigned n = 0; n < N; ++n)
            {
                auto &nns = graph[n];
                auto &knn = _knng[n];
                unsigned K = nns.size();
                knn.resize(K);
                for (unsigned k = 0; k < K; ++k)
                {
                    knn[k] = nns[k].id;
                }
            }
        }
    };

    class KGraphConstructor : public KGraphImpl
    {
        // The neighborhood structure maintains a pool of near neighbors of an object.
        // The neighbors are stored in the pool.  "n" (<="params.L") is the number of valid entries
        // in the pool, with the beginning "k" (<="n") entries sorted.
        struct Nhood
        { // neighborhood
            
            float radius; // distance of interesting range
            float radiusM;
            unsigned L; // # valid items in the pool,  L + 1 <= pool.size()
            unsigned M; // we only join items in pool[0..M)
            bool found; // helped found new NN in this round
            Neighbors pool;
            vector<Neighbor> nn_nsg;
            vector<unsigned> nn_old;
            vector<unsigned> nn_new;
            vector<unsigned> rnn_old;
            vector<unsigned> rnn_new;
            vector<unsigned> old_nsg;

            vector<bool> rnn_new_flag;
            Lock lock;
            // only non-readonly method which is supposed to be called in parallel
            unsigned parallel_try_insert(unsigned id, float dist)
            {
                if (dist > radius)
                    return pool.size();
                LockGuard guard(lock);
                // unsigned l = UpdateKnnList(&pool[0], L, Neighbor(id, dist, true));
                unsigned l=InsertIntoPool(&pool[0],L,Neighbor(id,dist,true));
                if (l <= L)
                { // inserted
                    if (L + 1 < pool.size())
                    { // if l == L + 1, there's a duplicate
                        ++L;
                    }
                    else
                    {
                        radius = pool[L - 1].dist;
                    }
                }
                return l;
            }

            // join should not be conflict with insert
            template <typename C>
            void join(C callback) const
            {
                for (unsigned const i : nn_new)
                {
                    for (unsigned const j : nn_new)
                    {
                        if (i < j)
                        {
                            callback(i, j);
                        }
                    }
                    for (unsigned j : nn_old)
                    {
                        callback(i, j);
                    }
                }
            }
        };

        unsigned ep_;
        unsigned width;
        IndexOracle const &oracle;
        IndexParams params;
        IndexInfo *pinfo;
        vector<Nhood> nhoods;
        size_t n_comps;
        vector<Control> controls;
        unsigned MAXL=0;
        

        void init()
        {
            // cout<<"init begin"<<endl;
            unsigned N = oracle.size();
            unsigned seed = params.seed;
            mt19937 rng(seed);
            for (auto &nhood : nhoods)
            {
                nhood.nn_new.resize(params.S * 2);
                nhood.pool.resize(params.L + 1);
                nhood.radius = numeric_limits<float>::max();
            }
#pragma omp parallel
            {
#ifdef _OPENMP
                mt19937 rng(seed ^ omp_get_thread_num());
#else
                mt19937 rng(seed);
#endif
                vector<unsigned> random(params.S + 1);
#pragma omp for
                for (unsigned n = 0; n < N; ++n)
                {
                    auto &nhood = nhoods[n];
                    Neighbors &pool = nhood.pool;
                    GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(), N);
                    GenRandom(rng, &random[0], random.size(), N);
                    nhood.L = params.S;
                    nhood.M = params.S;
                    unsigned i = 0;
                    for (unsigned l = 0; l < nhood.L; ++l)
                    {
                        if (random[i] == n)
                            ++i;
                        auto &nn = nhood.pool[l];
                        nn.id = random[i++];
                        nn.dist = oracle(nn.id, n);
                        nn.flag = true;
                    }
                    sort(pool.begin(), pool.begin() + nhood.L);
                }
            }
            // cout<<"init end"<<endl;
        }

        void init2()
        {
            // cout<<"init begin"<<endl;
            unsigned N = oracle.size();
            unsigned seed = params.seed;
            mt19937 rng(seed);
            for (auto &nhood : nhoods)
            {
                nhood.nn_new.resize(params.S);
                nhood.pool.resize(params.L + 1);
                nhood.radius = numeric_limits<float>::max();
            }
#pragma omp parallel
            {
#ifdef _OPENMP
                mt19937 rng(seed ^ omp_get_thread_num());
#else
                mt19937 rng(seed);
#endif
                vector<unsigned> random(params.L + 1);
#pragma omp for
                for (unsigned n = 0; n < N; ++n)
                {
                    auto &nhood = nhoods[n];
                    Neighbors &pool = nhood.pool;
                    // GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(), N);
                    GenRandom(rng, &random[0], random.size(), N);
                    nhood.L = params.L;
                    nhood.M = params.S;
                    unsigned i = 0;
                    for (unsigned l = 0; l < nhood.L; ++l)
                    {
                        if (random[i] == n)
                            ++i;
                        auto &nn = nhood.pool[l];
                        nn.id = random[i++];
                        nn.dist = oracle(nn.id, n);
                        nn.flag = true;
                    }
                    sort(pool.begin(), pool.begin() + nhood.L);

                    for(unsigned i = 0; i < params.S; i++)
                    {
                        nhood.nn_new[i]=nhood.pool[i].id;
                    }

                }
            }
            // cout<<"init end"<<endl;
        }

        void join()
        {
            size_t cc = 0;
#pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+ : cc)
            for (unsigned n = 0; n < oracle.size(); ++n)
            {
                size_t uu = 0;
                nhoods[n].found = false;
                nhoods[n].join([&](unsigned i, unsigned j)
                               {
                        float dist = oracle(i, j);
                        ++cc;
                        unsigned r;
                        r = nhoods[i].parallel_try_insert(j, dist);
                        if (r < params.K) ++uu;
                        r = nhoods[j].parallel_try_insert(i, dist);
                        if (r < params.K) ++uu; });
                nhoods[n].found = uu > 0;
            }
            n_comps += cc;
        }
        void update()
        {
            unsigned N = oracle.size();
            for (auto &nhood : nhoods)
            {
                nhood.nn_new.clear();
                nhood.nn_old.clear();
                nhood.rnn_new.clear();
                nhood.rnn_old.clear();
                nhood.radius = nhood.pool.back().dist;
                //nhood.radius = nhood.pool[nhood.L-1].dist;
            }
            //!!! compute radius2
#pragma omp parallel for
            for (unsigned n = 0; n < N; ++n)
            {
                auto &nhood = nhoods[n];
                if (nhood.found)
                {
                    unsigned maxl = std::min(nhood.M + params.S, nhood.L);
                    unsigned c = 0;
                    unsigned l = 0;
                    while ((l < maxl) && (c < params.S))
                    {
                        if (nhood.pool[l].flag)
                            ++c;
                        ++l;
                    }
                    nhood.M = l;
                }
                BOOST_VERIFY(nhood.M > 0);
                nhood.radiusM = nhood.pool[nhood.M - 1].dist;
            }
#pragma omp parallel for
            for (unsigned n = 0; n < N; ++n)
            {
                auto &nhood = nhoods[n];
                //cout<<nhood.L<<endl;
                auto &nn_new = nhood.nn_new;
                auto &nn_old = nhood.nn_old;
                for (unsigned l = 0; l < nhood.M; ++l)
                {
                    auto &nn = nhood.pool[l];
                    auto &nhood_o = nhoods[nn.id]; // nhood on the other side of the edge
                    if (nn.flag)
                    {
                        nn_new.push_back(nn.id);
                        if (nn.dist > nhood_o.radiusM)
                        {
                            LockGuard guard(nhood_o.lock);
                            nhood_o.rnn_new.push_back(n);
                        }
                        nn.flag = false;
                    }
                    else
                    {
                        nn_old.push_back(nn.id);
                        if (nn.dist > nhood_o.radiusM)
                        {
                            LockGuard guard(nhood_o.lock);
                            nhood_o.rnn_old.push_back(n);
                        }
                    }
                }
            }
            for (unsigned i = 0; i < N; ++i)
            {
                auto &nn_new = nhoods[i].nn_new;
                auto &nn_old = nhoods[i].nn_old;
                auto &rnn_new = nhoods[i].rnn_new;
                auto &rnn_old = nhoods[i].rnn_old;
                if (params.R && (rnn_new.size() > params.R))
                {
                    random_shuffle(rnn_new.begin(), rnn_new.end());
                    rnn_new.resize(params.R);
                }
                nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
                if (params.R && (rnn_old.size() > params.R))
                {
                    random_shuffle(rnn_old.begin(), rnn_old.end());
                    rnn_old.resize(params.R);
                }
                nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            }
        }

    public:
        unsigned loopcount = 0;
        vector<double> prune_time;
        vector<double> inter_time;
        vector<double> tree_grow_time;
        vector<double> buildnsg_time;
        vector<double> con_time;
        vector<double> search_time;
        vector<float> prune_cc;
        vector<float> search_cc;
        vector<float> allrecall;
        vector<double> kgraph_time;
        double final_prune_time=-1.0;
        unsigned AOD;
        int PG_type=-1;

        boost::dynamic_bitset<> pruneAgain;
        virtual void init_nhoods();

        virtual unsigned find_init_point();
        virtual void ComputeEp();
        virtual void nsg_Link();

        virtual void SaveNsg(const char *filename);
        virtual void SaveResult(const char *filename);

        virtual void get_neighbors(const float *query, unsigned init_point, unsigned L, std::vector<Neighbor> &pool);
        virtual void get_neighbors(unsigned query, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset);
        virtual unsigned parallel_InsertIntoPool(vector<Neighbor> &pool, unsigned &L, unsigned id,float dist, float &radius, Lock &lock);
        //virtual unsigned parallel_InsertIntoPool(vector<Neighbor> &pool, unsigned &L, unsigned id,float dist, float &radius, Lock &lock);

        virtual void tree_grow();
        virtual void findroot(boost::dynamic_bitset<> &flag, unsigned &root,unsigned &start);
        virtual void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
        virtual float nsg_sync_prune(unsigned q);
        virtual float nsg_sync_prune2(unsigned q);
        virtual void nsg_reversed_prune(unsigned q);
        virtual unsigned InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks);
        virtual float poolSearch(unsigned query,boost::dynamic_bitset<> &flags);
        virtual unsigned epSearch(unsigned query,vector<Neighbor> &retset,boost::dynamic_bitset<> &flags);
        virtual void buildNSG();
        virtual void buildHNSWLayer();
        virtual void inSearch();
        virtual unsigned BridgeView();
        virtual unsigned BridgeView_init();
        virtual unsigned BridgeView_update(bool flag);
        virtual float BridgeView_join();
        virtual unsigned BridgeView_clear();
        virtual unsigned CagraComputeRoute(vector<vector<unsigned>> &indexes,unsigned prune_K);
        virtual unsigned CagraReverse(vector<vector<unsigned>> &indexes,vector<vector<RankNeighbor>> &regra);
        virtual unsigned CagraMerge(vector<vector<unsigned>> &indexes,vector<vector<RankNeighbor>> &regra);
        virtual unsigned BuildCagra();
        virtual float taumg_sync_prune(unsigned q);
        virtual float taumg_sync_prune2(unsigned q);
        virtual void taumg_reversed_prune(unsigned q);
        virtual void taumg_Link();
        virtual void buildtauMG();
        virtual float alphapg_sync_prune(unsigned q);
        virtual float alphapg_sync_prune2(unsigned q);
        virtual void alphapg_reversed_prune(unsigned q);
        virtual void alphapg_Link();
        virtual void buildAlphaPG();
        virtual unsigned InsertKNN();
        virtual void UpdateNhoods();
        virtual void SwapHNSWLayer(char **&linkLists_,unsigned level);
        virtual void SwapHNSWLayer0(char *&data_level0_memory_);
        virtual unsigned getHNSWEp();
        virtual unsigned getLoop_i();
        virtual void final_prune();
        #ifdef DEBUG
        virtual void CalculateNSG();
        virtual void SwapPGraph();
        virtual void swapResults(std::vector<std::vector<int>> &knng, unsigned K);
        virtual float evaluate();
        #endif


        KGraphConstructor(IndexOracle const &o, IndexParams const &p, IndexInfo *r)
            : oracle(o), params(p), pinfo(r), nhoods(o.size()), n_comps(0)
        {
            no_dist = false;
            boost::timer::cpu_timer timer;
            // params.check();
            
            unsigned N = oracle.size();
            if (N <= params.controls)
            {
                cerr << "Warning: small dataset, shrinking control size to " << (N-1) << "." << endl;
                params.controls = N-1;
            }
            if (N <= params.L)
            {
                cerr << "Warning: small dataset, shrinking L to " << (N - 1) << "." << endl;
                params.L = N - 1;
            }
            if (N <= params.S)
            {
                cerr << "Warning: small dataset, shrinking S to " << (N - 1) << "." << endl;
                params.S = N - 1;
            }
            if (params.iterations < 0)
            {
                cout<<params.iterations<<endl;
                printf("The iterations must be larger than 0.\n");
                exit(-1);
            }
            if (N <= params.search_L)
            {
                cerr << "Warning: small dataset, shrinking search_L to " << (N - 1) << "." << endl;
                params.search_L = N - 1;
            }
            if (N <= params.search_K)
            {
                cerr << "Warning: small dataset, shrinking search_K to " << (N - 1) << "." << endl;
                params.search_K = N - 1;
            }
            if (N <= params.K)
            {
                cerr << "Warning: small dataset, shrinking K to " << (N - 1) << "." << endl;
                params.K = N - 1;
                params.loop_i=0;
                params.S=N-1;
                params.iterations=0;
            }
            // Here, we do not need such controlling mechanisms.
            //vector<Control> controls;
            MAXL=0;
            if(params.L>MAXL)   MAXL=params.L;
            if(params.K>MAXL)   MAXL=params.K;
            if(params.search_L>MAXL)   MAXL=params.search_L;
            if(params.search_K>MAXL)   MAXL=params.search_K;
            if (verbosity > 0)
                cout << "Generating control..." << endl;
            GenerateControl(oracle, params.controls, MAXL, &controls);
            if (verbosity > 0)
                cout << "Initializing..." << endl;

            cout<<"K: "<<params.K<<endl;
            // initialize nhoods

            // if(params.iterations!=0)
            // {
            //     init();
            // }
            // else
            // {
            //     init2();
            // }
            init();

            // iterate until converge
            float total = N * float(N - 1) / 2;
            IndexInfo info;
            info.stop_condition = IndexInfo::ITERATION;
            info.recall = 0;
            info.accuracy = numeric_limits<float>::max();
            info.cost = 0;
            info.iterations = 0;
            info.delta = 1.0;

            // for (unsigned it = 0; (params.iterations <= 0) || (it < params.iterations); ++it) {
            for (unsigned it = 0; it < params.iterations; ++it)
            {
                ++info.iterations;
                join();
                {
                    info.cost = n_comps / total;
                    accumulator_set<float, stats<tag::mean>> one_exact;
                    accumulator_set<float, stats<tag::mean>> one_approx;
                    accumulator_set<float, stats<tag::mean>> one_recall;
                    accumulator_set<float, stats<tag::mean>> recall;
                    accumulator_set<float, stats<tag::mean>> accuracy;
                    accumulator_set<float, stats<tag::mean>> M;
                    accumulator_set<float, stats<tag::mean>> delta;
                    for (auto const &nhood : nhoods)
                    {
                        M(nhood.M);
                        delta(EvaluateDelta(nhood.pool, params.K));
                    }
                    for (auto const &c : controls)
                    {
                        one_approx(nhoods[c.id].pool[0].dist);
                        one_exact(c.neighbors[0].dist);
                        one_recall(EvaluateOneRecall(nhoods[c.id].pool, c.neighbors));
                        recall(EvaluateRecall(nhoods[c.id].pool, nhoods[c.id].L, c.neighbors,params.K));
                        accuracy(EvaluateAccuracy(nhoods[c.id].pool, c.neighbors));
                    }
                    info.delta = mean(delta);
                    info.recall = mean(recall);
                    info.accuracy = mean(accuracy);
                    info.M = mean(M);
                    auto times = timer.elapsed();
                    kgraph_time.push_back(times.wall/1e9);
                    allrecall.push_back(info.recall);
                    if (verbosity > 0)
                    {
                        cout << "iteration: " << info.iterations
                             << " recall: " << info.recall
                             << " accuracy: " << info.accuracy
                             << " cost: " << info.cost
                             << " M: " << info.M
                             << " delta: " << info.delta
                             << " time: " << times.wall / 1e9
                             << " one-recall: " << mean(one_recall)
                             << " one-ratio: " << mean(one_approx) / mean(one_exact)
                             << endl;
                    }
                }
                if (info.delta <= params.delta)
                {
                    info.stop_condition = IndexInfo::DELTA;
                    break;
                }
                if (info.recall >= params.recall)
                {
                    info.stop_condition = IndexInfo::RECALL;
                    break;
                }
                if (info.iterations >= params.iterations)
                {   
                    cout<<"kgraph iter end"<<endl;
                    break;
                }
                update();
            }
            // M.resize(N);
            // graph.resize(N);
            // if (params.prune > 2)
            //     throw runtime_error("prune level not supported.");
            // for (unsigned n = 0; n < N; ++n)
            // {
            //     auto &knn = graph[n];
            //     M[n] = nhoods[n].M;
            //     auto const &pool = nhoods[n].pool;
            //     unsigned K = params.K;
            //     knn.resize(K);
            //     for (unsigned k = 0; k < K; ++k)
            //     {
            //         knn[k].id = pool[k].id;
            //         knn[k].dist = pool[k].dist;
            //     }
            // }
            // nhoods.clear();
            // if (params.reverse)
            // {
            //     reverse(params.reverse);
            // }
            // if (params.prune)
            // {
            //     prune(o, params.prune);
            // }
            // if (pinfo)
            // {
            //     *pinfo = info;
            // }
        }
    };

    void KGraphConstructor::init_nhoods() {
        unsigned nd_ = oracle.size();
        unsigned range = params.nsg_R;
        for(unsigned i = 0; i < nd_; i++) {
            auto &nhood=nhoods[i];
            vector<unsigned>().swap(nhood.nn_new);
            vector<unsigned>().swap(nhood.nn_old);
            vector<unsigned>().swap(nhood.rnn_new);
            vector<unsigned>().swap(nhood.rnn_old);
            nhood.nn_nsg.reserve(range/2);
            nhood.pool.resize(MAXL+1);
        }
//         unsigned seed = params.seed;
//         mt19937 rng(seed);
//         #pragma omp parallel
//         {
// #ifdef _OPENMP
//             mt19937 rng(seed ^ omp_get_thread_num());
// #else
//             mt19937 rng(seed);
// #endif
// #pragma omp for
//             for (unsigned i = 0; i < nd_; ++i)
//             {
//                 auto &nhood = nhoods[i];
//                 while(nhood.L<params.L)
//                 {
//                     unsigned id=rng()%nd_;
//                     if(id==i) continue;
//                     parallel_InsertIntoPool(nhood.pool,nhood.L,id,oracle(i,id),nhood.radius,nhood.lock);
//                 }
//             }
//         }

    }

    unsigned KGraphConstructor::find_init_point() {
        unsigned nd_ = oracle.size();
        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned root = rand() % nd_;
        unsigned unlinked_cnt = 0;
        unsigned maxnum = 0;
        unsigned res;
        while(unlinked_cnt < nd_) {
            unsigned pre_unlinked_cnt = unlinked_cnt;
            DFS(flags, root, unlinked_cnt);
            maxnum = std::max(maxnum, unlinked_cnt - pre_unlinked_cnt);
            if(maxnum == unlinked_cnt - pre_unlinked_cnt)
            {
                res = root;
            }
            if(unlinked_cnt < nd_)
            {
                for (unsigned i = 0; i < nd_; i++) 
                {
                    if (flags[i] == false)
                    {
                        root = i;
                        break;
                    }
                }
            }
        }
        return res;
    }

    void KGraphConstructor::ComputeEp() {
        unsigned nd_ = oracle.size();
        float *center = nullptr;
        center = oracle.Calcenter();
        std::vector<Neighbor> pool;
        unsigned init_point = rand()% nd_;
        get_neighbors(center, init_point, params.search_L, pool);
        ep_ = pool[0].id;
    }


    unsigned KGraphConstructor::parallel_InsertIntoPool(vector<Neighbor> &pool, unsigned &L, unsigned id,float dist, float &radius, Lock &lock)
    {
        if (dist > radius)
        {
            return pool.size();
        }
        LockGuard guard(lock);
        unsigned l = InsertIntoPool(pool.data(), L, Neighbor(true,id, dist, true));
        if (l <= L)
        { // inserted
            if (L + 1 < pool.size())
            { // if l == L + 1, there's a duplicate
                ++L;
            }
            else
            {
                radius = pool[L - 1].dist;
            }
        }
        return l;
    }
    void KGraphConstructor::get_neighbors(const float *query, unsigned init_point, unsigned L, std::vector<Neighbor> &pool) {
        unsigned nd_ = oracle.size();
        pool.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < nhoods[init_point].pool.size(); i++)
        {
            init_ids[i] = nhoods[init_point].pool[i].id;
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size())
        {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;

        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            if (id >= nd_)
                continue;
            float dist = oracle(query, id);
            pool[i] = Neighbor(id, dist, true);
            L++;
        }
        std::sort(pool.begin(), pool.begin() + L);
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (pool[k].flag)
            {
                pool[k].flag = false;
                unsigned n = pool[k].id;
                for (unsigned m = 0; m < nhoods[n].nn_nsg.size(); ++m)
                {
                    unsigned id = nhoods[n].nn_nsg[m].id;
                    if (flags[id])
                        continue;
                    flags[id] = 1;
                    float dist = oracle(query, id);
                    if (dist >= pool[L - 1].dist)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(pool.data(), L, nn);

                    if (L + 1 < pool.size())
                        ++L;
                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    void KGraphConstructor::get_neighbors(unsigned query, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset) {
        unsigned L = params.search_L; 
        unsigned nd_ = oracle.size();
        unsigned dimension_ = oracle.dim();

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L); 
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < nhoods[ep_].L; i++)
        {                                         
            init_ids[i] = nhoods[ep_].pool[i].id; 
            flags[init_ids[i]] = true;            
            L++;
        }
        while (L < init_ids.size())
        { 
            unsigned id = rand() % nd_;
            if (flags[id])
                continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++)
        {                              
            unsigned id = init_ids[i]; 
            if (id >= nd_)
                continue;
            // std::cout<<id<<std::endl;
            float dist = oracle(query, id);
            // distance_->compare(data_ + dimension_ * (size_t)id, query,
            //                               (unsigned)dimension_); 
            retset[i] = Neighbor(id, dist, true);
            fullset.push_back(retset[i]);
            // flags[id] = 1;
            L++;
        }
        std::sort(retset.begin(), retset.begin() + L); 
        int k = 0;
        while (k < (int)L)
        { 
            int nk = L;

            if (retset[k].flag)
            {
                retset[k].flag = false;
                unsigned n = retset[k].id; 

                for (unsigned m = 0; m < nhoods[n].L; ++m)
                { 
                    unsigned id = nhoods[n].pool[m].id;
                    if (flags[id])
                        continue;
                    flags[id] = 1;

                    float dist = oracle(query, id);
                    // distance_->compare(query, data_ + dimension_ * (size_t)id,
                    //                               (unsigned)dimension_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].dist)
                        continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size())
                        ++L;
                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    void KGraphConstructor::SaveNsg(const char *filename) {
        unsigned nd_ = oracle.size();
        std::ofstream out(filename, std::ios::binary | std::ios::out);
        assert(nhoods.size() == nd_);

        out.write((char *)&width, sizeof(unsigned));
        out.write((char *)&ep_, sizeof(unsigned));
        for (unsigned i = 0; i < nd_; i++)
        {
            unsigned GK = nhoods[i].nn_nsg.size();
            out.write((char *)&GK, sizeof(unsigned));
            vector<unsigned> res;
            res.resize(GK);
            for (int j = 0; j < GK; j++)
            {
                res[j] = nhoods[i].nn_nsg[j].id;
            }
            out.write((char *)res.data(), GK * sizeof(unsigned));
        }
        out.close();
    }
    void KGraphConstructor::SaveResult(const char *filename) {
        ofstream ofs(filename,std::ios::out | std::ios::app);
        if (ofs.is_open()) 
        {
        //     vector<double> prune_time;
        // vector<double> inter_time;
        // vector<double> tree_grow_time;
        // vector<double> buildnsg_time;
        // vector<double> con_time;
        // vector<double> search_time;
        // vector<float> prune_cc;
        // vector<float> search_cc;
        // double kgraph_time;
            ofs<<"PG_type,K,L,S,R,I,searchL,searchK,nsgR,massqS,tau,angle,step"<<endl;
            ofs<<PG_type<<","<<params.K<<","<<params.L<<","<<params.S<<","<<params.R<<","<<params.iterations<<","<<params.search_L<<","
            <<params.search_K<<","<<params.nsg_R<<","<<params.massq_S<<","<<params.tau<<","<<params.angle<<","<<params.step<<","<<endl;
            ofs<<"kgraph time"<<",";
            for(unsigned i=0;i<kgraph_time.size();i++)
            {
                ofs<<kgraph_time[i]<<",";
            }
            ofs<<endl;
            ofs<<"recall"<<",";
            for(unsigned i=0;i<allrecall.size();i++)
            {
                ofs<<allrecall[i]<<",";
            }
            ofs<<endl;
            ofs<<"prune time"<<",";
            for(unsigned i=0;i<prune_time.size();i++)
            {
                ofs<<prune_time[i]<<",";
            }
            ofs<<endl;
            ofs<<"inter time"<<",";
            for(unsigned i=0;i<inter_time.size();i++)
            {
                ofs<<inter_time[i]<<",";
            }
            ofs<<endl;
            ofs<<"tree_grow time"<<",";
            for(unsigned i=0;i<tree_grow_time.size();i++)
            {
                ofs<<tree_grow_time[i]<<",";
            }
            ofs<<endl;
            ofs<<"buildnsg time"<<",";
            for(unsigned i=0;i<buildnsg_time.size();i++)
            {
                ofs<<buildnsg_time[i]<<",";
            }
            ofs<<endl;
            ofs<<"search time"<<",";
            for(unsigned i=0;i<search_time.size();i++)
            {
                ofs<<search_time[i]<<",";
            }
            ofs<<endl;
            ofs<<"con time"<<",";
            for(unsigned i=0;i<con_time.size();i++)
            {
                ofs<<con_time[i]<<",";
            }
            ofs<<endl;
            ofs<<"prune cc"<<",";
            for(unsigned i=0;i<prune_cc.size();i++)
            {
                ofs<<prune_cc[i]<<",";
            }
            ofs<<endl;
            ofs<<"search cc"<<",";
            for(unsigned i=0;i<search_cc.size();i++)
            {
                ofs<<search_cc[i]<<",";
            }
            ofs<<endl;
            ofs<<"AOD"<<","<<AOD;
            ofs<<endl;
            ofs<<"final_prune_time"<<","<<final_prune_time;
            ofs<<endl;
            ofs.close();
        }
    }

    float KGraphConstructor::nsg_sync_prune(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        width = range; 
        unsigned cc=0;
        unsigned start = 0;
        nhoods[q].nn_nsg.clear();
        if (nhoods[q].pool[start].id == q)
        {
            start++;
        }
        nhoods[q].nn_nsg.push_back(nhoods[q].pool[start]);
        while (nhoods[q].nn_nsg.size() < range && (++start) < nhoods[q].L && start < maxc) {
            auto &p = nhoods[q].pool[start];
            if(p.id>=oracle.size()) continue;
            bool occlude = false;
            for (unsigned t = 0; t < nhoods[q].nn_nsg.size(); t++)
            {
                if (p.id == nhoods[q].nn_nsg[t].id)
                {
                    occlude = true;
                    break;
                }
                cc++;
                float djk = oracle(nhoods[q].nn_nsg[t].id, p.id);
                if (djk < p.dist)
                {
                    occlude = true;
                    p.pid=t;
                    break;
                }
            }
            if (!occlude)
            {
                nhoods[q].nn_nsg.push_back(p);
            }
        }
        for(unsigned i=0;i<nhoods[q].nn_nsg.size();i++)
        {
            nhoods[q].nn_nsg[i].flag=false;
        }
        return float(cc)/float(oracle.size()*float(oracle.size()-1)/2);
    }

    float KGraphConstructor::nsg_sync_prune2(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        width = range;
        unsigned cc=0;
        unsigned start = 0;
        auto &nhood=nhoods[q];
        vector<unsigned> temp_nsg;
        unsigned pass=0;
        for(unsigned i=0;i<nhood.nn_nsg.size();i++)
        {
            if(!nhood.nn_nsg[i].flag)
            {
                temp_nsg.push_back(nhood.nn_nsg[i].id);
            }
        }
        nhood.nn_nsg.clear();
        if (nhood.pool[start].id == q)
        {
            start++;
        }
        nhood.nn_nsg.push_back(nhood.pool[start]);
        if(find(temp_nsg.begin(),temp_nsg.end(),nhood.nn_nsg.back().id)==temp_nsg.end())
        {
            nhood.nn_nsg.back().isnew=true;
        }
        else
        {
            nhood.nn_nsg.back().isnew=false;
        }
        while (nhood.nn_nsg.size() < range && (++start) < nhood.L && start < maxc) {
            auto &p = nhood.pool[start];
            if(p.id>=oracle.size()) continue;
            bool occlude = false;
            if(p.isnew)
            {
                for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                {
                    if (p.id == nhood.nn_nsg[t].id)
                    {
                        occlude = true;
                        break;
                    }
                    cc++;
                    float djk = oracle(nhood.nn_nsg[t].id, p.id);
                    if (djk < p.dist)
                    {
                        occlude = true;
                        p.pid=t;
                        break;
                    }
                }
                if (!occlude)
                {
                    nhood.nn_nsg.push_back(p);
                    nhood.nn_nsg.back().isnew=true;
                }
            }
            else
            {
                if(find(temp_nsg.begin(),temp_nsg.end(),p.id)==temp_nsg.end()&&((p.pid>=0)&&(p.pid<temp_nsg.size())))
                {
                    unsigned prune_p=temp_nsg[p.pid];
                    auto iter=find_if(nhood.nn_nsg.begin(),nhood.nn_nsg.end(),[prune_p](Neighbor neighbor)
                    {
                        return neighbor.id==prune_p;
                    });
                    if(iter!=nhood.nn_nsg.end())
                    {
                        pass++;
                        p.pid=distance(nhood.nn_nsg.begin(),iter);
                        continue;
                    }
                    else
                    {
                        for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                        {
                            if (p.id == nhood.nn_nsg[t].id)
                            {
                                occlude = true;
                                break;
                            }
                            cc++;
                            float djk = oracle(nhood.nn_nsg[t].id, p.id);
                            if (djk < p.dist)
                            {
                                occlude = true;
                                p.pid=t;
                                break;
                            }
                        }
                        if (!occlude)
                        {
                            nhood.nn_nsg.push_back(p);
                            nhood.nn_nsg.back().isnew=true;
                        }
                    }
                }
                else if(find(temp_nsg.begin(),temp_nsg.end(),p.id)==temp_nsg.end()&&((p.pid<0)||(p.pid>=temp_nsg.size())))
                {
                    for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                    {
                        if (p.id == nhood.nn_nsg[t].id)
                        {
                            occlude = true;
                            break;
                        }
                        cc++;
                        float djk = oracle(nhood.nn_nsg[t].id, p.id);
                        if (djk < p.dist)
                        {
                            occlude = true;
                            p.pid=t;
                            break;
                        }
                    }
                    if (!occlude)
                    {
                        nhood.nn_nsg.push_back(p);
                        nhood.nn_nsg.back().isnew=true;
                    }
                }
                else 
                {
                    for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                    {
                        if (p.id == nhood.nn_nsg[t].id)
                        {
                            occlude = true;
                            break;
                        }
                        if(!nhood.nn_nsg[t].isnew)
                        {
                            continue;
                        }
                        cc++;
                        float djk = oracle(nhood.nn_nsg[t].id, p.id);
                        if (djk < p.dist)
                        {
                            occlude = true;
                            p.pid=t;
                            break;
                        }
                    }
                    if (!occlude)
                    {
                        nhood.nn_nsg.push_back(p);
                        nhood.nn_nsg.back().isnew=false;
                    }
                }
            }
        }
        for(unsigned i=0;i<nhoods[q].nn_nsg.size();i++)
        {
            nhoods[q].nn_nsg[i].flag=false;
        }
        return float(cc)/float(oracle.size()*float(oracle.size()-1)/2);
    }

    void KGraphConstructor::nsg_reversed_prune(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        width = range;
        unsigned start = 0;
        auto temp_pool=nhoods[q].nn_nsg;
        //cout<<"     "<<temp_pool.size()<<"   "<<endl;
        sort(temp_pool.begin(),temp_pool.end());
        nhoods[q].nn_nsg.clear();  
        if (temp_pool[start].id == q)
        {
            start++;
        }
        nhoods[q].nn_nsg.push_back(temp_pool[start]); 
        while (nhoods[q].nn_nsg.size() < range && (++start) < temp_pool.size()) {
            auto &p = temp_pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < nhoods[q].nn_nsg.size(); t++)
            {
                if (p.id == nhoods[q].nn_nsg[t].id)
                {
                    occlude = true;
                    break;
                }
                float djk = oracle(nhoods[q].nn_nsg[t].id, p.id);
                if (djk < p.dist)
                {
                    occlude = true;
                    break;
                }
            }
            if (!occlude)
            {
                nhoods[q].nn_nsg.push_back(p);
            }
        }
    }


    unsigned KGraphConstructor::InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks) {

        for (size_t i = 0; i < nhoods[n].nn_nsg.size(); i++)
        {
            size_t des = nhoods[n].nn_nsg[i].id;
            if(des>=oracle.size())
            {
                continue;
            }
            //cout<<des<<endl;
            int dup = 0;
            {
                LockGuard1 guard(locks[des]);
                for (size_t j = 0; j < nhoods[des].nn_nsg.size(); j++)
                {
                    if (n == nhoods[des].nn_nsg[j].id)
                    {
                        dup = 1;
                        break;
                    }
                }
            }
            if (dup)
            {
                continue;
            }
            Neighbor nn(n,oracle(n,des), true);
            {
                LockGuard1 guard(locks[des]);
                nhoods[des].nn_nsg.push_back(nn);
            }
        }
        return 0;
    }


    void KGraphConstructor::nsg_Link()
    {
        cout<<"Link Begin"<<endl;
        unsigned range = params.nsg_R;
        unsigned nd_ = oracle.size();
        std::vector<std::mutex> locks(nd_);
        float scan=0;
        unsigned cc=0;
        auto s_prune = std::chrono::high_resolution_clock::now();
        unsigned pass=0;
#pragma omp parallel
{    
#pragma omp for schedule(dynamic, 100) reduction(+ : scan)
        for (unsigned n = 0; n < nd_; ++n)
        {
            if(pruneAgain[n])
            {
                scan+=nsg_sync_prune(n);
            }
            else
            {
                scan+=nsg_sync_prune2(n);
            }
            //  scan+=nsg_sync_prune(n);
        }
      
}

        prune_cc.push_back(scan);
        auto e_prune = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_prune = e_prune - s_prune;
        double Time_prune = diff_prune.count();   
        prune_time.push_back(Time_prune);
        printf("prune end. time: %f\n", Time_prune);
        cout<<"interinsert begin"<<endl;
        auto s_inter = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n)
        {
            cc+=InterInsert(n, range, locks);
        }
        pruneAgain.reset();
#pragma omp parallel for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n)
        {
            if(nhoods[n].nn_nsg.size()>range)
            {
                pruneAgain[n]=1;
                nsg_reversed_prune(n);
            }
        }
        auto e_inter = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_inter = e_inter - s_inter;
        double Time_inter = diff_inter.count();   
        inter_time.push_back(Time_inter);
        printf("InterInsert end. time: %f\n", Time_inter);
    }

    void KGraphConstructor::tree_grow() {
        cout<<"Tree grow begin"<<endl;
        auto s_inter = std::chrono::high_resolution_clock::now();
        unsigned nd_ = oracle.size();
        unsigned root = ep_;
        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned unlinked_cnt = 0;
        unsigned start=0;
        while (unlinked_cnt < nd_)
        {
            DFS(flags, root, unlinked_cnt);
            if (unlinked_cnt >= nd_)
                break;
            findroot(flags, root,start);
        }
        for (size_t i = 0; i < nd_; ++i)
        {
            if (nhoods[i].nn_nsg.size() > width)
            {
                width = nhoods[i].nn_nsg.size();
            }
        }
        auto e_inter = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_inter = e_inter - s_inter;
        double Time_inter = diff_inter.count(); 
        tree_grow_time.push_back(Time_inter);  
        printf("Tree grow end. time: %f\n", Time_inter);
    }

    void KGraphConstructor::findroot(boost::dynamic_bitset<> &flag, unsigned &root,unsigned &start) {
        unsigned nd_ = oracle.size();
        unsigned id = nd_;
        for (unsigned i = start; i < nd_; i++) 
        {
            if (flag[i] == false)
            {
                id = i;
                start=i;
                break;
            }
        }
        if (id == nd_)
            return;

        // std::vector<Neighbor> tmp, pool;
        // boost::dynamic_bitset<> flag_00(nd_, 0);
        // get_neighbors(id, flag_00, tmp, pool); 
        // std::sort(pool.begin(), pool.end());
        // unsigned found = 0;
        // for (unsigned i = 0; i < pool.size(); i++)
        // {
        //     if (flag[pool[i].id])
        //     {
        //         root = pool[i].id;
        //         found = 1; 
        //         break;
        //     }
        // }

        unsigned found = 0;
        for (unsigned i = 0; i < nhoods[id].L; i++)
        {
            unsigned id=nhoods[id].pool[i].id;
            if (flag[id]&&nhoods[id].nn_nsg.size()<params.nsg_R)
            {
                root = id;
                found = 1; 
                break;
            }
        }

        if (found == 0)  
        {
            unsigned temp_count = 0;
            int factor = 1;
            while (true)
            {
                temp_count++;
                factor = int(temp_count / nd_) + 1;
                unsigned rid = rand() % nd_;  
                if (flag[rid]&&nhoods[rid].nn_nsg.size()<params.nsg_R*factor)
                {
                    root = rid;
                    break;
                }
            }
        }
        
 
        nhoods[root].nn_nsg.push_back(Neighbor(id,oracle(root,id),true));

    }

    void KGraphConstructor::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {  
        unsigned nd_ = oracle.size();
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root])
            cnt++;
        flag[root] = true;
        while (!s.empty())
        {
            unsigned next = nd_ + 1;
            for (unsigned i = 0; i < nhoods[tmp].nn_nsg.size(); i++)  
            {
                if (flag[nhoods[tmp].nn_nsg[i].id] == false)
                { //
                    next = nhoods[tmp].nn_nsg[i].id;
                    break;
                }
            }
            if (next == (nd_ + 1))
            {
                s.pop();
                if (s.empty())
                    break;
                tmp = s.top();
                    continue;
            }
            tmp = next;
            flag[tmp] = true;
            s.push(tmp);
            cnt++;
        }
    }

#ifdef DEBUG  //CalculateNSG
    void KGraphConstructor::CalculateNSG()
    {
        cout<<"start compute AOD"<<endl;
        unsigned nd_ = oracle.size();
        unsigned max = 0, min = 1e6, avg = 0;
        for (size_t i = 0; i < nd_; i++)
        {
            unsigned size = 0;
            for (unsigned j = 0; j < nhoods[i].nn_nsg.size(); j++)
            {
                size++;
            }
            max = max < size ? size : max;
            min = min > size ? size : min;
            avg += size;
        }

        avg /= 1.0 * nd_;
        AOD=avg;
        printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);
    }
#endif

    void KGraphConstructor::SwapPGraph()
    {
        unsigned nd_ = oracle.size();
        temp_graph.resize(nd_);
        for(int i = 0; i < nd_; i++) {
            auto &nhood=nhoods[i];
            temp_graph[i].resize(nhood.nn_nsg.size());
            for(unsigned j=0;j<nhood.nn_nsg.size();j++)
            {
                unsigned id = nhood.nn_nsg[j].id;
                temp_graph[i][j] = id;
            }
        }
        nhoods.clear();
        vector<Nhood>().swap(nhoods);
    }


    void KGraphConstructor::buildNSG() {
        nsg_Link();
        // if(loopcount==0)
        // {
        //     ComputeEp();
        //     cout<<ep_<<endl;
        // }
        ComputeEp();
        cout<<"ep: "<<ep_<<endl;
        tree_grow();
        UpdateNhoods();
    }
    void KGraphConstructor::buildHNSWLayer() {
        nsg_Link();
        // if(loopcount==0)
        // {
        //     ComputeEp();
        //     cout<<ep_<<endl;
        // }
        ComputeEp();
        cout<<"ep: "<<ep_<<endl;
        //tree_grow();
        UpdateNhoods();
    }
    

    float KGraphConstructor::poolSearch(unsigned query,boost::dynamic_bitset<> &flags) {
        unsigned num_break=0;
        unsigned dim_=oracle.dim();
        unsigned nd_ = oracle.size();
        nhoods[query].pool.resize(MAXL+1);
        //boost::dynamic_bitset<> flags{nd_, 0};
        //nhoods[query].radius=numeric_limits<float>::max();
        nhoods[query].radius=nhoods[query].pool.back().dist;
        size_t cc = 0;
        unsigned tmp_l = 0;
        unsigned cur_loop=loopcount-1;
        unsigned cur_L=min(params.search_L+cur_loop*params.step,MAXL);
        
        if(cur_loop==0)
        {
            for (;tmp_l < nhoods[query].L; tmp_l++) {
                //init_ids[tmp_l] = nhoods[query].pool[tmp_l].id;
                flags[nhoods[query].pool[tmp_l].id] = true;
                nhoods[query].pool[tmp_l].isnew=false;
                nhoods[query].pool[tmp_l].flag=true;
            }
        }
        else
        {
            unsigned pre_L=min(params.search_L+(cur_loop-1)*params.step,MAXL);
            for(;tmp_l<pre_L;tmp_l++)
            {
                flags[nhoods[query].pool[tmp_l].id] = true;
                nhoods[query].pool[tmp_l].isnew=false;
                nhoods[query].pool[tmp_l].flag=true;
            }
            for(;tmp_l<cur_L;tmp_l++)
            {
                flags[nhoods[query].pool[tmp_l].id] = true;
                auto isexpand=nhoods[query].pool[tmp_l].flag;
                nhoods[query].pool[tmp_l].isnew=isexpand;
                nhoods[query].pool[tmp_l].flag=true;
            }
            for(;tmp_l<nhoods[query].L;tmp_l++)
            {
                flags[nhoods[query].pool[tmp_l].id] = true;
                nhoods[query].pool[tmp_l].isnew=false;
                nhoods[query].pool[tmp_l].flag=true;
            }
        }
        int k = 0;
        while (k<cur_L) {
            int nk = cur_L;

            if (nhoods[query].pool[k].flag) 
            {
                nhoods[query].pool[k].flag = false;
                unsigned n = nhoods[query].pool[k].id;
                //cout<<nhoods[query].L<<endl;
                auto temp_flag=nhoods[query].pool[k].isnew;
                for (unsigned m = 0; m < nhoods[n].nn_nsg.size(); m++) 
                {
                    unsigned id = nhoods[n].nn_nsg[m].id;
                    if (flags[id]) continue;
                    if(id==query) continue;
                    // if(temp_flag==false&&nhoods[n].nn_nsg[m].isnew==false)    continue;
                    flags[id] = true;
                    cc++;
                    float dist = oracle(query, id);
                    int r=parallel_InsertIntoPool(nhoods[query].pool,nhoods[query].L,id,dist,nhoods[query].radius,nhoods[query].lock);
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        return (float)cc/(nd_*float(nd_-1)/2);
    }

    
    unsigned KGraphConstructor::epSearch(unsigned query,vector<Neighbor> &retset,boost::dynamic_bitset<> &flags)
    {
        unsigned nd_ = oracle.size();
        const unsigned L = params.search_L;
        nhoods[query].pool.resize(MAXL+1);
        // nhoods[query].radius=numeric_limits<float>::max();
        nhoods[query].radius=nhoods[query].pool.back().dist;
        size_t cc = 0;
        retset.resize(L + 1);

        
        unsigned tmp_l = 0;
        for (;tmp_l < L && tmp_l < nhoods[ep_].L; tmp_l++) {
            flags[nhoods[ep_].pool[tmp_l].id] = true;
            retset[tmp_l]=Neighbor(nhoods[ep_].pool[tmp_l].id,oracle(query,nhoods[ep_].pool[tmp_l].id),true);
            cc++;
        }
        while(tmp_l<L)
        {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            if(ep_==id) continue;
            flags[id] = true;
            float dist=oracle(query,id);
            cc++;
            retset[tmp_l]=Neighbor(id,dist,true);
            tmp_l++;
        }
        
        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k<L) {
            int nk = L;
            if (retset[k].flag) 
            {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < nhoods[n].nn_nsg.size(); m++) 
                {
                    unsigned id = nhoods[n].nn_nsg[m].id;
                    if (flags[id]) continue;
                    if(id==query) continue;
                    flags[id] = true;
                    cc++;
                    float dist = oracle(query, id);
                    
                    
                    parallel_InsertIntoPool(nhoods[query].pool,nhoods[query].L,id,dist,nhoods[query].radius,nhoods[query].lock);
                    if (dist >= retset[L - 1].dist) continue;
                    int r=InsertIntoPool(retset.data(),L,Neighbor(id,dist,true));
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        return cc;
    }

    unsigned KGraphConstructor::BridgeView()
    {
        float delta = 1.0;
        float scan_rate=0;
        unsigned iter=1;
        unsigned cnt=0;
        bool flag=true;
        unsigned nd_=oracle.size();
        unsigned S_=params.massq_S;
        BridgeView_init();
        while(true)
        {
            cnt = BridgeView_update(true);
            delta = (float) cnt/ nd_/S_;
            if(delta<0.5)
            {
                cnt+=BridgeView_update(false);
                delta = (float) cnt / nd_/S_;
                flag=false;
            }
            scan_rate+=BridgeView_join();
            iter++;
            if(!flag) break;
        }
        BridgeView_clear();
        search_cc.push_back(scan_rate);
        cout<<"bridge view scan_rate:  "<<scan_rate<<endl;
        return 0;
    }
    unsigned KGraphConstructor::BridgeView_init()
    {
        // cout<<"Bridge init begin"<<endl;
        unsigned nd_=oracle.size();
        unsigned S_=params.massq_S;
        unsigned L = MAXL;
        unsigned cur_loop=loopcount-1;
        unsigned cur_L=min(params.search_L+cur_loop*params.step,MAXL);
#pragma omp parallel for
        for (unsigned i=0;i<nd_;i++) {
            auto& nhood = nhoods[i];
            if(cur_loop==0)
            {
                for(unsigned j=0;j<nhood.L;j++)
                {
                    nhood.pool[j].flag=true;
                    nhood.pool[j].isnew=false;
                }
            }
            else
            {
                unsigned pre_L=min(params.search_L+(cur_loop-1)*params.step,MAXL);
                for(unsigned j=0;j<pre_L;j++)
                {
                    nhood.pool[j].flag=true;
                    nhood.pool[j].isnew=false;
                }
                for(unsigned j=pre_L;j<cur_L;j++)
                {
                    auto isexpand=nhood.pool[j].flag;
                    nhood.pool[j].isnew=isexpand;
                    nhood.pool[j].flag=true;
                }
                for(unsigned j=cur_L;j<nhood.L;j++)
                {
                    nhood.pool[j].flag=true;
                    nhood.pool[j].isnew=false;
                }
            }
            
            // for(unsigned j=min(params.search_L+temp2*40,MAXL);j<min(params.search_L+temp2*40,MAXL)+40&&j<nhood.L;j++)
            // {
            //     nhood.pool[j].isnew=true;
            // }
            nhood.pool.resize(L+1);
            //nhood.radius = numeric_limits<float>::max();
            nhood.radius=nhood.pool.back().dist;
            nhood.rnn_new.clear();
            nhood.rnn_new.resize(S_/2);
            nhood.rnn_new_flag.clear();
            nhood.rnn_new_flag.resize(S_);
        }
        // cout<<"Bridge init end"<<endl;
        return 0;
    }
    unsigned KGraphConstructor::BridgeView_update(bool flag)
    {
        // cout<<"Bridge update begin"<<endl;
        unsigned count=0;
        unsigned nd_=oracle.size();
        unsigned S_=params.massq_S;
        unsigned L=min(params.search_L+(loopcount-1)*params.step,MAXL);
        //unsigned L = params.search_L;
        if(flag)
        {
        #pragma omp parallel for
            for(unsigned i=0;i<nd_;i++)
            {
                auto &nhood = nhoods[i];
                nhood.rnn_new.clear();
                nhood.rnn_new_flag.clear();
            }
        }
        #pragma omp parallel for reduction(+:count)
        for(unsigned i=0;i<nd_;i++)
        {
            auto &nhood=nhoods[i];
            unsigned cnt=0;
            for(unsigned j=0;j<L&&j<nhood.L&&(cnt<S_||!flag);j++)
            {
                if(nhood.pool[j].flag){

                    unsigned id=nhood.pool[j].id;
                    {
                        LockGuard guard(nhoods[id].lock);
                        nhoods[id].rnn_new.push_back(i);
                        nhoods[id].rnn_new_flag.push_back(nhood.pool[j].isnew);
                    }
                    nhood.pool[j].flag=false;
                    cnt++;
                }
            }
            count+=cnt;
        }
        // cout<<"Bridge update end"<<endl;
        return count;
    }
    float KGraphConstructor::BridgeView_join()
    {
        // cout<<"Bridge join start"<<endl;
        unsigned cc=0;
        unsigned nd_=oracle.size();
        unsigned dim_=oracle.dim();
#pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+ : cc)
        for(unsigned i=0;i<nd_;i++)
        {
            unsigned q=nd_-i-1;
            auto &rnn_new=nhoods[q].rnn_new;
            auto &nhood=nhoods[q];
            auto &rnn_new_flag=nhoods[q].rnn_new_flag;
            auto &nn_nsg=nhoods[q].nn_nsg;
            if(rnn_new.empty())
            {
                continue;
            }
            // for(unsigned &id:rnn_old)
            // {
            //     for(unsigned j=0;j<nhood.nn_nsg.size();j++)
            //     {
            //         unsigned nsg_id=nhood.nn_nsg[j].id;
            //         if(id==nsg_id) continue;
            //         if(nhood.nn_nsg[j].isnew==false) continue;                    
            //         cc++;
            //         float dist=oracle(id,nsg_id);
            //         parallel_InsertIntoPool(nhoods[id].pool,nhoods[id].L,nsg_id,dist,nhoods[id].radius,nhoods[id].lock);
            //     }
            // }
            for(unsigned k=0;k<rnn_new.size();k++)
            {
                unsigned id=rnn_new[k];
                bool ifnew=rnn_new_flag[k];
                for(unsigned j=0;j<nn_nsg.size();j++)
                {
                    unsigned nsg_id=nn_nsg[j].id;
                    if(id==nsg_id) continue;
                    if(ifnew==false&&nn_nsg[j].isnew==false)   continue;
                    cc++;
                    float dist=oracle(id,nsg_id);
                    parallel_InsertIntoPool(nhoods[id].pool,nhoods[id].L,nsg_id,dist,nhoods[id].radius,nhoods[id].lock);
                }
                //cout<<q<<endl;
            }
        }
        // cout<<"Bridge join end"<<endl;
        return (float)cc/(nd_*float(nd_-1)/2);
    }

    unsigned KGraphConstructor::BridgeView_clear()
    {
        unsigned nd_ = oracle.size();
#pragma omp parallel for
        for (unsigned i=0;i<nd_;i++) {
            auto& nhood = nhoods[i];
            vector<unsigned>().swap(nhood.rnn_new);
        }
        return 0;
    }

    unsigned KGraphConstructor::CagraComputeRoute(vector<vector<unsigned>> &indexes,unsigned prune_K)
    {
        auto s_cagraRoute = std::chrono::high_resolution_clock::now();
        unsigned nd_=oracle.size();
        unsigned K=params.K;
        vector<unsigned> temp(K,0);
        vector<vector<unsigned>> twoHop(nd_,temp);
        unsigned cc=0;
// #pragma omp parallel for
//         for(unsigned i=0;i<nd_;i++)
//         {
//             auto &nhood=nhoods[i];
//             vector<unsigned> &route=twoHop[i];
//             route.resize(K);
//             for(unsigned j=0;j<K;j++)
//             {
//                 route[j]=0;
//             }
//         }
#pragma omp parallel for
        for(unsigned i=0;i<nd_;i++)
        {
            auto &nhood=nhoods[i];
            vector<unsigned> &route=twoHop[i];


            // for(unsigned j=0;j<K;j++)
            // {
            //     unsigned nid=nhood.pool[j].id;
            //     for(unsigned k=0;k<j;k++)
            //     {
            //         unsigned oneid=nhood.pool[k].id;
            //         for(unsigned l=0;l<j;l++)
            //         {
            //             if(nhoods[oneid].pool[l].id==nid)
            //             {
            //                 route[j]++;
            //                 break;
            //             }
            //         }
            //     }
            // }


            for(unsigned k=0;k<K;k++)
            {
                unsigned oneid=nhood.pool[k].id;
                for(unsigned l=0;l<K;l++)
                {
                    unsigned twoid=nhoods[oneid].pool[l].id;
                    for(unsigned j=max(k,l)+1;j<K;j++)
                    {
                        if(twoid==nhood.pool[j].id)
                        {
                            route[j]++;
                            break;
                        }
                    }
                }
            }


            auto index_=sort_indexes<unsigned>(route);  // 0 1 1 3 2       0 1 2 4 3     
                                                        // 0 1 2 3 4
            vector<unsigned> &index=indexes[i];
            index.resize(prune_K);
            for(unsigned j=0;j<prune_K;j++)
            {
                index[j]=index_[j];
            }

            
        }
        auto e_cagraRoute = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_cagraRoute = e_cagraRoute - s_cagraRoute;
        double Time_cagraRoute = diff_cagraRoute.count();
        printf("cagraRoute end, time: %f\n",Time_cagraRoute);
// #pragma omp parallel for
//         for(unsigned i=0;i<nd_;i++)
//         {
//             vector<unsigned> &route=twoHop[i];
//             auto index_=sort_indexes<unsigned>(route);  // 0 1 1 3 2       0 1 2 4 3     
//                                                         // 0 1 2 3 4
//             vector<unsigned> &index=indexes[i];
//             index.resize(prune_K);
//             for(unsigned j=0;j<prune_K;j++)
//             {
//                 index[j]=index_[j];
//             }
//         }
        
        return 0;
    }
    
    unsigned KGraphConstructor::CagraReverse(vector<vector<unsigned>> &indexes,vector<vector<RankNeighbor>> &regra)
    {
        unsigned nd_=oracle.size();
        std::vector<std::mutex> locks(nd_);
#pragma omp parallel for
        for(unsigned i=0;i<nd_;i++)
        {
            vector<unsigned> &index=indexes[i];
            for(unsigned j=0;j<index.size();j++)
            {
                unsigned id=nhoods[i].pool[index[j]].id;
                {
                    LockGuard1 guard(locks[id]);
                    regra[id].push_back(RankNeighbor(i,j));
                }
            }
        }
#pragma omp parallel for
        for(unsigned i=0;i<nd_;i++)
        {
            sort(regra[i].begin(),regra[i].end());
        }

        // for(unsigned i=0;i<regra[0].size();i++)
        // {
        //     cout<<regra[0][i].rank<<"  ";
        // }
        //cout<<endl;
        return 0;
    }

    unsigned KGraphConstructor::CagraMerge(vector<vector<unsigned>> &indexes,vector<vector<RankNeighbor>> &regra)
    {
        unsigned nd_=oracle.size();
#pragma omp parallel for
        for(unsigned i=0;i<nd_;i++)
        {
            auto &nhood=nhoods[i];
            nhood.nn_nsg.clear();
            auto &index=indexes[i];
            auto &iregra=regra[i];
            unsigned k=0,j=0;
            while(nhood.nn_nsg.size()<index.size())
            {
                if(k<index.size())
                {
                    nhood.nn_nsg.push_back(nhood.pool[index[k]]);
                    k++;
                }
                if(j<iregra.size())
                {
                    nhood.nn_nsg.push_back(Neighbor(iregra[j].id,oracle(i,iregra[j].id),true));
                    j++;
                }
            }
           // cout<<" "<<nhood.nn_nsg.size()<<"   ";
        }
        return 0;
    }
    unsigned KGraphConstructor::BuildCagra()
    {
        unsigned nd_=oracle.size();
        unsigned prune_K=params.K/2;
        vector<vector<unsigned>> indexes(nd_);  
        vector<vector<RankNeighbor>> regra(nd_);  // reversed graph
        CagraComputeRoute(indexes,prune_K);
        CagraReverse(indexes,regra);
        CagraMerge(indexes,regra);
        UpdateNhoods();
        return 0;
    }



    float KGraphConstructor::taumg_sync_prune(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        float tau_=params.tau;
        width = range; 
        unsigned cc=0;
        unsigned start = 0;
        nhoods[q].nn_nsg.clear();
        if (nhoods[q].pool[start].id == q)
        {
            start++;
        }
        nhoods[q].nn_nsg.push_back(nhoods[q].pool[start]);
        while (nhoods[q].nn_nsg.size() < range && (++start) < nhoods[q].L && start < maxc) {
            auto &p = nhoods[q].pool[start];
            if(p.id>=oracle.size()) continue;
            bool occlude = false;
            for (unsigned t = 0; t < nhoods[q].nn_nsg.size(); t++)
            {
                if (p.id == nhoods[q].nn_nsg[t].id)
                {
                    occlude = true;
                    break;
                }
                cc++;
                float djk = oracle(nhoods[q].nn_nsg[t].id, p.id);
                if (djk < p.dist-3*tau_)
                {
                    occlude = true;
                    p.pid=t;
                    break;
                }
            }
            if (!occlude)
            {
                nhoods[q].nn_nsg.push_back(p);
            }
        }
        for(unsigned i=0;i<nhoods[q].nn_nsg.size();i++)
        {
            nhoods[q].nn_nsg[i].flag=false;
        }
        return float(cc)/float(oracle.size()*float(oracle.size()-1)/2);
    }

    float KGraphConstructor::taumg_sync_prune2(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        float tau_=params.tau;
        width = range;
        unsigned cc=0;
        unsigned start = 0;
        auto &nhood=nhoods[q];
        vector<unsigned> temp_nsg;
        unsigned pass=0;
        for(unsigned i=0;i<nhood.nn_nsg.size();i++)
        {
            if(!nhood.nn_nsg[i].flag)
            {
                temp_nsg.push_back(nhood.nn_nsg[i].id);
            }
        }
        nhood.nn_nsg.clear();
        if (nhood.pool[start].id == q)
        {
            start++;
        }
        nhood.nn_nsg.push_back(nhood.pool[start]);
        if(find(temp_nsg.begin(),temp_nsg.end(),nhood.nn_nsg.back().id)==temp_nsg.end())
        {
            nhood.nn_nsg.back().isnew=true;
        }
        else
        {
            nhood.nn_nsg.back().isnew=false;
        }
        while (nhood.nn_nsg.size() < range && (++start) < nhood.L && start < maxc) {
            auto &p = nhood.pool[start];
            if(p.id>=oracle.size()) continue;
            bool occlude = false;
            if(p.isnew)
            {
                for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                {
                    if (p.id == nhood.nn_nsg[t].id)
                    {
                        occlude = true;
                        break;
                    }
                    cc++;
                    float djk = oracle(nhood.nn_nsg[t].id, p.id);
                    if (djk < p.dist-3*tau_)
                    {
                        occlude = true;
                        p.pid=t;
                        break;
                    }
                }
                if (!occlude)
                {
                    nhood.nn_nsg.push_back(p);
                    nhood.nn_nsg.back().isnew=true;
                }
            }
            else
            {
                if(find(temp_nsg.begin(),temp_nsg.end(),p.id)==temp_nsg.end()&&((p.pid>=0)&&(p.pid<temp_nsg.size())))
                {
                    unsigned prune_p=temp_nsg[p.pid];
                    auto iter=find_if(nhood.nn_nsg.begin(),nhood.nn_nsg.end(),[prune_p](Neighbor neighbor)
                    {
                        return neighbor.id==prune_p;
                    });
                    if(iter!=nhood.nn_nsg.end())
                    {
                        pass++;
                        p.pid=distance(nhood.nn_nsg.begin(),iter);
                        continue;
                    }
                    else
                    {
                        for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                        {
                            if (p.id == nhood.nn_nsg[t].id)
                            {
                                occlude = true;
                                break;
                            }
                            cc++;
                            float djk = oracle(nhood.nn_nsg[t].id, p.id);
                            if (djk < p.dist-3*tau_)
                            {
                                occlude = true;
                                p.pid=t;
                                break;
                            }
                        }
                        if (!occlude)
                        {
                            nhood.nn_nsg.push_back(p);
                            nhood.nn_nsg.back().isnew=true;
                        }
                    }
                }
                else if(find(temp_nsg.begin(),temp_nsg.end(),p.id)==temp_nsg.end()&&((p.pid<0)||(p.pid>=temp_nsg.size())))
                {
                    for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                    {
                        if (p.id == nhood.nn_nsg[t].id)
                        {
                            occlude = true;
                            break;
                        }
                        cc++;
                        float djk = oracle(nhood.nn_nsg[t].id, p.id);
                        if (djk < p.dist-3*tau_)
                        {
                            occlude = true;
                            p.pid=t;
                            break;
                        }
                    }
                    if (!occlude)
                    {
                        nhood.nn_nsg.push_back(p);
                        nhood.nn_nsg.back().isnew=true;
                    }
                }
                else 
                {
                    for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                    {
                        if (p.id == nhood.nn_nsg[t].id)
                        {
                            occlude = true;
                            break;
                        }
                        if(!nhood.nn_nsg[t].isnew)
                        {
                            continue;
                        }
                        cc++;
                        float djk = oracle(nhood.nn_nsg[t].id, p.id);
                        if (djk < p.dist-3*tau_)
                        {
                            occlude = true;
                            p.pid=t;
                            break;
                        }
                    }
                    if (!occlude)
                    {
                        nhood.nn_nsg.push_back(p);
                        nhood.nn_nsg.back().isnew=false;
                    }
                }
            }
        }
        for(unsigned i=0;i<nhoods[q].nn_nsg.size();i++)
        {
            nhoods[q].nn_nsg[i].flag=false;
        }
        return float(cc)/float(oracle.size()*float(oracle.size()-1)/2);
    }

    void KGraphConstructor::taumg_reversed_prune(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        float tau_=params.tau;
        width = range;
        unsigned start = 0;
        auto temp_pool=nhoods[q].nn_nsg;
        //cout<<"     "<<temp_pool.size()<<"   "<<endl;
        sort(temp_pool.begin(),temp_pool.end());
        // if(q==0)
        // {
        //     for(unsigned i=0;i<temp_pool.size();i++)
        //     {
        //         cout<<temp_pool[i].dist<<"  ";
        //     }
        //     cout<<endl;
        // }
        nhoods[q].nn_nsg.clear();  
        if (temp_pool[start].id == q)
        {
            start++;
        }
        nhoods[q].nn_nsg.push_back(temp_pool[start]);
        while (nhoods[q].nn_nsg.size() < range && (++start) < temp_pool.size()) {
            auto &p = temp_pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < nhoods[q].nn_nsg.size(); t++)
            {
                if (p.id == nhoods[q].nn_nsg[t].id)
                {
                    occlude = true;
                    break;
                }
                float djk = oracle(nhoods[q].nn_nsg[t].id, p.id);
                if (djk < p.dist-3*tau_)
                {
                    occlude = true;
                    break;
                }
            }
            if (!occlude)
            {
                nhoods[q].nn_nsg.push_back(p);
            }
        }
    }
    
    void KGraphConstructor::taumg_Link()
    {
        cout<<"Link Begin"<<endl;
        unsigned range = params.nsg_R;
        unsigned nd_ = oracle.size();
        std::vector<std::mutex> locks(nd_);
        float scan=0;
        unsigned cc=0;
        auto s_prune = std::chrono::high_resolution_clock::now();
        unsigned pass=0;
#pragma omp parallel
{    
#pragma omp for schedule(dynamic, 100) reduction(+ : scan)
        for (unsigned n = 0; n < nd_; ++n)
        {
            if(pruneAgain[n])
            {
                scan+=taumg_sync_prune(n);
            }
            else
            {
                scan+=taumg_sync_prune2(n);
            }
             //scan+=nsg_sync_prune(n);
        }
      
}

        prune_cc.push_back(scan);
        auto e_prune = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_prune = e_prune - s_prune;
        double Time_prune = diff_prune.count();   
        prune_time.push_back(Time_prune);
        printf("prune end. time: %f\n", Time_prune);
        cout<<"interinsert begin"<<endl;
        auto s_inter = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n)
        {
            cc+=InterInsert(n, range, locks);
        }
        pruneAgain.reset();
#pragma omp parallel for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n)
        {
            if(nhoods[n].nn_nsg.size()>range)
            {
                pruneAgain[n]=1;
                taumg_reversed_prune(n);
            }
        }
        auto e_inter = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_inter = e_inter - s_inter;
        double Time_inter = diff_inter.count();   
        inter_time.push_back(Time_inter);
        printf("InterInsert end. time: %f\n", Time_inter);
    }
    
    void KGraphConstructor::buildtauMG() {
        taumg_Link();
        // if(loopcount==0)
        // {
        //     ComputeEp();
        //     cout<<ep_<<endl;
        // }
        ComputeEp();
        cout<<"ep: "<<ep_<<endl;
        tree_grow();
        UpdateNhoods();
    }
    

    float KGraphConstructor::alphapg_sync_prune(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        float angle=params.angle;
        float threshold = std::cos(angle / 180 * kPi);
        width = range; 
        unsigned cc=0;
        unsigned start = 0;
        nhoods[q].nn_nsg.clear();
        if (nhoods[q].pool[start].id == q)
        {
            start++;
        }
        nhoods[q].nn_nsg.push_back(nhoods[q].pool[start]);
        while (nhoods[q].nn_nsg.size() < range && (++start) < nhoods[q].L && start < maxc) {
            auto &p = nhoods[q].pool[start];
            if(p.id>=oracle.size()) continue;
            bool occlude = false;
            for (unsigned t = 0; t < nhoods[q].nn_nsg.size(); t++)
            {
                if (p.id == nhoods[q].nn_nsg[t].id)
                {
                    occlude = true;
                    break;
                }
                cc++;
                float djk = oracle(nhoods[q].nn_nsg[t].id, p.id);
                float cos_ij = (nhoods[q].nn_nsg[t].dist + djk - p.dist) / 2 /
                     sqrt(nhoods[q].nn_nsg[t].dist * djk);
                if (djk < p.dist&&cos_ij<threshold)
                {
                    occlude = true;
                    p.pid=t;
                    break;
                }
            }
            if (!occlude)
            {
                nhoods[q].nn_nsg.push_back(p);
            }
        }
        for(unsigned i=0;i<nhoods[q].nn_nsg.size();i++)
        {
            nhoods[q].nn_nsg[i].flag=false;
        }
        return float(cc)/float(oracle.size()*float(oracle.size()-1)/2);
    }

    float KGraphConstructor::alphapg_sync_prune2(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        float angle=params.angle;
        float threshold = std::cos(angle / 180 * kPi);
        width = range;
        unsigned cc=0;
        unsigned start = 0;
        auto &nhood=nhoods[q];
        vector<unsigned> temp_nsg;
        unsigned pass=0;
        for(unsigned i=0;i<nhood.nn_nsg.size();i++)
        {
            if(!nhood.nn_nsg[i].flag)
            {
                temp_nsg.push_back(nhood.nn_nsg[i].id);
            }
        }
        nhood.nn_nsg.clear();
        if (nhood.pool[start].id == q)
        {
            start++;
        }
        nhood.nn_nsg.push_back(nhood.pool[start]);
        if(find(temp_nsg.begin(),temp_nsg.end(),nhood.nn_nsg.back().id)==temp_nsg.end())
        {
            nhood.nn_nsg.back().isnew=true;
        }
        else
        {
            nhood.nn_nsg.back().isnew=false;
        }
        while (nhood.nn_nsg.size() < range && (++start) < nhood.L && start < maxc) {
            auto &p = nhood.pool[start];
            if(p.id>=oracle.size()) continue;
            bool occlude = false;
            if(p.isnew)
            {
                for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                {
                    if (p.id == nhood.nn_nsg[t].id)
                    {
                        occlude = true;
                        break;
                    }
                    cc++;
                    float djk = oracle(nhood.nn_nsg[t].id, p.id);
                    float cos_ij = (nhoods[q].nn_nsg[t].dist + djk - p.dist) / 2 /
                            sqrt(nhoods[q].nn_nsg[t].dist * djk);
                    if (djk < p.dist&&cos_ij<threshold)
                    {
                        occlude = true;
                        p.pid=t;
                        break;
                    }
                }
                if (!occlude)
                {
                    nhood.nn_nsg.push_back(p);
                    nhood.nn_nsg.back().isnew=true;
                }
            }
            else
            {
                if(find(temp_nsg.begin(),temp_nsg.end(),p.id)==temp_nsg.end()&&((p.pid>=0)&&(p.pid<temp_nsg.size())))
                {
                    unsigned prune_p=temp_nsg[p.pid];
                    auto iter=find_if(nhood.nn_nsg.begin(),nhood.nn_nsg.end(),[prune_p](Neighbor neighbor)
                    {
                        return neighbor.id==prune_p;
                    });
                    if(iter!=nhood.nn_nsg.end())
                    {
                        pass++;
                        p.pid=distance(nhood.nn_nsg.begin(),iter);
                        continue;
                    }
                    else
                    {
                        for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                        {
                            if (p.id == nhood.nn_nsg[t].id)
                            {
                                occlude = true;
                                break;
                            }
                            cc++;
                            float djk = oracle(nhood.nn_nsg[t].id, p.id);
                            float cos_ij = (nhoods[q].nn_nsg[t].dist + djk - p.dist) / 2 /
                                            sqrt(nhoods[q].nn_nsg[t].dist * djk);
                            if (djk < p.dist&&cos_ij<threshold)
                            {
                                occlude = true;
                                p.pid=t;
                                break;
                            }
                        }
                        if (!occlude)
                        {
                            nhood.nn_nsg.push_back(p);
                            nhood.nn_nsg.back().isnew=true;
                        }
                    }
                }
                else if(find(temp_nsg.begin(),temp_nsg.end(),p.id)==temp_nsg.end()&&((p.pid<0)||(p.pid>=temp_nsg.size())))
                {
                    for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                    {
                        if (p.id == nhood.nn_nsg[t].id)
                        {
                            occlude = true;
                            break;
                        }
                        cc++;
                        float djk = oracle(nhood.nn_nsg[t].id, p.id);
                        float cos_ij = (nhoods[q].nn_nsg[t].dist + djk - p.dist) / 2 /
                                        sqrt(nhoods[q].nn_nsg[t].dist * djk);
                        if (djk < p.dist&&cos_ij<threshold)
                        {
                            occlude = true;
                            p.pid=t;
                            break;
                        }
                    }
                    if (!occlude)
                    {
                        nhood.nn_nsg.push_back(p);
                        nhood.nn_nsg.back().isnew=true;
                    }
                }
                else 
                {
                    for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                    {
                        if (p.id == nhood.nn_nsg[t].id)
                        {
                            occlude = true;
                            break;
                        }
                        if(!nhood.nn_nsg[t].isnew)
                        {
                            continue;
                        }
                        cc++;
                        float djk = oracle(nhood.nn_nsg[t].id, p.id);
                        float cos_ij = (nhoods[q].nn_nsg[t].dist + djk - p.dist) / 2 /
                                        sqrt(nhoods[q].nn_nsg[t].dist * djk);
                        if (djk < p.dist&&cos_ij<threshold)
                        {
                            occlude = true;
                            p.pid=t;
                            break;
                        }
                    }
                    if (!occlude)
                    {
                        nhood.nn_nsg.push_back(p);
                        nhood.nn_nsg.back().isnew=false;
                    }
                }
            }
        }
        for(unsigned i=0;i<nhoods[q].nn_nsg.size();i++)
        {
            nhoods[q].nn_nsg[i].flag=false;
        }
        return float(cc)/float(oracle.size()*float(oracle.size()-1)/2);
    }

    void KGraphConstructor::alphapg_reversed_prune(unsigned q) {
        unsigned range = params.nsg_R;
        unsigned maxc = params.search_K;
        float angle=params.angle;
        float threshold = std::cos(angle / 180 * kPi);
        width = range;
        unsigned start = 0;
        auto temp_pool=nhoods[q].nn_nsg;
        //cout<<"     "<<temp_pool.size()<<"   "<<endl;
        sort(temp_pool.begin(),temp_pool.end());
        // if(q==0)
        // {
        //     for(unsigned i=0;i<temp_pool.size();i++)
        //     {
        //         cout<<temp_pool[i].dist<<"  ";
        //     }
        //     cout<<endl;
        // }
        nhoods[q].nn_nsg.clear();  
        if (temp_pool[start].id == q)
        {
            start++;
        }
        nhoods[q].nn_nsg.push_back(temp_pool[start]);
        while (nhoods[q].nn_nsg.size() < range && (++start) < temp_pool.size()) {
            auto &p = temp_pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < nhoods[q].nn_nsg.size(); t++)
            {
                if (p.id == nhoods[q].nn_nsg[t].id)
                {
                    occlude = true;
                    break;
                }
                float djk = oracle(nhoods[q].nn_nsg[t].id, p.id);
                float cos_ij = (nhoods[q].nn_nsg[t].dist + djk - p.dist) / 2 /
                     sqrt(nhoods[q].nn_nsg[t].dist * djk);
                if (djk < p.dist&&cos_ij<threshold)
                {
                    occlude = true;
                    break;
                }
            }
            if (!occlude)
            {
                nhoods[q].nn_nsg.push_back(p);
            }
        }
    }
    
    void KGraphConstructor::alphapg_Link()
    {
        cout<<"Link Begin"<<endl;
        unsigned range = params.nsg_R;
        unsigned nd_ = oracle.size();
        std::vector<std::mutex> locks(nd_);
        float scan=0;
        unsigned cc=0;
        auto s_prune = std::chrono::high_resolution_clock::now();
        unsigned pass=0;
#pragma omp parallel
{    
#pragma omp for schedule(dynamic, 100) reduction(+ : scan)
        for (unsigned n = 0; n < nd_; ++n)
        {
            if(pruneAgain[n])
            {
                scan+=alphapg_sync_prune(n);
            }
            else
            {
                scan+=alphapg_sync_prune2(n);
            }
             //scan+=nsg_sync_prune(n);
        }
      
}

        prune_cc.push_back(scan);
        auto e_prune = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_prune = e_prune - s_prune;
        double Time_prune = diff_prune.count();   
        prune_time.push_back(Time_prune);
        printf("prune end. time: %f\n", Time_prune);
        cout<<"interinsert begin"<<endl;
        auto s_inter = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n)
        {
            cc+=InterInsert(n, range, locks);
        }
        pruneAgain.reset();
#pragma omp parallel for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n)
        {
            if(nhoods[n].nn_nsg.size()>range)
            {
                pruneAgain[n]=1;
                alphapg_reversed_prune(n);
            }
        }
        auto e_inter = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_inter = e_inter - s_inter;
        double Time_inter = diff_inter.count();   
        inter_time.push_back(Time_inter);
        printf("InterInsert end. time: %f\n", Time_inter);
    }
    
    void KGraphConstructor::buildAlphaPG() {
        alphapg_Link();
        // if(loopcount==0)
        // {
        //     ComputeEp();
        //     cout<<ep_<<endl;
        // }
        ComputeEp();
        cout<<"ep: "<<ep_<<endl;
        tree_grow();
        UpdateNhoods();
    }
    
    unsigned KGraphConstructor::InsertKNN()
    {
        unsigned nd_=oracle.size();
        unsigned size_=params.nsg_R;
#pragma omp parallel for
        for(unsigned i=0;i<nd_;i++)
        {
            auto &nhood=nhoods[i];
            if(nhood.nn_nsg.size()>=size_)
            {
                continue;
            }
            unsigned j=0,k=0;
            while(nhood.nn_nsg.size()<size_&&k<nhood.nn_nsg.size()&&j<params.K)
            {
                if(nhood.pool[j].dist==nhood.nn_nsg[k].dist)
                {
                    j++;
                    k++;
                }
                else if(nhood.pool[j].dist<nhood.nn_nsg[k].dist)
                {
                    nhood.nn_nsg.push_back(nhood.pool[j]);
                    j++;
                }
                else
                {
                    k++;
                }
            }
            while(nhood.nn_nsg.size()<size_&&j<params.K)
            {
                nhood.nn_nsg.push_back(nhood.pool[j]);
                j++;
            }
            // cout<<k<<"  "<<j<<endl;
            
        }
        CalculateNSG();
        return 0;
    }

    void KGraphConstructor::UpdateNhoods()
    {
        if(loopcount!=0)
        {
#pragma omp parallel for schedule(dynamic, 100)
            for(unsigned i=0;i<oracle.size();i++)
            {
                auto &nhood=nhoods[i];
                for(unsigned j=0;j<nhood.nn_nsg.size();j++)
                {
                    unsigned id=nhood.nn_nsg[j].id;
                    if(find(nhood.old_nsg.begin(),nhood.old_nsg.end(),id)==nhood.old_nsg.end())
                    // if(nhood.old_nsg.find(id)==nhood.old_nsg.end())
                    {
                        nhood.nn_nsg[j].isnew=true;
                    }
                    else
                    {
                        nhood.nn_nsg[j].isnew=false;
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for schedule(dynamic, 100)
            for(unsigned i=0;i<oracle.size();i++)
            {
                auto &nhood=nhoods[i];
                for(unsigned j=0;j<nhood.nn_nsg.size();j++)
                {
                    nhood.nn_nsg[j].isnew=true;
                }
            }   
        }
#pragma omp parallel for schedule(dynamic, 100)
        for(unsigned i=0;i<oracle.size();i++)
        {
            auto &nhood=nhoods[i];
            nhood.old_nsg.clear();
            nhood.old_nsg.reserve(nhood.nn_nsg.size());
            for(unsigned j=0;j<nhood.nn_nsg.size();j++)
            {
                unsigned id=nhood.nn_nsg[j].id;
                nhood.old_nsg.emplace_back(id);
            }
        }
    }
#ifdef DEBUG //swapResults
    void KGraphConstructor::swapResults(std::vector<std::vector<int>> &knng, unsigned K)
    {
        unsigned nq = oracle.size();
        knng.resize(nq);
        for (unsigned i = 0; i < nq; i++)
        {
            knng[i].resize(K);
            for (unsigned j = 0; j < K; j++)
                knng[i][j] = nhoods[i].pool[j].id;
        }
    }
#endif
    void KGraphConstructor::SwapHNSWLayer(char **&linkLists_,unsigned level)
    {
        size_t size_links_per_element_ = params.nsg_R * sizeof(unsigned) + sizeof(unsigned);
        for(unsigned i=0;i<oracle.size();i++)
        {
            unsigned* cll=(unsigned *)(linkLists_[i] + (level - 1) * size_links_per_element_);
            int size=nhoods[i].nn_nsg.size();
            unsigned short linkCount = static_cast<unsigned short>(size);
            *((unsigned short int*)(cll))=*((unsigned short int *)&linkCount);
            unsigned *nei=(unsigned *)(cll+1);
            for(unsigned j=0;j<nhoods[i].nn_nsg.size();j++)
            {
                nei[j]=nhoods[i].nn_nsg[j].id;
            }
        }
        
    }
    void KGraphConstructor::SwapHNSWLayer0(char *&data_level0_memory_)
    {
        size_t size_links_level0_=params.nsg_R*sizeof(unsigned)+sizeof(unsigned);
        size_t size_data_per_element_=size_links_level0_+sizeof(float)*oracle.dim()+sizeof(size_t);
        size_t offsetData_=size_links_level0_;
        size_t label_offset_ = size_links_level0_+sizeof(float)*oracle.dim();
        size_t offsetLevel0_ = 0;

        for(unsigned i=0;i<oracle.size();i++)
        {
            unsigned *cll= (unsigned *)(data_level0_memory_+i*size_data_per_element_+offsetLevel0_);
            int size=nhoods[i].nn_nsg.size();
            unsigned short linkCount = static_cast<unsigned short>(size);
            *((unsigned short int*)(cll))=*((unsigned short int *)&linkCount);
            unsigned *nei=(unsigned *)(cll+1);
            for(unsigned j=0;j<nhoods[i].nn_nsg.size();j++)
            {
                nei[j]=nhoods[i].nn_nsg[j].id;
            }

        }
    }

    unsigned KGraphConstructor::getHNSWEp()
    {
        return ep_;
    }
    unsigned KGraphConstructor::getLoop_i()
    {
        return params.loop_i;
    }

    void KGraphConstructor::final_prune()
    {
        unsigned range =params.nsg_R;
        unsigned nd_=oracle.size();
        auto s_final_prune = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n)
        {
            if(nhoods[n].nn_nsg.size()>range)
            {
                nsg_reversed_prune(n);
            }
        }
        auto e_final_prune = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_final_prune = e_final_prune - s_final_prune;
        double Time_final_prune = diff_final_prune.count(); 
        final_prune_time=Time_final_prune;  
        printf("final_prune end. time: %f\n", Time_final_prune);
    }

    void KGraphConstructor::inSearch() {
        unsigned L = params.search_L;
        unsigned nd_ = oracle.size();
        float scan=0;
        unsigned cc = 0;
#pragma omp parallel
{
        vector<Neighbor> retset(L+1);
        boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100) reduction(+ : scan)
        for (unsigned i = 0; i < nd_; i++)
        {
            retset.clear();
            flags.reset();
            scan += poolSearch(i,flags);
            // cc+=epSearch(i,retset,flags);
        }
        
}
        search_cc.push_back(scan);
    cout<<"scan_rate: "<<scan<<endl;
    }

#ifdef DEBUG //evaluate
    // float KGraphConstructor::evaluate() {
    //     accumulator_set<float, stats<tag::mean>> recall;
    //     for (auto const &c : controls)
    //     {
    //         recall(EvaluateRecall(nhoods[c.id].pool, nhoods[c.id].L,c.neighbors,MAXL));
    //     }
    //     float rec= mean(recall);
    //     cout<<"recall:  "<<rec<<endl;
    //     return rec;
    // }
    float KGraphConstructor::evaluate() {
        auto s_computegt = std::chrono::high_resolution_clock::now();
        double recall=0.0;
#pragma omp parallel for reduction(+:recall)
        for (size_t i = 0; i < controls.size(); ++i)
        {
            auto &c=controls[i];
            recall+=EvaluateRecall(nhoods[c.id].pool, nhoods[c.id].L,c.neighbors,MAXL);
        }
        float rec= recall/controls.size();
        cout<<"recall:  "<<rec<<endl;
        auto e_computegt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_computegt = e_computegt-s_computegt;
        double Time_computegt = diff_computegt.count();
        cout<<"compute recall time: "<<Time_computegt<<endl;
        return rec;
    }
#endif

    void KGraphImpl::buildFastHNSW(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name,const char *res_name,char *&data_level0_memory_,char **linkLists_,unsigned level,unsigned* ep_ptr,size_t size_data_per_element_) {
        auto s = std::chrono::high_resolution_clock::now();
        auto s_kgraph = std::chrono::high_resolution_clock::now();
        KGraphConstructor con(oracle, param, info);
        // cout<<"param.K in buildFastHNSW"<<param.K<<endl;
        // cout<<"all opt"<<endl;
        float recall=con.evaluate();
        con.allrecall.push_back(recall);
        auto e_kgraph = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_kgraph = e_kgraph - s_kgraph;
        double Time_kgraph = diff_kgraph.count();
        con.kgraph_time.push_back(Time_kgraph);
        con.pruneAgain.resize(oracle.size());
        con.pruneAgain.set();
        con.init_nhoods();
        con.loopcount=0;
        unsigned nsg_iteration=con.getLoop_i()+1;
        for(unsigned iter=1;iter<=nsg_iteration;iter++)
        {
            auto s_nsg = std::chrono::high_resolution_clock::now();
            con.buildHNSWLayer();
            con.PG_type=2;
            printf("buildHNSWLayer end, ");
            auto e_nsg = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_nsg = e_nsg - s_nsg;
            double Time_nsg = diff_nsg.count();
            con.buildnsg_time.push_back(Time_nsg);
            printf(" time: %f\n",Time_nsg);
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            double Time = diff.count();
            con.con_time.push_back(Time);
            printf("construct time: %f\n", Time);
            con.loopcount++;
            if(iter==nsg_iteration)
            {
                //con.final_prune();
                con.CalculateNSG();
                break;
            }
            auto s_search = std::chrono::high_resolution_clock::now();
            //con.inSearch();
            con.BridgeView();
            auto e_search = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_search = e_search - s_search;
            double Time_search = diff_search.count();
            con.search_time.push_back(Time_search);
            printf("Search end, time: %f\n",Time_search);
            recall=con.evaluate();
            con.allrecall.push_back(recall);
            auto e_kgraph2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_kgraph2 = e_kgraph2 - s;
            double Time_kgraph2 = diff_kgraph2.count();
            con.kgraph_time.push_back(Time_kgraph2);
        }
        *ep_ptr=con.getHNSWEp();
        // cout<<"save begin"<<endl;
        //con.SaveNsg(nsg_name);
        auto s_prune = std::chrono::high_resolution_clock::now();
        if(level>0)
        {
            con.SwapHNSWLayer(linkLists_,level);
        }
        if(level==0)
        {
            // data_level0_memory_=(char *)malloc(oracle.size()*size_data_per_element_);
            // memset(data_level0_memory_,0,oracle.size()*size_data_per_element_);
            // con.SwapHNSWLayer0(data_level0_memory_);
            con.SwapPGraph();
            temp_graph.swap(con.temp_graph);
        }
        

        con.SaveResult(res_name);
        // cout<<"save end"<<endl;
    }
    void KGraphImpl::buildFastNSG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name,const char *res_name) {
        auto s = std::chrono::high_resolution_clock::now();
        auto s_kgraph = std::chrono::high_resolution_clock::now();
        KGraphConstructor con(oracle, param, info);
        cout<<"all opt"<<endl;
        float recall=con.evaluate();
        con.allrecall.push_back(recall);
        auto e_kgraph = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_kgraph = e_kgraph - s_kgraph;
        double Time_kgraph = diff_kgraph.count();
        con.kgraph_time.push_back(Time_kgraph);
        con.pruneAgain.resize(oracle.size());
        con.pruneAgain.set();
        con.init_nhoods();
        con.loopcount=0;
        unsigned nsg_iteration=param.loop_i+1;
        for(unsigned iter=1;iter<=nsg_iteration;iter++)
        {
            auto s_nsg = std::chrono::high_resolution_clock::now();
            con.buildNSG();
            con.PG_type=1;
            printf("buildNSG end, ");
            auto e_nsg = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_nsg = e_nsg - s_nsg;
            double Time_nsg = diff_nsg.count();
            con.buildnsg_time.push_back(Time_nsg);
            printf(" time: %f\n",Time_nsg);
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            double Time = diff.count();
            con.con_time.push_back(Time);
            printf("construct time: %f\n", Time);
            con.loopcount++;
            if(iter==nsg_iteration)
            {
                con.CalculateNSG();
                break;
            }
            auto s_search = std::chrono::high_resolution_clock::now();
            // con.inSearch();
            con.BridgeView();
            auto e_search = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_search = e_search - s_search;
            double Time_search = diff_search.count();
            con.search_time.push_back(Time_search);
            printf("Search end, time: %f\n",Time_search);
            recall=con.evaluate();
            con.allrecall.push_back(recall);
            auto e_kgraph2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_kgraph2 = e_kgraph2 - s;
            double Time_kgraph2 = diff_kgraph2.count();
            con.kgraph_time.push_back(Time_kgraph2);
        }
        con.SaveNsg(nsg_name);
        con.SaveResult(res_name);
    }


    void KGraphImpl::buildFastTauMNG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name,const char *res_name) {
        auto s = std::chrono::high_resolution_clock::now();
        auto s_kgraph = std::chrono::high_resolution_clock::now();
        KGraphConstructor con(oracle, param, info);
        cout<<"all opt"<<endl;
        float recall=con.evaluate();
        con.allrecall.push_back(recall);
        auto e_kgraph = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_kgraph = e_kgraph - s_kgraph;
        double Time_kgraph = diff_kgraph.count();
        con.kgraph_time.push_back(Time_kgraph);
        con.pruneAgain.resize(oracle.size());
        con.pruneAgain.set();
        con.init_nhoods();
        con.loopcount=0;
        unsigned nsg_iteration=param.loop_i+1;
        for(unsigned iter=1;iter<=nsg_iteration;iter++)
        {
            auto s_nsg = std::chrono::high_resolution_clock::now();
            con.buildtauMG();
            con.PG_type=3;
            printf("buildTauMNG end, ");
            auto e_nsg = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_nsg = e_nsg - s_nsg;
            double Time_nsg = diff_nsg.count();
            con.buildnsg_time.push_back(Time_nsg);
            printf(" time: %f\n",Time_nsg);
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            double Time = diff.count();
            con.con_time.push_back(Time);
            printf("construct time: %f\n", Time);
            con.loopcount++;
            if(iter==nsg_iteration)
            {
                con.CalculateNSG();
                break;
            }
            auto s_search = std::chrono::high_resolution_clock::now();
            // con.inSearch();
            con.BridgeView();
            auto e_search = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_search = e_search - s_search;
            double Time_search = diff_search.count();
            con.search_time.push_back(Time_search);
            printf("Search end, time: %f\n",Time_search);
            recall=con.evaluate();
            con.allrecall.push_back(recall);
            auto e_kgraph2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_kgraph2 = e_kgraph2 - s;
            double Time_kgraph2 = diff_kgraph2.count();
            con.kgraph_time.push_back(Time_kgraph2);
        }
        con.SaveNsg(nsg_name);
        con.SaveResult(res_name);
    }

    void KGraphImpl::buildFastAlphaPG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name,const char *res_name) {
        auto s = std::chrono::high_resolution_clock::now();
        auto s_kgraph = std::chrono::high_resolution_clock::now();
        KGraphConstructor con(oracle, param, info);
        cout<<"all opt"<<endl;
        float recall=con.evaluate();
        con.allrecall.push_back(recall);
        auto e_kgraph = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_kgraph = e_kgraph - s_kgraph;
        double Time_kgraph = diff_kgraph.count();
        con.kgraph_time.push_back(Time_kgraph);
        con.pruneAgain.resize(oracle.size());
        con.pruneAgain.set();
        con.init_nhoods();
        con.loopcount=0;
        unsigned nsg_iteration=param.loop_i+1;
        for(unsigned iter=1;iter<=nsg_iteration;iter++)
        {
            auto s_nsg = std::chrono::high_resolution_clock::now();
            con.buildAlphaPG();
            con.PG_type=4;
            printf("buildAlphaPG end, ");
            auto e_nsg = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_nsg = e_nsg - s_nsg;
            double Time_nsg = diff_nsg.count();
            con.buildnsg_time.push_back(Time_nsg);
            printf(" time: %f\n",Time_nsg);
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            double Time = diff.count();
            con.con_time.push_back(Time);
            printf("construct time: %f\n", Time);
            con.loopcount++;
            if(iter==nsg_iteration)
            {
                con.CalculateNSG();
                break;
            }
            auto s_search = std::chrono::high_resolution_clock::now();
            // con.inSearch();
            con.BridgeView();
            auto e_search = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_search = e_search - s_search;
            double Time_search = diff_search.count();
            con.search_time.push_back(Time_search);
            printf("Search end, time: %f\n",Time_search);
            recall=con.evaluate();
            con.allrecall.push_back(recall);
            auto e_kgraph2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_kgraph2 = e_kgraph2 - s;
            double Time_kgraph2 = diff_kgraph2.count();
            con.kgraph_time.push_back(Time_kgraph2);
        }
        con.SaveNsg(nsg_name);
        con.SaveResult(res_name);
    }

    KGraph *KGraph::create()
    {
        return new KGraphImpl;
    }

    char const *KGraph::version()
    {
        return kgraph_version;
    }
}

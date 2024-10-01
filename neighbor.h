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
#include <sys/resource.h>   
#include <limits>
using std::numeric_limits;


   namespace kgraph{

        struct Neighbor {
        uint32_t id;
        float dist;
        bool flag;  // whether this entry is a newly found one
        bool isnew; // whether the new one in (i)th compare by (i-1)th
        short pid;
        Neighbor () {
            dist=numeric_limits<float>::max();
            isnew=true;
            pid=-1;
        }
        Neighbor (unsigned i, float d, bool f = true): id(i), dist(d), flag(f) {
            isnew=true;
            pid=-1;
        }
        Neighbor (bool is,unsigned i, float d, bool f = true): id(i), dist(d), flag(f), isnew(is){
            pid=-1;
        }
    };

    struct NSGNeighbor {
        uint32_t id;
        bool flag;
        NSGNeighbor () {}
        NSGNeighbor(unsigned i, bool f = true) : id(i), flag(f) {
        }
    };

    struct SimpleNeighbor {
        uint32_t id;
        float dist;
        SimpleNeighbor () {}
        SimpleNeighbor (unsigned i, float d): id(i), dist(d) {
        }
    };

    // extended neighbor structure for search time
    struct NeighborX: public Neighbor {
        uint16_t m;
        uint16_t M; // actual M used
        NeighborX () {}
        NeighborX (unsigned i, float d): Neighbor(i, d, true), m(0), M(0) {
        }
    };

    struct RankNeighbor{
        unsigned id;
        unsigned rank;
        RankNeighbor () {}
        RankNeighbor (unsigned i, unsigned r): id(i),rank(r){}
    };

    static inline bool operator < (Neighbor const &n1, Neighbor const &n2) {
        return n1.dist < n2.dist;
    }

    static inline bool operator < (SimpleNeighbor const &n1, SimpleNeighbor const &n2) {
        return n1.dist < n2.dist;
    }


    static inline bool operator == (Neighbor const &n1, Neighbor const &n2) {
        return n1.id == n2.id;
    }

    static inline bool operator < (RankNeighbor const &n1, RankNeighbor const &n2) {
        return n1.rank < n2.rank;
    }
   }

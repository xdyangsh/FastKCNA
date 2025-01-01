#ifndef WDONG_KGRAPH
#define WDONG_KGRAPH

#include <stdexcept>
#include <vector>
#include <neighbor.h>


using namespace std;

namespace kgraph {
    static unsigned const default_iterations =  30;
    static unsigned const default_L = 100;
    static unsigned const default_K = 25;
    static unsigned const default_P = 100;
    static unsigned const default_M = 0;
    static unsigned const default_T = 1;
    static unsigned const default_S = 10;
    static unsigned const default_R = 100;
    static unsigned const default_controls = 100;
    static unsigned const default_seed = 2024;
    static float const default_delta = 0.002;
    static float const default_recall = 0.95;
    static float const default_epsilon = 1e30;
    static unsigned const default_verbosity = 1;
    static unsigned const default_nthreads = 40;
    static unsigned const default_num0 = 100;
    static unsigned const default_iter = 3;
    static float const default_incr_r = 10.0;
    static unsigned const default_loop_i=2;
    
    static unsigned const default_nsgr = 50;

    static unsigned const default_searchl = 40;
    static unsigned const default_searchk = 500;
    static unsigned const default_step=10;
    //static unsigned const default_nt = 88;
    static unsigned const default_massqS = 10;
    static float const default_tau = 0;
    static float const default_angle=60;

    enum {
        PRUNE_LEVEL_1 = 1,
        PRUNE_LEVEL_2 = 2
    };
    enum {
        REVERSE_AUTO = -1,
        REVERSE_NONE = 0,
    };
    static unsigned const default_prune = 0;
    static int const default_reverse = REVERSE_NONE;

    /// Verbosity control
    /** Set verbosity = 0 to disable information output to stderr.
     */
    extern unsigned verbosity;

    /// Index oracle
    /** The index oracle is the user-supplied plugin that computes
     * the distance between two arbitrary objects in the dataset.
     * It is used for offline k-NN graph construction.
     */
    class IndexOracle {   
    public:
        /// Returns the size of the dataset.
        virtual unsigned size () const = 0;
        /// Computes similarity
        virtual unsigned dim() const = 0;
        /**
         * 0 <= i, j < size() are the index of two objects in the dataset.
         * This method return the distance between objects i and j.
         */
        virtual float operator () (unsigned i, unsigned j) const = 0;
        virtual float operator () (unsigned i, unsigned j, unsigned offset,unsigned dim) const = 0;
        virtual float operator () (const float *query, unsigned i) const = 0;
        virtual float computeDot(unsigned i, unsigned j) const=0;
        virtual void computeSelfDotPtr() =0;
        virtual float* Calcenter() const = 0;
        virtual unsigned data_size () const =0;
        virtual void set_size (unsigned mysize) =0;
    };

    /// Search oracle
    /** The search oracle is the user-supplied plugin that computes
     * the distance between the query and a arbitrary object in the dataset.
     * It is used for online k-NN search.
     */
    class SearchOracle {
    public:
        /// Returns the size of the dataset.
        virtual unsigned size () const = 0;
        /// Computes similarity
        /**
         * 0 <= i < size() are the index of an objects in the dataset.
         * This method return the distance between the query and object i.
         */
        virtual float operator () (unsigned i) const = 0;
        /// Search with brutal force.
        /**
         * Search results are guaranteed to be ranked in ascending order of distance.
         *
         * @param K Return at most K nearest neighbors.
         * @param epsilon Only returns nearest neighbors within distance epsilon.
         * @param ids Pointer to the memory where neighbor IDs are returned.
         * @param dists Pointer to the memory where distance values are returned, can be nullptr.
         */
        unsigned search (unsigned K, float epsilon, unsigned *ids, float *dists = nullptr) const;
    };

    /// The KGraph index.
    /** This is an abstract base class.  Use KGraph::create to create an instance.
     */
    class KGraph {
    public:
        /// Indexing parameters.
        struct IndexParams {

            //NSG
            unsigned nsg_R;
            unsigned search_L;
            unsigned search_K;
            unsigned step;
            
            //tauMG
            float tau;
            
            //NSSG
            
            float angle;
            unsigned nt;


            unsigned massq_S;

            unsigned loop_i;
            unsigned iterations; 
            unsigned L;
            unsigned K;
            unsigned S;
            unsigned R;
            unsigned controls;
            unsigned seed;
            unsigned nthreads;      
            float delta;
            float recall;
            unsigned prune;
            int reverse;

            unsigned num0;      // the parameter to control the number of points in the initial step
            float incr_r;       // the parameter to control the increasing rate of vectors in each step
            unsigned sl;       // the parameter to control the search accuracy in each step
            unsigned iter;      // the parameter to control the propagation in each step

            // used for testing 
            unsigned qid;
            unsigned Lcon;
            unsigned sidx;
            unsigned B;     // the neighborhood size of the bridge nodes.
            unsigned P;     // the number of the starting points for the search
            /// Construct with default values.
            IndexParams (): iterations(default_iterations), L(default_L), K(default_K), S(default_S), R(default_R), controls(default_controls), seed(default_seed), delta(default_delta), recall(default_recall), prune(default_prune), reverse(default_reverse) {
                num0 = default_num0;
                nthreads = default_nthreads;
                incr_r = default_incr_r;
                iter = default_iter;
                massq_S=default_massqS;
                loop_i=default_loop_i;
                angle = default_angle;
            }
        };

        /// Search parameters.
        struct SearchParams {
            unsigned K;
            unsigned M;
            unsigned P;
            unsigned S;
            unsigned T;
            float epsilon;
            unsigned seed;
            unsigned init;

            /// Construct with default values.
            SearchParams (): K(default_K), M(default_M), P(default_P), S(default_S), T(default_T), epsilon(default_epsilon), seed(1998), init(0) {
            }
        };

        enum {
            FORMAT_DEFAULT = 0,
            FORMAT_NO_DIST = 1
        };

        /// Information and statistics of the indexing algorithm.
        struct IndexInfo {
            enum StopCondition {
                ITERATION = 0,
                DELTA,
                RECALL
            } stop_condition;
            unsigned iterations;
            float cost;
            float recall;
            float accuracy;
            float delta;
            float M;
        };

        /// Information and statistics of the search algorithm.
        struct SearchInfo {
            float cost;
            unsigned updates;
        };



        virtual ~KGraph () {
        }
        /// Load index from file.
        /**
         * @param path Path to the index file.
         */
        virtual void load (char const *path) = 0;
        /// Save index to file.
        /**
         * @param path Path to the index file.
         */
        virtual void save (char const *path, int format = FORMAT_DEFAULT) const = 0; // save to file
        /// Build the index
        //virtual void build (IndexOracle const &oracle, IndexParams const &params, IndexInfo *info = 0) = 0;
        virtual void buildFastNSG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info = 0,const char *nsg_name=0,const char *res_name=0) = 0;
        virtual void buildFastTauMNG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info = 0,const char *nsg_name=0,const char *res_name=0) = 0;
        virtual void buildFastAlphaPG(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info = 0,const char *nsg_name=0,const char *res_name=0) = 0;
        virtual void buildFastHNSW(IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,const char *nsg_name,const char *res_name,char *&data_level0_memory_,char** linkLists_=0,unsigned level=0,unsigned* ep_ptr=0,size_t size_data_per_element_=0) = 0;
        virtual void FatherSwapHNSWLayer0(IndexOracle const &oracle,IndexParams const &params,char *&data_level0_memory_) = 0;
        /**
         * Pruning makes the index smaller to save memory, and makes online search on the pruned index faster.
         * (The cost parameters of online search must be enlarged so accuracy is not reduced.)
         *
         * Currently only two pruning levels are supported:
         * - PRUNE_LEVEL_1 = 1: Only reduces index size, fast.
         * - PRUNE_LEVEL_2 = 2: For improve online search speed, slow.
         *
         * No pruning is done if level = 0.
         */
        virtual void prune (IndexOracle const &oracle, unsigned level) = 0;
        /// Online k-NN search.
        /**
         * Search results are guaranteed to be ranked in ascending order of distance.
         *
         * @param ids Pointer to the memory where neighbor IDs are stored, must have space to save params.K ids.
         */
        unsigned search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, SearchInfo *info = 0) const {
            return search(oracle, params, ids, nullptr, info);
        }
        /// Online k-NN search.
        /**
         * Search results are guaranteed to be ranked in ascending order of distance.
         *
         * @param ids Pointer to the memory where neighbor IDs are stored, must have space to save params.K values.
         * @param dists Pointer to the memory where distances are stored, must have space to save params.K values.
         */
        virtual unsigned search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, float *dists, SearchInfo *info) const = 0;
        /// Constructor.
        static KGraph *create ();
        /// Returns version string.
        static char const* version ();

        //virtual KGraph* createBuilder();

        /// Get offline computed k-NNs of a given object.
        /**
         * See the full version of get_nn.
         */
        virtual void get_nn (unsigned id, unsigned *nns, unsigned *M, unsigned *L) const {
            get_nn(id, nns, nullptr, M, L);
        }
        /// Get offline computed k-NNs of a given object.
        /**
         * The user must provide space to save IndexParams::L values.
         * The actually returned L could be smaller than IndexParams::L, and
         * M <= L is the number of neighbors KGraph thinks
         * could be most useful for online search, and is usually < L.
         * If the index has been pruned, the returned L could be smaller than
         * IndexParams::L used to construct the index.
         *
         * @params id Object ID whose neighbor information are returned.
         * @params nns Neighbor IDs, must have space to save IndexParams::L values. 
         * @params dists Distance values, must have space to save IndexParams::L values.
         * @params M Useful number of neighbors, output only.
         * @params L Actually returned number of neighbors, output only.
         */
        virtual void get_nn (unsigned id, unsigned *nns, float *dists, unsigned *M, unsigned *L) const = 0;

        virtual void reverse (int) = 0;

        virtual void swapResult(vector<vector<int> >& _knng) {
            printf("HaHa\n");
        }
        virtual void swapResult(vector<vector<unsigned> >& _knng) {
            printf("HaHa\n");
        }

        virtual void test() {printf("From KGraph.\n");}
    };
}

#if __cplusplus > 199711L
#include <functional>
namespace kgraph {
    /// Oracle adapter for datasets stored in a vector-like container.
    /**
     * If the dataset is stored in a container of CONTAINER_TYPE that supports
     * - a size() method that returns the number of objects.
     * - a [] operator that returns the const reference to an object.
     * This class can be used to provide a wrapper to facilitate creating
     * the index and search oracles.
     *
     * The user must provide a callback function that takes in two
     * const references to objects and returns a distance value.
     */
    template <typename CONTAINER_TYPE, typename OBJECT_TYPE>
    class VectorOracle: public IndexOracle {
    public:
        typedef std::function<float(OBJECT_TYPE const &, OBJECT_TYPE const &)> METRIC_TYPE;
    private:
        CONTAINER_TYPE const &data;
        METRIC_TYPE dist;
    public:
        class VectorSearchOracle: public SearchOracle {
            CONTAINER_TYPE const &data;
            OBJECT_TYPE const query;
            METRIC_TYPE dist;
        public:
            VectorSearchOracle (CONTAINER_TYPE const &p, OBJECT_TYPE const &q, METRIC_TYPE m): data(p), query(q), dist(m) {
            }
            virtual unsigned size () const {
                return data.size();
            }
            virtual float operator () (unsigned i) const {
                return dist(data[i], query);
            }
        };
        /// Constructor.
        /**
         * @param d: the container that holds the dataset.
         * @param m: a callback function for distance computation.  m(d[i], d[j]) must be
         *  a valid expression to compute distance.
         */
        VectorOracle (CONTAINER_TYPE const &d, METRIC_TYPE m): data(d), dist(m) {
        }
        virtual unsigned size () const {
            return data.size();
        }
        virtual float operator () (unsigned i, unsigned j) const {
            return dist(data[i], data[j]);
        }
        /// Constructs a search oracle for query object q.
        VectorSearchOracle query (OBJECT_TYPE const &q) const {
            return VectorSearchOracle(data, q, dist);
        }
    };

    class invalid_argument: public std::invalid_argument {
    public:
        using std::invalid_argument::invalid_argument;
    };

    class runtime_error: public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    class io_error: public runtime_error {
    public:
        using runtime_error::runtime_error;
    };

}
#endif

#endif


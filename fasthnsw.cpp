#include <sys/time.h>
#include <cctype>
#include <random>
#include <iomanip>
#include <type_traits>
#include <string.h>
#include <boost/timer/timer.hpp>
#include <boost/random.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <string>
#include "kgraph.h"
#include "kgraph-data.h" 
#include <omp.h>
//#include "groundtruth.h"

using namespace std;
using namespace boost;
using namespace boost::timer;
using namespace kgraph;

namespace po = boost::program_options; 

#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif

//typedef KGRAPH_VALUE_TYPE value_type;

typedef KGRAPH_VALUE_TYPE value_type;

#include <sys/resource.h>
#include <memory>
long getPeakMemoryUsage() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return r_usage.ru_maxrss / 1024;
}

template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}


int main (int argc, char *argv[]) {
    string data_path;
    string output_path;
    string res_path;
    KGraph::IndexParams params;
    unsigned D;
    unsigned skip;
    unsigned gap;
    unsigned synthetic;
    float noise;

    bool lshkit = true;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("version,v", "print version information.")
    ("data", po::value(&data_path), "data name")
    ("output", po::value(&output_path), "output path")
    ("result", po::value(&res_path), "result path")
    (",K", po::value(&params.K)->default_value(default_K), "number of nearest neighbor")
    ("nthreads,nt", po::value(&params.nthreads)->default_value(default_nthreads), "number of threads")
    ("controls,C", po::value(&params.controls)->default_value(default_controls), "number of control pounsigneds")
    

    ("nsg_R", po::value(&params.nsg_R)->default_value(default_nsgr), "")
    ("loop_i", po::value(&params.loop_i)->default_value(default_loop_i), "")

    ("search_L", po::value(&params.search_L)->default_value(default_searchl), "")
    ("search_K", po::value(&params.search_K)->default_value(default_searchk), "")
    ("massq_S", po::value(&params.massq_S)->default_value(default_massqS), "")
    ("angle", po::value(&params.angle)->default_value(default_angle), "")
    ("step", po::value(&params.step)->default_value(default_step), "")
    ("tau", po::value(&params.tau)->default_value(default_tau), "");
    //("nt", po::value(&params.nt)->default_value(default_nt), "");

    po::options_description desc_hidden("Expert options");
    desc_hidden.add_options()
    ("iterations,I", po::value(&params.iterations)->default_value(default_iterations), "")
    (",S", po::value(&params.S)->default_value(default_S), "")
    (",R", po::value(&params.R)->default_value(default_R), "")
    (",L", po::value(&params.L)->default_value(default_L), "")
    ("delta", po::value(&params.delta)->default_value(default_delta), "")
    ("recall", po::value(&params.recall)->default_value(default_recall), "")
    ("prune", po::value(&params.prune)->default_value(default_prune), "")
    ("reverse", po::value(&params.reverse)->default_value(default_reverse), "")
    ("noise", po::value(&noise)->default_value(0), "noise")
    ("seed", po::value(&params.seed)->default_value(default_seed), "")
    ("dim,D", po::value(&D), "dimension, see format")
    ("skip", po::value(&skip)->default_value(0), "see format")
    ("gap", po::value(&gap)->default_value(0), "see format")
    ("raw", "read raw binary file, need to specify D.")
    ("synthetic", po::value(&synthetic)->default_value(0), "generate synthetic data, for performance evaluation only, specify number of points")
    ("l2norm", "l2-normalize data, so as to mimic cosine similarity")
    ;
    
    po::options_description desc("Allowed options");
    desc.add(desc_visible).add(desc_hidden);

    po::positional_options_description p;
    p.add("data", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 
    omp_set_num_threads(params.nthreads);
    if (vm.count("raw") == 1) {
        lshkit = false;
    }

    if (vm.count("version")) {
        cout << "KGraph version " << KGraph::version() << endl;
        return 0;
    }

    if (vm.count("help")
            || (synthetic && (vm.count("dim") == 0 || vm.count("data")))
            || (!synthetic && (vm.count("data") == 0 || (vm.count("dim") == 0 && !lshkit)))) {
        cout << "Usage: index [OTHER OPTIONS]... INPUT [OUTPUT]" << endl;
        cout << desc_visible << endl;
        cout << desc_hidden << endl;
        return 0;
    }

    if (params.S == 0) {
        params.S = params.K;
    }


    if (lshkit && (synthetic == 0)) {   // read dimension information from the data file
        static const unsigned LSHKIT_HEADER = 3;
        ifstream is(data_path.c_str(), ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof header);
        BOOST_VERIFY(is);
        BOOST_VERIFY(header[0] == sizeof(value_type));
        is.close();
        D = header[2];
        skip = LSHKIT_HEADER * sizeof(unsigned);
        gap = 0;
    }

    Matrix<value_type> data;
    vector<int> labels;
    vector<int> points_num;
    double get_level_time;
    if (synthetic) {
        if (!std::is_floating_point<value_type>::value) {
            throw std::runtime_error("synthetic data not implemented for non-floating-point values.");
        }
        data.resize(synthetic, D);
        cerr << "Generating synthetic data..." << endl;
        default_random_engine rng(params.seed);
        uniform_real_distribution<double> distribution(-1.0, 1.0);
        data.zero(); // important to do that
        for (unsigned i = 0; i < synthetic; ++i) {
            value_type *row = data[i];
            for (unsigned j = 0; j < D; ++j) {
                row[j] = distribution(rng);
            }
        }
    }
    else {
        get_level_time=data.hnsw_dataload(data_path, D, skip, gap,labels,points_num,params.nsg_R,params.seed);
    }
    if (noise != 0) {
        if (!std::is_floating_point<value_type>::value) {
            throw std::runtime_error("noise injection not implemented for non-floating-point value.");
        }
        boost::random::ranlux64_base_01 rng;
        double sum = 0, sum2 = 0;
        for (unsigned i = 0; i < data.size(); ++i) {
            for (unsigned j = 0; j < data.dim(); ++j) {
                value_type v = data[i][j];
                sum += v;
                sum2 += v * v;
            }
        }
        double total = double(data.size()) * data.dim();
        double avg2 = sum2 / total, avg = sum / total;
        double dev = sqrt(avg2 - avg * avg);
        cerr << "Adding Gaussian noise w/ " << noise << "x sigma(" << dev << ")..." << endl;
        boost::normal_distribution<double> gaussian(0, noise * dev);
        for (unsigned i = 0; i < data.size(); ++i) {
            for (unsigned j = 0; j < data.dim(); ++j) {
                data[i][j] += gaussian(rng);
            }
        }
    }
    if (vm.count("l2norm")) {
        cerr << "L2-normalizing data..." << endl;
        data.normalize2();
    }
    
    printf("Size1: %d\n", data.size());
    std::ofstream savestart(res_path,std::ios::out | std::ios::app);
    if (savestart.is_open())
    {
        savestart<<"start*************************************************************************************************************"<<endl;
        savestart<<"get_level_time"<<","<<get_level_time<<endl;
    }
    savestart.close();
    auto s = std::chrono::high_resolution_clock::now();
    vector<int> levels;
    for(int  i=points_num.size()-1;i>=0;i--)
    {
        for(int j=0;j<points_num[i];j++)
        {
            levels.push_back(i);
        }
    }
    KGraph::IndexInfo info;
    KGraph *kgraph = KGraph::create(); //(oracle, params, &info);
    MatrixOracle<value_type, metric::l2sqr> oracle(data);
    size_t max_elements_=data.size();
    size_t cur_element_count=data.size();
    size_t M_=params.nsg_R;
    size_t maxM_=params.nsg_R;
    size_t maxM0_=2*params.nsg_R;
    size_t ef_construction_;
    size_t size_links_per_element_ = maxM_ * sizeof(unsigned) + sizeof(unsigned);
    size_t size_links_level0_=maxM0_*sizeof(unsigned)+sizeof(unsigned);
    size_t size_data_per_element_=size_links_level0_+sizeof(float)*data.dim()+sizeof(size_t);
    size_t offsetData_=size_links_level0_;
    size_t label_offset_ = size_links_level0_+sizeof(float)*data.dim();
    size_t offsetLevel0_ = 0;
    char *data_level0_memory_=nullptr;
    char **linkLists_ = (char **) malloc(sizeof(void *) * data.size());
    int maxlevel_=levels[0];
    double mult_=1 / log(1.0 * M_);
    ef_construction_=params.search_K;
    unsigned int enterpoint_node_=0;
    for(int cur_c=0;cur_c<levels.size();cur_c++)
    {
        if(levels[cur_c]>0)
        {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * levels[cur_c] + 1);
            memset(linkLists_[cur_c], 0, size_links_per_element_ * levels[cur_c] + 1);
        }
        else
        {
            break;
        }
    }
    int size=0;
    unsigned temp_ep=0;
    unsigned *ep_ptr=&temp_ep;
    for(int level=points_num.size()-1;level>=0;level--)
    {
        size+=points_num[level];
        oracle.set_size(size);
        cout<<"oracle size: "<<oracle.size()<<endl;
        if(size==1&&level>0)
        {
            unsigned* cll=(unsigned *)(linkLists_[0] + (level - 1) * size_links_per_element_);
            unsigned short linkCount = 0;
            *((unsigned short int*)(cll))=*((unsigned short int *)&linkCount);
            if(level==points_num.size()-1)
            {
                enterpoint_node_=0;
            }
            continue;
        }
        if(level>0)
        {
            kgraph->buildFastHNSW(oracle, params, &info,output_path.c_str(),res_path.c_str(),data_level0_memory_,linkLists_,level,ep_ptr);
            if(level==points_num.size()-1)
            {
                enterpoint_node_=temp_ep;
            }
        }
        if(level==0)
        {
            params.nsg_R*=2;
            kgraph->buildFastHNSW(oracle, params, &info,output_path.c_str(),res_path.c_str(),data_level0_memory_,linkLists_,level,ep_ptr,size_data_per_element_);
            long peakmemory=getPeakMemoryUsage();
  	        std::cout<<"fasthnsw peakmemory: "<<peakmemory<<std::endl;
            data_level0_memory_=(char *)malloc(oracle.size()*size_data_per_element_);
            memset(data_level0_memory_,0,oracle.size()*size_data_per_element_);
            kgraph->FatherSwapHNSWLayer0(oracle, params,data_level0_memory_);
            if(level==points_num.size()-1)
            {
                enterpoint_node_=temp_ep;
            }
        }
    }
    delete kgraph;
    if(data_level0_memory_==nullptr)
    {
        cout<<"error"<<endl;
    }
    for(int internal_id=0;internal_id<labels.size();internal_id++)
    {
        size_t label=labels[internal_id];
        char* c_data=data.getdata(internal_id);
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_),c_data,sizeof(float)*data.dim());
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(label));
    }
    auto e = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> diff = e - s;
    double time=diff.count()+get_level_time;
  	std::cout << "fasthnsw indexing time: " << time << "\n";
    
    std::ofstream saveresult(res_path,std::ios::out | std::ios::app);
    if (saveresult.is_open())
    {
        saveresult<<"fasthnsw indexing time"<<","<<time<<endl;
        saveresult<<"end***************************************************************************************************************"<<endl<<endl;
    }
    saveresult.close();
    std::ofstream output(output_path, std::ios::binary);
    std::streampos position;
    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);

    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);

    output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count; i++) {
        unsigned int linkListSize = levels[i] > 0 ? size_links_per_element_ * levels[i] : 0;
        writeBinaryPOD(output, linkListSize);
        if (linkListSize)
            output.write(linkLists_[i], linkListSize);
    }
    output.close();
    
    
    free(data_level0_memory_);
    for (int i = 0; i < cur_element_count; i++) {
        if (levels[i] > 0)
            free(linkLists_[i]);
    }
    free(linkLists_);
    


    cout<<"final"<<endl;
    long peakmemory=getPeakMemoryUsage();
  	std::cout<<"fasthnsw peakmemory: "<<peakmemory<<std::endl;
    return 0;
}


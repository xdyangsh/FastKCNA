# FastKCNA

## Experimental Environment

All experiments are conducted on a server equipped with 2 Intel(R) Xeon(R) Silver 4210R CPUs, each of which has 10 cores, and 256 GB DRAM as the main memory. The OS version is CentOS 7.9.2009. All codes were written by {C++} and compiled by {g++ 11.3}. The SIMD instructions are enabled to accelerate the distance computations.

## Building Instruction

### Prerequisites

cmake g++ OpenMP Boost

### Compile

```
cd FastKCNA/
cmake .
make
```

## Usage

```
./fastnsg --data yourdatapath.lshkit -K 500 -L 500 -S 12 -R 100 -I 6 --search_L 80 --nsg_R 50 --search_K 500 --massq_S 10 --loop_i 2 --step 10 --angle 60 --output ./fastnsg.index --result ./fastnsg.csv
```

```
./fasthnsw --data yourdatapath.lshkit -K 200 -L 200 -S 10 -R 100 -I 6 --search_L 40 --nsg_R 40 --search_K 200 --massq_S 10 --loop_i 2 --step 10 --angle 60 --output ./fasthnsw.index --result ./fasthnsw.csv
```

```
./fasttaumng --data yourdatapath.lshkit -K 500 -L 500 -S 12 -R 100 -I 6 --search_L 80 --nsg_R 50 --search_K 500 --massq_S 10 --tau 10 --loop_i 2 --step 10 --angle 60 --output ./fasttaumng.index --result ./fasttaumng.csv
```

```
./fastnsw --data yourdatapath.lshkit -K 200 -L 200 -S 10 -R 100 -I 6 --search_L 40 --nsg_R 40 --search_K 200 --massq_S 10 --loop_i 2 --step 10 --angle 60 --output ./fastnsw.index --result ./fastnsw.csv
```

Note : You can use fvec2lshkit.cpp to convert fvec format data into lshkit format data. In FastHNSW, nsg_R has the same effect as M in the original HNSW.

The candidate neighbor set size will be the largest of K, L, search_L, search_K.

The index structure of FastHNSW is the same as that of HNSW in hnswlib.

#include "gg.h"
#include "csr_graph.h"
#include <cusparse_v2.h>

unsigned CSRGraph::init() {
  row_start = edge_dst = NULL;
  edge_data            = NULL;
  node_data            = NULL;
  nnodes = nedges = 0;
  device_graph    = false;

  return 0;
}

unsigned CSRGraph::allocOnHost(bool no_edge_data) {
  assert(nnodes > 0);
  assert(!device_graph);

  if (row_start != NULL) // already allocated
    return true;

  size_t mem_usage = ((nnodes + 1) + nedges) * sizeof(index_type) +
                     (nnodes) * sizeof(node_data_type);
  if (!no_edge_data)
    mem_usage += (nedges) * sizeof(edge_data_type);

  printf("Host memory for graph: %3u MB\n", mem_usage / 1048756);

  row_start = (index_type*)calloc(nnodes + 1, sizeof(index_type));
  edge_dst  = (index_type*)calloc(nedges, sizeof(index_type));
  if (!no_edge_data)
    edge_data = (edge_data_type*)calloc(nedges, sizeof(edge_data_type));
  node_data = (node_data_type*)calloc(nnodes, sizeof(node_data_type));

  return ((no_edge_data || edge_data) && row_start && edge_dst && node_data);
}

unsigned CSRGraph::allocOnDevice(bool no_edge_data) {
  if (edge_dst != NULL) // already allocated
    return true;

  assert(edge_dst == NULL); // make sure not already allocated

  if (nedges > 0)
    check_cuda(cudaMalloc((void**)&edge_dst, nedges * sizeof(index_type)));
  check_cuda(cudaMalloc((void**)&row_start, (nnodes + 1) * sizeof(index_type)));

  if (!no_edge_data && (nedges > 0))
    check_cuda(cudaMalloc((void**)&edge_data, nedges * sizeof(edge_data_type)));
  if (nnodes > 0)
    check_cuda(cudaMalloc((void**)&node_data, nnodes * sizeof(node_data_type)));

  device_graph = true;

  assert(((nedges == 0) || edge_dst) &&
         (no_edge_data || (nedges == 0) || edge_data) && row_start &&
         ((nnodes == 0) || node_data));
  return true;
}

unsigned CSRGraphTex::allocOnDevice(bool no_edge_data) {
  if (CSRGraph::allocOnDevice(no_edge_data)) {
    assert(sizeof(index_type) <= 4);     // 32-bit only!
    assert(sizeof(node_data_type) <= 4); // 32-bit only!

    cudaResourceDesc resDesc;

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType           = cudaResourceTypeLinear;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32; // bits per channel

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    resDesc.res.linear.devPtr      = edge_dst;
    resDesc.res.linear.sizeInBytes = nedges * sizeof(index_type);
    check_cuda(cudaCreateTextureObject(&edge_dst_tx, &resDesc, &texDesc, NULL));

    resDesc.res.linear.devPtr      = row_start;
    resDesc.res.linear.sizeInBytes = (nnodes + 1) * sizeof(index_type);
    check_cuda(
        cudaCreateTextureObject(&row_start_tx, &resDesc, &texDesc, NULL));

    resDesc.res.linear.devPtr      = node_data;
    resDesc.res.linear.sizeInBytes = (nnodes) * sizeof(node_data_type);
    check_cuda(
        cudaCreateTextureObject(&node_data_tx, &resDesc, &texDesc, NULL));

    return 1;
  }

  return 0;
}


CSRGraph::CSRGraph() { this->row_start= NULL; init(); }


unsigned CSRGraph::readFromGR(char file[], char binFile[], int* d_row_start, int* d_edge_dst, float* d_B, int FV_size ) {
  std::ifstream cfile;
  cfile.open(file);

  int masterFD = open(file, O_RDONLY);
  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
    abort();
  }
  size_t masterLength = buf.st_size;

  int _MAP_BASE = MAP_PRIVATE;

  void* m1 = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m1 == MAP_FAILED) {
    m1 = 0;
    printf("FileGraph::structureFromFile: mmap failed.\n");
    abort();
  }

  ggc::Timer t("graphreader");
  t.start();

  // parse file
  uint64_t* fptr                           = (uint64_t*)m1;
  __attribute__((unused)) uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t numNodes   = le64toh(*fptr++);
  uint64_t numEdges   = le64toh(*fptr++);
  uint64_t* outIdx    = fptr;
  fptr += numNodes;
  uint32_t* fptr32 = (uint32_t*)fptr;
  uint32_t* outs   = fptr32;
  fptr32 += numEdges;
  if (numEdges % 2)
    fptr32 += 1;
 // edge_data_type* edgeData = (edge_data_type*)fptr32;

  // cuda.
  nnodes = numNodes;
  nedges = numEdges;
  printf("nnodes=%d, nedges=%d, sizeEdge=%d.\n", nnodes, nedges, sizeEdgeTy);
  allocOnHost(true);

  row_start[0] = 0;

  for (unsigned ii = 0; ii < nnodes; ++ii) {
    row_start[ii + 1] = le64toh(outIdx[ii]);
    index_type degree = row_start[ii + 1] - row_start[ii];

    for (unsigned jj = 0; jj < degree; ++jj) {
      unsigned edgeindex = row_start[ii] + jj;

      unsigned dst = le32toh(outs[edgeindex]);
      if (dst >= nnodes)
        printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj,
               edgeindex);

      edge_dst[edgeindex] = dst;

    }
  }


int m = nnodes;
int n = FV_size;
//Extracting features from bin file
float f1;
float* h_B = (float*) malloc(n * m *sizeof(float));
std::ifstream fin(binFile, std::ios::binary);
int i=0;
while (fin.read(reinterpret_cast<char*>(&f1), sizeof(float))) {
    h_B[i] = f1;
    i++;
}
//Copying features matrix(B) to device memory
cudaError_t alloc;
alloc = cudaMemcpy(d_B, h_B, (m * n *sizeof(float)), cudaMemcpyHostToDevice);
if(alloc != cudaSuccess) {
    printf("Feature matrix memcpy failed\n");
}

//Creating row and col data for the Adj matrix(CSR format)
int* h_row_start ;
int* h_edge_dst ;
h_row_start = (int*)malloc((nnodes+1) * sizeof(int));
h_edge_dst = (int*)malloc((nedges+2708) * sizeof(int));
h_edge_dst[0] = 0;
for(int i=0;i<(nnodes+1);i++) {
    h_row_start[i] = int(row_start[i]) + i;
}
int deg,flag;
for(int i=0; i< nnodes; ++i) {
deg = int(row_start[i+1]) - int(row_start[i]);
//printf("deg is %d row is %d\n", deg,i);
flag = 0;
for(int j=0;j<(deg+1);j++) {
    //printf("edge_dst[i+j] is %d\n",edge_dst[i+j]);
    if((edge_dst[int(row_start[i])+j] > i) && !flag) {
        h_edge_dst[i+int(row_start[i])+j] = i;
//          printf("h_edge_dst is %d in i index is %d\n",h_edge_dst[i+j+int(row_start[i])], (i+j+int(row_start[i])));
        flag = 1;
    }
    else {
        if(flag == 0) {
            h_edge_dst[i+int(row_start[i])+j] = edge_dst[int(row_start[i]) + j];
//          printf("h_edge_dst is %d index is %d\n",h_edge_dst[i+j+int(row_start[i])], (i+j+int(row_start[i])));
        } else {
            h_edge_dst[i+int(row_start[i])+j] = edge_dst[int(row_start[i]) + j - 1];
//          printf("h_edge_dst is %d index is %d\n",h_edge_dst[i+j+int(row_start[i])], (i+j+int(row_start[i])));
}
}

}
}

alloc = cudaMemcpy(d_row_start, h_row_start,((nnodes+1) * sizeof(int)), cudaMemcpyHostToDevice);
if(alloc != cudaSuccess) {
    printf("row info memcpy failed \n");
}
alloc = cudaMemcpy(d_edge_dst, h_edge_dst,((nedges+nnodes) * sizeof(int)), cudaMemcpyHostToDevice);
if(alloc != cudaSuccess) {
    printf("col info memcpy failed \n");
}
  free(h_B);
  free(h_row_start);
  free(h_edge_dst);
  fin.close(); 
  cfile.close(); // probably galois doesn't close its file due to mmap.
  t.stop();
  return 0;
}

 void SpMM(float* nnz_data, int* row, int* col, float* d_B, float* d_C, int FV_size, int m, int nnz) {
//Allocating memory for output matrix
int n = FV_size;
//float *h_C = (float*)malloc(m * m * sizeof(float));
//if(!h_C) {
//    printf("malloc for output matrix failed\n");
//}
//cudaError_t alloc;

//Prepping for Cusparse function
cusparseHandle_t cusparse1 = NULL;
cusparseMatDescr_t      descrA  ;
cusparseCreateMatDescr(&descrA);
cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

const float alp = 1;
const float bet = 1;
const float* alpha = &alp;
const float* beta = &bet;

cusparseStatus_t result;
cusparseCreate(&cusparse1);
result = cusparseScsrmm(cusparse1,CUSPARSE_OPERATION_NON_TRANSPOSE, m, n , m, nnz, alpha, descrA, nnz_data, row, col , d_B, m , beta , d_C, m);
if(result != CUSPARSE_STATUS_SUCCESS) {
    printf("Cusparse failed\n");
}

//alloc = cudaMemcpy(h_C, d_C, m*m*sizeof(float), cudaMemcpyDeviceToHost);
//if(alloc != cudaSuccess) {
//    printf("output matrix memcpy failed\n");
//}
//int count = 0;
//for(int i=0;i<(2708*2708);i++) {
//    if(h_C[i] != 0.0) {
//	count++;
//    }
//}
//printf("\n %d \n", count);
  return;
}

unsigned CSRGraph::read(char file[], int* num_nodes, int* num_edges) {
  std::ifstream cfile;
  cfile.open(file);

  int masterFD = open(file, O_RDONLY);
  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
    abort();
  }
  size_t masterLength = buf.st_size;

  int _MAP_BASE = MAP_PRIVATE;

  void* m1 = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m1 == MAP_FAILED) {
    m1 = 0;
    printf("FileGraph::structureFromFile: mmap failed.\n");
    abort();
  }

  ggc::Timer t("graphreader");
  t.start();

  // parse file
  uint64_t* fptr                           = (uint64_t*)m1;
  __attribute__((unused)) uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t numNodes   = le64toh(*fptr++);
  uint64_t numEdges   = le64toh(*fptr++);
  uint64_t* outIdx    = fptr;
  fptr += numNodes;
  uint32_t* fptr32 = (uint32_t*)fptr;
  uint32_t* outs   = fptr32;
  fptr32 += numEdges;
  if (numEdges % 2)
    fptr32 += 1;
  int num_node = numNodes;
  int num_edge = numEdges;
  num_nodes = &num_node;
  num_edges = &num_edge;
  cfile.close();
  return 0;
}

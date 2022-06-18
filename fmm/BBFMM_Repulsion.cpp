#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "BBFMM2D/header/BBFMM2D.hpp"

namespace py = pybind11;
using namespace std;

//The properties of a numpy array are obtained through the request method
// that returns a structure of the form :

//struct buffer_info {
//    void *ptr;
//    size_t itemsize;
//    std::string format;
//    int ndim;
//    std::vector<size_t> shape;
//    std::vector<size_t> strides;
//};

//Laplacian kernel is not among the predefined kernels of BBFMM2D
class laplacian: public kernel_Base {
public:
virtual double kernel_Func(Point r0, Point r1){
   
    double r = sqrt((r0.x-r1.x)*(r0.x-r1.x) + (r0.y-r1.y)*(r0.y-r1.y));
    double result = 1.0/r; 
    if(isinf(result) || isnan(result))
    {
        return 0;
    }
    else
    {
        return result;
    }
  }
};

//Tranform the input into a vector of points (structure of BBFMM2D)
void loc2Points(py::array_t<double, py::array::c_style | py::array::forcecast> locations, unsigned long N, vector<Point>& particles){
    
    py::buffer_info loc = locations.request();
    double *ptrLoc = (double *) loc.ptr;
    
    //cout << "max coordinate is: " << *max_element(ptrLoc , ptrLoc + N) << endl;
    //cout << "min coordinate is: " << *min_element(ptrLoc , ptrLoc + N) << endl;
    
    for(int k = 0 ; k < N ; k++){
    
        Point newPoint(ptrLoc[k], ptrLoc[N+k]); //locations is 2*N array so xi at position i means yi at i+N
        particles.push_back(newPoint);   
    } 
}

//Construct the set of weights
double* chargesSet_multiple(py::array_t<double, py::array::c_style | py::array::forcecast> locations, unsigned long N, unsigned m){ //in 2D, there are three set of charges (kx, ky and a vector of ones)
   
   py::buffer_info loc = locations.request();
   double *ptrLoc = (double *) loc.ptr;
   double* charges = new double[N*m];
   
   //first set is unitary weights
   for(int k = 0 ; k < N ; k++){
       charges[k] = 1.0;
   }
   
   //second and first set are locations.x and locations.y
   
   for(int j = 0; j < 2*N ; j++){
       charges[N+j] = ptrLoc[j];
   }
   
   return charges;

}

//The output will be a N*3 array, each column corresponding to a set of charges
py::array_t<double, py::array::c_style | py::array::forcecast> bbfmm_multiple(py::array_t<double, py::array::c_style | py::array::forcecast> locations, unsigned short nCheb){
    
    //Pybind11 buffers
    py::buffer_info loc = locations.request();
    
    //cout<< "loc.shape[0] = "<< loc.shape[0] << endl;
    //cout<< "loc.shape[1] = "<< loc.shape[1] << endl;
    //cout << "loc.size = " << loc.size << endl;
     
     //Check the number of dimensions
    if (loc.ndim != 2){
        throw runtime_error("Input locations should be of shape (2,N)");
    }
    
    unsigned m = 3;
    unsigned long N = loc.shape[1];
    //cout << endl << "Number of charges:"  << N << endl;

    py::array_t<double, py::array::c_style | py::array::forcecast> output({int(N*m), 1});
    //py::array_t<double> output = py::array_t<double>(loc.size); does not work

    py::buffer_info out = output.request();
    
    double *ptrOut = (double *) out.ptr;
    
    vector<Point> particles;
    
    loc2Points(locations, N, particles);
    
    double* charges = chargesSet_multiple(locations, N, m);
    
    /****************      Building fmm tree     **************/
    
    H2_2D_Tree Atree(nCheb, charges, particles, N, m);// Build the fmm tree

    /****************    Calculating potential   *************/
    
    double* potentials;
    potentials = new double[N*m];
    /* Other options of kernel:
     LOGARITHM:          kernel_Logarithm
     ONEOVERR2:          kernel_OneOverR2
     GAUSSIAN:           kernel_Gaussian
     QUADRIC:            kernel_Quadric
     INVERSEQUADRIC:     kernel_InverseQuadric
     THINPLATESPLINE:    kernel_ThinPlateSpline
     */
    //kernel_OneOverR2 A;
    laplacian A;
    A.calculate_Potential(Atree, potentials);
    
    for(int l = 0 ; l < N*m ; l++){
        ptrOut[l] = potentials[l];   
    }
    
    //Resize as a standard FMM output (ie found in the literature)
    //output.resize({int(N),int(m)});
    
         /********       Clean Up        *******/
    delete []charges;
    delete []potentials;
    
    return output;
}

//Construct the set of weights
double* chargesSet_single(py::array_t<double, py::array::c_style | py::array::forcecast> weights, unsigned long N, unsigned m){

   py::buffer_info H = weights.request();
   double *ptrH = (double *) H.ptr;
   double* charges = new double[N];

   for(int k = 0 ; k < N ; k++){
       charges[k] = ptrH[k];
   }
   return charges;
}

//The output will be a N*3 array, each column corresponding to a set of charges
py::array_t<double, py::array::c_style | py::array::forcecast> bbfmm_single(py::array_t<double, py::array::c_style | py::array::forcecast> locations, py::array_t<double, py::array::c_style | py::array::forcecast> weights, int kernelID, unsigned short nCheb){

    //Avoid getting printed info from BBFMM2D
    //streambuf* cout_sbuf = std::cout.rdbuf(); // save original sbuf
    //ofstream   fout("/dev/null");
    //cout.rdbuf(fout.rdbuf()); // redirect 'cout' to a 'fout'
    //cout.rdbuf(cout_sbuf); // restore the original stream buffer

    //Pybind11 buffers
    py::buffer_info loc = locations.request();

    //Check the number of dimensions
    if (loc.ndim != 2){
        throw runtime_error("Input locations should be of shape (2,N)");
    }

    unsigned m = 1;
    unsigned long N = loc.shape[1];

    py::array_t<double, py::array::c_style | py::array::forcecast> output({int(N), int(m)});
    //py::array_t<double> output = py::array_t<double>(loc.size) -> does not work

    py::buffer_info out = output.request();

    double *ptrOut = (double *) out.ptr;

    vector<Point> particles;

    loc2Points(locations, N, particles);

    double* charges = chargesSet_single(weights, N, m);

    /****************      Building fmm tree     **************/

    H2_2D_Tree Atree(nCheb, charges, particles, N, m);// Build the fmm tree

    /****************    Calculating potential   *************/

    double* potentials;
    potentials = new double[N*m];

    /* Other options of kernel:
     LOGARITHM:          kernel_Logarithm
     ONEOVERR2:          kernel_OneOverR2
     GAUSSIAN:           kernel_Gaussian
     QUADRIC:            kernel_Quadric
     INVERSEQUADRIC:     kernel_InverseQuadric
     THINPLATESPLINE:    kernel_ThinPlateSpline
     */

    //Choice of kernel, be careful of switch scope
    switch(kernelID){
        case 1:
        {
            laplacian A;
            A.calculate_Potential(Atree, potentials);
            break;
        }
        case 2:
        {
            kernel_OneOverR2 A;
            A.calculate_Potential(Atree, potentials);
            break;
        }
        case 3:
        {
            kernel_Gaussian A;
            A.calculate_Potential(Atree, potentials);
            break;
        }
    }


    for(int l = 0 ; l < N*m ; l++){
        ptrOut[l] = potentials[l];
    }

     /********       Clean Up        *******/
    delete []charges;
    delete []potentials;

    return output;
}


PYBIND11_MODULE(bbfmm, m) {
    m.doc() = "Calculate the Repulsive Force gradient with BBFMM2D"; // optional
    m.def("multiple", &bbfmm_multiple, py::return_value_policy::take_ownership);

    m.doc() = "Calculate the Repulsive Force gradient with BBFMM2D"; // optional
    m.def("single", &bbfmm_single, py::return_value_policy::reference);
}

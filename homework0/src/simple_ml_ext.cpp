#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t batch_size = m;
    size_t num_classes = k;
    size_t input_dimension = n;
    for (size_t b = 0; b < batch_size; b += batch)
    {
        size_t base = input_dimension * b;
        float* output = new float[batch * num_classes];
        for (size_t i = 0; i < batch; i++ )
            for (size_t j = 0; j < num_classes; j++)
            {
                output[ i * num_classes + j ] = 0;
                for (size_t t=0; t < input_dimension; t++)
                    output[ i * num_classes + j ] += X[ base + i * input_dimension + t] * theta[ t * num_classes + j ];
            }
        
        float* Z = new float[batch * num_classes];
        for (size_t i = 0; i < batch; i++ )
        {
            float sum = 0;
            for (size_t j = 0; j < num_classes; j++)
                Z[ i * num_classes + j ] = exp(output[ i * num_classes + j ]),sum += Z[ i * num_classes + j ];
            for (size_t j = 0; j < num_classes; j++)
                Z[ i * num_classes + j ] /= sum;//,printf("%d %d %f\n",i,j,Z[i*num_classes+j]);
            Z[ i * num_classes + y[b + i] ] -= 1;
        }
        
        float* grad = new float[input_dimension * num_classes];
        for (size_t i = 0; i < input_dimension; i++)
            for (size_t j = 0; j < num_classes; j++)
            {
                grad[ i * num_classes + j ] = 0;
                for (size_t t=0; t < batch; t++)
                    grad[ i * num_classes + j ] += X[ base + t * input_dimension + i] * Z[ t * num_classes + j ];
            }
        for (size_t i = 0; i < input_dimension * num_classes; i++) theta[i] -= grad[i] * lr / batch;
        delete output;
        delete Z;
        delete grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

#ifndef TENSOR_IMPL_HPP
#define TENSOR_IMPL_HPP

#include "base.hpp"
#include <memory>
#include <armadillo>

namespace Malphax
{
    class TensorImpl : public std::enable_shared_from_this<TensorImpl>
    {
    public:
        arma::mat data;
        arma::mat grad;
        unsigned long long n_rows;
        unsigned long long n_cols;
        bool requires_grad;
        std::shared_ptr<autograd::Function> grad_fn;

        TensorImpl() : requires_grad(false), n_rows(0), n_cols(0)
        {}

        TensorImpl(unsigned long n_rows, unsigned long n_cols, const std::string &init = "norm",
                   bool requires_grad = true)
                : requires_grad(requires_grad), n_rows(n_rows), n_cols(n_cols)
        {
            if (init == "norm")
                data = arma::randn(n_rows, n_cols);
            else if (init == "zeros")
                data = arma::zeros(n_rows, n_cols);
            else if (init == "ones")
                data = arma::ones(n_rows, n_cols);
            else
                data = arma::randn(n_rows, n_cols);

            grad = arma::zeros(n_rows, n_cols);
        }

        explicit TensorImpl(const arma::mat &data_in, bool requires_grad = true)
                : data(data_in), n_rows(data_in.n_rows), n_cols(data_in.n_cols), requires_grad(requires_grad)
        {
            grad = arma::zeros(data.n_rows, data.n_cols);
        }

        std::shared_ptr<TensorImpl> shared_this()
        {
            return shared_from_this();
        }

        void zero_grad()
        {
            grad.zeros(data.n_rows, data.n_cols);
        }
    };
}

#endif // TENSOR_IMPL_HPP
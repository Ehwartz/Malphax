#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include "tensor.hpp"
#include "autograd.hpp"

namespace Malphax
{
    inline Tensor operator+(const Tensor &A, const Tensor &B)
    {
        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = A.n_cols();
        result_impl->data = A.data() + B.data();
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad() || B.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Add_>(
                    const_cast<Tensor *>(&A),
                    const_cast<Tensor *>(&B),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor operator-(const Tensor &A, const Tensor &B)
    {
        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = A.n_cols();
        result_impl->data = A.data() - B.data();
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad() || B.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Sub_>(
                    const_cast<Tensor *>(&A),
                    const_cast<Tensor *>(&B),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor operator*(const Tensor &A, const Tensor &B)
    {
        if ((A.n_rows() != B.n_rows() || A.n_cols() != B.n_cols()) &&
            (A.data().size() != 1 && B.data().size() != 1))
        {
            throw std::runtime_error("Element-wise multiplication requires tensors of the same shape");
        }

        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = B.data().size() == 1 ? A.n_rows() : B.n_rows();
        result_impl->n_cols = B.data().size() == 1 ? A.n_cols() : B.n_cols();

        if (A.data().size() != 1 && B.data().size() != 1)
        {
            result_impl->data = A.data() % B.data();
        }
        else if (A.data().size() == 1)
        {
            result_impl->data = A.data()(0, 0) * B.data();
        }
        else if (B.data().size() == 1)
        {
            result_impl->data = A.data() * B.data()(0, 0);
        }

        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad() || B.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Dot_>(
                    const_cast<Tensor *>(&A),
                    const_cast<Tensor *>(&B),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor operator*(const Tensor &A, const double &B)
    {
        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = A.n_cols();
        result_impl->data = A.data() * B;
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::ScalarDot_>(
                    const_cast<Tensor *>(&A),
                    B,
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor operator*(const double &A, const Tensor &B)
    {
        return B * A;
    }

    inline Tensor operator/(const Tensor &A, const Tensor &B)
    {
        if ((A.n_rows() != B.n_rows() || A.n_cols() != B.n_cols()) &&
            (A.data().size() != 1 && B.data().size() != 1))
        {
            throw std::runtime_error("Element-wise division requires tensors of the same shape");
        }

        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = B.data().size() == 1 ? A.n_rows() : B.n_rows();
        result_impl->n_cols = B.data().size() == 1 ? A.n_cols() : B.n_cols();

        if (A.data().size() != 1 && B.data().size() != 1)
        {
            result_impl->data = A.data() / B.data();
        }
        else if (A.data().size() == 1)
        {
            result_impl->data = A.data()(0, 0) / B.data();
        }
        else if (B.data().size() == 1)
        {
            result_impl->data = A.data() / B.data()(0, 0);
        }

        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad() || B.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Div_>(
                    const_cast<Tensor *>(&A),
                    const_cast<Tensor *>(&B),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor operator/(const Tensor &A, const double &B)
    {
        if (B == 0.0)
        {
            throw std::runtime_error("Division by zero");
        }

        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = A.n_cols();
        result_impl->data = A.data() / B;
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::ScalarDiv_>(
                    const_cast<Tensor *>(&A),
                    B,
                    result_impl,
                    true
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor operator/(const double &A, const Tensor &B)
    {
        if (arma::any(arma::vectorise(B.data()) == 0.0))
        {
            throw std::runtime_error("Division by zero in tensor elements");
        }

        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = B.n_rows();
        result_impl->n_cols = B.n_cols();
        result_impl->data = A / B.data();
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (B.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::ScalarDiv_>(
                    const_cast<Tensor *>(&B),
                    A,
                    result_impl,
                    false
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor matmul(const Tensor &A, const Tensor &B)
    {
        if (A.n_cols() != B.n_rows())
        {
            throw std::runtime_error("Matrix multiplication dimension mismatch");
        }

        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = B.n_cols();
        result_impl->data = A.data() * B.data();
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad() || B.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::MatMul_>(
                    const_cast<Tensor *>(&A),
                    const_cast<Tensor *>(&B),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor dot(const Tensor &A, const Tensor &B)
    {
        if (A.n_rows() != B.n_rows() || A.n_cols() != B.n_cols())
        {
            throw std::runtime_error("Element-wise multiplication requires tensors of the same shape");
        }

        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = A.n_cols();
        result_impl->data = A.data() % B.data();
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad() || B.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Dot_>(
                    const_cast<Tensor *>(&A),
                    const_cast<Tensor *>(&B),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor sum(const Tensor &A, unsigned long long dim)
    {
        if (dim > 1)
        {
            throw std::runtime_error("Dimension must be either 0 (rows) or 1 (columns)");
        }

        auto result_impl = std::make_shared<TensorImpl>();

        if (dim == 0)
        {
            result_impl->data = arma::sum(A.data(), 0);
            result_impl->n_rows = 1;
            result_impl->n_cols = A.n_cols();
        }
        else
        {
            result_impl->data = arma::sum(A.data(), 1);
            result_impl->n_rows = A.n_rows();
            result_impl->n_cols = 1;
        }

        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Sum_>(
                    const_cast<Tensor *>(&A),
                    result_impl,
                    dim
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor mean(const Tensor &A, unsigned long long dim)
    {
        if (dim > 1)
        {
            throw std::runtime_error("Dimension must be either 0 (rows) or 1 (columns)");
        }

        auto result_impl = std::make_shared<TensorImpl>();

        if (dim == 0)
        {
            result_impl->data = arma::mean(A.data(), 0);
            result_impl->n_rows = 1;
            result_impl->n_cols = A.n_cols();
        }
        else
        {
            result_impl->data = arma::mean(A.data(), 1);
            result_impl->n_rows = A.n_rows();
            result_impl->n_cols = 1;
        }

        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Mean_>(
                    const_cast<Tensor *>(&A),
                    result_impl,
                    dim
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor exp(const Tensor &A)
    {
        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = A.n_cols();
        result_impl->data = arma::exp(A.data());
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Exp_>(
                    const_cast<Tensor *>(&A),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor log(const Tensor &A)
    {
        if (arma::any(arma::vectorise(A.data()) <= 0.0))
        {
            throw std::runtime_error("Log of zero or negative value");
        }

        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = A.n_cols();
        result_impl->data = arma::log(A.data());
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Log_>(
                    const_cast<Tensor *>(&A),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

    inline Tensor abs(const Tensor &A)
    {
        auto result_impl = std::make_shared<TensorImpl>();
        result_impl->n_rows = A.n_rows();
        result_impl->n_cols = A.n_cols();
        result_impl->data = arma::abs(A.data());
        result_impl->grad = arma::zeros(result_impl->n_rows, result_impl->n_cols);

        if (A.requires_grad())
        {
            result_impl->requires_grad = true;

            result_impl->grad_fn = std::make_shared<autograd::Abs_>(
                    const_cast<Tensor *>(&A),
                    result_impl
            );
        }

        return Tensor(result_impl);
    }

}

#endif // OPERATORS_HPP
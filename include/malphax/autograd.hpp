#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include "base.hpp"
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <armadillo>

namespace Malphax
{
    namespace autograd
    {
        class Add_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> B_impl;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;
            Tensor *B;

            Add_(Tensor *A, Tensor *B, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), B(B), A_impl(A->get_impl()), B_impl(B->get_impl()), C_impl(C_impl)
            {
                set_inputs(A_impl, B_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad) A_impl->grad += C_impl->grad;
                if (B_impl->requires_grad) B_impl->grad += C_impl->grad;
            }

            std::vector<Tensor *> parents() override
            {
                return {A, B};
            }
        };

        class Sub_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> B_impl;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;
            Tensor *B;

            Sub_(Tensor *A, Tensor *B, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), B(B), A_impl(A->get_impl()), B_impl(B->get_impl()), C_impl(C_impl)
            {
                set_inputs(A_impl, B_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad) A_impl->grad += C_impl->grad;
                if (B_impl->requires_grad) B_impl->grad -= C_impl->grad;
            }

            std::vector<Tensor *> parents() override
            {
                return {A, B};
            }
        };

        class MatMul_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> B_impl;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;
            Tensor *B;

            MatMul_(Tensor *A, Tensor *B, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), B(B), A_impl(A->get_impl()), B_impl(B->get_impl()), C_impl(C_impl)
            {

                set_inputs(A_impl, B_impl);
            }

            void backward() override
            {

                if (A_impl->requires_grad) A_impl->grad += C_impl->grad * B_impl->data.t();

                if (B_impl->requires_grad) B_impl->grad += A_impl->data.t() * C_impl->grad;
            }

            std::vector<Tensor *> parents() override
            {
                return {A, B};
            }
        };

        class Dot_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> B_impl;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;
            Tensor *B;

            Dot_(Tensor *A, Tensor *B, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), B(B), A_impl(A->get_impl()), B_impl(B->get_impl()), C_impl(C_impl)
            {

                set_inputs(A_impl, B_impl);
            }

            void backward() override
            {
                if (A_impl->data.n_rows == B_impl->data.n_rows && A_impl->data.n_cols == B_impl->data.n_cols)
                {
                    if (A_impl->requires_grad) A_impl->grad += C_impl->grad % B_impl->data;
                    if (B_impl->requires_grad) B_impl->grad += C_impl->grad % A_impl->data;
                }
                else if (A_impl->data.size() == 1)
                {
                    if (B_impl->requires_grad) B_impl->grad += C_impl->grad * A_impl->data(0, 0);
                }
                else if (B_impl->data.size() == 1)
                {
                    if (A_impl->requires_grad) A_impl->grad += C_impl->grad * B_impl->data(0, 0);
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A, B};
            }
        };

        class ScalarDot_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            double scalar;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;

            ScalarDot_(Tensor *A, double scalar, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), A_impl(A->get_impl()), scalar(scalar), C_impl(C_impl)
            {
                set_inputs(A_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad)
                {
                    A_impl->grad += C_impl->grad * scalar;
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A};
            }
        };

        class Div_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> B_impl;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;
            Tensor *B;

            Div_(Tensor *A, Tensor *B, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), B(B), A_impl(A->get_impl()), B_impl(B->get_impl()), C_impl(C_impl)
            {
                set_inputs(A_impl, B_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad)
                {
                    if (B_impl->data.size() == 1)
                    {
                        A_impl->grad += C_impl->grad / B_impl->data(0, 0);
                    }
                    else
                    {
                        A_impl->grad += C_impl->grad % (1.0 / B_impl->data);
                    }
                }

                if (B_impl->requires_grad)
                {
                    if (A_impl->data.size() == 1)
                    {
                        B_impl->grad -= C_impl->grad % (A_impl->data(0, 0) / arma::square(B_impl->data));
                    }
                    else if (B_impl->data.size() == 1)
                    {
                        double b_val = B_impl->data(0, 0);
                        B_impl->grad -= arma::accu(C_impl->grad % (A_impl->data / (b_val * b_val)));
                    }
                    else
                    {
                        B_impl->grad -= C_impl->grad % (A_impl->data / arma::square(B_impl->data));
                    }
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A, B};
            }
        };

        class ScalarDiv_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            double scalar;
            std::shared_ptr<TensorImpl> C_impl;
            bool tensor_numerator;
            Tensor *A;

            ScalarDiv_(Tensor *A, double scalar, std::shared_ptr<TensorImpl> C_impl, bool tensor_numerator)
                    : A(A), A_impl(A->get_impl()), scalar(scalar), C_impl(C_impl), tensor_numerator(tensor_numerator)
            {
                set_inputs(A_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad)
                {
                    if (tensor_numerator)
                    {
                        A_impl->grad += C_impl->grad * (1.0 / scalar);
                    }
                    else
                    {
                        A_impl->grad -= C_impl->grad % (scalar / arma::square(A_impl->data));
                    }
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A};
            }
        };

        class Sum_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> C_impl;
            unsigned long long dim;
            arma::uvec orig_dims;
            Tensor *A;

            Sum_(Tensor *A, std::shared_ptr<TensorImpl> C_impl, unsigned long long dim)
                    : A(A), A_impl(A->get_impl()), C_impl(C_impl), dim(dim)
            {
                orig_dims = {A_impl->n_rows, A_impl->n_cols};
                set_inputs(A_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad)
                {
                    if (dim == 0)
                    {
                        arma::mat ones_mat = arma::ones(orig_dims[0], 1);
                        A_impl->grad += ones_mat * C_impl->grad;
                    }
                    else if (dim == 1)
                    {
                        arma::mat ones_mat = arma::ones(1, orig_dims[1]);
                        A_impl->grad += C_impl->grad * ones_mat;
                    }
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A};
            }
        };

        class Mean_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> C_impl;
            unsigned long long dim;
            arma::uvec orig_dims;
            Tensor *A;


            Mean_(Tensor *A, std::shared_ptr<TensorImpl> C_impl, unsigned long long dim)
                    : A(A), A_impl(A->get_impl()), C_impl(C_impl), dim(dim)
            {
                orig_dims = {A_impl->n_rows, A_impl->n_cols};
                set_inputs(A_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad)
                {
                    if (dim == 0)
                    {
                        arma::mat ones_mat = arma::ones(orig_dims[0], 1);
                        A_impl->grad += ones_mat * C_impl->grad / static_cast<double>(orig_dims[0]);
                    }
                    else if (dim == 1)
                    {
                        arma::mat ones_mat = arma::ones(1, orig_dims[1]);
                        A_impl->grad += C_impl->grad * ones_mat / static_cast<double>(orig_dims[1]);
                    }
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A};
            }
        };


        class Exp_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;

            Exp_(Tensor *A, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), A_impl(A->get_impl()), C_impl(C_impl)
            {
                set_inputs(A_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad)
                {
                    A_impl->grad += C_impl->grad % arma::exp(A_impl->data);
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A};
            }
        };


        class Log_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;

            Log_(Tensor *A, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), A_impl(A->get_impl()), C_impl(C_impl)
            {
                set_inputs(A_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad)
                {
                    A_impl->grad += C_impl->grad % (1.0 / A_impl->data);
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A};
            }
        };

        class Abs_ : public Function
        {
        public:
            std::shared_ptr<TensorImpl> A_impl;
            std::shared_ptr<TensorImpl> C_impl;
            Tensor *A;

            Abs_(Tensor *A, std::shared_ptr<TensorImpl> C_impl)
                    : A(A), A_impl(A->get_impl()), C_impl(C_impl)
            {
                set_inputs(A_impl);
            }

            void backward() override
            {
                if (A_impl->requires_grad)
                {
                    arma::mat sign_matrix = arma::sign(A_impl->data);
                    sign_matrix.elem(arma::find(A_impl->data == 0)).zeros();

                    A_impl->grad += C_impl->grad % sign_matrix;
                }
            }

            std::vector<Tensor *> parents() override
            {
                return {A};
            }
        };
    }
}

#endif // AUTOGRAD_HPP
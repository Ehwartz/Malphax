#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "base.hpp"
#include "tensor_impl.hpp"
#include <memory>
#include <utility>
#include <vector>
#include <queue>
#include <armadillo>
#include <iostream>
#include <unordered_set>

namespace Malphax
{
    class Tensor
    {
    private:

        std::shared_ptr<TensorImpl> impl;

    public:

        Tensor() : impl(std::make_shared<TensorImpl>())
        {}


        Tensor(unsigned long n_rows, unsigned long n_cols, const std::string &init = "norm", bool requires_grad = true)
                : impl(std::make_shared<TensorImpl>(n_rows, n_cols, init, requires_grad))
        {}

        explicit Tensor(const arma::mat &data, bool requires_grad = true)
                : impl(std::make_shared<TensorImpl>(data, requires_grad))
        {}

        explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl(impl)
        {}

        Tensor(const Tensor &other) = default;

        Tensor(Tensor &&other) noexcept = default;

        Tensor &operator=(const Tensor &other) = default;

        Tensor &operator=(Tensor &&other) noexcept = default;

        const arma::mat &data() const
        { return impl->data; }

        arma::mat &data()
        { return impl->data; }

        const arma::mat &grad() const
        { return impl->grad; }

        arma::mat &grad()
        { return impl->grad; }

        unsigned long long n_rows() const
        { return impl->n_rows; }

        unsigned long long n_cols() const
        { return impl->n_cols; }

        bool requires_grad() const
        { return impl->requires_grad; }

        void set_requires_grad(bool requires_grad)
        { impl->requires_grad = requires_grad; }

        std::shared_ptr<autograd::Function> grad_fn() const
        { return impl->grad_fn; }

        void set_grad_fn(std::shared_ptr<autograd::Function> fn)
        { impl->grad_fn = fn; }

        std::shared_ptr<TensorImpl> get_impl() const
        { return impl; }

        void zero_grad()
        {
            impl->zero_grad();
        }

        void backward()
        {
            if (!impl->requires_grad)
            {
                return;
            }

            if (impl->grad.n_elem == 0 || (impl->grad.n_rows == 0 || impl->grad.n_cols == 0))
            {

                impl->grad = arma::zeros(impl->data.n_rows, impl->data.n_cols);
            }

            if (arma::accu(impl->grad) == 0)
            {
                impl->grad.ones(impl->data.n_rows, impl->data.n_cols);
            }

            std::unordered_set<std::shared_ptr<TensorImpl>> impl_refs;

            std::queue<std::shared_ptr<TensorImpl>> queue;
            queue.push(impl);

            std::unordered_set<TensorImpl *> visited;

            while (!queue.empty())
            {
                auto tensor_impl = queue.front();
                queue.pop();

                if (visited.find(tensor_impl.get()) != visited.end())
                {
                    continue;
                }
                visited.insert(tensor_impl.get());

                if (tensor_impl->grad_fn)
                {
                    tensor_impl->grad_fn->backward();

                    for (auto parent: tensor_impl->grad_fn->parents())
                    {
                        if (parent && parent->requires_grad())
                        {
                            impl_refs.insert(parent->get_impl());

                            if (parent->get_impl().get() != tensor_impl.get())
                                queue.push(parent->get_impl());
                        }
                    }

                    for (const auto &saved_impl: tensor_impl->grad_fn->input_tensor_impls)
                    {
                        impl_refs.insert(saved_impl);
                    }
                }
            }
        }
    };
}

#endif // TENSOR_HPP
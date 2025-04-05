#ifndef MALPHAX_BASE_HPP
#define MALPHAX_BASE_HPP

#include <armadillo>
#include <memory>
#include <vector>
#include <string>

namespace Malphax
{
    class TensorImpl;

    class Tensor;

    namespace autograd
    {
        class Function
        {
        public:
            virtual void backward() = 0;

            virtual std::vector<Tensor *> parents() = 0;

            std::vector<std::shared_ptr<TensorImpl>> input_tensor_impls;

            void set_inputs(const std::shared_ptr<TensorImpl> &A_impl, const std::shared_ptr<TensorImpl> &B_impl)
            {
                input_tensor_impls.push_back(A_impl);
                input_tensor_impls.push_back(B_impl);
            }

            void set_inputs(const std::shared_ptr<TensorImpl> &A_impl)
            {
                input_tensor_impls.push_back(A_impl);
            }

            virtual ~Function() = default;
        };

        class Add_;

        class Sub_;

        class MatMul_;

        class Dot_;

        class ScalarDot_;

        class Div_;

        class ScalarDiv_;

        class Sum_;

        class Mean_;

        class Exp_;

        class Log_;

        class Abs_;
    }

    Tensor operator+(const Tensor &A, const Tensor &B);

    Tensor operator-(const Tensor &A, const Tensor &B);

    Tensor operator*(const Tensor &A, const Tensor &B);

    Tensor operator*(const Tensor &A, const double &B);

    Tensor operator*(const double &A, const Tensor &B);

    Tensor operator/(const Tensor &A, const Tensor &B);

    Tensor operator/(const Tensor &A, const double &B);

    Tensor operator/(const double &A, const Tensor &B);

    Tensor matmul(const Tensor &A, const Tensor &B);

    Tensor dot(const Tensor &A, const Tensor &B);

    Tensor sum(const Tensor &A, unsigned long long dim);

    Tensor mean(const Tensor &A, unsigned long long dim);

    Tensor exp(const Tensor &A);

    Tensor log(const Tensor &A);

    Tensor abs(const Tensor &A);

}

#endif // MALPHAX_BASE_HPP
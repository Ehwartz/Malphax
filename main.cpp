
#include <armadillo>
#include <memory>
#include "include/malphax/malphax.hpp"
#include <iostream>

int main()
{

    {
        Malphax::Tensor a(4, 4, "ones");
        Malphax::Tensor b(4, 4, "ones");
        Malphax::Tensor c(4, 4, "ones");
        auto d = Malphax::exp(Malphax::matmul(a, b) * c + c / 16);
        auto e = Malphax::sum(d, 1);
        auto f = Malphax::sum(e, 0);
        f.backward();
        std::cout << "Gradient of a:\n" << a.grad() << std::endl;
        std::cout << "Gradient of b:\n" << b.grad() << std::endl;
        std::cout << "Gradient of c:\n" << c.grad() << std::endl;
    }

    {
        Malphax::Tensor a(4, 4, "ones");
        Malphax::Tensor b(4, 4, "ones");
        Malphax::Tensor c(4, 4, "ones");
        auto d = Malphax::mean(Malphax::mean(a + b * b * b * b, 1), 0);
        d.backward();
        std::cout << "Gradient of a:\n" << a.grad() << std::endl;
        std::cout << "Gradient of b:\n" << b.grad() << std::endl;
        std::cout << "Gradient of c:\n" << c.grad() << std::endl;
    }

    {
        Malphax::Tensor a(4, 4, "ones");
        Malphax::Tensor b(4, 4, "ones");
        Malphax::Tensor c(4, 4, "ones");
        auto d = Malphax::log(Malphax::matmul(a * a, b + c + a) * c + c * c);
        auto e = Malphax::sum(d, 1);
        auto f = Malphax::sum(e, 0);
        f.backward();
        std::cout << "Gradient of a:\n" << a.grad() << std::endl;
        std::cout << "Gradient of b:\n" << b.grad() << std::endl;
        std::cout << "Gradient of c:\n" << c.grad() << std::endl;
    }


    return 0;
}

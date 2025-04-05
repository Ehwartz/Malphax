
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

//int main()
//{
////    Malphax::Tensor t(3, 3);
////    std::cout<<t.data<<std::endl;
////    arma::mat A = arma::randn(3, 4);
////    arma::mat B = arma::randn(4, 5);
////    arma::mat C = arma::randn(3, 4);
////    std::cout<<A*B<<std::endl;
////    std::cout<<arma::dot(A, C)<<std::endl;
//
////    Malphax::Tensor A(4, 4), B(4, 4);
//
////    Malphax::Tensor a(2, 2);
////    Malphax::Tensor b(2, 2);
////    Malphax::Tensor c = a + b;
////    c.backward();
////    std::cout<<a.grad<<std::endl;
////    c.backward();
////    std::cout<<a.grad<<std::endl;
//
//    Malphax::Tensor a(1, 1, "ones");
//    Malphax::Tensor b(4, 4, "ones");
//    double c = 4.0;
//    Malphax::Tensor f = a * b;
//    std::cout<<f.data<<std::endl;
//    f.backward();
//    std::cout << a.grad << std::endl;
//    std::cout << b.grad << std::endl;
////    Malphax::Tensor d = Malphax::matmul(a, b);
////    d.backward();
////    std::cout << a.grad << std::endl;
////    std::cout << b.grad << std::endl;
//
//
////    arma::mat d(3, 3);
////    arma::mat a1(3, 4);
////    arma::mat a2(3, 4);
////    a1.ones();
////    a1 = a1 * 6;
//////    a1 = a1 * 3;
////    a2.ones();
////    std::cout << a1 % a2 << std::endl;
//////    d.ones();
////    a1.ones();
//////    a1 = a1 * 3;
////    a2.ones();
////    auto c = a1 * 6;
////    std::cout<<c<<std::endl;
////    a2 = a2 * 2;
////
////    auto e = d*c;
////    std::cout<<e<<std::endl;
////
////    auto c1 = a1 % a2;
////    std::cout<<c1<<std::endl;
////    f.backward();
////    std::cout<<a.data<<std::endl;
////    std::cout<<b.data<<std::endl;
////
////    std::cout<<a.grad<<std::endl;
////    std::cout<<b.grad<<std::endl;
//
//
//    return 0;
//
//}

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include <dlib/clustering.h>
#include <dlib/rand.h>
#include <random>

using namespace std;
using namespace dlib;

typedef matrix<double, 2, 1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;

void generate()
{
    std::ofstream file("kkmeans_ex.txt");
    if (file.is_open())
    {
        dlib::rand rnd;
        file << std::fixed;
        file.precision(5);
        for (int i = 0; i < 200; i++)
        {
            double x = rnd.get_random_double() * 200 - 100;
            double y = rnd.get_random_double() * 200 - 100;
            file << x << ';' << y << '\n';
        }
        file.close();
    }

    std::ofstream file2("example.txt");
    if (file2.is_open())
    {
        dlib::rand rnd;
        file2 << std::fixed;
        const long num = 50;
        double radius = 0.5;
        for (long i = 0; i < num; ++i)
        {
            double sign = 1;
            if (rnd.get_random_double() < 0.5)
                sign = -1;
            double x = 2 * radius * rnd.get_random_double() - radius;
            double y = sign * sqrt(radius * radius - x * x);

            file2 << x << ';' << y << '\n';
        }


        radius = 10.0;
        for (long i = 0; i < num; ++i)
        {
            double sign = 1;
            if (rnd.get_random_double() < 0.5)
                sign = -1;
            double x = 2 * radius * rnd.get_random_double() - radius;
            double y = sign * sqrt(radius * radius - x * x);

            file2 << x << ';' << y << '\n';
        }

        radius = 4.0;
        for (long i = 0; i < num; ++i)
        {
            double sign = 1;
            if (rnd.get_random_double() < 0.5)
                sign = -1;
            double x = 2 * radius * rnd.get_random_double() - radius;
            double y = sign * sqrt(radius * radius - x * x);
            x += 25;
            y += 25;

            file2 << x << ';' << y << '\n';
        }
        file2.close();
    }
}

bool isPositiveInteger(const std::string &&s)
{
    return !s.empty() &&
           (std::count_if(s.begin(), s.end(), [](auto a) { return (a >= '0' && a <= '9'); }) == (long)s.size());
}

bool isDouble(const std::string &s)
{
    return !s.empty() &&
           (std::count_if(s.begin(), s.end(), [](auto a) { return (a >= '0' && a <= '9') || (a == '-') || (a == '.'); }) == (long)s.size());
}

auto main(int argc, char *argv[]) -> int
{
    if (argc > 1)
    {
        if (isPositiveInteger(argv[1]))
        {
            size_t clusters = atoi(argv[1]);
            kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);
            kkmeans<kernel_type> test(kc);
            std::vector<sample_type> samples;
            std::vector<sample_type> initial_centers;

            std::string input;
            while (std::getline(std::cin, input))
            {
                size_t delim = input.find(';');
                std::string first = input.substr(0, delim);
                std::string second = input.substr(delim + 1, input.size() - delim - 1);
                if (isDouble(first))
                    if (isDouble(second))
                    {
                        sample_type m;
                        m(0) = atof(first.c_str());
                        m(1) = atof(second.c_str());
                        samples.push_back(m);
                    }
            }
            test.set_number_of_centers(clusters);
            pick_initial_centers(clusters, initial_centers, samples, test.get_kernel());
            test.train(samples, initial_centers);

            for (unsigned long i = 0; i < samples.size(); ++i)
            {
                cout << samples[i](0) << ';' << samples[i](1) << ';' << test(samples[i]) << "\n";
            }
        }
    }
    else
    {
        generate();
    }
}
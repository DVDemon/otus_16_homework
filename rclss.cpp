#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <exception>
#include <fstream>
#include <set>

#include <dlib/clustering.h>
#include <dlib/svm_threaded.h>
#include <dlib/any.h>
#include <dlib/svm.h>

using namespace std;
using namespace dlib;
static const size_t columns = 7;

using sample_type = matrix<double, columns, 1>;
using kernel_type = dlib::radial_basis_kernel<sample_type>;
using ovo_trainer = dlib::one_vs_one_trainer<dlib::any_trainer<sample_type>>;
using my_decision_function = dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<kernel_type>>;
using my_normalized_function = dlib::normalized_function<my_decision_function>;

bool isDouble(const std::string &s)
{
    return !s.empty() &&
           (std::count_if(s.begin(), s.end(), [](auto a) { return (a >= '0' && a <= '9') || (a == '-') || (a == '.'); }) == (long)s.size());
}

bool load_sample(std::istream& stream, sample_type &result)
{
    std::string input;

    if (!std::getline(stream, input))
        return false;
    int pos = 0;
    int index = 0;
    bool valid = true;
    do
    {
        int next_pos = input.find(';', pos);
        if (next_pos == std::string::npos)
            next_pos = input.length();

        std::string val = input.substr(pos, next_pos - pos).c_str();

        if (val.length() == 0)
        {
            valid = false;
            break;
        }
        if (!isDouble(val))
        {
            throw std::logic_error("invalid format");
        }
        result(index) = atof(val.c_str());

        pos = next_pos + 1;
        index++;
    } while (index < 7);

    return true;
}

auto main(int argc, char *argv[]) -> int
{
    if (argc == 2)
    {

        try
        {
            if(!ifstream(argv[1], std::ios::in)) throw std::runtime_error("file not found");
            sample_type input;
            while (load_sample(std::cin,input))
            {
                my_normalized_function dec_function;
                deserialize(argv[1]) >> dec_function;
                int cluster = dec_function(input);


                std::string file_name = argv[1];
                file_name+="."+std::to_string(cluster);
                
                std::vector<sample_type> array;
                std::ifstream file(file_name);
                if(file.is_open()){
                    sample_type i;
                    while(load_sample(file,i)){
                        array.push_back(i);
                    }
                }
                std::sort(std::begin(array),std::end(array),
                        [&](const sample_type& lhv, const sample_type& rhv){
                            double d1=  (lhv(0)-input(0))*(lhv(0)-input(0))+
                                        (lhv(1)-input(1))*(lhv(1)-input(1));
                            double d2=  (rhv(0)-input(0))*(rhv(0)-input(0))+
                                        (rhv(1)-input(1))*(rhv(1)-input(1));
                            return d1<d2;
                        });

                std::cout << setw(5) << fixed;
                for(auto a: array){
                    for(size_t i=0;i<columns;++i){
                        std::cout << a(i);
                        if(i<columns-1) std::cout <<';';
                    }
                    std::cout << std::endl;
                }
            }
        }
        catch (std::exception& ex)
        {
            std::cout << ex.what() << std::endl;
        }
    }
    else
    {
        std::cout << "Uasge: rclss model_name" << std::endl;
    }
    return 1;
}
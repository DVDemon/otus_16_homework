#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <exception>
#include <fstream>
#include <memory>

#include <dlib/clustering.h>
#include <dlib/svm_threaded.h>
#include <dlib/any.h>
#include <dlib/svm.h>

using namespace std;
using namespace dlib;

static const double GAMMA_DEFAULT = 0.000625;
static const size_t columns = 7;

using sample_type = matrix<double, columns, 1>;
using kernel_type = dlib::radial_basis_kernel<sample_type>;
using ovo_trainer = dlib::one_vs_one_trainer<dlib::any_trainer<sample_type>>;
using poly_kernel = dlib::polynomial_kernel<sample_type>;
using rbf_kernel = dlib::radial_basis_kernel<sample_type>;
using my_decision_function = dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<kernel_type>>;
using my_normalized_function = dlib::normalized_function<my_decision_function>;

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

void load_data(std::vector<sample_type> &array)
{
    std::string input;
    while (std::getline(std::cin, input))
    {
        std::array<double, columns+1> input_array;

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
            input_array[index] = atof(val.c_str());

            pos = next_pos + 1;
            index++;
        } while (index < 8);

        if (valid)
        {
            sample_type s;
            s(0) = input_array[0];
            s(1) = input_array[1];
            s(2) = input_array[2];
            s(3) = input_array[3];
            s(4) = input_array[4];
            s(5) = input_array[5];
            if (input_array[6] == 1)
                s(6) = 1;
            else if (input_array[6] == input_array[7])
                s(6) = 1;
            else
                s(6) = 0;
            array.push_back(s);
        }
    }
}

void cluster_kkmeans(const kernel_type &kernel,
                     double tolerance,
                     unsigned long max_dictionary_size,
                     unsigned long cluster_num,
                     const std::vector<sample_type> &data,
                     std::vector<double> &labels)
{

    kcentroid<kernel_type> kc(kernel, tolerance, max_dictionary_size);

    kkmeans<kernel_type> kkmeansObj(kc);
    kkmeansObj.set_number_of_centers(cluster_num);

    std::vector<sample_type> initial_centers;

    pick_initial_centers(cluster_num, initial_centers, data, kkmeansObj.get_kernel());

    kkmeansObj.train(data, initial_centers);

    for (const auto &data : data)
    {
        labels.push_back(kkmeansObj(data));
    }
}

void save_sample_to_file(std::ofstream* file,sample_type& data){
    *file << std::setw(4)<<std::fixed;
    for(size_t i=0;i<columns;++i){
        *file << data(i);
        if(i<columns-1) *file<<';';
    }
    *file << std::endl;
}

void save_clustered_files(const std::vector<sample_type>& array,
                          my_normalized_function& dec_function,
                          const std::string &model_file,
                          size_t clusters_count){

    std::vector<std::unique_ptr<std::ofstream>> files;
    for(size_t i=0; i< clusters_count; ++i){
        std::string file_name = model_file;
        file_name+=".";
        file_name+=std::to_string(i);
        files.push_back( std::unique_ptr<std::ofstream>(new std::ofstream(file_name)));
    }

    for(auto s : array){
        int cluster = dec_function(s);
        save_sample_to_file(files[cluster].get(),s);

    }
}

void generate_model(size_t clusters_count, const std::string &model_file)
{
    std::vector<sample_type> array;
    std::vector<double> labels;

    load_data(array);

    std::vector<sample_type> normalized_data;
    normalized_data.reserve(array.size());

    vector_normalizer<sample_type> normalizer;
    normalizer.train(array);

    std::transform(array.cbegin(),
                   array.cend(),
                   std::back_inserter(normalized_data),
                   [&normalizer](const auto &data) {
                       return normalizer(data);
                   });

    cluster_kkmeans(kernel_type(0.01),
                    0.01,
                    8,
                    clusters_count,
                    normalized_data,
                    labels);

    randomize_samples(normalized_data, labels);

    ovo_trainer trainer;
    krr_trainer<rbf_kernel> rbf_trainer;
    rbf_trainer.set_kernel(rbf_kernel(GAMMA_DEFAULT));
    trainer.set_trainer(rbf_trainer);

    my_normalized_function dec_function;
    dec_function.normalizer = normalizer;
    dec_function.function = trainer.train(normalized_data, labels);
    serialize(model_file) << dec_function;

    save_clustered_files(array,dec_function,model_file,clusters_count);
}



auto main(int argc, char *argv[]) -> int
{
    if (argc == 3)
    {
        if (isPositiveInteger(argv[1]))
        {
            size_t clusters_count = atoi(argv[1]);
            std::string model_file = argv[2];

            std::string rmfiles;
            rmfiles = "rm -f " + model_file + '*';
            std::system(rmfiles.c_str());

            try
            {
                generate_model(clusters_count, model_file);
            }
            catch (std::exception& ex)
            {
                std::cout << ex.what() << std::endl;
            }
        }
    }
    else
    {
        std::cout << "Usgae: rclst clusters_count model_file_name" << std::endl;
    }
}
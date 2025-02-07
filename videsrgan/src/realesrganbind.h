#ifndef _REALESRGANBIND_H
#define _REALESRGANBIND_H

#include "realesrgan.h"
#include "pybind11/include/pybind11/pybind11.h"
#include <locale>
#include <codecvt>
#include <utility>

// wrapper class of ncnn::Mat
class RealESRGANImage
{
public:
    std::string d;
    int w;
    int h;
    int c;

    RealESRGANImage(std::string d, int w, int h, int c);

    void set_data(std::string data);

    pybind11::bytes get_data() const;
};

class RealESRGANBind : public RealESRGAN
{
public:
    RealESRGANBind(int gpuid, bool tta_mode);

    int get_tilesize() const;

    // realesrgan parameters
    void set_parameters(int _tilesize, int _scale);

    int load(const std::string &parampath, const std::string &modelpath);

    int process(const RealESRGANImage &inimage, RealESRGANImage &outimage) const;

private:
    int gpuid;
};

int get_gpu_count();

void destroy_gpu_instance();

#endif // _REALESRGANBIND_H

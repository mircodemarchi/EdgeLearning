/***************************************************************************
 *            parser/mnist.hpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of EdgeLearning.
 *
 *  EdgeLearning is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  EdgeLearning is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with EdgeLearning.  If not, see <https://www.gnu.org/licenses/>.
 */

/*! \file  parser/mnist.hpp
 *  \brief MNIST Dataset Parser implementation.
 */

#ifndef EDGE_LEARNING_PARSER_MNIST_HPP
#define EDGE_LEARNING_PARSER_MNIST_HPP

#include "parser.hpp"
#include "data/path.hpp"

#include <cstddef>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace EdgeLearning {

namespace fs = std::filesystem;

/**
 * \brief Calculate an uint32 value in the right endian order according to
 * the processor characteristics.
 * \param i The input value.
 * \return std::uint32_t& The output value in the right endian order.
 */
static inline std::uint32_t& uint32_endian_order(std::uint32_t& i)
{
    const std::uint32_t N = 1;
    if (*reinterpret_cast<const std::uint8_t *>(&N) == 1)
    {
        char* buf = reinterpret_cast<char*>(&i);
        std::swap(buf[0], buf[3]);
        std::swap(buf[1], buf[2]);
    }
    return i;
}

/**
 * \brief Read from a bytes stream an uint32 value in the right endian order
 * according to the processor characteristics.
 * \param ifs The input bytes stream.
 * \return uint32_t The uint32 value in the right endian order.
 */
static inline uint32_t read_uint32_endian_order(std::ifstream& ifs)
{
    std::uint32_t ret;
    char* buf = reinterpret_cast<char*>(&ret);
    ifs.read(buf, 4);
    uint32_endian_order(ret);
    return ret;
}

/**
 * \brief Class for a single image of the Mnist dataset.
 */
class MnistImage : public Parser
{
public:
    /// @brief The image side. Image shape: side x side.
    static const std::uint32_t IMAGE_SIDE = 28;

    /**
     * \brief Mnist Image constructor.
     * \param data Input bytes stream with the Mnist image values.
     * \param idx The image index in the dataset.
     */
    MnistImage(std::ifstream& data, std::size_t idx)
        : Parser()
        , _image{}
        , _idx{idx}
    {
        _image.resize(IMAGE_SIDE * IMAGE_SIDE);
        data.read(reinterpret_cast<char*>(_image.data()),
                  IMAGE_SIDE * IMAGE_SIDE);
    }

    /**
     * \brief MnistImage default deconstruct.
     */
    ~MnistImage() = default;

    /**
     * \brief Getter of the vector image field.
     * \return const std::vector<std::uint8_t>& The Mnist Image in a vector.
     */
    [[nodiscard]] const std::vector<std::uint8_t>& data() const
    {
        return _image;
    }
    std::vector<std::uint8_t>& data() { return _image; }

    /**
     * \brief Getter of the _idx field.
     * \return std::size_t The index of the image in the Mnist dataset.
     */
    [[nodiscard]] std::size_t idx() const { return _idx; }

    /**
     * \brief Convert the Mnist image in a string representation.
     * \return std::string A string representation of the Mnist image.
     */
    operator std::string() const
    {
        // const char map[] = "@#%xo;:,. ";
        const char map[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
        std::string ret = "\n";
        for (size_t i = 0; i < IMAGE_SIDE; ++i)
        {
            size_t offset = i * IMAGE_SIDE;
            for (size_t j = 0; j < IMAGE_SIDE; ++j)
            {
                auto grayscale = _image[offset + j];
                auto c = map[static_cast<std::size_t>(
                    grayscale * ((sizeof(map) - 1) / 256.0))];
                ret += c;
                ret += c;
            }
            ret += "\n";
        }
        ret += "\n";
        return ret;
    }

    /**
     * \brief Output stream operator for MnistImage object.
     * \param os  Output stream object.
     * \param obj Mnist Image object to print.
     * \return Overwritten output stream object.
     */
    friend std::ostream& operator<< (
        std::ostream& os, const MnistImage& obj)
    {
        os << std::string(obj);
        return os;
    }

private:
    /**
     * \brief The MNIST image of values in range [0,255].
     * Size: side*side.
     */
    std::vector<std::uint8_t> _image;
    std::size_t _idx; ///< \brief Index of the image in the Mnist dataset.
};

/**
 * \brief Class for a single label of the Mnist dataset.
 */
class MnistLabel : public Parser
{
public:
    /**
     * \brief Mnist Label constructor.
     * \param data Input bytes stream with the Mnist label value.
     * \param idx The label index in the dataset.
     */
    MnistLabel(std::ifstream& data, std::size_t idx)
        : Parser()
        , _label{}
        , _idx{idx}
    {
        data.read(reinterpret_cast<char *>(&_label), 1);
    }

    /**
     * \brief MnistLabel default deconstruct.
     */
    ~MnistLabel() = default;

    /**
     * \brief Getter of the label field.
     * \return std::uint8_t& The label value.
     */
    [[nodiscard]] const std::uint8_t& data() const { return _label; }
    std::uint8_t& data() { return _label; }

    /**
     * \brief Getter of the _idx field.
     * \return std::size_t The index of the label in the Mnist dataset.
     */
    [[nodiscard]] std::size_t idx() const { return _idx; }

    /**
     * \brief Convert Mnist label to string.
     * \return std::string The string value of the Mnist label.
     */
    operator std::string() const
    {
        return std::to_string(static_cast<std::uint32_t>(_label));
    }

    /**
     * \brief Output stream operator for MnistLabel object.
     * \param os  Output stream object.
     * \param obj Mnist Label object to print.
     * \return Overwritten output stream object.
     */
    friend std::ostream& operator<< (
        std::ostream& os, const MnistLabel& obj)
    {
        os << std::string(obj);
        return os;
    }

private:
    std::uint8_t _label; ///< \brief The MNIST label with value in range [0,9].
    std::size_t _idx;    ///< \brief Index of the image in the Mnist dataset.
};

/**
 * \brief Mnist dataset item composed by the image and the label.
 */
struct MnistItem
{
    MnistImage image; ///< \brief A MNIST dataset image.
    MnistLabel label; ///< \brief A MNIST dataset label.
};

/**
 * \brief MNIST Dataset core class.
 */
class Mnist : public Parser
{
public:
    /// \brief Image file magic number.
    static const std::uint32_t IMAGE_MAGIC = 0x00000803;
    /// \brief Label file magic number.
    static const std::uint32_t LABEL_MAGIC = 0x00000801;
    /// \brief The header size in bytes of the image file.
    static const std::size_t IMAGE_HEADER_SIZE = 16;
    /// \brief The header size in bytes of the label file.
    static const std::size_t LABEL_HEADER_SIZE = 8;

    /**
     * \brief Mnist core class constructor.
     * \param image_fp File path of the images.
     * \param label_fp File path of the labels.
     */
    Mnist(fs::path image_fp, fs::path label_fp)
        : Parser()
        , _image_ifs{image_fp}
        , _label_ifs{label_fp}
    {
        if(!_image_ifs.is_open() || !_image_ifs.good())
        {
            _image_ifs.close();
            _label_ifs.close();
            throw std::runtime_error("Images malformed: could not open file");
        }
        if(!_label_ifs.is_open() || !_label_ifs.good())
        {
            _image_ifs.close();
            _label_ifs.close();
            throw std::runtime_error("Labels malformed: could not open file");
        }

        if (_is_malformed(_image_ifs, IMAGE_MAGIC))
        {
            _image_ifs.close();
            _label_ifs.close();
            throw std::runtime_error("Images malformed: magic number error");
        }
        auto image_count = read_uint32_endian_order(_image_ifs);

        if (_is_malformed(_label_ifs, LABEL_MAGIC))
        {
            _image_ifs.close();
            _label_ifs.close();
            throw std::runtime_error("Labels malformed: magic number error");
        }
        auto label_count = read_uint32_endian_order(_label_ifs);

        if (label_count != image_count)
        {
            _image_ifs.close();
            _label_ifs.close();
            throw std::runtime_error("Data malformed: "
                                     "labels amount not match images amount");
        }
        _size = image_count;

        std::uint32_t columns, rows;
        rows = read_uint32_endian_order(_image_ifs);
        columns = read_uint32_endian_order(_image_ifs);
        if (rows != MnistImage::IMAGE_SIDE || columns != MnistImage::IMAGE_SIDE)
        {
            _image_ifs.close();
            _label_ifs.close();
            throw std::runtime_error("Data malformed: "
                                     "not expected image shape");
        }
    }

    /**
     * \brief Mnist object deconstruct.
     * Clone image and label file streams.
     */
    ~Mnist()
    {
        _image_ifs.close();
        _label_ifs.close();
    }

    /**
     * \brief Getter of the data amount in the dataset.
     * \return std::size_t The data amount in the dataset.
     */
    std::size_t size() const { return _size; }

    /**
     * \brief Getter of the image side.
     * \return std::size_t The image side.
     */
    std::size_t side() const   { return MnistImage::IMAGE_SIDE; }

    /**
     * \brief Getter of the image height. Equals to the image side.
     * \return std::size_t The image height.
     */
    std::size_t height() const { return MnistImage::IMAGE_SIDE; }

    /**
     * \brief Getter of the image width. Equals to the image side.
     * \return std::size_t The image width.
     */
    std::size_t width() const  { return MnistImage::IMAGE_SIDE; }

    /**
     * \brief Getter of the image shape: side x side.
     * \return std::tuple<std::size_t, std::size_t> The image shape tuple.
     */
    std::tuple<std::size_t, std::size_t> shape() const
    {
        return {side(), side()};
    }

    /**
     * \brief Get an image from the Mnist dataset at index.
     * \param idx The index of the Mnist image.
     * \return MnistImage The image object at idx index.
     */
    MnistImage image(std::size_t idx)
    {
        idx = idx % _size;
        _image_ifs.seekg(static_cast<std::int64_t>(
            IMAGE_HEADER_SIZE + idx * height() * width()), _image_ifs.beg);
        return MnistImage{_image_ifs, idx};
    }

    /**
     * \brief Get a label from the Mnist dataset at index.
     * \param idx The index of the Mnist label.
     * \return MnistLabel The label object at idx index.
     */
    MnistLabel label(std::size_t idx)
    {
        idx = idx % _size;
        _label_ifs.seekg(static_cast<std::int64_t>(
            LABEL_HEADER_SIZE + idx), _label_ifs.beg);
        return MnistLabel{_label_ifs, idx};
    }

    /**
     * \brief Get a Mnist item at the index specified.
     * \param idx The index of the Mnist item.
     * \return MnistItem The requested index item.
     */
    MnistItem operator[](std::size_t idx)
    {
        return {image(idx), label(idx)};
    }

private:
    /**
     * \brief Check if the Mnist file stream is malformed in relation to the
     * number to compare.
     * \param data The data stream from which obtain the value to compare.
     * \param num_to_check The value to compare.
     * \return bool True if the Mnist file is malformed otherwise false.
     */
    static bool _is_malformed(
        std::ifstream& data, std::uint32_t num_to_check)
    {
        auto magic = read_uint32_endian_order(data);
        return magic != num_to_check;
    }

    std::ifstream _image_ifs; ///< \brief Images input stream.
    std::ifstream _label_ifs; ///< \brief Labels input stream.

    std::uint32_t _size; ///< \brief Number of elements in the MNIST dataset.
};



} // namespace EdgeLearning

#endif // EDGE_LEARNING_PARSER_MNIST_HPP

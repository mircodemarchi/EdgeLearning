/***************************************************************************
 *            parser/cifar.hpp
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

/*! \file  parser/cifar.hpp
 *  \brief Cifar Dataset Parser implementation.
 */

#ifndef EDGE_LEARNING_PARSER_CIFAR_HPP
#define EDGE_LEARNING_PARSER_CIFAR_HPP

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

enum class CifarDataset
{
    CIFAR_10,
    CIFAR_100
};

enum class CifarShapeOrder
{
    CHN_ROW_COL,
    ROW_COL_CHN
};

/**
 * \brief NTSC Formula.
 * \param r std::uint8_t Red channel value.
 * \param g std::uint8_t Green channel value.
 * \param b std::uint8_t Blue channel value.
 * \return std::uint8_t Brightness perceived from RGB.
 */
static std::uint8_t perception_brightness(
    std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    return static_cast<std::uint8_t>(r * 0.21 + g * 0.72 * b * 0.07);
}

/**
 * \brief Class for a single image of the Cifar dataset.
 */
class CifarImage : public Parser
{
public:
    /// @brief The image side. Image shape: side x side.
    static const std::uint32_t IMAGE_SIDE = 32;
    /// @brief The image number of channels.
    static const std::uint32_t IMAGE_CHANNELS = 3;

    /**
     * \brief Cifar Image constructor.
     * \param data Input bytes stream with the Cifar image values.
     * \param idx The image index in the dataset.
     * \param order The order of shape of Cifar image requested later in output.
     */
    CifarImage(std::ifstream& data, std::size_t idx,
               CifarShapeOrder order = CifarShapeOrder::CHN_ROW_COL)
        : Parser()
        , _image{}
        , _idx{idx}
        , _order{order}
    {
        _image.resize(IMAGE_SIDE * IMAGE_SIDE * IMAGE_CHANNELS);
        data.read(reinterpret_cast<char*>(_image.data()),
                  IMAGE_SIDE * IMAGE_SIDE * IMAGE_CHANNELS);
    }

    /**
     * \brief CifarImage default deconstruct.
     */
    ~CifarImage() = default;

    /**
     * \brief Getter of the vector image field.
     * \return const std::vector<std::uint8_t>& The Cifar Image in a vector.
     */
    [[nodiscard]] std::vector<std::uint8_t> data() const
    {
        switch (_order) {
            case CifarShapeOrder::ROW_COL_CHN:
            {
                std::vector<std::uint8_t> ret(_image.size());
                const auto r_offset = 0;
                const auto g_offset = IMAGE_SIDE * IMAGE_SIDE;
                const auto b_offset = 2 * IMAGE_SIDE * IMAGE_SIDE;
                for (size_t i = 0; i < IMAGE_SIDE; ++i) {
                    size_t offset = i * IMAGE_SIDE;
                    size_t new_offset = i * IMAGE_SIDE * IMAGE_CHANNELS;
                    for (size_t j = 0; j < IMAGE_SIDE; ++j) {
                        ret[new_offset + j * IMAGE_CHANNELS]
                            = _image[r_offset + offset + j];
                        ret[new_offset + j * IMAGE_CHANNELS + 1]
                            = _image[g_offset + offset + j];
                        ret[new_offset + j * IMAGE_CHANNELS + 2]
                            = _image[b_offset + offset + j];
                    }
                }
                return ret;
            }
            case CifarShapeOrder::CHN_ROW_COL:
            default:
                return _image;
        }
    }

    /**
     * \brief Getter of the _idx field.
     * \return std::size_t The index of the image in the Cifar dataset.
     */
    [[nodiscard]] std::size_t idx() const { return _idx; }

    /**
     * \brief Convert the Cifar image in a string representation.
     * \return std::string A string representation of the Cifar image.
     */
    operator std::string() const
    {
        // const char map[] = "@#%xo;:,. ";
        const char map[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
        std::string ret = "\n";
        const auto r_offset = 0;
        const auto g_offset = IMAGE_SIDE * IMAGE_SIDE;
        const auto b_offset = 2 * IMAGE_SIDE * IMAGE_SIDE;
        for (size_t i = 0; i < IMAGE_SIDE; ++i)
        {
            size_t offset = i * IMAGE_SIDE;
            for (size_t j = 0; j < IMAGE_SIDE; ++j)
            {
                auto grayscale = perception_brightness(
                    _image[r_offset + offset + j],
                    _image[g_offset + offset + j],
                    _image[b_offset + offset + j]);

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
     * \brief Output stream operator for CifarImage object.
     * \param os  Output stream object.
     * \param obj Cifar Image object to print.
     * \return Overwritten output stream object.
     */
    friend std::ostream& operator<< (
        std::ostream& os, const CifarImage& obj)
    {
        os << std::string(obj);
        return os;
    }

private:
    /**
     * \brief The Cifar image of values in range [0,255].
     * Size: side*side.
     */
    std::vector<std::uint8_t> _image;
    std::size_t _idx; ///< \brief Index of the image in the Cifar dataset.
    CifarShapeOrder _order; ///< \brief Order of shape in output.
};

/**
 * \brief Class for a single label of the Cifar dataset.
 */
class CifarLabel : public Parser
{
public:
    /**
     * \brief Cifar Label constructor.
     * \param data    Input bytes stream with the Cifar label value.
     * \param idx     The label index in the dataset.
     * \param dataset The dataset format in input data.
     */
    CifarLabel(std::ifstream& data, std::size_t idx,
               CifarDataset dataset = CifarDataset::CIFAR_10)
        : Parser()
        , _dataset_format{dataset}
        , _coarse_label{}
        , _fine_label{0}
        , _idx{idx}
    {
        data.read(reinterpret_cast<char*>(&_coarse_label), 1);
        switch(_dataset_format)
        {
            case CifarDataset::CIFAR_100:
            {
                data.read(reinterpret_cast<char*>(&_fine_label), 1);
                break;
            }
            case CifarDataset::CIFAR_10:
            default:
            {
                break;
            }
        }
    }

    /**
     * \brief CifarLabel default deconstruct.
     */
    ~CifarLabel() = default;

    /**
     * \brief Getter of the label field.
     * \return std::uint8_t& The label value.
     */
    [[nodiscard]] const std::uint8_t& data() const { return _coarse_label; }
    std::uint8_t& data() { return _coarse_label; }

    /**
     * \brief Getter of the coarse label field.
     * \return std::uint8_t& The coarse label value.
     */
    [[nodiscard]] const std::uint8_t& coarse_label() const
    { return _coarse_label; }
    std::uint8_t& coarse_label() { return _coarse_label; }

    /**
     * \brief Getter of the fine label field.
     * \return std::uint8_t& The fine label value.
     */
    [[nodiscard]] const std::uint8_t& fine_label() const { return _fine_label; }
    std::uint8_t& fine_label() { return _fine_label; }

    /**
     * \brief Getter of the _idx field.
     * \return std::size_t The index of the label in the Cifar dataset.
     */
    [[nodiscard]] std::size_t idx() const { return _idx; }

    /**
     * \brief Convert Cifar label to string.
     * \return std::string The string value of the Cifar label.
     */
    operator std::string() const
    {
        switch(_dataset_format)
        {
            case CifarDataset::CIFAR_100:
            {
                return "("
                    + std::to_string(static_cast<std::uint32_t>(_coarse_label))
                    + ","
                    + std::to_string(static_cast<std::uint32_t>(_fine_label))
                    + ")";
            }
            case CifarDataset::CIFAR_10:
            default:
            {
                return std::to_string(
                    static_cast<std::uint32_t>(_coarse_label));
            }
        }
    }

    /**
     * \brief Output stream operator for CifarLabel object.
     * \param os  Output stream object.
     * \param obj Cifar Label object to print.
     * \return Overwritten output stream object.
     */
    friend std::ostream& operator<< (
        std::ostream& os, const CifarLabel& obj)
    {
        os << std::string(obj);
        return os;
    }

private:
    CifarDataset _dataset_format; ///< @brief The Cifar dataset format.

    /**
     * \brief The Cifar coarse label.
     * In case of Cifar-10 dataset the range values is in [0-10).
     * In case of Cifar-100 dataset the range values is in [0-20).
     */
    std::uint8_t _coarse_label;

    /**
     * \brief The Cifar coarse label.
     * In case of Cifar-10 dataset the range value is always 0.
     * In case of Cifar-100 dataset the range values is in [0-5).
     */
    std::uint8_t _fine_label;

    std::size_t _idx;    ///< \brief Index of the image in the Cifar dataset.
};

/**
 * \brief Cifar dataset item composed by the image and the label.
 */
struct CifarItem
{
    CifarImage image; ///< \brief A Cifar dataset image.
    CifarLabel label; ///< \brief A Cifar dataset label.
};

/**
 * \brief Cifar Dataset core class.
 */
class Cifar : public DatasetParser
{
public:
    /// @brief The dataset size.
    static const std::uint32_t SIZE = 10000;

    /**
     * \brief Cifar core class constructor.
     * \param batch_fp File path of the batch dataset.
     */
    Cifar(fs::path batch_fp,
          fs::path coarse_label_meta_fp,
          CifarShapeOrder order = CifarShapeOrder::CHN_ROW_COL,
          CifarDataset dataset = CifarDataset::CIFAR_10,
          fs::path fine_label_meta_fp = fs::path())
        : DatasetParser()
        , _batch_ifs{batch_fp}
        , _order{order}
        , _dataset{dataset}
        , _coarse_label_names{}
        , _fine_label_names{}
        , _label_offset{}
    {
        std::ifstream coarse_label_meta_ifs(coarse_label_meta_fp);
        std::string line;
        while (std::getline(coarse_label_meta_ifs, line))
        {
            if (!line.empty()) _coarse_label_names.push_back(line);
        }
        coarse_label_meta_ifs.close();

        if (!fine_label_meta_fp.empty() && _dataset == CifarDataset::CIFAR_100)
        {
            std::ifstream fine_label_meta_ifs(fine_label_meta_fp);
            while (std::getline(fine_label_meta_ifs, line))
            {
                if (!line.empty()) _fine_label_names.push_back(line);
            }
            fine_label_meta_ifs.close();
        }
        else if (fine_label_meta_fp.empty())
        {
            _dataset = CifarDataset::CIFAR_10;
        }

        switch(_dataset)
        {
            case CifarDataset::CIFAR_100:
            {
                _label_offset = 2;
                break;
            }
            case CifarDataset::CIFAR_10:
            default:
            {
                _label_offset = 1;
                break;
            }
        }
    }

    /**
     * \brief Cifar object deconstruct.
     * Clone image and label file streams.
     */
    ~Cifar()
    {
        _batch_ifs.close();
    }

    /**
     * \brief Getter of the data amount in the dataset.
     * \return std::size_t The data amount in the dataset.
     */
    std::size_t size() const { return SIZE; }

    /**
     * \brief Getter of the image side.
     * \return std::size_t The image side.
     */
    std::size_t side() const   { return CifarImage::IMAGE_SIDE; }

    /**
     * \brief Getter of the image height. Equals to the image side.
     * \return std::size_t The image height.
     */
    std::size_t height() const { return CifarImage::IMAGE_SIDE; }

    /**
     * \brief Getter of the image width. Equals to the image side.
     * \return std::size_t The image width.
     */
    std::size_t width() const  { return CifarImage::IMAGE_SIDE; }

    /**
     * \brief Getter of the image channels.
     * \return std::size_t The image channels.
     */
    std::size_t channels() const { return CifarImage::IMAGE_CHANNELS; }

    /**
     * \brief Getter of the image shape: side x side.
     * \return std::tuple<std::size_t, std::size_t> The image shape tuple.
     */
    std::tuple<std::size_t, std::size_t, std::size_t> shape() const
    {
        switch (_order)
        {
            case CifarShapeOrder::ROW_COL_CHN:
                return {side(), side(), channels()};
            case CifarShapeOrder::CHN_ROW_COL:
            default:
                return {channels(), side(), side()};
        }
    }

    /**
     * \brief Getter of the dataset label names (aka coarse label names).
     * \return const std::vector<std::string>& The label names.
     */
    const std::vector<std::string>& label_names() const
    { return _coarse_label_names; }

    /**
     * \brief Getter of the dataset coarse label names.
     * \return const std::vector<std::string>& The coarse label names.
     */
    const std::vector<std::string>& coarse_label_names() const
    { return _coarse_label_names; }

    /**
     * \brief Getter of the dataset fine label names.
     * \return const std::vector<std::string>& The fine label names.
     */
    const std::vector<std::string>& fine_label_names() const
    { return _fine_label_names; }

    /**
     * \brief Get an image from the Cifar dataset at index.
     * \param idx The index of the Cifar image.
     * \return CifarImage The image object at idx index.
     */
    CifarImage image(std::size_t idx)
    {
        idx = idx % SIZE;
        _batch_ifs.seekg(
            static_cast<std::int64_t>(
                (idx * (height() * width() * channels() + _label_offset))
                + _label_offset),
            _batch_ifs.beg);
        return CifarImage{_batch_ifs, idx, _order};
    }

    /**
     * \brief Get a label from the Cifar dataset at index.
     * \param idx The index of the Cifar label.
     * \return CifarLabel The label object at idx index.
     */
    CifarLabel label(std::size_t idx)
    {
        idx = idx % SIZE;
        _batch_ifs.seekg(static_cast<std::int64_t>(
            idx * (height() * width() * channels() + _label_offset)),
            _batch_ifs.beg);
        return CifarLabel{_batch_ifs, idx, _dataset};
    }

    /**
     * \brief Get a Cifar item at the index specified.
     * \param idx The index of the Cifar item.
     * \return CifarItem The requested index item.
     */
    CifarItem operator[](std::size_t idx)
    {
        return {image(idx), label(idx)};
    }

    std::vector<NumType> entry(SizeType i) override
    {
        auto cifar_image = image(i).data();
        auto cifar_label = label(i);
        std::vector<NumType> ret(cifar_image.begin(), cifar_image.end());

        switch(_dataset)
        {
            case CifarDataset::CIFAR_100:
            {
                ret.push_back(cifar_label.fine_label());
                break;
            }
            case CifarDataset::CIFAR_10:
            default:
            {
                ret.push_back(cifar_label.data());
                break;
            }
        }
        return ret;
    }

    SizeType entries_amount() const override
    {
        return size();
    }

    SizeType feature_size() const override
    {
        switch(_dataset)
        {
            case CifarDataset::CIFAR_100:
            {
                return height() * width() * channels() + 1;
            }
            case CifarDataset::CIFAR_10:
            default:
            {
                return height() * width() * channels() + 1;
            }
        }
    }

    std::set<SizeType> labels_idx() const override
    {
        switch(_dataset)
        {
            case CifarDataset::CIFAR_100:
            {
                return {feature_size() - 1};
            }
            case CifarDataset::CIFAR_10:
            default:
            {
                return {feature_size() - 1};
            }
        }
    }

private:

    std::ifstream _batch_ifs; ///< \brief Images input stream.
    CifarShapeOrder _order;   ///< \brief Images shape order in output.
    CifarDataset _dataset;    ///< \brief Dataset type format.

    /**
     * \brief The coarse label names.
     * If Cifar dataset is Cifar-10, then coarse label names contains the
     * Cifar-10 labels (amount of 10), if Cifar dataset is Cifar-100, then
     * coarse label names contains the Cifar-100 coarse labels (amount of 20).
     */
    std::vector<std::string> _coarse_label_names;

    /**
     * \brief The fine label names.
     * If Cifar dataset is Cifar-10, then fine label names are empty,
     * if Cifar dataset is Cifar-100, then fine label names contains
     * the Cifar-100 fine labels (amount of 100).
     */
    std::vector<std::string> _fine_label_names;

    std::size_t _label_offset;
};



} // namespace EdgeLearning

#endif // EDGE_LEARNING_PARSER_CIFAR_HPP

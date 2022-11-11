/***************************************************************************
 *            example/custom_loss.cpp
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


#include "util.hpp"

// In order to invoke the EdgeLearning functionalities, you need to include
// only this header file, which includes all the library headers.
#include "edge_learning.hpp"

#include <cmath>      //< For std::sin(), std::abs() and std::sqrt() function.
#include <functional> //< For std::function type.
#include <iomanip>    //< For std::setprecision() function.

// The EdgeLeaning namespace is the only namespace in which all the features of
// the library are included.
using namespace EdgeLearning;

class CustomLossLayer : public LossLayer {
public:
    CustomLossLayer(
        std::string name = std::string(),
        SizeType input_size = 0, SizeType batch_size = 1)
        : LossLayer(input_size, batch_size, name)
    { }

    /*
     * \brief Clone method.
     * \return SharedPtr Layer pointer.
     * This implementation is standard for each layer. Do not edit.
     */
    [[nodiscard]] SharedPtr clone() const override
    { return std::make_shared<CustomLossLayer>(*this); }

    /*
     * \brief Forward step of the loss layer.
     * \param inputs The prediction of the neural network.
     * \return Not used.
     *
     * Implement here a custom cost function, that takes as inputs the current
     * predicted output of the neural network (inputs) and the ground truth
     * (_target), and return a single loss value, assigned to the _loss field.
     *
     * Important fields summary:
     * - inputs: the predicted output of the model;
     * - _target: the current ground truth to compare with the prediction;
     * - _loss: the loss value to extrapolate;
     * - _input_size: the input size of the layer aka the output size of the
     *                neural network;
     * - _cumulative_loss: accumulated loss;
     * - _correct: number of correct prediction;
     * - _incorrect: number of incorrect prediction;
     */
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override
    {
        _loss = 0.0;

        // Suppose to implement MeanAbsoluteError.
        for (SizeType i = 0; i < _input_size; ++i)
        {
            _loss += std::abs(inputs[i] - _target[i]);
        }

        _cumulative_loss += _loss;

        // Evaluate correctness.
        if (-0.1 <= _loss && _loss <= 0.1) _correct++;
        else _incorrect++;
        return inputs; //< Return value never used.
    }

    /*
     * \brief Forward step of the loss layer.
     * \param gradients Parameter not used (empty).
     * \return The gradients to pass to the previous layer.
     *
     * Implement here the first derivative of the custom cost function used,
     * that takes the last input of the loss layer (_last_input) and the ground
     * truth (_target), and fill the gradients (_gradients) to pass backward
     * to the previous layers.
     *
     * Important fields summary:
     * - _last_input: pointer to the last input in forward of the layer,
     *                automatically updated by the training process;
     * - _target: the current ground truth to compare with the prediction;
     */
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override
    {
        (void) gradients; //< Input gradients never used.

        // Implement MeanAbsoluteError first derivative.
        for (SizeType i = 0; i < _gradients.size(); ++i)
        {
            _gradients[i] = (_last_input[i] - _target[i]) > 0 ? 1.0 : -1.0;
        }

        return _gradients;
    }

private:
};

/*
 * \brief Generate the input dataset (a vector of feature vectors).
 * \param random       Enable random generation of the dataset, otherwise the
 *                     dataset is taken from a constant built-in vector of
 *                     elements.
 * \param entry_amount Used only if the random generation is enabled. It defines
 *                     the amount of input entry in the dataset.
 * \param input_size   Used only if the random generation is enabled. It defines
 *                     the size of each input entry of the dataset.
 * \param value_from   Used only if the random generation is enabled. It defines
 *                     the value from which generate the random.
 * \param value_to     Used only if the random generation is enabled. It defines
 *                     the value to which generate the random.
 * \param seed         Used only if the random generation is enabled. It defines
 *                     the seed of the random generator.
 * \return The input dataset: a vector of feature vectors.
 *
 * RneType is the default Random number engine chosen in EdgeLearning. By
 * default, RneType is defined as std::mt19937_64.
 */
static std::vector<std::vector<NumType>> generate_inputs(
    bool random = false,
    SizeType entry_amount = 0,
    SizeType input_size = 0,
    NumType value_from = 0.0,
    NumType value_to = 0.0,
    RneType::result_type seed = std::random_device{}());

/*
 * \brief Generate the labels from an input dataset (a vector of feature
 * vectors) applying a multivariate linear regression (through a set of
 * regression coefficients) and a set of non-linear functions used to generate
 * the individual labels.
 * \param inputs    The input dataset constructed as a vector of feature
 *                  vectors.
 * \param functions Vector of linear functions to apply after the linear
 *                  regression over inputs. The size of each labels entry is
 *                  defined by the size of this vector.
 * \return A vector of label vectors generated for each input entry.
 */
static std::vector<std::vector<NumType>> generate_labels(
    const std::vector<std::vector<NumType>>& inputs,
    std::vector<std::function<NumType(std::vector<NumType>)>> functions);

/*
 * \brief Visualize the input, the expected output from the training set
 * and the model predictions in order to compare each other.
 * \param inputs            Dataset<NumType> Inputs.
 * \param labels            Dataset<NumType> Labels.
 * \param model_predictions Dataset<NumType> Model predictions set.
 */
static void check_predictions(Dataset<NumType>& inputs,
                              Dataset<NumType>& labels,
                              Dataset<NumType>& model_predictions);


int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // SizeType and NumType are custom types defined in EdgeLearning.
    // By default, SizeType is defined as a std::size_t and NumType is double,
    // for more compatibility with future versions of EdgeLearning I suggest you
    // to use the SizeType and NumType custom types.

    // Random generator seed, in order to obtain predictable results.
    // Put it to 0 for random generation of model parameters.
    const SizeType SEED          = 134234563;
    // Number of data entries used to update the model gradients
    // (stochastic gradient descent).
    const SizeType BATCH_SIZE    = 8;
    // Number of iterations performed over the whole dataset.
    const SizeType EPOCHS        = 50;
    // Step size of the optimizer.
    const NumType  LEARNING_RATE = 0.01;

    // Size of the first hidden layer.
    const SizeType HIDDEN1 = 200;
    // Size of the second hidden layer.
    const SizeType HIDDEN2 = 100;

    Time elapsed; //< Utility for performance measure.

    // Create dummy data as inputs for the following regression process.
    // auto inputs = generate_inputs();
    // By default, the generate_inputs() function uses a built-in dataset for
    // debug purposes, in alternatives you can generate a random dataset as
    // follows:
    const SizeType ENTRY_AMOUNT = 1000;
    const SizeType INPUT_SIZE = 4;
    const NumType FROM_RANDOM_VALUE = -1.0;
    const NumType TO_RANDOM_VALUE = 1.0;
    auto inputs = generate_inputs(true, ENTRY_AMOUNT, INPUT_SIZE,
                                  FROM_RANDOM_VALUE, TO_RANDOM_VALUE, SEED);
    const SizeType input_size = inputs[0].size();


    const std::vector<std::function<NumType(std::vector<NumType>)>> USER_DEFINED_FUNCS = {
        // Label 0: norm 2
        [](std::vector<NumType> v) {
            NumType ret = 0;
            for (const auto& e: v) ret += e*e;
            return std::sqrt(ret);
        },
    };
    auto labels = generate_labels(inputs, USER_DEFINED_FUNCS);
    SizeType output_size = labels[0].size();

    // Transform the inputs and the labels in Dataset class.
    Dataset<NumType> inputs_ds(inputs);
    Dataset<NumType> labels_ds(labels);

    // Normalize inputs.
    inputs_ds.min_max_normalization();


    // Create an optimizer object that requires the learning step size.
    GradientDescentOptimizer optimizer(LEARNING_RATE);
    // The library support also the following optimizers:
    // AdamOptimizer optimizer(LEARNING_RATE);

    // Create the model object with the low level interface.
    Model m_ll{"regressor"};

    // Construct the model with the following structure:
    // ------------- IN[input_size] -------------
    //           Dense[HIDDEN1] + ReLU
    // #params: (input_size * HIDDEN1) + HIDDEN1
    // ------------------------------------------
    //           Dense[HIDDEN1] + ReLU
    // #params: (HIDDEN1 * HIDDEN2) + HIDDEN2
    // ------------------------------------------
    //           Dense[output_size]
    // #params: (HIDDEN2 * output_size) + output_size
    // ------------ OUT[output_size] ------------
    auto h1      = m_ll.add_layer<DenseLayer>("h1", input_size, HIDDEN1);
    auto h1_relu = m_ll.add_layer<ReluLayer>("h1_relu", HIDDEN1);
    auto h2      = m_ll.add_layer<DenseLayer>("h2", HIDDEN1, HIDDEN2);
    auto h2_relu = m_ll.add_layer<ReluLayer>("h2_relu", HIDDEN2);
    auto out     = m_ll.add_layer<DenseLayer>("out", HIDDEN2, output_size);
    m_ll.create_edge(h1, h1_relu);
    m_ll.create_edge(h1_relu, h2);
    m_ll.create_edge(h2, h2_relu);
    m_ll.create_edge(h2_relu, out);

    // Define the loss for training.
    // It requires the batch size for normalization purpose in gradient update.
    auto loss = m_ll.add_loss<CustomLossLayer>("mse", output_size, BATCH_SIZE);
    m_ll.create_loss_edge(out, loss);

    // Initialize the model.
    // Available initialization methods: XAVIER, KAIMING.
    // Available PDF to random generate values: NORMAL, UNIFORM.
    // By default, the automatic model parameters initialization performs
    // KAIMING initialization on layers with ReLU activations, otherwise it
    // performs XAVIER initialization.
    m_ll.init(Model::InitializationFunction::AUTO,
              Model::ProbabilityDensityFunction::NORMAL, SEED);

    // Train over epochs.
    std::cout << "--- Training" << std::endl;
    elapsed.start();
    for (SizeType e = 0; e < EPOCHS; ++e)
    {
        std::cout << "[ EPOCH " << e << " ] ";
        for (SizeType i = 0; i < inputs_ds.size();)
        {
            // Reset the model loss scores.
            m_ll.reset_score();

            // Stochastic gradient descent.
            for (SizeType b = 0; b < BATCH_SIZE && i < inputs_ds.size(); ++b, ++i)
            {
                // Crosses forward and backward the model, and generates the
                // gradients.
                m_ll.step(inputs_ds.entry(i), labels_ds.entry(i));
            }

            // Update the model parameters with the optimizer and the generated
            // gradients.
            m_ll.train(optimizer);
        }
        std::cout << "loss: " << m_ll.avg_loss()
                  << ", accuracy: " << m_ll.accuracy() * 100.0 << "%"
                  << std::endl;
    }
    elapsed.stop();
    std::cout << "elapsed: " << std::string(elapsed) << std::endl;

    // Validate the trained model.
    std::cout << "--- Validation" << std::endl;
    std::vector<std::vector<NumType>> predictions;
    for (SizeType i = 0; i < inputs_ds.size(); ++i)
    {
        predictions.push_back(m_ll.predict(inputs_ds.entry(i)));
    }
    auto predictions_ds = Dataset<NumType>(predictions);
    check_predictions(inputs_ds, labels_ds, predictions_ds);

    std::cout << "End" << std::endl;
}

//==============================================================================
static std::vector<std::vector<NumType>> generate_inputs(
    bool random,
    SizeType entry_amount,
    SizeType input_size,
    NumType value_from,
    NumType value_to,
    RneType::result_type seed)
{
    if (!random) {
        // Return a constant built-in dataset.
        std::vector<std::vector<NumType>> INPUTS = {
            {10.0,  1.0,  10.0, 1.0},
            { 1.0,  3.0,  8.0,  3.0},
            { 8.0,  1.0,  8.0,  1.0},
            { 1.0,  1.5,  8.0,  1.5},
            {-1.0,  2.5, -1.0,  1.5},
            { 8.0, -2.5,  1.0, -3.0},
            { 1.0,  2.5, -1.0,  1.5},
            { 8.0,  2.5,  1.0, -3.0},
            { 0.0,  0.0,  0.0,  0.0},
            { 1.0,  1.0,  1.0,  1.0},
        };
        return INPUTS;
    }

    // Generate random values for a dataset of entry_amount x input_size shape.
    RneType rne(seed);
    std::vector<std::vector<NumType>> ret;
    for (SizeType i = 0; i < entry_amount; ++i)
    {
        std::vector<NumType> input_entry;
        for (SizeType j = 0; j < input_size; ++j)
        {
            input_entry.push_back(
                DLMath::rand<NumType>(value_from, value_to, rne));
        }
        ret.push_back(input_entry);
    }
    return ret;
}

static std::vector<std::vector<NumType>> generate_labels(
    const std::vector<std::vector<NumType>>& inputs,
    std::vector<std::function<NumType(std::vector<NumType>)>> functions)
{
    std::vector<std::vector<NumType>> labels;

    // Generate the labels for each input entry in dataset.
    for (const auto& input_entry: inputs)
    {
        // Apply non-linearity to the multivariate linear regression.
        std::vector<NumType> label_entry;
        for (const auto& f: functions)
        {
            label_entry.push_back(f(input_entry));
        }
        labels.push_back(label_entry);
    }
    return labels;
}

static void check_predictions(Dataset<NumType>& inputs,
                              Dataset<NumType>& labels,
                              Dataset<NumType>& model_predictions)
{
    const SizeType MAX_ENTRY = 10;
    for (SizeType i = 0; i < std::min(MAX_ENTRY, inputs.size()); ++i)
    {
        const auto& input_entry = inputs.entry(i);
        const auto& expected_output = labels.entry(i);
        const auto& predicted_output = model_predictions.entry(i);

        std::cout << "INPUT" << i << ": { ";
        for (SizeType j = 0; j < input_entry.size() - 1; ++j)
        {
            std::cout << input_entry[j] << ", ";
        }
        std::cout << input_entry[input_entry.size() - 1] << " } ";

        std::cout << "EXPECTED: { ";
        for (SizeType j = 0; j < expected_output.size() - 1; ++j)
        {
            std::cout << expected_output[j] << ", ";
        }
        std::cout << expected_output[expected_output.size() - 1] << " } ";

        std::cout << "PREDICTED: { ";
        for (SizeType j = 0; j < predicted_output.size() - 1; ++j)
        {
            std::cout << predicted_output[j] << ", ";
        }
        std::cout << predicted_output[predicted_output.size() - 1] << " } ";

        std::cout << std::endl;
    }
    if (inputs.size() > MAX_ENTRY) std::cout << " ... " << std::endl;
}
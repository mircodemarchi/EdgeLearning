/***************************************************************************
 *            example/simple_regression.cpp
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
 * \param inputs                  The input dataset constructed as a vector of
 *                                feature vectors.
 * \param regression_coefficients A vector of the same size of the input entries
 *                                plus 1, where the first element is the
 *                                regression bias and the others the regression
 *                                coefficients applied for each feature.
 * \param non_linear_functions    Vector of linear functions to apply after the
 *                                linear regression over inputs.
 *                                The size of each labels entry is defined by
 *                                the size of this vector.
 * \return A vector of label vectors generated for each input entry.
 */
static std::vector<std::vector<NumType>> generate_labels(
    const std::vector<std::vector<NumType>>& inputs,
    const std::vector<NumType>& regression_coefficients,
    std::vector<std::function<NumType(NumType)>> non_linear_functions);

/**
 * \brief Visualize the input, the expected output from the training set
 * and the model predictions in order to compare each other.
 * \param trainset          Dataset<NumType> Training set.
 * \param model_predictions Dataset<NumType> Model predictions set.
 */
static void check_predictions(Dataset<NumType>& trainset,
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

    // Define the regression weights and the non-linear functions for the
    // labels generation.
    const std::vector<NumType> REGRESSION_WEIGHTS = {
        // Bias.
        0.03,
        // 0.0,
        // Weights on input: same size as each individual input entry.
        0.05, 0.5, 0.3, 0.15
        // 1.0, 1.0, 1.0, 1.0
    };
    const std::vector<std::function<NumType(NumType)>> NON_LINEAR_FUNCTIONS = {
        // Label 0: square root function.
        [](NumType v) { return std::sqrt(std::abs(v)); },
        // Label 1: sin function.
        [](NumType v) { return std::sin(v); }
    };
    auto labels = generate_labels(inputs,
                                  REGRESSION_WEIGHTS, NON_LINEAR_FUNCTIONS);
    SizeType output_size = labels[0].size();

    // Transform the inputs and the labels in Dataset class.
    Dataset<NumType> inputs_ds(inputs);
    Dataset<NumType> labels_ds(labels);

    // Normalize inputs.
    inputs_ds.min_max_normalization();

    // Concatenate inputs and labels in order to build the training set.
    // concatenate(inputs, labels, axis=2) => [ input | labels ]
    auto training_set = Dataset<NumType>::concatenate(inputs_ds, labels_ds, 2);

    // Set labels indexes in the training set.
    std::vector<SizeType> labels_idx(output_size);
    std::iota(labels_idx.begin(), labels_idx.end(), input_size);
    training_set.label_idx(std::set(labels_idx.begin(), labels_idx.end()));

    //================================ MODEL DEFINITION: LOW LEVEL INTERFACE ===
    std::cout
        << "Example simple_regression with LOW LEVEL INTERFACE"
        << std::endl;

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
    auto loss = m_ll.add_loss<MeanSquaredLossLayer>("mse", output_size,
                                                    BATCH_SIZE);
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
                m_ll.step(training_set.input(i), training_set.label(i));
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
    for (SizeType i = 0; i < training_set.size(); ++i)
    {
        predictions.push_back(m_ll.predict(training_set.input(i)));
    }
    auto predictions_ds = Dataset<NumType>(predictions);
    check_predictions(training_set, predictions_ds);


    //=============================== MODEL DEFINITION: HIGH LEVEL INTERFACE ===
    std::cout
        << "Example simple_regression with HIGH LEVEL INTERFACE"
        << std::endl;

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
    NeuralNetworkDescriptor layers_descriptor(
        {
            Input{"input_layer",   input_size},
            Dense{"hidden_layer1", HIDDEN1,     ActivationType::ReLU   },
            Dense{"hidden_layer2", HIDDEN2,     ActivationType::ReLU   },
            Dense{"output_layer",  output_size, ActivationType::Linear }
        }
    );

    // Create the model object with the high level interface.
    CompileFeedforwardNeuralNetwork<LossType::MSE, InitType::AUTO> m_hl(
        layers_descriptor, //< Model descriptor.
        "regressor"        //< Model name.
    );

    // Training.
    std::cout << "--- Training" << std::endl;
    elapsed.start();
    m_hl.fit(training_set,                      //< Labeled dataset.
             OptimizerType::GRADIENT_DESCENT,   //< Optimizer.
             EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED);
    elapsed.stop();
    std::cout << "elapsed: " << std::string(elapsed) << std::endl;

    // Validation.
    std::cout << "--- Validation" << std::endl;
    auto score = m_hl.evaluate(training_set);
    std::cout
        << "Loss: " << score.loss << ", "
        << "Accuracy: " << score.accuracy_perc << "%, "
        << "Error rate: " << score.error_rate_perc << "%"
        << std::endl;

    // Prediction.
    auto observations = training_set.inputs();
    auto prediction = m_hl.predict(observations);
    check_predictions(training_set, prediction);

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
    const std::vector<NumType>& regression_coefficients,
    std::vector<std::function<NumType(NumType)>> non_linear_functions)
{
    std::vector<std::vector<NumType>> labels;

    // Generate the labels for each input entry in dataset.
    for (const auto& input_entry: inputs)
    {
        // Solve the multivariate linear regression applying the coefficients.
        NumType mlr = regression_coefficients[0];
        auto regression_size = std::min(
            regression_coefficients.size() - 1, input_entry.size());
        for (std::size_t w_i = 0; w_i < regression_size; ++w_i)
        {
            mlr += regression_coefficients[w_i + 1] * input_entry[w_i];
        }

        // Apply non-linearity to the multivariate linear regression.
        std::vector<NumType> label_entry;
        for (const auto& non_linear_f: non_linear_functions)
        {
            label_entry.push_back(non_linear_f(mlr));
        }
        labels.push_back(label_entry);
    }
    return labels;
}

static void check_predictions(Dataset<NumType>& trainset,
                              Dataset<NumType>& model_predictions)
{
    const SizeType MAX_ENTRY = 10;
    for (SizeType i = 0; i < std::min(MAX_ENTRY, trainset.size()); ++i)
    {
        const auto& input_entry = trainset.input(i);
        const auto& expected_output = trainset.label(i);
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
    if (trainset.size() > MAX_ENTRY) std::cout << " ... " << std::endl;
}
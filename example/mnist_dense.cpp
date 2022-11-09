/***************************************************************************
 *            example/mnist_dense.cpp
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

#include <iomanip>    //< For std::setprecision() function.

// The EdgeLeaning namespace is the only namespace in which all the features of
// the library are included.
using namespace EdgeLearning;


/*
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
    const SizeType BATCH_SIZE    = 128;
    // Number of iterations performed over the whole dataset.
    const SizeType EPOCHS        = 1;
    // Step size of the optimizer.
    const NumType  LEARNING_RATE = 0.01;

    // Size of the hidden layers.
    const SizeType HIDDEN1 = 250;
    const SizeType HIDDEN2 = 200;
    const SizeType HIDDEN3 = 150;
    const SizeType HIDDEN4 = 100;
    const SizeType HIDDEN5 = 50;

    const NumType PERCENTAGE_EVALUATION_DATASET = 0.1;

    Time elapsed; //< Utility for performance measure.

    // Mnist filepaths.
    const std::string MNIST_TRAINING_IMAGES_FN =
        "train-images.idx3-ubyte";
    const std::string MNIST_TRAINING_LABELS_FN =
        "train-labels.idx1-ubyte";
    const std::string MNIST_TESTING_IMAGES_FN =
        "t10k-images.idx3-ubyte";
    const std::string MNIST_TESTING_LABELS_FN =
        "t10k-labels.idx1-ubyte";
    const std::filesystem::path MNIST_RESOURCE_ROOT =
        std::filesystem::path(__FILE__).parent_path() / ".." / "data";
    const std::filesystem::path MNIST_TRAINING_IMAGES_FP =
        MNIST_RESOURCE_ROOT / MNIST_TRAINING_IMAGES_FN;
    const std::filesystem::path MNIST_TRAINING_LABELS_FP =
        MNIST_RESOURCE_ROOT / MNIST_TRAINING_LABELS_FN;
    const std::filesystem::path MNIST_TESTING_IMAGES_FP =
        MNIST_RESOURCE_ROOT / MNIST_TESTING_IMAGES_FN;
    const std::filesystem::path MNIST_TESTING_LABELS_FP =
        MNIST_RESOURCE_ROOT / MNIST_TESTING_LABELS_FN;

    // Parse Mnist whole dataset.
    auto mnist_training = Mnist(
        MNIST_TRAINING_IMAGES_FP,
        MNIST_TRAINING_LABELS_FP);
    auto mnist_testing = Mnist(
        MNIST_TESTING_IMAGES_FP,
        MNIST_TESTING_LABELS_FP);

    // Parse Mnist training set and normalize.
    auto data_training = Dataset<NumType>::parse(
        mnist_training,
        DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
    data_training = data_training.min_max_normalization(
        0, 255, data_training.input_idx());
    // Parse Mnist testing set and normalize.
    auto data_testing = Dataset<NumType>::parse(
        mnist_testing,
        DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
    data_testing = data_testing.min_max_normalization(
        0, 255, data_testing.input_idx());
    // Select Mnist evaluation dataset.
    auto data_evaluation = data_training.subdata(
        PERCENTAGE_EVALUATION_DATASET);

    auto input_shape = DLMath::Shape3d{
        MnistImage::IMAGE_SIDE, MnistImage::IMAGE_SIDE};
    auto input_size = input_shape.size();
    auto output_size = data_training.label_idx().size();

    //================================ MODEL DEFINITION: LOW LEVEL INTERFACE ===
    std::cout
        << "Example mnist_dense with LOW LEVEL INTERFACE"
        << std::endl;

    // Create an optimizer object that requires the learning step size.
    GradientDescentOptimizer optimizer(LEARNING_RATE);
    // The library support also the following optimizers:
    // AdamOptimizer optimizer(LEARNING_RATE);

    // Create the model object with the low level interface.
    Model m_ll{"mnist_classifier"};

    // Construct the model with the following structure:
    // ------------- IN[input_size] -------------
    //           Dense[HIDDEN1] + ReLU
    // #params: (input_size * HIDDEN1) + HIDDEN1
    // ------------------------------------------
    //           Dense[HIDDEN2] + ReLU
    // #params: (HIDDEN1 * HIDDEN2) + HIDDEN2
    // ------------------------------------------
    //           Dense[HIDDEN3] + ReLU
    // #params: (HIDDEN2 * HIDDEN3) + HIDDEN3
    // ------------------------------------------
    //           Dense[HIDDEN4] + ReLU
    // #params: (HIDDEN3 * HIDDEN4) + HIDDEN4
    // ------------------------------------------
    //           Dense[HIDDEN5] + ReLU
    // #params: (HIDDEN4 * HIDDEN5) + HIDDEN5
    // ------------------------------------------
    //           Dense[output_size] + Softmax
    // #params: (HIDDEN6 * output_size) + output_size
    // ------------ OUT[output_size] ------------
    auto h1          = m_ll.add_layer<DenseLayer>("h1", input_size, HIDDEN1);
    auto h1_relu     = m_ll.add_layer<ReluLayer>("h1_relu", HIDDEN1);
    auto h2          = m_ll.add_layer<DenseLayer>("h2", HIDDEN1, HIDDEN2);
    auto h2_relu     = m_ll.add_layer<ReluLayer>("h2_relu", HIDDEN2);
    auto h3          = m_ll.add_layer<DenseLayer>("h3", HIDDEN2, HIDDEN3);
    auto h3_relu     = m_ll.add_layer<ReluLayer>("h3_relu", HIDDEN3);
    auto h4          = m_ll.add_layer<DenseLayer>("h4", HIDDEN3, HIDDEN4);
    auto h4_relu     = m_ll.add_layer<ReluLayer>("h4_relu", HIDDEN4);
    auto h5          = m_ll.add_layer<DenseLayer>("h5", HIDDEN4, HIDDEN5);
    auto h5_relu     = m_ll.add_layer<ReluLayer>("h5_relu", HIDDEN5);
    auto out         = m_ll.add_layer<DenseLayer>("out", HIDDEN5, output_size);
    auto out_softmax = m_ll.add_layer<SoftmaxLayer>("out_softmax", output_size);
    m_ll.create_edge(h1, h1_relu);
    m_ll.create_edge(h1_relu, h2);
    m_ll.create_edge(h2, h2_relu);
    m_ll.create_edge(h2_relu, h3);
    m_ll.create_edge(h3, h3_relu);
    m_ll.create_edge(h3_relu, h4);
    m_ll.create_edge(h4, h4_relu);
    m_ll.create_edge(h4_relu, h5);
    m_ll.create_edge(h5, h5_relu);
    m_ll.create_edge(h5_relu, out);
    m_ll.create_edge(out, out_softmax);

    // Define the loss for training.
    // It requires the batch size for normalization purpose in gradient update.
    auto loss = m_ll.add_loss<CategoricalCrossEntropyLossLayer>(
        "cce", output_size, BATCH_SIZE);
    m_ll.create_loss_edge(out_softmax, loss);

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
        for (SizeType i = 0; i < data_training.size();)
        {
            // Reset the model loss scores.
            m_ll.reset_score();

            // Stochastic gradient descent.
            for (SizeType b = 0; b < BATCH_SIZE && i < data_training.size(); ++b, ++i)
            {
                // Crosses forward and backward the model, and generates the
                // gradients.
                m_ll.step(data_training.input(i), data_training.label(i));
            }

            // Update the model parameters with the optimizer and the generated
            // gradients.
            m_ll.train(optimizer);

            std::cout << "step " << i << " "
                      << "loss: " << m_ll.avg_loss()
                      << ", accuracy: " << m_ll.accuracy() * 100.0 << "%"
                      << std::endl;
        }
        std::cout << "loss: " << m_ll.avg_loss()
                  << ", accuracy: " << m_ll.accuracy() * 100.0 << "%"
                  << std::endl;
    }
    elapsed.stop();
    std::cout << "elapsed: " << std::string(elapsed) << std::endl;
    return 0;

    std::cout << "--- Predicting" << std::endl;
    std::vector<std::vector<NumType>> predictions;
    for (SizeType i = 0; i < data_evaluation.size(); ++i)
    {
        predictions.push_back(m_ll.predict(data_evaluation.input(i)));
    }
    auto predictions_ds = Dataset<NumType>(predictions);
    check_predictions(data_evaluation, predictions_ds);

    //=============================== MODEL DEFINITION: HIGH LEVEL INTERFACE ===
    std::cout
        << "Example simple_classification with HIGH LEVEL INTERFACE"
        << std::endl;

    // Construct the model with the following structure:
    // ------------- IN[input_size] -------------
    //           Dense[HIDDEN1] + ReLU
    // #params: (input_size * HIDDEN1) + HIDDEN1
    // ------------------------------------------
    //           Dense[HIDDEN2] + ReLU
    // #params: (HIDDEN1 * HIDDEN2) + HIDDEN2
    // ------------------------------------------
    //           Dense[HIDDEN3] + ReLU
    // #params: (HIDDEN2 * HIDDEN3) + HIDDEN3
    // ------------------------------------------
    //           Dense[HIDDEN4] + ReLU
    // #params: (HIDDEN3 * HIDDEN4) + HIDDEN4
    // ------------------------------------------
    //           Dense[HIDDEN5] + ReLU
    // #params: (HIDDEN4 * HIDDEN5) + HIDDEN5
    // ------------------------------------------
    //           Dense[output_size] + Softmax
    // #params: (HIDDEN5 * output_size) + output_size
    // ------------ OUT[output_size] ------------
    NeuralNetworkDescriptor layers_descriptor(
        {
            Input{"input_layer",   input_size},
            Dense{"hidden_layer1", HIDDEN1,     ActivationType::ReLU   },
            Dense{"hidden_layer2", HIDDEN2,     ActivationType::ReLU   },
            Dense{"hidden_layer3", HIDDEN3,     ActivationType::ReLU   },
            Dense{"hidden_layer4", HIDDEN4,     ActivationType::ReLU   },
            Dense{"hidden_layer5", HIDDEN5,     ActivationType::ReLU   },
            Dense{"output_layer",  output_size, ActivationType::Softmax }
        }
    );

    // Create the model object with the high level interface.
    CompileFeedforwardNeuralNetwork<LossType::CCE, InitType::AUTO> m_hl(
        layers_descriptor, //< Model descriptor.
        "classifier"       //< Model name.
    );

    // Training.
    std::cout << "--- Training" << std::endl;
    elapsed.start();
    m_hl.fit(data_training,                      //< Labeled dataset.
             OptimizerType::GRADIENT_DESCENT,   //< Optimizer.
             EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED);
    elapsed.stop();
    std::cout << "elapsed: " << std::string(elapsed) << std::endl;

    // Evaluation.
    std::cout << "--- Evaluation" << std::endl;
    auto evaluation_score = m_hl.evaluate(data_evaluation);
    std::cout
        << "Loss: " << evaluation_score.loss << ", "
        << "Accuracy: " << evaluation_score.accuracy_perc << "%, "
        << "Error rate: " << evaluation_score.error_rate_perc << "%"
        << std::endl;

    // Testing.
    std::cout << "--- Testing" << std::endl;
    auto testing_score = m_hl.evaluate(data_testing);
    std::cout
        << "Loss: " << testing_score.loss << ", "
        << "Accuracy: " << testing_score.accuracy_perc << "%, "
        << "Error rate: " << testing_score.error_rate_perc << "%"
        << std::endl;

    std::cout << "--- Predicting" << std::endl;
    auto prediction = m_hl.predict(data_evaluation);
    check_predictions(data_evaluation, prediction);

    std::cout << "End" << std::endl;
}

//==============================================================================
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
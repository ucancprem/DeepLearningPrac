package org.example.classification;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MultiClassClassification {

    private static int CLASS_COUNT = 3;
    private static int FEATURE_COUNT = 4;
    private static float SIXTY_FIVE_PERCENT = 65/100;
    private static String fileName = "IRIS.csv";


    private static void normalizeData(DataSet dataSet){
        DataNormalization dataNormalization = new NormalizerStandardize();
        dataNormalization.fit(dataSet); //Calculates the mean and std dev
        dataNormalization.transform(dataSet); // Applies normalization to dataset ensuring that each feature has a mean of zero and  a std deviation of 1
//        System.out.println("---------POST NORMALIZATION--------------");
//        System.out.println(dataSet);
    }

    private static MultiLayerConfiguration getNNConfig() {
        return new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .iterations(10000)
                .weightInit(WeightInit.XAVIER)
                .regularization(true)
                .learningRate(0.1).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURE_COUNT).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASS_COUNT).build())
                .backpropType(BackpropType.Standard).pretrain(false)
                .build();
    }

    private static MultiLayerNetwork trainModelWithDataSet(MultiLayerConfiguration configuration, DataSet trainingDataSet) {
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.fit(trainingDataSet);
        return model;
    }

    private static void evaluateModelWithTestData(MultiLayerNetwork model, DataSet testDataSet) {
        INDArray output = model.output(testDataSet.getFeatures());
        Evaluation evaluation =  new Evaluation(CLASS_COUNT);
        evaluation.eval(testDataSet.getLabels(), output);
        System.out.println(evaluation.stats());
    }

    public static void main(String[] args) {
        DataSet dataSet = CSVReader.fetchShuffledCSVData(fileName, 150, FEATURE_COUNT, CLASS_COUNT);
        //System.out.println(dataSet.toString());
        normalizeData(dataSet);
        SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.65);
        DataSet trainingDataSet = splitTestAndTrain.getTrain();
        DataSet testDataSet = splitTestAndTrain.getTest();

        MultiLayerConfiguration configuration =  getNNConfig();
        MultiLayerNetwork model = trainModelWithDataSet(configuration, trainingDataSet);

        evaluateModelWithTestData(model, testDataSet);
    }
}

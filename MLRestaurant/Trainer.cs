using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLRestaurant
{
    public class Trainer:BaseML
    {
        public void Train(string trainingFile)
        {
            if (!File.Exists(trainingFile))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFile}");
                return;
            }

            var trainingDataView = MlContext.Data.LoadFromTextFile<RestaurantFeedback>(trainingFile);
            Console.WriteLine("''''''''''");

            var dataSplit = MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            var dataProcessPipeline = MlContext.Transforms.Text.FeaturizeText(
                                        outputColumnName: "Features",
                                        inputColumnName: nameof(RestaurantFeedback.Text));

            var sdcaRegressionTrainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                                        labelColumnName: nameof(RestaurantFeedback.Label),
                                        featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);

            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            MlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);

            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            var modelMetrics = MlContext.BinaryClassification.Evaluate(data: testSetTransform,
                                labelColumnName: nameof(RestaurantFeedback.Label),
                                scoreColumnName: nameof(RestaurantPrediction.Score));

            Console.WriteLine($"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}{Environment.NewLine}" +
            $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}{Environment.NewLine}" +
            $"Accuracy: {modelMetrics.Accuracy:P2}{Environment.NewLine}" +
            $"F1Score: {modelMetrics.F1Score:P2}{Environment.NewLine}" +
            $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
            $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}");
        }
    }
}

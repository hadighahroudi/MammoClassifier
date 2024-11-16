using Microsoft.ML.Transforms.Image;
using Microsoft.ML;
using System.Drawing;
using Microsoft.ML.Data;
using System.Runtime.CompilerServices;
using System.IO;
using static System.Net.Mime.MediaTypeNames;
using System.ComponentModel.DataAnnotations;
using OpenCvSharp;
using System;

namespace MammoClassifier.Model
{
    public class BCDetector
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;

        public BCDetector()
        {
            _mlContext = new MLContext();
            _model = LoadModel();
        }

        public struct BCDetectorConfig
        {
            public const string ModelFilePath = "D:\\Study\\Proposal\\Breast cancer(v2)\\MammoClassifier\\MammoClassifier.Model\\model\\convnextv2_base-AdamW-up_sample-pos_smooth-mixup-cmmd_vindr-VOILUT_Flipped_pect_imgs-bs8x8-s0_e36_seed0.onnx";

            public const int ImageWidth = 1024;
            public const int ImageHeight = 1024;

            public const string InputLayer = "input";
            public const string OutputLayer = "output";

            public static readonly string[] Labels = { "non-malignant", "malignant" };

            public static float[] Softmax(float[] values)
            {
                var exponentialValues = values.Select(v => Math.Exp(v))
                    .ToArray();

                return exponentialValues.Select(exp => (float)(exp / exponentialValues.Sum()))
                    .ToArray();
            }
        }

        public ModelOutput DetectBCInMLImage(MLImage image)
        {
            var imageInputs = new List<ModelInput>
            {
                new ModelInput() {ImageAsBitmap = image}
            };

            float[] scoredImage = ScoreImageList(imageInputs).Single();

            //Format the score with labels
            return WrapModelOutput(scoredImage);
        }

        public IEnumerable<ModelOutput> DetectBCInImageFiles(string[] imagePaths)
        {
            var imageInputs = imagePaths
                            .Select(path => new ModelInput()
                            {
                                ImageAsBitmap = load_and_preprocess_image(path)
                            }
                            ).ToList();

            float[][] scoredImages = ScoreImageList(imageInputs);


            //Format the score with labels
            var result =
                imagePaths.Select(Path.GetFileName)
                    .Zip(
                        scoredImages,
                        (fileName, probabilities) => WrapModelOutput(probabilities, fileName)
                    );


            return result;
        }

        private float[] predict(string path)
        {
            //preprocess_pipeline = _mlContext.Transforms.NormalizeMinMax(outputColumnName: "scaled_float_image");
            var image = Cv2.ImRead(path, ImreadModes.Color);
            //var clahe = Cv2.CreateCLAHE(clipLimit: 2.0, tileGridSize: new OpenCvSharp.Size(8, 8));
            //var clahe_image = new Mat();
            //clahe.Apply(image, clahe_image);

            // Scale and normalize the image
            image.MinMaxIdx(out double minval, out double maxval);
            image.ConvertTo(image, MatType.CV_32FC3, alpha: 1 / maxval); // alpha is the scale factor which the image is multiplyed by
            image = image.Subtract(new Scalar(0.20275));
            image = image.Divide(0.19875);
            
            var input_blob = OpenCvSharp.Dnn.CvDnn.BlobFromImage(image);
            _opencv_net.SetInput(input_blob);
            var output = opencv_net.Forward();
            output.GetArray<float>(out float[] logits);
            var probs = BCDetectorConfig.Softmax(logits);

            return probs;
        }

        private static ModelOutput WrapModelOutput(float[] probabilities, string filename = null)
        {
            List<(string label, float probability)> mergedLabelsWithProbabilities =
                                                          BCDetectorConfig
                                                          .Labels
                                                          .Zip(
                                                                probabilities,
                                                                (label, probability) => (label, probability))
                                                          .ToList();

            return new ModelOutput()
            {
                Filename = filename,
                Probabilities = mergedLabelsWithProbabilities
            };
        }

        private ITransformer LoadModel()
        {
            var onnxScorer = _mlContext
                .Transforms
                .ApplyOnnxModel(
                    modelFile: BCDetectorConfig.ModelFilePath,
                    inputColumnNames: new[] { BCDetectorConfig.InputLayer },
                    outputColumnNames: new[] { BCDetectorConfig.OutputLayer }

                );

            var preProcessingPipeline = _mlContext
                    .Transforms
                    .ResizeImages(
                        inputColumnName: nameof(ModelInput.ImageAsBitmap),
                        imageWidth: BCDetectorConfig.ImageWidth,
                        imageHeight: BCDetectorConfig.ImageHeight,
                        outputColumnName: nameof(ModelInput.ImageAsBitmap)
                    ).Append(_mlContext
                    .Transforms
                    .ExtractPixels(
                        inputColumnName: nameof(ModelInput.ImageAsBitmap),
                        outputColumnName: BCDetectorConfig.InputLayer,
                        outputAsFloatArray: true
                    ));

            var completePipeline = preProcessingPipeline.Append(onnxScorer);

            // Fit scoring pipeline to the ModelInput structure to create a model
            var emptyInput = _mlContext.Data.LoadFromEnumerable(new List<ModelInput>());
            var model = completePipeline.Fit(emptyInput);

            return model;
        }

        private float[][] ScoreImageList(List<ModelInput> imageInputs)
        {
            // Create an IDataView from the image list
            IDataView imageDataView = _mlContext.Data.LoadFromEnumerable(imageInputs);

            // Transform the IDataView with the model
            IDataView scoredData = _model.Transform(imageDataView);

            // Extract the scores from the output layer
            var scoringValues = scoredData.GetColumn<float[]>(BCDetectorConfig.OutputLayer);

            //// Run the scores through the SoftMax function
            //float[][] probabilities;
            //probabilities = scoringValues.Select(BCDetectorConfig.Softmax)
            //                            .ToArray();

            return scoringValues.Cast<float[]>().ToArray();//probabilities;
        }
    }
}

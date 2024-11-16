using OpenCvSharp;
using System.Net.NetworkInformation;
using System.IO;

namespace MammoClassifier.Model
{
    public class BCDetector
    {
        private readonly OpenCvSharp.Dnn.Net _classifier_net;
        private readonly OpenCvSharp.Dnn.Net _segmentor_net;


        public BCDetector()
        {
            _classifier_net = OpenCvSharp.Dnn.CvDnn.ReadNetFromOnnx(BCDetectorConfig.ModelFilePath);
            _segmentor_net = OpenCvSharp.Dnn.CvDnn.ReadNetFromOnnx(BCDetectorConfig.PectRemoverPath);
        }

        public struct BCDetectorConfig
        {
            public static readonly string ModelFilePath = Path.Join(new string[] {Directory.GetCurrentDirectory(), "model", "convnextv2_base-AdamW-up_sample-pos_smooth-mixup-cmmd_vindr-VOILUT_Flipped_pect_imgs-bs8x8-s0_e36_seed0.onnx"});
            public static readonly string PectRemoverPath = Path.Join(new string[] {Directory.GetCurrentDirectory(), "model", "last_resnet101_unet3-inbreast_mias-breast_roi-adam-no_cls_guide-no_mixup-elastic_flip-output_resized-bs8_e100.onnx" });

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

        //public ModelOutput DetectBCInMLImage(MLImage image)
        //{
        //    var imageInputs = new List<ModelInput>
        //    {
        //        new ModelInput() {ImageAsBitmap = image}
        //    };

        //    float[] scoredImage = ScoreImageList(imageInputs).Single();

        //    //Format the score with labels
        //    return WrapModelOutput(scoredImage);
        //}

        public IEnumerable<ModelOutput> DetectBCInImageFiles(string[] imagePaths)
        {
            var imageProbs = new float[imagePaths.Length][];


            for(int i = 0; i < imagePaths.Length; i++)
            {
                imageProbs[i] = predict(imagePaths[i]);
            }


            //Format the score with labels
            var result = imagePaths.Select(Path.GetFileName).Zip(
                        imageProbs,
                        (fileName, probabilities) => WrapModelOutput(fileName, probabilities)
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

            _classifier_net.SetInput(input_blob);
            var output = _classifier_net.Forward();
            output.GetArray<float>(out float[] logits);
            var probs = BCDetectorConfig.Softmax(logits);

            return probs;
        }

        private void RemovePectMuscle(Mat image)
        {
            Cv2.Resize(image, image, new Size(512, 512));
            image = image.Normalize(0, 1, NormTypes.MinMax);
            var input_blob = OpenCvSharp.Dnn.CvDnn.BlobFromImage(image);
            _segmentor_net.SetInput(input_blob);
            var output = _segmentor_net.Forward();
            output = output.Multiply(255);
            output.ConvertTo(output, MatType.CV_8UC3);
            Cv2.ImShow("output", output);
            output.GetArray<float>(out float[] array);
        } 

        private static ModelOutput WrapModelOutput(string filename, float[] probabilities)
        {
            List<(string label, float prob)> mergedLabelsWithProbabilities = 
                BCDetectorConfig.Labels.Zip(probabilities, (label, prob) => (label, prob)).ToList();

            return new ModelOutput()
            {
                Filename = filename,
                Probabilities = mergedLabelsWithProbabilities
            };
        }
    }
}

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using static MammoClassifier.Model.BCDetector;

namespace MammoClassifier.Model
{
    public class ModelInput
    {
        [ImageType(BCDetectorConfig.ImageHeight, BCDetectorConfig.ImageWidth)]
        public MLImage ImageAsBitmap { get; set; }
    }
}

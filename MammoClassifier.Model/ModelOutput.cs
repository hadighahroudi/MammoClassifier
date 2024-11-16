using System.Numerics;

namespace MammoClassifier.Model
{
    public class ModelOutput
    {
        public string Filename { get; set; }
        public List<(string label, float prob)> Probabilities { get; set; }
    }
}

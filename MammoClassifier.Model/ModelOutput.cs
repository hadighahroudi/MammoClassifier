namespace MammoClassifier.Model
{
    public class ModelOutput
    {
        public string Filename { get; set; }
        public List<(string label, float probability)> Probabilities { get; set; }
    }
}

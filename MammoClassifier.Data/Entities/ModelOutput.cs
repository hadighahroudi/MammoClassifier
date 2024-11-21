namespace MammoClassifier.Data.Entities
{
    public class ModelOutput : BaseEntity
    {
        public string BIRADS { get; set; }
        public string Label { get; set; }
        public float Probability { get; set; }
    }
}

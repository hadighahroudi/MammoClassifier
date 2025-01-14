namespace MammoClassifier.Data.Entities
{
    public class Image : BaseEntity
    {
        public string? SopInstanceUID { get; set; }
        public DateTime? AcquisitionDate { get; set; }
        public string DICOMPath { get; set; }
        public string? ThumbnailPath { get; set; }
        public string Projection { get; set; }
        public string? MapPath { get; set; }
        public string? BIRADS { get; set; }
        public ICollection<ModelOutput>? ClassProbs { get; set; }
    }
}

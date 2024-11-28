namespace MammoClassifier.Data.Entities
{
    public class Study : BaseEntity
    {
        public string FirstName { get; set; }
        public string LastName { get; set; }
        public int? Age { get; set; }
        public string? PatientID { get; set; }
        public string? StudyUID { get; set; }
        public ICollection<Image> Images { get; set; }
    }
}

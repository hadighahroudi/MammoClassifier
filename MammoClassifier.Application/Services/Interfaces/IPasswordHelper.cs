namespace MammoClassifier.Application.Services.Interfaces
{
    public interface IPasswordHelper
    {
        string EncodePassword(string password);
        bool ComparePasswordWithHash(string savedHash,  string password);
    }
}

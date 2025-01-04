namespace MammoClassifier.Application.Services.Interfaces
{
    public interface IPasswordHelper
    {
        string EncodePasswordMD5(string password);
    }
}

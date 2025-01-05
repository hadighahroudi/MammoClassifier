using MammoClassifier.Application.Services.Interfaces;
using MammoClassifier.Data.Entities;
using Microsoft.EntityFrameworkCore;
using System.Security.Cryptography;
using Microsoft.AspNetCore.Identity;
using System.Text;

namespace MammoClassifier.Application.Services.Implementations
{
    public class PasswordHelper : IPasswordHelper
    {
        public string EncodePassword(string password)
        {
            return new PasswordHasher<object?>().HashPassword(null, password);

        }

        public bool ComparePasswordWithHash(string savedHash, string password)
        {
            var passwordVerificationResult = new PasswordHasher<object?>().VerifyHashedPassword(null, savedHash, password);
            switch (passwordVerificationResult)
            {
                case PasswordVerificationResult.Failed:
                    return false;

                case PasswordVerificationResult.Success:
                    return true;

                case PasswordVerificationResult.SuccessRehashNeeded:
                    Console.WriteLine("Password ok but should be rehashed and updated.");
                    return true;

                default:
                    throw new ArgumentOutOfRangeException();
            }

        }
    }
}

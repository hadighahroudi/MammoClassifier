using MammoClassifier.Data.DTOs;
using MammoClassifier.Data.Entities;

namespace MammoClassifier.Application.Services.Interfaces
{
    public interface IUserService
    {
        Task<RegisterUserResult> RegisterUser(RegisterUserDTO dto);
        Task<LoginUserResult> AuthenticateUser(LoginUserDTO dto);
        Task<User> GetUserByUsername(string username);
    }
}

using MammoClassifier.Data.DTOs;
using MammoClassifier.Data.Entities;

namespace MammoClassifier.Application.Services.Interfaces
{
    public interface IUserService
    {
        Task<LoginUserResult> AuthenticateUser(LoginUserDTO dto);
        Task<User> GetUserByUsername(string username);
    }
}

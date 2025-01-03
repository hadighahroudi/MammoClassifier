using MammoClassifier.Application.Services.Interfaces;
using MammoClassifier.Data.Context;
using MammoClassifier.Data.DTOs;
using MammoClassifier.Data.Entities;
using Microsoft.EntityFrameworkCore;

namespace MammoClassifier.Application.Services.Implementations
{
    public class UserService : IUserService
    {
        private readonly MammoClassifierDbContext _context;

        public UserService(MammoClassifierDbContext context)
        {
            _context = context;
        }

        public async Task<LoginUserResult> AuthenticateUser(LoginUserDTO dto)
        {
            try
            {
                var user = await _context.Users.SingleOrDefaultAsync(u => u.Username == dto.Username);
                if (user == null || user.IsDeleted || user.Password != dto.Password) return LoginUserResult.NotFound;
                else if (user.IsBlocked) return LoginUserResult.NotActivated;
                return LoginUserResult.Success;
            }
            catch (Exception) 
            {
                return LoginUserResult.Error;    
            }
        }

        public async Task<User> GetUserByUsername(string username)
        {
            return await _context.Users.SingleOrDefaultAsync(u => u.Username == username);
        }
    }
}

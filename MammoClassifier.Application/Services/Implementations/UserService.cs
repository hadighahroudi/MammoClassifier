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
        private readonly IPasswordHelper _passwordHelper;

        public UserService(MammoClassifierDbContext context, IPasswordHelper passwordHelper)
        {
            _context = context;
            _passwordHelper = passwordHelper;
        }

        public async Task<LoginUserResult> AuthenticateUser(LoginUserDTO dto)
        {
            try
            {
                var user = await GetUserByUsername(dto.Username);
                if (user == null ||
                    user.IsDeleted ||
                    !_passwordHelper.ComparePasswordWithHash(user.Password, dto.Password)) return LoginUserResult.NotFound;
                else if (user.IsBlocked) return LoginUserResult.NotActivated;
                return LoginUserResult.Success;
            }
            catch (Exception) 
            {
                return LoginUserResult.Error;    
            }
        }

        public async Task<RegisterUserResult> RegisterUser(RegisterUserDTO dto)
        {
            try
            {
                if (!await UserExistsByUsername(dto.Username))
                {
                    var user = new User()
                    { 
                        Username = dto.Username,
                        Password = _passwordHelper.EncodePassword(dto.Password),
                        CreateDate = DateTime.Now,
                        LastUpdateDate = DateTime.Now
                    };

                    await _context.Users.AddAsync(user);
                    await _context.SaveChangesAsync();

                    return RegisterUserResult.Success;
                }
                else
                {
                    return RegisterUserResult.UsernameExists;
                }
            }
            catch (Exception)
            {
                return RegisterUserResult.Error;
            }
        }


        public async Task<User> GetUserByUsername(string username)
        {
            return await _context.Users.SingleOrDefaultAsync(u => u.Username == username);
        }

        public async Task<bool> UserExistsByUsername(string username)
        {
            var user = await GetUserByUsername(username);
            return (user != null) && (user.IsDeleted == false);
        }
    }
}

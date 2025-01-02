using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using MammoClassifier.Web.Models;
using MammoClassifier.Data.DTOs;
using MammoClassifier.Data.Context;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication;

namespace MammoClassifier.Web.Controllers;

public class HomeController : Controller
{
    private readonly MammoClassifierDbContext _context;

    public HomeController(MammoClassifierDbContext context)
    {
        _context = context;
    }

    public IActionResult Index()
    {
        return View();
    }

    [HttpGet("login")]
    public IActionResult Login(string returnUrl = null)
    {
        if (User.Identity.IsAuthenticated) return Redirect("/");
        ViewData["ReturnUrl"] = returnUrl;
        return View();
    }

    [HttpPost("login"), ValidateAntiForgeryToken]
    public async Task<IActionResult> Login(LoginUserDTO dto, string returnUrl = null)
    {
        ViewData["ReturnUrl"] = returnUrl; //In case any condition leads to return View

        if (ModelState.IsValid)
        {
            var user = await _context.Users.SingleOrDefaultAsync(u => u.Username == dto.Username);
            if (user == null || user.Password != dto.Password) // Hash the password
            {
                ModelState.AddModelError(key: "Username", errorMessage: "نام کاربری یا کلمه عبور اشتباه است.");
                return View(dto);
            }

            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
                new Claim(ClaimTypes.Name, user.Username)
            };

            var identity = new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme);
            var principal = new ClaimsPrincipal(identity);
            var properties = new AuthenticationProperties()
            {
                IsPersistent = dto.RememberMe
            };

            await HttpContext.SignInAsync(principal, properties);

            if (!string.IsNullOrEmpty(returnUrl) && Url.IsLocalUrl(returnUrl)) { 
                return Redirect(returnUrl);
            }
            else
            {
                return Redirect("/");
            }

        }

        return View(dto);
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}

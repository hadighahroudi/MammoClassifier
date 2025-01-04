using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using MammoClassifier.Web.Models;
using MammoClassifier.Data.DTOs;
using MammoClassifier.Data.Context;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication;
using GoogleReCaptcha.V3.Interface;
using MammoClassifier.Application.Services.Interfaces;

namespace MammoClassifier.Web.Controllers;

public class HomeController : Controller
{
    private readonly IUserService _userService;
    private readonly ICaptchaValidator _captchaValidator;

    public HomeController(IUserService userService, ICaptchaValidator captchaValidator)
    {
        _userService = userService;
        _captchaValidator = captchaValidator;
    }

    public IActionResult Index()
    {
        return View();
    }

    [HttpGet("signup")]
    public IActionResult Signup()
    {
        return View();
    }

    [HttpPost("signup"), ValidateAntiForgeryToken]
    public async Task<IActionResult> Signup(RegisterUserDTO dto)
    {
        if (!await _captchaValidator.IsCaptchaPassedAsync(dto.Captcha))
        {
            ModelState.AddModelError(key: "Username", errorMessage: "عدم تایید اعتبارسنجی reCAPTCHA.");
        }
        else if (ModelState.IsValid)
        {
            var res = await _userService.RegisterUser(dto);

            switch(res)
            {
                case RegisterUserResult.Success:
                    return RedirectToAction("Login");

                case RegisterUserResult.UsernameExists:
                    ModelState.AddModelError(key: "Username", errorMessage: "اکانتی با این نام کاربری وجود دارد.");
                    break;

                case RegisterUserResult.Error:
                    ModelState.AddModelError(key: "Username", errorMessage: "خطا در ثبت نام.");
                    break;

            }
        }

        return View(dto);
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

        if (!await _captchaValidator.IsCaptchaPassedAsync(dto.Captcha))
        {
            ModelState.AddModelError(key: "Username", errorMessage: "عدم تایید اعتبارسنجی reCAPTCHA.");
        }
        else if (ModelState.IsValid)
        {
            var res = await _userService.AuthenticateUser(dto);
            switch (res)
            {
                case LoginUserResult.NotFound:
                    ModelState.AddModelError(key: "Username", errorMessage: "نام کاربری یا کلمه عبور اشتباه است.");
                    break;

                case LoginUserResult.NotActivated:
                    ModelState.AddModelError(key: "Username", errorMessage: "حساب کاربری شما فعال نشده است.");
                    break;

                case LoginUserResult.Error:
                    ModelState.AddModelError(key: "Username", errorMessage: "خطا در ورود.");
                    break;

                case LoginUserResult.Success:
                    var user = await _userService.GetUserByUsername(dto.Username);

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

                    if (!string.IsNullOrEmpty(returnUrl) && Url.IsLocalUrl(returnUrl))
                    {
                        return Redirect(returnUrl);
                    }
                    else
                    {
                        return Redirect("/");
                    }
            }
        }

        return View(dto);
    }

    [HttpGet("logout")]
    public async Task<IActionResult> Logout()
    {
        await HttpContext.SignOutAsync();

        return Redirect("/");
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}

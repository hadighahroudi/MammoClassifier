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
using Microsoft.AspNetCore.Authorization;

namespace MammoClassifier.Web.Controllers;

public class HomeController : Controller
{
    public IActionResult Index()
    {
        return View();
    }


    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}

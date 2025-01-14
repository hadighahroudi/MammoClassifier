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
using Newtonsoft.Json;
using MammoClassifier.Data.Entities;
using System.Reflection.Emit;
using Microsoft.IdentityModel.Tokens;

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

        // fill_database();

        return View();
    }

    private void fill_database()
    {
        // Path to your file
        string filePath = @"D:\Study\Proposal\Breast cancer(v3)\Sampled test images for system\cmmd model results.txt";

        // Read all lines from the file
        string[] lines = System.IO.File.ReadAllLines(filePath);

        // List to store dictionaries
        var dictionaries = new List<Dictionary<string, object>>();

        foreach (string line in lines)
        {
            // Skip empty lines
            if (string.IsNullOrWhiteSpace(line)) continue;

            // Parse the line into a dictionary
            try
            {
                var dictionary = JsonConvert.DeserializeObject<Dictionary<string, object>>(line.Trim());
                dictionaries.Add(dictionary);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error parsing line: {line}. Exception: {ex.Message}");
            }
        }


        // Example usage: Print all dictionaries
        foreach (var dict in dictionaries)
        {
            var study = new Study()
            {
                PatientID = (string)dict["patId"],
                StudyUID = (string)dict["patId"],
                CreateDate = DateTime.Now,
                IsDeleted = false,
                LastUpdateDate = DateTime.Now
            };

            study.Age = ((string)dict["age"] != "NA") ? int.Parse((string)dict["age"]) : 0;

            float prob = 0;
            string label = null;
            try
            {
                prob = float.Parse(((Newtonsoft.Json.Linq.JArray)dict["probs"]).Last.Last.ToString());

                if (prob < 0.5)
                {
                    label = "Non-Malignant";
                    prob = 1 - prob;
                }
                else
                {
                    label = "Malignant";
                }

                if (prob == 1.0)
                {
                    prob = 0.99f;
                }
            }
            catch // Prob is empty
            {

            }


            var image_cc = new Image()
            {
                DICOMPath = ((string)dict["cc_map_path"]).Replace("_map.png", ".dcm"),
                IsDeleted = false,
                CreateDate = DateTime.Now,
                LastUpdateDate = DateTime.Now,
                Projection = "CC",
                SopInstanceUID = (string)dict["patId"],
                AcquisitionDate = DateTime.Now
            };

            var image_mlo = new Image()
            {
                DICOMPath = ((string)dict["mlo_map_path"]).Replace("_map.png", ".dcm"),
                IsDeleted = false,
                CreateDate = DateTime.Now,
                LastUpdateDate = DateTime.Now,
                Projection = "MLO",
                SopInstanceUID = (string)dict["patId"],
                AcquisitionDate = DateTime.Now
            };

            if (prob != 0)
            {
                image_cc.ClassProbs = new List<ModelOutput>() { new ModelOutput() { Label = label,
                                                                           Probability = prob,
                                                                           CreateDate = DateTime.Now,
                                                                           IsDeleted = false,
                                                                           LastUpdateDate = DateTime.Now}};
                image_mlo.ClassProbs = new List<ModelOutput>() { new ModelOutput() { Label = label,
                                                                           Probability = prob,
                                                                           CreateDate = DateTime.Now,
                                                                           IsDeleted = false,
                                                                           LastUpdateDate = DateTime.Now}};
                image_cc.MapPath = (string)dict["cc_map_path"];
                image_mlo.MapPath = (string)dict["mlo_map_path"];
            }

            study.Images = new List<Image>() { image_cc, image_mlo };

            _context.Studies.Add(study);
            _context.SaveChanges();
        }
    }


    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}

using MammoClassifier.Data.Context;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Cors;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Authorization;
using Newtonsoft.Json;
using MammoClassifier.Data.DTOs;

namespace MammoClassifier.Web.Controllers
{
    [Authorize]
    public class ViewerController : Controller, IDisposable
    {
        private readonly MammoClassifierDbContext _context;

        public ViewerController(MammoClassifierDbContext context, IWebHostEnvironment webHostEnvironment)
        {
            _context = context;
        }
        
        public async Task<IActionResult> Index()
        {
            return View(_context.Studies);
        }

        public async Task<IActionResult> MainPanel(int id)
        {
            var study = await _context.Studies.Include(x => x.Images).ThenInclude(y => y.ClassProbs).FirstOrDefaultAsync(x => x.Id == id);

            if (study == null){
                return NotFound();
            }

            foreach (var image in study.Images)
            {
                var address = $"{HttpContext.Request.Scheme}://{HttpContext.Request.Host}";
                image.DICOMPath = $"{address}/{image.DICOMPath}";
                image.MapPath = $"{address}/{image.MapPath}";
                image.ThumbnailPath = $"{address}/{image.ThumbnailPath}";
            }

            ViewData["NextStudyId"] = (await _context.Studies.Where(s => s.Id > id).OrderBy(s => s.Id).FirstOrDefaultAsync())?.Id;
            ViewData["PrevStudyId"] = (await _context.Studies.Where(s => s.Id < id).OrderByDescending(s => s.Id).FirstOrDefaultAsync())?.Id;

            if (ViewData["NextStudyId"] == null) // If no next record exists, wrap around to the first record
            {
                ViewData["NextStudyId"] = (await _context.Studies.OrderBy(r => r.Id).FirstOrDefaultAsync())?.Id;
            }
            if (ViewData["PrevStudyId"] == null) // If no previous record exists, wrap around to the last record
            {
                ViewData["PrevStudyId"] = (await _context.Studies.OrderByDescending(r => r.Id).FirstOrDefaultAsync())?.Id;
            }
            return View(study);
        }

        [HttpPost("/savebirads")]
        public async Task<IActionResult> SaveBIRADS([FromBody] BiradsDTO dto)
        {
            if (dto == null || dto.StudyId == null)
            {
                return BadRequest(new { Message = "The final BIRADS score value can not be saved.", StudyId = dto.StudyId, BiradsScore = dto.Score});
            }
            else if (string.IsNullOrEmpty(dto.Score))
            {
                return Ok(new { Message = "No BIRADS value.", StudyId = dto.StudyId, BiradsScore = dto.Score});
            }

            var study = await _context.Studies.SingleOrDefaultAsync(s => s.Id == dto.StudyId);
            if (study == null)
            {
                return BadRequest(new { Message = "The final BIRADS score value can not be saved.", StudyId = dto.StudyId, BiradsScore = dto.Score});
            }

            study.BIRADS = dto.Score;
            _context.Studies.Update(study);
            await _context.SaveChangesAsync();

            return Ok(new { Message = "The final BIRADS score value was saved successfully", StudyId = dto.StudyId, BiradsScore = dto.Score});
        }

        protected override void Dispose(bool disposing)
        {
            _context.Dispose();
            base.Dispose(disposing);
        }
    }
}

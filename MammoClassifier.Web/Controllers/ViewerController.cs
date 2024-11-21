using MammoClassifier.Data.Context;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Cors;
using Microsoft.EntityFrameworkCore;

namespace MammoClassifier.Web.Controllers
{
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
            var study = await _context.Studies.Include(x => x.Images).FirstOrDefaultAsync(x => x.Id == id);

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

            return View(study);
        }

        protected override void Dispose(bool disposing)
        {
            _context.Dispose();
            base.Dispose(disposing);
        }
    }
}

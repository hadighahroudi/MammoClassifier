using MammoClassifier.Data.Entities;
using Microsoft.EntityFrameworkCore;

namespace MammoClassifier.Data.Context
{
    public class MammoClassifierDbContext : DbContext
    {
        public MammoClassifierDbContext(DbContextOptions<MammoClassifierDbContext> options) : base(options) { }

        public DbSet<Study> Studies { get; set; }
        public DbSet<Image> Images { get; set; }
        public DbSet<ModelOutput> ModelOutputs { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            foreach (var relationship in modelBuilder.Model.GetEntityTypes().SelectMany(e => e.GetForeignKeys()))
            {
                relationship.DeleteBehavior = DeleteBehavior.Cascade;
            }

            base.OnModelCreating(modelBuilder);
        }
    }
}

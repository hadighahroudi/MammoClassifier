using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MammoClassifier.Data.Migrations
{
    /// <inheritdoc />
    public partial class Added_study_birads : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "BIRADS",
                table: "Studies",
                type: "nvarchar(max)",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "BIRADS",
                table: "Studies");
        }
    }
}

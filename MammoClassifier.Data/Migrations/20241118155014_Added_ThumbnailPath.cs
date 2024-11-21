using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MammoClassifier.Data.Migrations
{
    /// <inheritdoc />
    public partial class Added_ThumbnailPath : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "ThumbnailPath",
                table: "Images",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ThumbnailPath",
                table: "Images");
        }
    }
}

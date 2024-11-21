using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MammoClassifier.Data.Migrations
{
    /// <inheritdoc />
    public partial class Added_Projection : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Projection",
                table: "Images",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Projection",
                table: "Images");
        }
    }
}

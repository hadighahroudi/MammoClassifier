using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MammoClassifier.Data.Migrations
{
    /// <inheritdoc />
    public partial class added_fields_to_study : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "FirstName",
                table: "Studies",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "LastName",
                table: "Studies",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "NationalID",
                table: "Studies",
                type: "nvarchar(max)",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "FirstName",
                table: "Studies");

            migrationBuilder.DropColumn(
                name: "LastName",
                table: "Studies");

            migrationBuilder.DropColumn(
                name: "NationalID",
                table: "Studies");
        }
    }
}

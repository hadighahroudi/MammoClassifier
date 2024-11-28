using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MammoClassifier.Data.Migrations
{
    /// <inheritdoc />
    public partial class corrected_study_props : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "age",
                table: "Studies",
                newName: "Age");

            migrationBuilder.RenameColumn(
                name: "NationalID",
                table: "Studies",
                newName: "PatientID");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "Age",
                table: "Studies",
                newName: "age");

            migrationBuilder.RenameColumn(
                name: "PatientID",
                table: "Studies",
                newName: "NationalID");
        }
    }
}

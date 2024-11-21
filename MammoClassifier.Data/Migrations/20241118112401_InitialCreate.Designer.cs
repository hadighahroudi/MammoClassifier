﻿// <auto-generated />
using System;
using MammoClassifier.Data.Context;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Migrations;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

#nullable disable

namespace MammoClassifier.Data.Migrations
{
    [DbContext(typeof(MammoClassifierDbContext))]
    [Migration("20241118112401_InitialCreate")]
    partial class InitialCreate
    {
        /// <inheritdoc />
        protected override void BuildTargetModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder
                .HasAnnotation("ProductVersion", "9.0.0")
                .HasAnnotation("Relational:MaxIdentifierLength", 128);

            SqlServerModelBuilderExtensions.UseIdentityColumns(modelBuilder);

            modelBuilder.Entity("MammoClassifier.Data.Entities.Image", b =>
                {
                    b.Property<long>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("bigint");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<long>("Id"));

                    b.Property<DateTime?>("AcquisitionDate")
                        .HasColumnType("datetime2");

                    b.Property<string>("BIRADS")
                        .HasColumnType("nvarchar(max)");

                    b.Property<DateTime>("CreateDate")
                        .HasColumnType("datetime2");

                    b.Property<string>("DICOMPath")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.Property<bool>("IsDeleted")
                        .HasColumnType("bit");

                    b.Property<DateTime>("LastUpdateDate")
                        .HasColumnType("datetime2");

                    b.Property<string>("MapPath")
                        .HasColumnType("nvarchar(max)");

                    b.Property<string>("SopInstanceUID")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.Property<long?>("StudyId")
                        .HasColumnType("bigint");

                    b.HasKey("Id");

                    b.HasIndex("StudyId");

                    b.ToTable("Images");
                });

            modelBuilder.Entity("MammoClassifier.Data.Entities.ModelOutput", b =>
                {
                    b.Property<long>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("bigint");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<long>("Id"));

                    b.Property<string>("BIRADS")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.Property<DateTime>("CreateDate")
                        .HasColumnType("datetime2");

                    b.Property<long?>("ImageId")
                        .HasColumnType("bigint");

                    b.Property<bool>("IsDeleted")
                        .HasColumnType("bit");

                    b.Property<string>("Label")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.Property<DateTime>("LastUpdateDate")
                        .HasColumnType("datetime2");

                    b.Property<float>("Probability")
                        .HasColumnType("real");

                    b.HasKey("Id");

                    b.HasIndex("ImageId");

                    b.ToTable("ModelOutputs");
                });

            modelBuilder.Entity("MammoClassifier.Data.Entities.Study", b =>
                {
                    b.Property<long>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("bigint");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<long>("Id"));

                    b.Property<DateTime>("CreateDate")
                        .HasColumnType("datetime2");

                    b.Property<bool>("IsDeleted")
                        .HasColumnType("bit");

                    b.Property<DateTime>("LastUpdateDate")
                        .HasColumnType("datetime2");

                    b.Property<string>("StudyUID")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.HasKey("Id");

                    b.ToTable("Studies");
                });

            modelBuilder.Entity("MammoClassifier.Data.Entities.Image", b =>
                {
                    b.HasOne("MammoClassifier.Data.Entities.Study", null)
                        .WithMany("Images")
                        .HasForeignKey("StudyId")
                        .OnDelete(DeleteBehavior.Cascade);
                });

            modelBuilder.Entity("MammoClassifier.Data.Entities.ModelOutput", b =>
                {
                    b.HasOne("MammoClassifier.Data.Entities.Image", null)
                        .WithMany("ClassProbs")
                        .HasForeignKey("ImageId")
                        .OnDelete(DeleteBehavior.Cascade);
                });

            modelBuilder.Entity("MammoClassifier.Data.Entities.Image", b =>
                {
                    b.Navigation("ClassProbs");
                });

            modelBuilder.Entity("MammoClassifier.Data.Entities.Study", b =>
                {
                    b.Navigation("Images");
                });
#pragma warning restore 612, 618
        }
    }
}

using MammoClassifier.Model;

Console.WriteLine("Breast cancer detector");
Console.WriteLine();

// Load all image paths
var imageFolder = GetImageFolderFromArgs(args);
var imagePaths = Directory.GetFiles(imageFolder,
                                    "*.png")
                          .Select(Path.GetFullPath)
                          .ToArray();


// Run the images through the ONNX model
var bcDetector = new BCDetector();

Console.WriteLine("Scoring all images...");
var timeStamp = DateTime.Now;

var BCProbabilities = bcDetector.DetectBCInImageFiles(imagePaths);

Console.WriteLine($"Images scored in {(DateTime.Now - timeStamp).TotalSeconds} seconds");
Console.WriteLine();


// Print the results
foreach (var fileWithScore in BCProbabilities)
{
    Console.Write("* ");

    Console.ForegroundColor = ConsoleColor.Cyan;

    Console.WriteLine($"{fileWithScore.Filename}");

    Console.ForegroundColor = ConsoleColor.Gray;

    foreach (var label_prob in fileWithScore.Probabilities)
    {
        if (label_prob.probability > 0.1)
        {
            Console.ForegroundColor = ConsoleColor.Green;
        }

        Console.Write($"[{label_prob.probability:P0}] {label_prob.label}  ");

        Console.ForegroundColor = ConsoleColor.Gray;
    }
    Console.WriteLine();
    Console.WriteLine();

}


string GetImageFolderFromArgs(string[] args)
{
    //Default to "images" folder if no arguments are given
    if (args.Length != 1) return "images";

    if (Directory.Exists(args[0])) return args[0];

    Console.WriteLine("Given image directory does not exist");
    Environment.Exit(1);

    return null;
}
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Reflection;
using Tensorflow;
using Console = Colorful.Console;
using static Tensorflow.Binding;
using CommandLine;

namespace TensorFlowClassification
{
    public class Arguments
    {
        public static int HandleParseError(IEnumerable<Error> errs)
        {
            var result = -2;
            Console.WriteLine("errors {0}", errs.Count());
            if (errs.Any(x => x is HelpRequestedError || x is VersionRequestedError))
                result = -1;
            Console.WriteLine("Exit code {0}", result);
            return result;
        }

        [Option('t', "method", Required = true, HelpText = "Prediction mode")]
        public string Method { get; set; }

        [Option('i', "image", Required = false, Separator = ',', HelpText = "Predicted image path list")]
        public IEnumerable<string> ImagePathList { get; set; }

        [Option('l', "image-list", Required = false, HelpText = "Predicted image path list file (*.csv)")]
        public string ImageListDefFilePath { get; set; }

        [Option('m', "model", Required = true, HelpText = "Deep learning model")]
        public string ModelFilePath { get; set; }

        [Option('c', "label", Required = true, HelpText = "Label name")]
        public string LabelFilePath { get; set; }

        [Option('s', "batch-size", HelpText = "Batch size")]
        public int BatchSize { get; set; } = 64;

        [Option('o', "output", Required = false, HelpText = "")]
        public string OutputPath { get; set; }

        [Option('v', "verbose", Required = false, HelpText = "Set output to verbose messages.")]
        public bool Verbose { get; set; }


    }
}

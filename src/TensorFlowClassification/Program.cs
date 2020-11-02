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
    class Program
    {
        static void Main(string[] args)
        {
            // Parse command line arguments
            Arguments parsedArgs = null;
            CommandLine.Parser.Default.ParseArguments<Arguments>(args)
                .WithParsed(opt => parsedArgs = opt)
                .WithNotParsed(errs => Arguments.HandleParseError(errs));
            if (parsedArgs == null)
            {
                return;
            }

            // Collect run mode class
            var examples = Assembly.GetEntryAssembly().GetTypes()
                .Where(x => x.GetInterfaces().Contains(typeof(IExample)))
                .Select(x => (IExample)Activator.CreateInstance(x))
                .Where(x => x.InitConfig(parsedArgs) != null)
                .Where(x => x.Config.Name == parsedArgs.Method)
                .OrderBy(x => x.Config.Priority)
                .ToArray();

            if (examples.Length != 1)
            {
                Console.WriteLine($"Error : Method {parsedArgs.Method} was not found.", Color.Yellow);
                return;
            }

            var errors = new List<string>();
            var success = new List<string>();

            // debug output
            if (parsedArgs.Verbose)
            {
                Console.WriteLine(Environment.OSVersion.ToString(), Color.Yellow);
                Console.WriteLine($"TensorFlow Binary v{tf.VERSION}", Color.Yellow);
                Console.WriteLine($"TensorFlow.NET v{Assembly.GetAssembly(typeof(TF_DataType)).GetName().Version}", Color.Yellow);
            }

            // parformance
            var sw = new Stopwatch();

            var example = examples[0];
            if (parsedArgs.Verbose)
            {
                Console.WriteLine($"{DateTime.UtcNow} Starting {example.Config.Name}", Color.White);
            }

            try
            {
                sw.Restart();
                bool isSuccess = example.Run();
                sw.Stop();

                if (isSuccess)
                    success.Add($"Example: {example.Config.Name} in {sw.Elapsed.TotalSeconds}s");
                else
                    errors.Add($"Example: {example.Config.Name} in {sw.Elapsed.TotalSeconds}s");
            }
            catch (Exception ex)
            {
                errors.Add($"Example: {example.Config.Name}");
                Console.WriteLine(ex);
            }

            if (parsedArgs.Verbose)
            {
                Console.WriteLine($"{DateTime.UtcNow} Completed {example.Config.Name}", Color.White);
            }

            success.ForEach(x => Console.WriteLine($"{x} is OK!", Color.Green));
            errors.ForEach(x => Console.WriteLine($"{x} is Failed!", Color.Red));
        }
    }
}

using NumSharp;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Console = Colorful.Console;
using Tensorflow;
using System.Drawing;
using static Tensorflow.Binding;

using SharpCV;
using static SharpCV.Binding;

namespace TensorFlowClassification
{
    /// <summary>
    /// Single image pair prediction
    /// </summary>
    public class RunModeSingle: RunModeBase, IExample
    {
        List<string[]> imagePathList = new List<string[]>();
        List<NDArray> file_ndarrays = new List<NDArray>();

        public ExampleConfig InitConfig(Arguments args)
            => Config = new ExampleConfig
            {
                Priority = 10,
                Name = "single",
                Args = args,
            };

        public bool Run()
        {
            PrepareData();
           
            var graph = new Graph();
            //import GraphDef from pb file
            graph.Import(Path.Join( Config.Args.ModelFilePath));

            // モデルに応じて決まる
            var input_name = graph.First().name;
            var output_name = graph.Last().name;

            var input_operation = graph.OperationByName(input_name);
            var output_operation = graph.OperationByName(output_name);

            var labels = File.ReadAllLines(Path.Join(Config.Args.LabelFilePath));
            var result_labels = new List<string>();
            var sw = new Stopwatch();

            var predicts = new PredictResult()
            {
                Items = new List<PredictResultItem>(),
            };

            using (var sess = tf.Session(graph))
            {
                //foreach (var nd in file_ndarrays)
                for(int i = 0; i < file_ndarrays.Count; i++)
                {
                    var nd = file_ndarrays[i];
                    var files = this.imagePathList[i];

                    sw.Restart();

                    var results = sess.run(output_operation.outputs[0], (input_operation.outputs[0], nd));
                    results = np.squeeze(results);
                    var singleNd = results.ToArray<float>();

                    var idxs = np.argsort<double>(results)["::-1"];  // high confident order
                    var idx = idxs[0];  // prediction result

                    Console.WriteLine($"{files[0]} in {sw.ElapsedMilliseconds}ms", Color.Tan);
                    for (int j = 0; j < idxs.Shape[0]; j++)
                    {
                        Console.WriteLine($"  {idxs[j]}) {labels[idxs[j]]} {singleNd[idxs[j]]}", Color.Tan);
                    }
                    result_labels.Add(labels[idx]);

                    var resultArr = results.ToArray<float>();

                    predicts.Items.Add(new PredictResultItem()
                    {
                        PathList = new List<string>(files),
                        Predicts = new List<float>(resultArr),
                        TopIndex = idx,
                    }); ;
                }
            }

            // output file
            predicts.WriteFile(Config.Args.OutputPath, labels);

            return predicts.Items.Count > 0;
        }

        public void PrepareData()
        {
            // single image pair
            string[] fileNames = Config.Args.ImagePathList.ToArray();
            this.imagePathList.Add(fileNames);

            // convert to TensorFlow object
            foreach(var files in this.imagePathList)
            {
                var nd = ReadTensorFromImageFile(files);
                file_ndarrays.Add(nd);
            }
        }

    }
}

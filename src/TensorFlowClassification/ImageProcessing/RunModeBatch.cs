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
    public class RunModeBatch: RunModeBase, IExample
    {
        public ExampleConfig InitConfig(Arguments args)
            => Config = new ExampleConfig
            {
                Priority = 20,
                Name = "batch",
                Args = args,
            };

        public bool Run()
        {
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
            sw.Start();

            var predicts = new PredictResult()
            {
                Items = new List<PredictResultItem>(),
            };

            using (var sess = tf.Session(graph))
            {
                Console.WriteLine($"Initialized in {sw.ElapsedMilliseconds}ms", Color.Tan);
                sw.Restart();

                // mini batch
                foreach(BatchContainer bc in PrepareData(Config.Args.BatchSize))
                {
                    // source
                    var files = new List<string[]>(bc.Files);
                    var nd = bc.Nd;

                    // prediction process
                    var results = sess.run(output_operation.outputs[0], (input_operation.outputs[0], nd));

                    // per file pair
                    double elapsed = (double)sw.ElapsedMilliseconds / files.Count;

                    // output
                    for(int i = 0; i < files.Count; i++)
                    {
                        var singleResults = results[i];
                        var singleNd = singleResults.ToArray<float>();
                        var singleFiles = files[i];

                        var idxs = np.argsort<double>(singleResults)["::-1"];  // high confident order
                        var idx = idxs[0];  // prediction result

                        Console.WriteLine($"{singleFiles[0]} in {elapsed}ms", Color.Tan);
                        for (int j = 0; j < idxs.Shape[0]; j++)
                        {
                            Console.WriteLine($"  {idxs[j]}) {labels[idxs[j]]} {singleNd[idxs[j]]}", Color.Tan);
                        }
                        result_labels.Add(labels[idx]);

                        predicts.Items.Add(new PredictResultItem()
                        {
                            PathList = new List<string>(singleFiles),
                            Predicts = new List<float>(singleNd),
                            TopIndex = idx,
                        }) ;
                    }

                    sw.Restart();
                }
            }

            // output file
            predicts.WriteFile(Config.Args.OutputPath, labels);

            return predicts.Items.Count > 0;
        }

        public IEnumerable<BatchContainer> PrepareData(int batchSize)
        {
            List<string[]> imagePathList = new List<string[]>();

            var dirInfo = System.IO.Directory.GetParent(Config.Args.ImageListDefFilePath);
            var dir = dirInfo.FullName;
            using (StreamReader reader = new StreamReader(Config.Args.ImageListDefFilePath, System.Text.Encoding.GetEncoding("UTF-8")))
            {
                while (reader.Peek() >= 0)
                {
                    string line = reader.ReadLine();
                    if (line.Length == 0)
                    {
                        continue;
                    }
                    string[] cols = line.Split(',').Select(p => System.IO.Path.Combine(dir, p)).ToArray();
                    imagePathList.Add(cols);
                }
            }

            foreach(IEnumerable<string[]>filesList in this.GetBatch<string>(imagePathList, batchSize))
            {
                List<NDArray> mats = new List<NDArray>();
                foreach(string[] files in filesList)
                {
                    var nd = ReadTensorFromImageFile(files);
                    mats.Add(nd);
                }
                var image_array = np.concatenate(mats.ToArray(), 0);
                yield return new BatchContainer(image_array, filesList, filesList.Count());
            }
        }

        IEnumerable<List<T[]>> GetBatch<T>(List<T[]> sources, int batchSize)
        {
            List<T[]> container = new List<T[]>();
            for(int i = 0; i < sources.Count; i++)
            {
                container.add(sources[i]);
                if (container.Count == batchSize)
                {
                    yield return container;
                    container.Clear();
                    continue;
                }
            }
            if (container.Count > 0)
            {
                yield return container;
                container.Clear();
            }
        }

    }

    public class BatchContainer
    {
        public BatchContainer(NDArray nd, IEnumerable<string[]> files, int count)
        {
            this.Nd = nd;
            this.Files = files;
            this.Count = Count;
        }

        public NDArray Nd { get; }

        public IEnumerable<string[]> Files { get; }

        public int Count { get; }
    }
}

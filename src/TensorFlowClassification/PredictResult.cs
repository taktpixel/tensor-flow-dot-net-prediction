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
using System.Text;

namespace TensorFlowClassification
{
    public class PredictResult
    {
        public List<PredictResultItem> Items { get; set; }

        public bool WriteFile(string filePath, IEnumerable<string> labels)
        {
            var outputDirInfo = System.IO.Directory.GetParent(filePath);
            if (!System.IO.Directory.Exists(outputDirInfo.FullName))
            {
                System.IO.Directory.CreateDirectory(outputDirInfo.FullName);
            }

            using (var sw = new System.IO.StreamWriter(filePath, false))
            {
                // header
                sw.Write("paths,");
                sw.Write(string.Join(",", labels));
                sw.Write(",index");
                sw.Write(Environment.NewLine);

                // result
                foreach(var item in this.Items)
                {
                    string line = item.GetCsv();
                    sw.WriteLine(line);
                }
            }


            return true;
        }
    }

    public class PredictResultItem
    {
        public List<string> PathList { get; set; }

        public List<float> Predicts { get; set; }

        public int TopIndex { get; set; }


        public string GetCsv()
        {
            var sb = new StringBuilder();

            sb.Append(string.Join("|", this.PathList));

            sb.Append(',');

            sb.Append(string.Join(",", this.Predicts));

            sb.Append(',');

            sb.Append(this.TopIndex);

            return sb.ToString();
        }
    }
}

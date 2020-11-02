using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowClassification
{
    public class ExampleConfig
    {
        public int Priority { get; set; } = 100;

        /// <summary>
        /// Example name
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Command line arguments
        /// </summary>
        public Arguments Args { get; set; }

    }
}

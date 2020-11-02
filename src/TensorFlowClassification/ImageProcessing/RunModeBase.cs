using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;
using SharpCV;
using static SharpCV.Binding;

namespace TensorFlowClassification
{
    public class RunModeBase
    {
        public ExampleConfig Config { get; set; }

        public virtual Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public virtual Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public virtual void Train()
        {
            
        }

        public virtual void Test()
        {

        }

        public virtual void Predict()
        {
        }

        public virtual string FreezeModel()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Read image files
        /// </summary>
        /// <param name="file_names"></param>
        /// <param name="input_height"></param>
        /// <param name="input_width"></param>
        /// <param name="input_std"></param>
        /// <returns></returns>
        protected NDArray ReadTensorFromImageFile(string[] file_names,
                                int input_height = 112,
                                int input_width = 112,
                                int input_std = 1)
        {
            NDArray[] mats = new NDArray[file_names.Length];
            int channels = 0;
            for (int i = 0; i < file_names.Length; i++)
            {
                if (!System.IO.File.Exists(file_names[i]))
                {
                    throw new FileNotFoundException($"File {file_names[i]} was not found.");
                }
                Mat original_image = cv2.imread(file_names[i]);
                
                if (original_image.size == 0)
                {
                    throw new FileNotFoundException($"File {file_names[i]} was not load.");
                }

                channels += original_image.Channels;

                Mat resized = cv2.resize(original_image, (input_height, input_width));
                mats[i] = np.array(resized);
            }
            var image_array_byte = np.concatenate(mats, 2);
            var image_array = image_array_byte / 255.0;

            // 一つのファイルだけの場合
            var reshaped = np.reshape(image_array, (1, input_height, input_width, channels));
            return reshaped;
        }

    }
}

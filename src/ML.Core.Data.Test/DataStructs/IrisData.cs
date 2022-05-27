using System;
using MathNet.Numerics.Random;
using ML.Core.Data.Loader;
using NumSharp;

namespace ML.Core.Data.Test.DataStructs
{
    [Serializable]
    public class IrisData : DataView
    {
        [LoadColumn(0)] public double Label;

        [LoadColumn(3)] public double PetalLength;

        [LoadColumn(4)] public double PetalWidth;

        [LoadColumn(1)] public double SepalLength;

        [LoadColumn(2)] public double SepalWidth;


        public override NDArray GetFeatureArray()
        {
            return np.array(PetalLength, PetalWidth, SepalLength, SepalWidth);
        }

        public override NDArray GetLabelArray()
        {
            return np.array(Label);
        }


        public override string ToString()
        {
            return
                $"Label:{Label}\t" +
                $"SepalLength:{SepalLength:F2}\tSepalWidth:{SepalWidth:F2}\t" +
                $"PetalLength:{PetalLength:F2}\tPetalWidth:{PetalWidth:F2}";
        }

        /// <summary>
        ///     return a random Iris
        /// </summary>
        /// <returns></returns>
        public static IrisData RandomIris()
        {
            var randomSource = SystemRandomSource.Default;
            return new IrisData
            {
                Label = randomSource.Next(0, 3),
                PetalLength = randomSource.NextDouble() * 4,
                PetalWidth = randomSource.NextDouble() * 4,
                SepalLength = randomSource.NextDouble() * 4,
                SepalWidth = randomSource.NextDouble() * 4
            };
        }
    }
}
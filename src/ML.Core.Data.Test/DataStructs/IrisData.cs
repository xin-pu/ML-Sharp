using System;
using MathNet.Numerics.Random;
using ML.Core.Data.Loader;

namespace ML.Core.Data.Test.DataStructs
{
    [Serializable]
    public class IrisData
    {
        [LoadColumn(0)] [LoadType(LoadType.Label)]
        public double Label;

        [LoadColumn(3)] [LoadType(LoadType.Feature)]
        public double PetalLength;

        [LoadColumn(3)] [LoadType(LoadType.Feature)]
        public double PetalWidth;

        [LoadColumn(3)] [LoadType(LoadType.Feature)]
        public double SepalLength;

        [LoadColumn(3)] [LoadType(LoadType.Feature)]
        public double SepalWidth;

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
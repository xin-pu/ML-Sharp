using System;
using MathNet.Numerics.Random;
using Numpy;

namespace ML.Core.Test.DataStructs
{
    [Serializable]
    public class IrisDataOneHot : IrisData
    {
        public override NDarray GetLabelArray()
        {
            var array = new double[3];
            array[(int) Label] = 1;
            return np.array(array);
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
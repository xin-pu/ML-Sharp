using ML.Core.Data.Loader;
using Numpy;

namespace ML.Core.Data.DataStructs
{
    public class LinearData : DataView
    {
        [LoadColumn(0)] public double X;
        [LoadColumn(1)] public double Y;

        public override NDarray GetFeatureArray()
        {
            return np.array(X);
        }

        public override NDarray GetLabelArray()
        {
            return np.array(Y);
        }


        public override string ToString()
        {
            return $"X:{X:F4},Y:{Y:F4}";
        }
    }
}
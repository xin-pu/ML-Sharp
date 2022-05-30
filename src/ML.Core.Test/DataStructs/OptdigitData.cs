using System;
using ML.Core.Data;
using ML.Core.Data.Loader;
using Numpy;

namespace ML.Core.Test.DataStructs
{
    [Serializable]
    public class OptdigitData : DataView
    {
        [LoadColumn(64)] public double Label;

        [LoadColumn(0, 63)] public double[] Pixel;


        public override string ToString()
        {
            return
                $"Label:{Label}\tPixel:{string.Join(',', Pixel)}";
        }


        public override NDarray GetFeatureArray()
        {
            return np.array(Pixel);
        }

        public override NDarray GetLabelArray()
        {
            return np.array(Label);
        }
    }
}
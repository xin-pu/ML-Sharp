using System;
using ML.Core.Data.Loader;
using NumSharp;

namespace ML.Core.Data.Test.DataStructs
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


        public override NDArray GetFeatureArray()
        {
            return np.array(Pixel);
        }

        public override NDArray GetLabelArray()
        {
            return np.array(Label);
        }
    }
}
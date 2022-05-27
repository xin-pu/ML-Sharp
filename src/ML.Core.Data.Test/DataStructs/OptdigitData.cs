using System;
using ML.Core.Data.Loader;

namespace ML.Core.Data.Test.DataStructs
{
    [Serializable]
    public class OptdigitData
    {
        [LoadColumn(64)] public double Label;

        [LoadColumn(0, 63)] public double[] Pixel;


        public override string ToString()
        {
            return
                $"Label:{Label}\tPixel:{string.Join(',', Pixel)}";
        }
    }
}
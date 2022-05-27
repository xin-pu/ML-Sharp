using System;
using NumSharp;

namespace ML.Core.Data
{
    [Serializable]
    public abstract class DataView
    {
        public abstract NDArray GetFeatureArray();

        public abstract NDArray GetLabelArray();


        public DatasetNDArray ToDatasetNdArray()
        {
            return new DatasetNDArray
            {
                Feature = GetFeatureArray(),
                Label = GetLabelArray()
            };
        }
    }
}
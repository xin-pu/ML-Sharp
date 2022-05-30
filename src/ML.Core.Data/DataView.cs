using System;
using Numpy;

namespace ML.Core.Data
{
    [Serializable]
    public abstract class DataView
    {
        public abstract NDarray GetFeatureArray();

        public abstract NDarray GetLabelArray();


        public DatasetNDarray ToDatasetNDarray()
        {
            return new DatasetNDarray
            {
                Feature = GetFeatureArray(),
                Label = GetLabelArray()
            };
        }
    }
}
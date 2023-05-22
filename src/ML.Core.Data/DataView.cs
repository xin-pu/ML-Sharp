﻿using CommunityToolkit.Mvvm.ComponentModel;
using Numpy;

namespace ML.Core.Data
{
    [Serializable]
    public abstract class DataView : ObservableObject
    {
        public int GetFeatures => GetFeatureArray().size;

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

        public NDarray ToFeatureNDarray()
        {
            return np.hstack(GetFeatureArray(), GetLabelArray());
        }
    }
}
using CommunityToolkit.Mvvm.ComponentModel;
using Numpy;

namespace ML.Core.Data
{
    public abstract class DataView : ObservableObject
    {
        public abstract NDarray GetFeatureArray();

        public abstract NDarray GetLabelArray();

        public int GetFeatures => GetFeatureArray().size;

        public NDarray ToFeatureNDarray()
        {
            return np.hstack(GetFeatureArray(), GetLabelArray());
        }

        public DatasetNDarray ToDatasetNDarray()
        {
            return new DatasetNDarray(GetFeatureArray(), GetLabelArray());
        }
    }
}